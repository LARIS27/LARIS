from __future__ import annotations

import json
import hashlib
import hmac
import math
import os
import stat
import sys
import threading
import time
import uuid
from collections.abc import Generator, Mapping
from dataclasses import dataclass
from typing import Callable, Literal, Protocol

from laris_onpremise.kv_state.acquisition.contracts import (
    RuntimeStatePayloadCommit,
)
from laris_onpremise.kv_state.acquisition.acquirer import (
    AcquisitionBindingError,
    AcquisitionObserverIntegrityError,
    AcquisitionStabilityError,
    AcquisitionTargetConflict,
)
from laris_onpremise.attribution.proxy.coordinator import AcquisitionCoordinationError
from laris_onpremise.kv_state.acquisition.reader import MemoryReadError
from laris_onpremise.kv_state.evidence.store import (
    PayloadStoreError,
    PayloadStoreRootIntegrityError,
)
from laris_onpremise.attribution.proxy.binding_ledger import LedgerError
from laris_onpremise.attribution.proxy.lease_controller import SequenceLeaseController
from laris_onpremise.runtime.llama_server_client import (
    CompletionStreamEvent,
    HttpSendEvidence,
    LlamaServerClient,
)
from laris_onpremise.core.records import (
    JsonValue,
    IdentityKeyRegistry,
    LeaseRecord,
    LeaseState,
    RecordValidationError,
    RequestIdentity,
    derive_user_hmac,
    new_request_ref,
)
from laris_onpremise.attribution.proxy.sequence_lease import LeaseTransitionError
from laris_onpremise.kv_state.observation.runtime_state_client import RuntimeStateClientError
from laris_onpremise.kv_state.observation.runtime_state_repository import (
    RuntimeStateObservationError,
)


class ProxyError(RuntimeError):
    """Base class for LARIS Proxy failures."""


class IngressValidationError(ProxyError, ValueError):
    """Raised when a client request is outside the strict P1 schema."""


class AdmissionClosedError(ProxyError):
    """Raised after durable evidence failure closes new admission."""


class AdmissionDrainTimeout(ProxyError, TimeoutError):
    """Raised when admitted requests do not drain before the deadline."""


class RuntimeStatePayloadVerificationError(ProxyError):
    """Raised when a returned selective payload file is not the sealed commit."""


class _AcquisitionCoordinator(Protocol):
    def capture(
        self,
        lease: LeaseRecord,
        *,
        on_copy_sealed: Callable[[LeaseRecord], None] | None = None,
    ) -> RuntimeStatePayloadCommit: ...


def _is_infrastructure_fatal_failure(error: BaseException) -> bool:
    return isinstance(error, (LedgerError, PayloadStoreRootIntegrityError))


def _pipeline_failure_reason(error: BaseException) -> str:
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        current = current.__cause__ or current.__context__
    classifications = (
        (RuntimeStateClientError, "runtime_state_socket_failure"),
        (RuntimeStateObservationError, "runtime_state_observation_failure"),
        (MemoryReadError, "process_memory_read_failure"),
        (AcquisitionBindingError, "acquisition_binding_failure"),
        (AcquisitionTargetConflict, "acquisition_target_conflict"),
        (AcquisitionObserverIntegrityError, "observer_integrity_failure"),
        (AcquisitionStabilityError, "acquisition_stability_failure"),
        (AcquisitionCoordinationError, "acquisition_coordination_failure"),
        (PayloadStoreRootIntegrityError, "payload_store_root_integrity_failure"),
        (PayloadStoreError, "payload_store_failure"),
        (RuntimeStatePayloadVerificationError, "payload_verification_failure"),
        (LedgerError, "ledger_failure"),
    )
    for candidate in reversed(chain):
        for failure_type, reason in classifications:
            if isinstance(candidate, failure_type):
                return reason
    return "unexpected_failure"


def _report_pipeline_failure(pipeline: str, error: BaseException) -> None:
    print(
        f"laris-proxy {pipeline} pipeline failed: "
        f"reason={_pipeline_failure_reason(error)}",
        file=sys.stderr,
        flush=True,
    )


_BINDING_FIELDS = frozenset(
    {
        "id_slot",
        "lease_id",
        "request_ref",
        "user_hmac",
        "identity_key_id",
        "slot_epoch",
        "seq_id",
        "expected_seq_id",
        "observed_seq_id",
        "runtime_instance_id",
        "context_ref",
    }
)

_ALLOWED_FIELDS = frozenset(
    {
        "prompt",
        "n_predict",
        "temperature",
        "top_k",
        "top_p",
        "min_p",
        "seed",
        "stop",
        "stream",
        "cache_prompt",
        "repeat_penalty",
        "presence_penalty",
        "frequency_penalty",
    }
)

_CHAT_ALLOWED_FIELDS = frozenset(
    {
        "messages",
        "max_tokens",
        "temperature",
        "top_k",
        "top_p",
        "min_p",
        "seed",
        "stream",
    }
)


def _finite_number(value: object, field: str, *, minimum: float) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise IngressValidationError(f"{field} must be a number")
    if isinstance(value, float) and not math.isfinite(value):
        raise IngressValidationError(f"{field} must be finite")
    if value < minimum:
        raise IngressValidationError(f"{field} must be at least {minimum}")
    return value


def sanitize_completion_request(
    payload: Mapping[str, object],
) -> dict[str, JsonValue]:
    if not isinstance(payload, Mapping):
        raise IngressValidationError("request must be a JSON object")
    filtered = {
        key: value for key, value in payload.items() if key not in _BINDING_FIELDS
    }
    unsupported = sorted(set(filtered) - _ALLOWED_FIELDS)
    if unsupported:
        raise IngressValidationError(
            f"unsupported completion field: {unsupported[0]}"
        )
    prompt = filtered.get("prompt")
    if not isinstance(prompt, str):
        raise IngressValidationError("prompt must be a text string")

    cleaned: dict[str, JsonValue] = {"prompt": prompt}
    if "n_predict" in filtered:
        n_predict = filtered["n_predict"]
        if (
            isinstance(n_predict, bool)
            or not isinstance(n_predict, int)
            or (n_predict != -1 and n_predict <= 0)
        ):
            raise IngressValidationError(
                "n_predict must be -1 or a positive integer"
            )
        cleaned["n_predict"] = n_predict
    for field in (
        "temperature",
        "top_p",
        "min_p",
        "repeat_penalty",
        "presence_penalty",
        "frequency_penalty",
    ):
        if field in filtered:
            cleaned[field] = _finite_number(filtered[field], field, minimum=0.0)
    if "top_k" in filtered:
        top_k = filtered["top_k"]
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
            raise IngressValidationError("top_k must be a non-negative integer")
        cleaned["top_k"] = top_k
    if "seed" in filtered:
        seed = filtered["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise IngressValidationError("seed must be an integer")
        cleaned["seed"] = seed
    if "stop" in filtered:
        stop = filtered["stop"]
        if isinstance(stop, str):
            cleaned["stop"] = stop
        elif isinstance(stop, list) and all(
            isinstance(item, str) for item in stop
        ):
            cleaned["stop"] = list(stop)
        else:
            raise IngressValidationError(
                "stop must be a string or an array of strings"
            )
    if "stream" in filtered:
        stream = filtered["stream"]
        if not isinstance(stream, bool):
            raise IngressValidationError("stream must be a boolean")
        cleaned["stream"] = stream
    cleaned["cache_prompt"] = False
    return cleaned


def sanitize_chat_request(
    payload: Mapping[str, object],
) -> dict[str, JsonValue]:
    if not isinstance(payload, Mapping):
        raise IngressValidationError("request must be a JSON object")
    filtered = {
        key: value for key, value in payload.items() if key not in _BINDING_FIELDS
    }
    unsupported = sorted(set(filtered) - _CHAT_ALLOWED_FIELDS)
    if unsupported:
        raise IngressValidationError(f"unsupported chat field: {unsupported[0]}")
    raw_messages = filtered.get("messages")
    if (
        not isinstance(raw_messages, list)
        or not raw_messages
        or len(raw_messages) > 512
    ):
        raise IngressValidationError(
            "messages must be a non-empty array with at most 512 entries"
        )
    messages: list[JsonValue] = []
    for raw_message in raw_messages:
        if not isinstance(raw_message, Mapping) or set(raw_message) != {
            "role",
            "content",
        }:
            raise IngressValidationError("chat message field set is invalid")
        role = raw_message.get("role")
        content = raw_message.get("content")
        if role not in {"system", "user", "assistant"}:
            raise IngressValidationError("chat message role is invalid")
        if not isinstance(content, str) or not content:
            raise IngressValidationError("chat message content must be non-empty text")
        messages.append({"role": role, "content": content})

    cleaned: dict[str, JsonValue] = {"messages": messages}
    if "max_tokens" in filtered:
        maximum = filtered["max_tokens"]
        if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum <= 0:
            raise IngressValidationError("max_tokens must be a positive integer")
        cleaned["max_tokens"] = maximum
    for field in ("temperature", "top_p", "min_p"):
        if field in filtered:
            cleaned[field] = _finite_number(filtered[field], field, minimum=0.0)
    if "top_k" in filtered:
        top_k = filtered["top_k"]
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
            raise IngressValidationError("top_k must be a non-negative integer")
        cleaned["top_k"] = top_k
    if "seed" in filtered:
        seed = filtered["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise IngressValidationError("seed must be an integer")
        cleaned["seed"] = seed
    if "stream" in filtered:
        stream = filtered["stream"]
        if not isinstance(stream, bool):
            raise IngressValidationError("stream must be a boolean")
        cleaned["stream"] = stream
    # Selective acquisition requires each Lease interval to contain a complete
    # sequence rewrite.  llama.cpp chat endpoints otherwise reuse a prior slot
    # prefix, leaving only the suffix observable inside a reused-slot interval.
    cleaned["cache_prompt"] = False
    return cleaned


@dataclass(frozen=True)
class ProxyCompletionResult:
    request_ref: str
    lease_id: str
    slot_id: int
    user_hmac: str
    identity_key_id: str
    content: str
    tokens_evaluated: int | None = None
    tokens_predicted: int | None = None


@dataclass(frozen=True)
class ProxyStreamEvent:
    slot_id: int
    content: str
    terminal: bool


@dataclass(frozen=True)
class ProxyStreamSession:
    request_ref: str
    lease_id: str
    slot_id: int
    slot_epoch: int
    seq_id: int
    user_hmac: str
    identity_key_id: str
    events: _TrackedStreamEvents


@dataclass(frozen=True)
class AdmissionDrainEvidence:
    closed_at_monotonic_ns: int
    drained_at_monotonic_ns: int
    active_request_count: int


class _TrackedStreamEvents:
    def __init__(
        self,
        events: Generator[ProxyStreamEvent, None, None],
        *,
        on_abandon: Callable[[], None],
        on_finish: Callable[[], None],
    ) -> None:
        self._events = events
        self._on_abandon = on_abandon
        self._on_finish = on_finish
        self._started = False
        self._finished = False

    def __iter__(self) -> _TrackedStreamEvents:
        return self

    def __next__(self) -> ProxyStreamEvent:
        self._started = True
        try:
            return next(self._events)
        except BaseException:
            self._finish_once()
            raise

    def close(self) -> None:
        try:
            if not self._started:
                self._on_abandon()
            self._events.close()
        finally:
            self._finish_once()

    def _finish_once(self) -> None:
        if self._finished:
            return
        self._finished = True
        self._on_finish()


class LarisProxyCore:
    @staticmethod
    def _validate_capture_policy(
        capture_policy: str,
        controller: SequenceLeaseController,
    ) -> None:
        if capture_policy not in {"FULL", "CONTROL_NO_KV"}:
            raise ValueError("unsupported Proxy capture policy")
        if capture_policy == "CONTROL_NO_KV" and not controller.allow_capture_skip:
            raise ValueError("control capture policy requires capture skip support")

    def __init__(
        self,
        *,
        controller: SequenceLeaseController,
        upstream: LlamaServerClient,
        acquisition_coordinator: _AcquisitionCoordinator,
        identity_key_registry: IdentityKeyRegistry,
        identity_key_id: str,
        tenant_id: str,
        queue_timeout_seconds: float,
        slot_idle_timeout_seconds: float,
        phase_sink: Callable[[str, int, int], None] | None = None,
        capture_policy: str = "FULL",
    ) -> None:
        if queue_timeout_seconds <= 0 or slot_idle_timeout_seconds <= 0:
            raise ValueError("Proxy timeouts must be positive")
        if not callable(getattr(acquisition_coordinator, "capture", None)):
            raise ValueError("acquisition_coordinator must implement capture")
        if not isinstance(identity_key_registry, IdentityKeyRegistry):
            raise ValueError("identity_key_registry must be an IdentityKeyRegistry")
        identity_key_registry.resolve(identity_key_id)
        self._validate_capture_policy(capture_policy, controller)
        self.controller = controller
        self.upstream = upstream
        self.acquisition_coordinator = acquisition_coordinator
        self.identity_key_registry = identity_key_registry
        self.identity_key_id = identity_key_id
        self.tenant_id = tenant_id
        self.queue_timeout_seconds = queue_timeout_seconds
        self.slot_idle_timeout_seconds = slot_idle_timeout_seconds
        self.phase_sink = phase_sink
        self.capture_policy = capture_policy
        self._admission_condition = threading.Condition()
        self._admission_open = True
        self._admission_closed_at_monotonic_ns: int | None = None
        self._active_request_count = 0

    def configure_capture_policy(self, capture_policy: str) -> None:
        """Select an E6 policy only before the first request is admitted."""

        self._validate_capture_policy(capture_policy, self.controller)
        with self._admission_condition:
            if (
                not self._admission_open
                or self._active_request_count
                or self.controller.ledger.read_events()
            ):
                raise RuntimeError(
                    "Proxy capture policy cannot change after admission begins"
                )
            self.capture_policy = capture_policy

    @property
    def admission_open(self) -> bool:
        with self._admission_condition:
            return self._admission_open

    @property
    def shutdown_ready(self) -> bool:
        with self._admission_condition:
            return not self._admission_open and self._active_request_count == 0

    def _close_admission(self) -> None:
        with self._admission_condition:
            if self._admission_open:
                self._admission_open = False
                self._admission_closed_at_monotonic_ns = time.monotonic_ns()
            self._admission_condition.notify_all()

    def _begin_request(self) -> None:
        with self._admission_condition:
            if not self._admission_open:
                raise AdmissionClosedError("new admission is closed")
            self._active_request_count += 1

    def _finish_request(self) -> None:
        with self._admission_condition:
            if self._active_request_count <= 0:
                raise RuntimeError("active request accounting underflow")
            self._active_request_count -= 1
            self._admission_condition.notify_all()

    def close_admission(self, *, deadline: float) -> AdmissionDrainEvidence:
        with self._admission_condition:
            if self._admission_open:
                self._admission_open = False
                self._admission_closed_at_monotonic_ns = time.monotonic_ns()
            assert self._admission_closed_at_monotonic_ns is not None
            while self._active_request_count:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise AdmissionDrainTimeout(
                        "active requests did not drain before the deadline"
                    )
                self._admission_condition.wait(timeout=remaining)
            return AdmissionDrainEvidence(
                closed_at_monotonic_ns=self._admission_closed_at_monotonic_ns,
                drained_at_monotonic_ns=time.monotonic_ns(),
                active_request_count=self._active_request_count,
            )

    @staticmethod
    def parse_ingress_body(raw_body: bytes) -> dict[str, object]:
        def reject_duplicates(
            pairs: list[tuple[str, object]],
        ) -> dict[str, object]:
            result: dict[str, object] = {}
            for key, value in pairs:
                if key in result:
                    raise IngressValidationError(
                        f"duplicate JSON key: {key}"
                    )
                result[key] = value
            return result

        try:
            payload = json.loads(raw_body, object_pairs_hook=reject_duplicates)
        except IngressValidationError:
            raise
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise IngressValidationError("request body is not valid JSON") from error
        if not isinstance(payload, dict):
            raise IngressValidationError("request must be a JSON object")
        return payload

    def _prepare(
        self,
        *,
        sanitizer: Callable[[Mapping[str, object]], dict[str, JsonValue]],
        asserted_subject: str,
        auth_receipt_id: str,
        session_ref: str,
        connection_ref: str,
        retry_of: str | None,
        raw_body: bytes,
    ) -> tuple[RequestIdentity, dict[str, JsonValue], LeaseRecord]:
        payload = sanitizer(self.parse_ingress_body(raw_body))
        try:
            user_hmac = derive_user_hmac(
                registry=self.identity_key_registry,
                identity_key_id=self.identity_key_id,
                tenant_id=self.tenant_id,
                normalized_authenticated_subject_id=asserted_subject,
            )
            request = RequestIdentity(
                user_hmac=user_hmac,
                identity_key_id=self.identity_key_id,
                auth_receipt_id=auth_receipt_id,
                session_ref=session_ref,
                request_ref=new_request_ref(),
                connection_ref=connection_ref,
                retry_of=retry_of,
            )
        except RecordValidationError as error:
            raise IngressValidationError("invalid identity receipt") from error
        try:
            self.controller.enqueue(request)
            lease = self.controller.wait_for_lease(
                request.request_ref,
                deadline=time.monotonic() + self.queue_timeout_seconds,
            )
            self.controller.authorize_dispatch(
                lease.lease_id, f"dispatch:{uuid.uuid4().hex}"
            )
        except LedgerError:
            self._close_admission()
            raise
        return request, payload, lease

    def _mark_actual_send(
        self, lease_id: str, evidence: HttpSendEvidence
    ) -> None:
        self.controller.mark_submitted(
            lease_id,
            evidence.evidence_ref,
            monotonic_ns=evidence.monotonic_ns,
        )

    def _finish_runtime_state_capture(self, lease_id: str) -> LeaseRecord:
        lease = self.controller.get_lease(lease_id)
        release = self.upstream.wait_for_slot_idle(
            lease.expected_slot_id,
            deadline=time.monotonic() + self.slot_idle_timeout_seconds,
        )
        self.controller.confirm_slot_idle(lease_id, release.evidence_ref)
        if self.capture_policy == "CONTROL_NO_KV":
            self.controller.mark_capture_skipped(
                lease_id,
                reason="e6-control-no-kv",
            )
            return self.controller.release(lease_id)
        capture_id = f"capture_{uuid.uuid4().hex}"
        capturing = self.controller.begin_capture(
            lease_id, capture_id
        )
        capturing = self.controller.record_acquisition_requested(
            lease_id,
            capture_id=capture_id,
            observed_seq_id=capturing.expected_seq_id,
            evidence_ref=f"proxy-runtime-state-capture:{capture_id}",
        )
        def record_copy_sealed(copied_lease: LeaseRecord) -> None:
            if (
                copied_lease.lease_id != lease_id
                or copied_lease.capture_id != capture_id
            ):
                raise AcquisitionCoordinationError(
                    "verified copy callback does not match active capture"
                )
            self.controller.record_acquisition_copied(
                lease_id,
                capture_id=capture_id,
                evidence_ref=f"proxy-runtime-state-copy:{capture_id}",
            )

        commit = self.acquisition_coordinator.capture(
            capturing,
            on_copy_sealed=record_copy_sealed,
        )
        _validate_runtime_state_payload_binding(commit, capturing)
        started_ns = time.monotonic_ns() if self.phase_sink is not None else 0
        _verify_runtime_state_payload_file(commit)
        if self.phase_sink is not None:
            self.phase_sink(
                "payload_core_rehash",
                time.monotonic_ns() - started_ns,
                commit.payload_size_bytes,
            )
        self.controller.record_acquisition_completed(
            lease_id,
            capture_id=commit.capture_id,
            request_ref=commit.request_ref,
            runtime_instance_id=commit.runtime_instance_id,
            slot_id=commit.slot_id,
            slot_epoch=commit.slot_epoch,
            observed_seq_id=commit.observed_seq_id,
            digest=commit.payload_digest,
            size_bytes=commit.payload_size_bytes,
            evidence_ref=commit.acquisition_evidence_ref,
        )
        self.controller.commit_runtime_state_payload(
            lease_id,
            capture_id=commit.capture_id,
            digest=commit.payload_digest,
            size_bytes=commit.payload_size_bytes,
            evidence_ref=commit.acquisition_evidence_ref,
        )
        return self.controller.release(lease_id)

    def _quarantine_after_failure(self, lease_id: str, reason: str) -> None:
        try:
            lease = self.controller.get_lease(lease_id)
        except KeyError:
            return
        if lease.state in {
            LeaseState.RELEASED,
            LeaseState.UNKNOWN,
            LeaseState.QUARANTINED,
        }:
            return
        try:
            self.controller.quarantine(lease_id, reason=reason)
        except LedgerError:
            self._close_admission()
            return
        except LeaseTransitionError:
            return

    def handle_completion(
        self,
        *,
        asserted_subject: str,
        auth_receipt_id: str,
        session_ref: str,
        connection_ref: str,
        retry_of: str | None = None,
        raw_body: bytes,
    ) -> ProxyCompletionResult:
        self._begin_request()
        try:
            request, payload, lease = self._prepare(
                sanitizer=sanitize_completion_request,
                asserted_subject=asserted_subject,
                auth_receipt_id=auth_receipt_id,
                session_ref=session_ref,
                connection_ref=connection_ref,
                retry_of=retry_of,
                raw_body=raw_body,
            )
            if payload.get("stream") is True:
                self._quarantine_after_failure(
                    lease.lease_id, "stream_request_sent_to_nonstream_handler"
                )
                raise IngressValidationError(
                    "streaming request requires the stream handler"
                )
            upstream_payload = dict(payload)
            upstream_payload["id_slot"] = lease.expected_slot_id
            try:
                result = self.upstream.complete(
                    upstream_payload,
                    on_request_sent=lambda evidence: self._mark_actual_send(
                        lease.lease_id, evidence
                    ),
                )
                self.controller.confirm_active(
                    lease.lease_id,
                    response_confirmed_slot_id=result.id_slot,
                    evidence_ref=result.response_evidence_ref,
                )
                self.controller.begin_quiescing(
                    lease.lease_id,
                    reason="completed",
                    evidence_ref=result.response_evidence_ref,
                )
                self._finish_runtime_state_capture(lease.lease_id)
                return ProxyCompletionResult(
                    request_ref=request.request_ref,
                    lease_id=lease.lease_id,
                    slot_id=result.id_slot,
                    user_hmac=request.user_hmac,
                    identity_key_id=request.identity_key_id,
                    content=result.content,
                    tokens_evaluated=result.tokens_evaluated,
                    tokens_predicted=result.tokens_predicted,
                )
            except Exception as error:
                _report_pipeline_failure("completion", error)
                if _is_infrastructure_fatal_failure(error):
                    self._close_admission()
                self._quarantine_after_failure(
                    lease.lease_id, "completion_pipeline_failure"
                )
                raise
        finally:
            self._finish_request()

    @staticmethod
    def _apply_stream_event(
        controller: SequenceLeaseController,
        lease_id: str,
        event: CompletionStreamEvent,
        *,
        active: bool,
        terminal: bool,
    ) -> tuple[bool, bool]:
        if not active and event.id_slot >= 0:
            controller.confirm_active(
                lease_id,
                response_confirmed_slot_id=event.id_slot,
                evidence_ref=event.response_evidence_ref,
            )
            active = True
        if event.terminal:
            if terminal:
                raise IngressValidationError("duplicate terminal stream event")
            controller.begin_quiescing(
                lease_id,
                reason="completed",
                evidence_ref=event.response_evidence_ref,
            )
            terminal = True
        return active, terminal

    def _stream_generator(
        self,
        *,
        lease: LeaseRecord,
        payload: dict[str, JsonValue],
        request_kind: Literal["completion", "chat"],
    ) -> Generator[ProxyStreamEvent, None, None]:
        upstream_payload = dict(payload)
        upstream_payload["stream"] = True
        upstream_payload["id_slot"] = lease.expected_slot_id
        callback = lambda evidence: self._mark_actual_send(lease.lease_id, evidence)
        if request_kind == "chat":
            upstream_events = self.upstream.stream_chat_completion(
                upstream_payload,
                on_request_sent=callback,
            )
        else:
            upstream_events = self.upstream.stream_completion(
                upstream_payload,
                on_request_sent=callback,
            )
        active = False
        terminal = False
        terminal_event: ProxyStreamEvent | None = None
        try:
            try:
                for event in upstream_events:
                    active, terminal = self._apply_stream_event(
                        self.controller,
                        lease.lease_id,
                        event,
                        active=active,
                        terminal=terminal,
                    )
                    proxy_event = ProxyStreamEvent(
                        slot_id=event.id_slot,
                        content=event.content,
                        terminal=event.terminal,
                    )
                    if event.terminal:
                        terminal_event = proxy_event
                    else:
                        yield proxy_event
            except GeneratorExit:
                for event in upstream_events:
                    active, terminal = self._apply_stream_event(
                        self.controller,
                        lease.lease_id,
                        event,
                        active=active,
                        terminal=terminal,
                    )
                if not terminal:
                    raise IngressValidationError(
                        "stream disconnect drain did not reach terminal"
                    )
                self._finish_runtime_state_capture(lease.lease_id)
                raise
            if not terminal:
                raise IngressValidationError(
                    "upstream stream ended without terminal evidence"
                )
            self._finish_runtime_state_capture(lease.lease_id)
            if terminal_event is None:
                raise IngressValidationError(
                    "terminal stream event is unavailable"
                )
            yield terminal_event
        except GeneratorExit:
            raise
        except Exception as error:
            _report_pipeline_failure("stream", error)
            if _is_infrastructure_fatal_failure(error):
                self._close_admission()
            self._quarantine_after_failure(
                lease.lease_id, "stream_pipeline_failure"
            )
            raise

    def start_stream_completion(
        self,
        *,
        asserted_subject: str,
        auth_receipt_id: str,
        session_ref: str,
        connection_ref: str,
        retry_of: str | None = None,
        raw_body: bytes,
    ) -> ProxyStreamSession:
        self._begin_request()
        try:
            request, payload, lease = self._prepare(
                sanitizer=sanitize_completion_request,
                asserted_subject=asserted_subject,
                auth_receipt_id=auth_receipt_id,
                session_ref=session_ref,
                connection_ref=connection_ref,
                retry_of=retry_of,
                raw_body=raw_body,
            )
            if payload.get("stream") is not True:
                self._quarantine_after_failure(
                    lease.lease_id, "nonstream_request_sent_to_stream_handler"
                )
                raise IngressValidationError("stream must be true")
        except BaseException:
            self._finish_request()
            raise
        events = _TrackedStreamEvents(
            self._stream_generator(
                lease=lease,
                payload=payload,
                request_kind="completion",
            ),
            on_abandon=lambda: self._quarantine_after_failure(
                lease.lease_id, "stream_abandoned_before_dispatch"
            ),
            on_finish=self._finish_request,
        )
        return ProxyStreamSession(
            request_ref=request.request_ref,
            lease_id=lease.lease_id,
            slot_id=lease.expected_slot_id,
            slot_epoch=(
                lease.slot_epoch
                if lease.slot_epoch is not None
                else lease.lease_generation
            ),
            seq_id=lease.expected_seq_id,
            user_hmac=request.user_hmac,
            identity_key_id=request.identity_key_id,
            events=events,
        )

    def start_stream_chat(
        self,
        *,
        asserted_subject: str,
        auth_receipt_id: str,
        session_ref: str,
        connection_ref: str,
        retry_of: str | None = None,
        raw_body: bytes,
    ) -> ProxyStreamSession:
        self._begin_request()
        try:
            request, payload, lease = self._prepare(
                sanitizer=sanitize_chat_request,
                asserted_subject=asserted_subject,
                auth_receipt_id=auth_receipt_id,
                session_ref=session_ref,
                connection_ref=connection_ref,
                retry_of=retry_of,
                raw_body=raw_body,
            )
            if payload.get("stream") is not True:
                self._quarantine_after_failure(
                    lease.lease_id, "nonstream_chat_sent_to_stream_handler"
                )
                raise IngressValidationError("stream must be true")
        except BaseException:
            self._finish_request()
            raise
        events = _TrackedStreamEvents(
            self._stream_generator(
                lease=lease,
                payload=payload,
                request_kind="chat",
            ),
            on_abandon=lambda: self._quarantine_after_failure(
                lease.lease_id, "stream_abandoned_before_dispatch"
            ),
            on_finish=self._finish_request,
        )
        return ProxyStreamSession(
            request_ref=request.request_ref,
            lease_id=lease.lease_id,
            slot_id=lease.expected_slot_id,
            slot_epoch=(
                lease.slot_epoch
                if lease.slot_epoch is not None
                else lease.lease_generation
            ),
            seq_id=lease.expected_seq_id,
            user_hmac=request.user_hmac,
            identity_key_id=request.identity_key_id,
            events=events,
        )


def _validate_runtime_state_payload_binding(
    commit: RuntimeStatePayloadCommit, lease: LeaseRecord
) -> None:
    expected = {
        "capture_id": lease.capture_id,
        "lease_id": lease.lease_id,
        "request_ref": lease.request.request_ref,
        "runtime_instance_id": lease.runtime_instance_id,
        "slot_id": lease.expected_slot_id,
        "slot_epoch": lease.slot_epoch,
        "observed_seq_id": lease.observed_seq_id,
    }
    for field, value in expected.items():
        if getattr(commit, field) != value:
            raise RuntimeStatePayloadVerificationError(
                f"payload {field} does not match the active Lease"
            )


def _verify_runtime_state_payload_file(commit: RuntimeStatePayloadCommit) -> None:
    path = commit.payload_path
    if path.name != "runtime_state_payload.bin":
        raise RuntimeStatePayloadVerificationError(
            "payload path must end with runtime_state_payload.bin"
        )
    try:
        before = path.lstat()
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise RuntimeStatePayloadVerificationError(
                "payload path must be a regular non-symlink file"
            )
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        try:
            opened = os.fstat(descriptor)
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_dev != before.st_dev
                or opened.st_ino != before.st_ino
            ):
                raise RuntimeStatePayloadVerificationError(
                    "payload file identity changed before verification"
                )
            digest = hashlib.sha256()
            size = 0
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                size += len(chunk)
        finally:
            os.close(descriptor)
    except RuntimeStatePayloadVerificationError:
        raise
    except OSError as error:
        raise RuntimeStatePayloadVerificationError(
            "payload file cannot be verified"
        ) from error
    observed_digest = "sha256:" + digest.hexdigest()
    if size != commit.payload_size_bytes or not hmac.compare_digest(
        observed_digest, commit.payload_digest
    ):
        raise RuntimeStatePayloadVerificationError(
            "payload file does not match the sealed commit"
        )
