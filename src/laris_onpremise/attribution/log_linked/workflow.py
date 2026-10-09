from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time
from typing import Protocol
import urllib.error
import urllib.request

import yaml

from laris_onpremise.bundle.rehost_metadata import RehostRuntimeMetadata
from laris_onpremise.core.records import canonical_json
from laris_onpremise.kv_state.observation.runtime_state_client import RuntimeStateClient
from laris_onpremise.kv_state.observation.runtime_state_profile import (
    load_runtime_state_observer_profile,
)
from laris_onpremise.kv_state.observation.runtime_state_service import (
    RuntimeStateTopologySummary,
)

from .acquisition import (
    LogLinkedStateUnitAcquirer,
)
from .command import LogLinkedCommandError, LogLinkedVerificationError
from .contracts import (
    AttributionDecision,
    SequenceGenerationRef,
    parse_state_unit_manifest,
    parse_writer_capture_receipt,
)
from .correlation import (
    AttributionResult,
    CorrelationInputs,
    EndpointCorrelator,
    RuntimeCorrelationProfile,
)
from .evidence import EvidenceInputs, LogLinkedEvidencePackage
from .export import EndpointBundleExportInputs, EndpointBundleExporter
from .generation import GenerationSeal, SequenceGenerationTracker
from .llama_server_log import LlamaServerLogProfile, LlamaServerTraceParser
from .materialize import EndpointStateMaterializer
from .preflight import (
    ExecutableIdentity,
    LogPrivacyPreflight,
    PreflightRequest,
)
from .recorder import (
    RecorderProgress,
    StateUnitRecorder,
    WriterCaptureStatistics,
)
from .state_units import StateUnitStore, StoredStateUnit


_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,191}\Z")
_RUNTIME_INSTANCE_ID = re.compile(
    r"(?:[A-Za-z0-9][A-Za-z0-9._-]{0,191}|sha256:[0-9a-f]{64})\Z"
)
_PROFILE_FIELDS = {
    "schema_version",
    "adapter_profile",
    "runtime",
    "observer",
    "correlation",
    "rehosting",
}
_RUNTIME_FIELDS = {
    "pid",
    "start_time_ticks",
    "executable",
    "executable_digest",
    "version",
    "application_log",
    "canary_url",
    "canary_authorization_file",
}
_OBSERVER_FIELDS = {
    "socket_path",
    "profile_path",
    "terminal_seal_path",
    "runtime_instance_id",
    "poll_interval_seconds",
    "observation_duration_seconds",
}
_CORRELATION_FIELDS = {
    "slot_sequence_map",
    "maximum_clock_error_ns",
    "maximum_access_delay_ns",
    "maximum_access_pre_terminal_skew_ns",
    "allowed_path_classes",
}
_REHOSTING_FIELDS = {
    "model_profile_id",
    "model_digest",
    "runtime_profile_digest",
    "n_ctx",
    "n_seq_max",
    "runtime_metadata",
}
_ADAPTER_FIELDS = {
    "schema_version",
    "adapter_id",
    "application",
    "upstream_commit",
    "maximum_line_bytes",
    "maximum_log_bytes",
    "required_arguments",
    "forbidden_arguments",
    "privacy_contract",
}
_PRIVACY_CONTRACT = {
    "allow_endpoint_value_in_restricted_evidence": True,
    "allow_request_body": False,
    "allow_response_body": False,
    "allow_prompt_text": False,
    "allow_token_ids": False,
}


@dataclass(frozen=True)
class RuntimeInputs:
    pid: int
    start_time_ticks: int
    executable: Path
    executable_digest: str
    version: str
    application_log: Path
    canary_url: str
    canary_authorization_file: Path | None


@dataclass(frozen=True)
class ObserverInputs:
    socket_path: Path
    profile_path: Path
    terminal_seal_path: Path
    runtime_instance_id: str
    poll_interval_seconds: float
    observation_duration_seconds: float


@dataclass(frozen=True)
class CorrelationConfiguration:
    slot_sequence_map: tuple[tuple[int, int], ...]
    maximum_clock_error_ns: int
    maximum_access_delay_ns: int
    maximum_access_pre_terminal_skew_ns: int
    allowed_path_classes: tuple[str, ...]


@dataclass(frozen=True)
class RehostingConfiguration:
    model_profile_id: str
    model_digest: str
    runtime_profile_digest: str
    n_ctx: int
    n_seq_max: int
    runtime_metadata: RehostRuntimeMetadata


@dataclass(frozen=True)
class LogLinkedWorkflowProfile:
    source_path: Path
    adapter_profile_path: Path
    adapter: LlamaServerLogProfile
    runtime: RuntimeInputs
    observer: ObserverInputs
    correlation: CorrelationConfiguration
    rehosting: RehostingConfiguration
    source_document: Mapping[str, object]


def load_workflow_profile(path: Path | str) -> LogLinkedWorkflowProfile:
    source = _regular_path(Path(path), "workflow profile")
    document = _read_mapping(source, maximum_bytes=256 * 1024)
    _exact(document, _PROFILE_FIELDS, "workflow profile")
    if document["schema_version"] != "1.0":
        raise LogLinkedCommandError("workflow profile schema is unsupported")
    adapter_path = Path(_text(document["adapter_profile"], "adapter_profile"))
    if not adapter_path.is_absolute():
        adapter_path = source.parent / adapter_path
    adapter_document = _read_mapping(adapter_path, maximum_bytes=64 * 1024)
    _exact(adapter_document, _ADAPTER_FIELDS, "adapter profile")
    if (
        adapter_document["schema_version"] != "1.0"
        or adapter_document["application"] != "llama-server"
        or adapter_document["privacy_contract"] != _PRIVACY_CONTRACT
    ):
        raise LogLinkedCommandError("adapter profile privacy contract is invalid")
    adapter = LlamaServerLogProfile(
        adapter_id=_text(adapter_document["adapter_id"], "adapter_id"),
        upstream_commit=_text(
            adapter_document["upstream_commit"], "upstream_commit"
        ),
        maximum_line_bytes=_positive_int(
            adapter_document["maximum_line_bytes"], "maximum_line_bytes"
        ),
        maximum_log_bytes=_positive_int(
            adapter_document["maximum_log_bytes"], "maximum_log_bytes"
        ),
        required_arguments=_text_tuple(
            adapter_document["required_arguments"], "required_arguments"
        ),
        forbidden_arguments=_text_tuple(
            adapter_document["forbidden_arguments"], "forbidden_arguments"
        ),
    )

    runtime_value = _mapping(document["runtime"], "runtime")
    _exact(runtime_value, _RUNTIME_FIELDS, "runtime")
    auth_value = runtime_value["canary_authorization_file"]
    auth_path = (
        None
        if auth_value is None
        else _absolute_path(auth_value, "canary_authorization_file")
    )
    runtime = RuntimeInputs(
        pid=_positive_int(runtime_value["pid"], "runtime.pid"),
        start_time_ticks=_positive_int(
            runtime_value["start_time_ticks"], "runtime.start_time_ticks"
        ),
        executable=_absolute_path(runtime_value["executable"], "runtime.executable"),
        executable_digest=_digest(
            runtime_value["executable_digest"], "runtime.executable_digest"
        ),
        version=_text(runtime_value["version"], "runtime.version"),
        application_log=_absolute_path(
            runtime_value["application_log"], "runtime.application_log"
        ),
        canary_url=_http_url(runtime_value["canary_url"]),
        canary_authorization_file=auth_path,
    )

    observer_value = _mapping(document["observer"], "observer")
    _exact(observer_value, _OBSERVER_FIELDS, "observer")
    observer = ObserverInputs(
        socket_path=_absolute_path(
            observer_value["socket_path"], "observer.socket_path"
        ),
        profile_path=_absolute_path(
            observer_value["profile_path"], "observer.profile_path"
        ),
        terminal_seal_path=_absolute_path(
            observer_value["terminal_seal_path"], "observer.terminal_seal_path"
        ),
        runtime_instance_id=_runtime_instance_id(
            observer_value["runtime_instance_id"], "observer.runtime_instance_id"
        ),
        poll_interval_seconds=_positive_number(
            observer_value["poll_interval_seconds"],
            "observer.poll_interval_seconds",
        ),
        observation_duration_seconds=_positive_number(
            observer_value["observation_duration_seconds"],
            "observer.observation_duration_seconds",
        ),
    )

    correlation_value = dict(
        _mapping(document["correlation"], "correlation")
    )
    correlation_value.setdefault(
        "maximum_access_pre_terminal_skew_ns", 0
    )
    _exact(correlation_value, _CORRELATION_FIELDS, "correlation")
    pairs_value = correlation_value["slot_sequence_map"]
    if not isinstance(pairs_value, list):
        raise LogLinkedCommandError("slot_sequence_map must be a list")
    pairs: list[tuple[int, int]] = []
    for item in pairs_value:
        if not isinstance(item, list) or len(item) != 2:
            raise LogLinkedCommandError("slot_sequence_map contains an invalid pair")
        pairs.append(
            (
                _nonnegative_int(item[0], "slot_id"),
                _nonnegative_int(item[1], "seq_id"),
            )
        )
    slot_sequence_map = tuple(sorted(pairs))
    if not slot_sequence_map or len({item[0] for item in pairs}) != len(pairs):
        raise LogLinkedCommandError("slot_sequence_map is empty or duplicated")
    allowed_paths = tuple(sorted(_text_tuple(
        correlation_value["allowed_path_classes"], "allowed_path_classes"
    )))
    correlation = CorrelationConfiguration(
        slot_sequence_map=slot_sequence_map,
        maximum_clock_error_ns=_nonnegative_int(
            correlation_value["maximum_clock_error_ns"],
            "maximum_clock_error_ns",
        ),
        maximum_access_delay_ns=_nonnegative_int(
            correlation_value["maximum_access_delay_ns"],
            "maximum_access_delay_ns",
        ),
        maximum_access_pre_terminal_skew_ns=_nonnegative_int(
            correlation_value["maximum_access_pre_terminal_skew_ns"],
            "maximum_access_pre_terminal_skew_ns",
        ),
        allowed_path_classes=allowed_paths,
    )

    rehosting_value = _mapping(document["rehosting"], "rehosting")
    _exact(rehosting_value, _REHOSTING_FIELDS, "rehosting")
    runtime_metadata = RehostRuntimeMetadata.from_mapping(
        _mapping(rehosting_value["runtime_metadata"], "runtime_metadata")
    )
    runtime_profile_digest = _digest(
        rehosting_value["runtime_profile_digest"], "runtime_profile_digest"
    )
    if runtime_metadata.source_runtime_profile_digest != runtime_profile_digest:
        raise LogLinkedCommandError("rehosting runtime profile digest differs")
    rehosting = RehostingConfiguration(
        model_profile_id=_safe_id(
            rehosting_value["model_profile_id"], "model_profile_id"
        ),
        model_digest=_digest(rehosting_value["model_digest"], "model_digest"),
        runtime_profile_digest=runtime_profile_digest,
        n_ctx=_positive_int(rehosting_value["n_ctx"], "n_ctx"),
        n_seq_max=_positive_int(rehosting_value["n_seq_max"], "n_seq_max"),
        runtime_metadata=runtime_metadata,
    )
    if rehosting.n_ctx % rehosting.n_seq_max:
        raise LogLinkedCommandError("n_ctx must be divisible by n_seq_max")
    return LogLinkedWorkflowProfile(
        source_path=source,
        adapter_profile_path=adapter_path,
        adapter=adapter,
        runtime=runtime,
        observer=observer,
        correlation=correlation,
        rehosting=rehosting,
        source_document=document,
    )


def run_preflight(profile_path: Path) -> Mapping[str, object]:
    profile = load_workflow_profile(profile_path)
    runtime = profile.runtime
    clock = _sample_runtime_clock(runtime.start_time_ticks)
    baseline_size = _regular_path(
        runtime.application_log, "application log"
    ).stat().st_size
    canary = "LARIS-CONTENT-FREE-CANARY-" + os.urandom(12).hex()
    authorization = "LARIS-NO-AUTHORIZATION-" + os.urandom(12).hex()
    if runtime.canary_authorization_file is not None:
        authorization = _read_secret(runtime.canary_authorization_file)
    response_fragment = _send_canary(
        runtime.canary_url, canary, authorization
    )

    def log_reader() -> bytes:
        deadline = time.monotonic() + 5.0
        while True:
            payload = _read_append_only_snapshot(
                runtime.application_log,
                maximum_bytes=profile.adapter.maximum_log_bytes,
            )
            suffix = payload[baseline_size:]
            if suffix.endswith(b"\n") and b"done request:" in suffix:
                return suffix
            if time.monotonic() >= deadline:
                return suffix
            time.sleep(0.05)

    expected = ExecutableIdentity(
        path=str(runtime.executable),
        digest=runtime.executable_digest,
        version=runtime.version,
    )
    launch_arguments = _process_arguments(runtime.pid, runtime.start_time_ticks)
    receipt = LogPrivacyPreflight(
        LlamaServerTraceParser(
            profile.adapter,
            elapsed_timestamp_origin_utc_ns=int(
                clock["application_timestamp_origin_utc_ns"]
            ),
        )
    ).verify(
        PreflightRequest(
            profile=profile.adapter,
            launch_arguments=launch_arguments,
            expected_identity=expected,
            process_identity_probe=lambda: _process_identity(runtime),
            canary_probe=lambda _text: None,
            log_reader=log_reader,
            canary_text=canary,
            prompt_fragment=canary,
            response_fragment=response_fragment,
            authorization_value=authorization,
        )
    )
    return {
        "schema_version": "1.0",
        "status": "SUPPORTED" if receipt.passed else "UNSUPPORTED",
        "adapter_id": receipt.adapter_id,
        "reason_codes": list(receipt.reasons),
        "application_log_digest": receipt.log_digest,
        "normalized_event_count": receipt.event_count,
        "endpoint_event_count": receipt.endpoint_event_count,
        "runtime_executable_digest": receipt.executable_identity.digest,
        "launch_arguments_digest": receipt.launch_arguments_digest,
    }


class _CompletedApplyClient(Protocol):
    def arm(self) -> None: ...

    def stop(self) -> None: ...


CaptureFactory = Callable[
    [LogLinkedWorkflowProfile, Path, _CompletedApplyClient],
    tuple[object, object, object],
]


class ObservationSession:
    """Bounded observation of an external runtime; it never owns that runtime."""

    def __init__(
        self,
        *,
        profile: Path,
        evidence_root: Path,
        preflight_runner: Callable[[Path], Mapping[str, object]] = run_preflight,
        client_factory: Callable[[Path], _CompletedApplyClient] = RuntimeStateClient,
        capture_factory: CaptureFactory | None = None,
        native_topology: RuntimeStateTopologySummary | None = None,
        wall_clock_ns: Callable[[], int] = time.time_ns,
        monotonic_clock_ns: Callable[[], int] = time.monotonic_ns,
        process_cpu_clock_ns: Callable[[], int] = time.process_time_ns,
        clock_ticks_per_second_supplier: Callable[[], int] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        launch_arguments_supplier: Callable[
            [int, int], tuple[str, ...]
        ] | None = None,
        completion_requested: Callable[[], bool] | None = None,
        terminal_drain: Callable[[], None] | None = None,
        completion_quiescence_polls: int = 1,
    ) -> None:
        if completion_requested is not None and not callable(completion_requested):
            raise TypeError("completion_requested must be callable")
        if terminal_drain is not None and not callable(terminal_drain):
            raise TypeError("terminal_drain must be callable")
        if native_topology is not None and not isinstance(
            native_topology, RuntimeStateTopologySummary
        ):
            raise TypeError("native_topology is invalid")
        if (
            isinstance(completion_quiescence_polls, bool)
            or not isinstance(completion_quiescence_polls, int)
            or completion_quiescence_polls <= 0
        ):
            raise ValueError("completion_quiescence_polls must be positive")
        self.profile_path = profile
        self.evidence_root = evidence_root
        self.preflight_runner = preflight_runner
        self.client_factory = client_factory
        if capture_factory is None:
            self.capture_factory = lambda profile, root, client: _capture_components(
                profile,
                root,
                client,
                native_topology=native_topology,
            )
        else:
            self.capture_factory = capture_factory
        self.wall_clock_ns = wall_clock_ns
        self.monotonic_clock_ns = monotonic_clock_ns
        self.process_cpu_clock_ns = process_cpu_clock_ns
        self.clock_ticks_per_second_supplier = (
            _clock_ticks_per_second
            if clock_ticks_per_second_supplier is None
            else clock_ticks_per_second_supplier
        )
        self.sleep = sleep
        self.launch_arguments_supplier = (
            _process_arguments
            if launch_arguments_supplier is None
            else launch_arguments_supplier
        )
        self.completion_requested = completion_requested
        self.terminal_drain = terminal_drain
        self.completion_quiescence_polls = completion_quiescence_polls
        self.finished = False
        self._profile: LogLinkedWorkflowProfile | None = None
        self._client: _CompletedApplyClient | None = None
        self._tracker: object | None = None
        self._store: object | None = None
        self._recorder: object | None = None
        self._log_start_offset = 0
        self._clock_sample: Mapping[str, object] | None = None
        self._progress: list[RecorderProgress] = []
        self._initial_apply_cursor = 0
        self._observer_cpu_start_ns: int | None = None
        self._terminal_drain_requested = False

    def run(self) -> Mapping[str, object]:
        profile = load_workflow_profile(self.profile_path)
        preflight = self.preflight_runner(self.profile_path)
        if preflight.get("status") != "SUPPORTED":
            return dict(preflight)
        if self.evidence_root.exists() or self.evidence_root.is_symlink():
            raise LogLinkedCommandError("evidence_root already exists")
        self.evidence_root.mkdir(mode=0o700, parents=True)
        _make_private(self.evidence_root, directory=True)
        self._profile = profile
        self._log_start_offset = _regular_path(
            profile.runtime.application_log, "application log"
        ).stat().st_size
        self._clock_sample = _runtime_clock_receipt(
            runtime_start_time_ticks=profile.runtime.start_time_ticks,
            wall_time_utc_ns=self.wall_clock_ns(),
            monotonic_ns=self.monotonic_clock_ns(),
            clock_ticks_per_second=self.clock_ticks_per_second_supplier(),
        )
        portable_profile = dict(profile.source_document)
        portable_profile["adapter_profile"] = "adapter_profile.json"
        adapter_document = _read_mapping(
            profile.adapter_profile_path, maximum_bytes=64 * 1024
        )
        _write_new(
            self.evidence_root / "adapter_profile.json",
            canonical_json(adapter_document) + b"\n",
        )
        _write_new(
            self.evidence_root / "workflow_profile.json",
            canonical_json(portable_profile) + b"\n",
        )
        _write_new(
            self.evidence_root / "preflight_receipt.json",
            canonical_json(dict(preflight)) + b"\n",
        )
        client = self.client_factory(profile.observer.socket_path)
        self._client = client
        tracker, store, recorder = self.capture_factory(
            profile, self.evidence_root, client
        )
        self._tracker = tracker
        self._store = store
        self._recorder = recorder
        prime = getattr(recorder, "prime", None)
        if callable(prime):
            initial_cursor = prime()
            if (
                isinstance(initial_cursor, bool)
                or not isinstance(initial_cursor, int)
                or initial_cursor < 0
            ):
                raise LogLinkedVerificationError(
                    "state-unit recorder returned an invalid initial cursor"
                )
            self._initial_apply_cursor = initial_cursor
        self._observer_cpu_start_ns = self.process_cpu_clock_ns()
        client.arm()
        try:
            started = self.monotonic_clock_ns()
            last_application_log_size = self._log_start_offset
            completion_quiet_polls = 0
            while True:
                progress = recorder.drain_once()
                if not isinstance(progress, RecorderProgress):
                    raise LogLinkedVerificationError(
                        "state-unit recorder returned an invalid progress record"
                    )
                self._progress.append(progress)
                now = self.monotonic_clock_ns()
                if self.completion_requested is not None:
                    completion = self.completion_requested()
                    if not isinstance(completion, bool):
                        raise LogLinkedVerificationError(
                            "observation completion predicate returned a non-boolean"
                        )
                    if (
                        completion
                        and self.terminal_drain is not None
                        and not self._terminal_drain_requested
                    ):
                        self.terminal_drain()
                        self._terminal_drain_requested = True
                        completion_quiet_polls = 0
                        continue
                    application_log_size = _regular_path(
                        profile.runtime.application_log,
                        "application log",
                    ).stat().st_size
                    if (
                        completion
                        and progress.next_cursor == progress.previous_cursor
                        and application_log_size == last_application_log_size
                    ):
                        completion_quiet_polls += 1
                    else:
                        completion_quiet_polls = 0
                    last_application_log_size = application_log_size
                    if (
                        completion_quiet_polls
                        >= self.completion_quiescence_polls
                    ):
                        break
                if (
                    now - started
                    >= int(profile.observer.observation_duration_seconds * 1e9)
                ):
                    if self.completion_requested is not None:
                        raise LogLinkedVerificationError(
                            "observation completion condition timed out"
                        )
                    break
                self.sleep(profile.observer.poll_interval_seconds)
            return self._finish(incomplete=False)
        except (Exception, KeyboardInterrupt):
            if not self.finished:
                try:
                    self._finish(incomplete=True)
                except Exception:
                    pass
            raise

    def interrupt(self) -> Mapping[str, object]:
        if self.finished:
            raise LogLinkedCommandError("observation is already finalized")
        if self._profile is None:
            return {"schema_version": "1.0", "status": "INCOMPLETE"}
        return self._finish(incomplete=True)

    def _finish(self, *, incomplete: bool) -> Mapping[str, object]:
        if self.finished or self._profile is None or self._client is None:
            raise LogLinkedCommandError("observation cannot be finalized")
        profile = self._profile
        tracker = self._tracker
        store = self._store
        state = "INCOMPLETE" if incomplete else "COMPLETE"
        reason_codes: list[str] = []
        try:
            close_recorder = getattr(self._recorder, "close", None)
            if callable(close_recorder):
                close_recorder()
            refs = tuple(tracker.active_refs())  # type: ignore[union-attr]
            for ref in refs:
                if incomplete:
                    tracker.mark_incomplete(  # type: ignore[union-attr]
                        ref, "OBSERVATION_INTERRUPTED"
                    )
                else:
                    tracker.seal(ref)  # type: ignore[union-attr]
            self._client.stop()
            for seal in tracker.seals():  # type: ignore[union-attr]
                store.seal_generation(seal)  # type: ignore[union-attr]
        except Exception:
            state = "INCOMPLETE"
            reason_codes.append("OBSERVER_FINALIZATION_FAILED")
        application_root = self.evidence_root / "application"
        observer_root = self.evidence_root / "observer"
        application_root.mkdir(mode=0o700)
        observer_root.mkdir(mode=0o700)
        _make_private(application_root, directory=True)
        _make_private(observer_root, directory=True)
        raw_log = _read_append_only_snapshot(
            profile.runtime.application_log,
            maximum_bytes=profile.adapter.maximum_log_bytes,
        )
        _write_new(
            application_root / "raw_llama_server.log",
            raw_log[self._log_start_offset :],
        )
        try:
            terminal = _read_regular(profile.observer.terminal_seal_path)
        except LogLinkedCommandError:
            terminal = b""
            state = "INCOMPLETE"
            reason_codes.append("OBSERVER_TERMINAL_SEAL_MISSING")
        _write_new(observer_root / "runtime_state_terminal_seal.json", terminal)
        writer_receipt = _build_writer_capture_receipt(
            profile=profile,
            recorder=self._recorder,
            progress=tuple(self._progress),
            terminal=terminal,
            observer_cpu_time_ns=(
                0
                if self._observer_cpu_start_ns is None
                else max(
                    0,
                    self.process_cpu_clock_ns()
                    - self._observer_cpu_start_ns,
                )
            ),
        )
        writer_receipt_bytes = canonical_json(writer_receipt) + b"\n"
        parse_writer_capture_receipt(writer_receipt_bytes)
        _write_new(
            self.evidence_root / "writer_capture_receipt.json",
            writer_receipt_bytes,
        )
        clock_receipt = dict(self._clock_sample or {})
        clock_bytes = canonical_json(clock_receipt) + b"\n"
        _write_new(self.evidence_root / "clock_receipt.json", clock_bytes)
        arguments = self.launch_arguments_supplier(
            profile.runtime.pid, profile.runtime.start_time_ticks
        )
        manifest = {
            "schema_version": "1.0",
            "state": state,
            "reason_codes": sorted(set(reason_codes)),
            "adapter_id": profile.adapter.adapter_id,
            "runtime_instance_id": profile.observer.runtime_instance_id,
            "runtime_pid": profile.runtime.pid,
            "runtime_start_time_ticks": profile.runtime.start_time_ticks,
            "runtime_executable_digest": profile.runtime.executable_digest,
            "launch_arguments": list(arguments),
            "application_log_file": "application/raw_llama_server.log",
            "observer_terminal_seal_file": (
                "observer/runtime_state_terminal_seal.json"
            ),
            "writer_capture_receipt_file": "writer_capture_receipt.json",
            "writer_capture_receipt_digest": _sha256(writer_receipt_bytes),
            "state_units_root": "state-units",
            "clock_receipt_file": "clock_receipt.json",
            "clock_receipt_digest": _sha256(clock_bytes),
            "drain_count": len(self._progress),
            "initial_apply_cursor": self._initial_apply_cursor,
            "captured_unit_count": sum(
                item.captured_unit_count for item in self._progress
            ),
            "observation_truncated": any(
                item.observation_truncated for item in self._progress
            ),
        }
        _write_new(
            self.evidence_root / "observation_manifest.json",
            canonical_json(manifest) + b"\n",
        )
        self.finished = True
        return {
            "schema_version": "1.0",
            "status": state,
            "evidence_root": str(self.evidence_root),
            "captured_unit_count": manifest["captured_unit_count"],
            "reason_codes": manifest["reason_codes"],
        }


def _build_writer_capture_receipt(
    *,
    profile: LogLinkedWorkflowProfile,
    recorder: object | None,
    progress: tuple[RecorderProgress, ...],
    terminal: bytes,
    observer_cpu_time_ns: int,
) -> dict[str, object]:
    statistics = getattr(recorder, "writer_capture_statistics", None)
    if statistics is None:
        captured = sum(item.captured_unit_count for item in progress)
        statistics = WriterCaptureStatistics(
            apply_count=captured,
            unit_count=captured,
            fragment_count=captured,
            copied_byte_count=0,
            native_ring_frame_high_water_mark=(1 if captured else 0),
            persistence_queue_high_water_mark=(1 if captured else 0),
            incomplete_unit_count=sum(
                item.incomplete_generation_count for item in progress
            ),
            persistence_failure_count=0,
        )
    if not isinstance(statistics, WriterCaptureStatistics):
        raise LogLinkedVerificationError(
            "writer capture statistics are invalid"
        )
    native_ring_frames, counters, terminal_failure = (
        _writer_terminal_counters(terminal)
    )
    return {
        "schema_version": "1.0",
        "capture_mode": "writer-boundary",
        "runtime_instance_id": profile.observer.runtime_instance_id,
        "runtime_pid": profile.runtime.pid,
        "runtime_start_time_ticks": profile.runtime.start_time_ticks,
        "runtime_executable_digest": profile.runtime.executable_digest,
        "observer_profile_digest": _sha256(
            _read_regular(profile.observer.profile_path)
        ),
        "apply_count": statistics.apply_count,
        "unit_count": statistics.unit_count,
        "fragment_count": statistics.fragment_count,
        "copied_byte_count": statistics.copied_byte_count,
        "native_ring_frame_count": native_ring_frames,
        "native_ring_high_water_mark": (
            statistics.native_ring_frame_high_water_mark
        ),
        "native_ring_high_water_unit": "FRAMES_PER_SEALED_UNIT",
        "persistence_queue_high_water_mark": (
            statistics.persistence_queue_high_water_mark
        ),
        "observer_cpu_time_ns": observer_cpu_time_ns,
        "safe_failure_counts": {
            "ring_reserve_failure_count": counters[
                "ringbuf_reserve_fail_count"
            ],
            "user_memory_read_failure_count": counters[
                "user_memory_read_failure_count"
            ],
            "unmatched_call_count": counters["unmatched_call_count"],
            "unsupported_mutation_count": counters[
                "unsupported_mutation_count"
            ],
            "mapping_mutation_count": counters[
                "mapping_mutation_count"
            ],
            "final_inflight_count": counters["final_inflight_count"],
            "incomplete_unit_count": statistics.incomplete_unit_count,
            "persistence_failure_count": statistics.persistence_failure_count,
            "terminal_receipt_failure_count": terminal_failure,
        },
        "contains_prompt_text": False,
        "contains_response_text": False,
        "contains_token_ids": False,
        "contains_endpoint_values": False,
        "contains_request_refs": False,
        "contains_private_truth": False,
    }


def _writer_terminal_counters(
    terminal: bytes,
) -> tuple[int, dict[str, int], int]:
    names = (
        "ringbuf_reserve_fail_count",
        "user_memory_read_failure_count",
        "unmatched_call_count",
        "unsupported_mutation_count",
        "mapping_mutation_count",
        "final_inflight_count",
    )
    empty = {name: 0 for name in names}
    try:
        document = json.loads(terminal.decode("utf-8"))
        if (
            not isinstance(document, Mapping)
            or document.get("artifact") != "runtime_state_terminal_seal"
            or not isinstance(document.get("terminal_counters"), Mapping)
        ):
            raise ValueError("terminal seal counter document is incomplete")
        raw_frame_count = document.get("native_unit_frame_count")
        if (
            isinstance(raw_frame_count, bool)
            or not isinstance(raw_frame_count, int)
            or raw_frame_count < 0
        ):
            raise ValueError("terminal seal frame count is invalid")
        raw_counters = document["terminal_counters"]
        counters: dict[str, int] = {}
        for name in names:
            value = raw_counters.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("terminal seal safety counter is invalid")
            counters[name] = value
        return raw_frame_count, counters, 0
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return 0, empty, 1


def _capture_components(
    profile: LogLinkedWorkflowProfile,
    evidence_root: Path,
    client: _CompletedApplyClient,
    *,
    native_topology: RuntimeStateTopologySummary | None = None,
) -> tuple[SequenceGenerationTracker, StateUnitStore, StateUnitRecorder]:
    observer_profile = load_runtime_state_observer_profile(
        profile.observer.profile_path
    )
    store = StateUnitStore(evidence_root)
    tracker = SequenceGenerationTracker()
    acquirer = LogLinkedStateUnitAcquirer(
        store=store,
        observer_profile=observer_profile,
        native_topology=native_topology,
    )
    return tracker, store, StateUnitRecorder(
        client=client,
        tracker=tracker,
        acquirer=acquirer,
        runtime_pid=profile.runtime.pid,
        runtime_start_time_ticks=profile.runtime.start_time_ticks,
    )


def correlate_observation(
    *,
    evidence_root: Path,
    correlation_key_file: Path,
    correlation_key_id: str,
) -> Mapping[str, object]:
    profile = load_workflow_profile(evidence_root / "workflow_profile.json")
    observation = _read_mapping(
        evidence_root / "observation_manifest.json", maximum_bytes=256 * 1024
    )
    raw_path = evidence_root / str(observation["application_log_file"])
    raw_log = _read_regular(raw_path)
    clock_path = evidence_root / str(observation["clock_receipt_file"])
    clock_bytes = _read_regular(clock_path)
    clock = _parse_json(clock_bytes, "clock receipt")
    parser = LlamaServerTraceParser(
        profile.adapter,
        elapsed_timestamp_origin_utc_ns=int(
            clock["application_timestamp_origin_utc_ns"]
        ),
    )
    events = parser.parse(raw_log)
    if not events:
        raise LogLinkedVerificationError("application log has no normalized events")
    units, seals = _load_state_units(evidence_root / str(observation["state_units_root"]))
    if not units or not seals:
        raise LogLinkedVerificationError("state-unit evidence is incomplete")
    terminal_path = evidence_root / str(observation["observer_terminal_seal_file"])
    terminal_bytes = _read_regular(terminal_path)
    writer_receipt_path = evidence_root / str(
        observation["writer_capture_receipt_file"]
    )
    writer_receipt_bytes = _read_regular(writer_receipt_path)
    parse_writer_capture_receipt(writer_receipt_bytes)
    if _sha256(writer_receipt_bytes) != observation["writer_capture_receipt_digest"]:
        raise LogLinkedVerificationError("writer capture receipt digest differs")
    key = _read_regular(correlation_key_file)
    if len(key) != 32:
        raise LogLinkedCommandError("correlation key must be exactly 32 bytes")
    correlation_profile = RuntimeCorrelationProfile(
        adapter_id=profile.adapter.adapter_id,
        runtime_instance_id=profile.observer.runtime_instance_id,
        slot_sequence_map=profile.correlation.slot_sequence_map,
        wall_to_monotonic_offset_ns=int(clock["wall_to_monotonic_offset_ns"]),
        maximum_clock_error_ns=profile.correlation.maximum_clock_error_ns,
        maximum_access_delay_ns=profile.correlation.maximum_access_delay_ns,
        maximum_access_pre_terminal_skew_ns=(
            profile.correlation.maximum_access_pre_terminal_skew_ns
        ),
        allowed_path_classes=profile.correlation.allowed_path_classes,
        source_application_log_digest=_sha256(raw_log),
        source_observer_terminal_seal_digest=_sha256(terminal_bytes),
        clock_normalization_receipt_digest=_sha256(clock_bytes),
    )
    results = EndpointCorrelator().analyze(
        CorrelationInputs(
            application_events=events,
            generation_seals=seals,
            state_units=units,
            runtime_profile=correlation_profile,
        ),
        correlation_key=key,
        correlation_key_id=correlation_key_id,
    )
    if not results:
        raise LogLinkedVerificationError("correlation produced no result")
    wall_times = [event.wall_time_utc_ns for event in events]
    log_provenance = {
        "schema_version": "1.0",
        "adapter_id": profile.adapter.adapter_id,
        "upstream_commit": profile.adapter.upstream_commit,
        "server_executable_sha256": profile.runtime.executable_digest,
        "logging_arguments": list(observation["launch_arguments"]),
        "log_started_at_utc_ns": min(wall_times),
        "log_ended_at_utc_ns": max(wall_times),
        "parser_profile_id": profile.adapter.adapter_id,
        "clock_sources": [
            "llama_cpp_elapsed_us",
            "runtime_start_time_ticks",
            "application_wall_utc_ns",
            "observer_monotonic_ns",
        ],
    }
    package = LogLinkedEvidencePackage.create(
        evidence_root / "package",
        EvidenceInputs(
            adapter_id=profile.adapter.adapter_id,
            runtime_instance_id=profile.observer.runtime_instance_id,
            created_at_utc_ns=time.time_ns(),
            raw_application_log=raw_path,
            log_provenance=log_provenance,
            observer_terminal_seal=terminal_path,
            writer_capture_receipt=writer_receipt_path,
            state_units_root=evidence_root / str(observation["state_units_root"]),
            normalized_application_events=events,
            attribution_results=results,
            endpoint_derivation_receipt={
                "schema_version": "1.0",
                "algorithm": "HMAC-SHA256",
                "correlation_key_id": correlation_key_id,
                "raw_endpoint_retention": "RESTRICTED_APPLICATION_LOG_ONLY",
            },
            evaluation_metrics=None,
        ),
    )
    verified = LogLinkedEvidencePackage.verify(package.root)
    counts = {
        decision: sum(result.final_decision.value == decision for result in results)
        for decision in ("CERTIFIED", "RANKED", "AMBIGUOUS", "INCOMPLETE")
    }
    status = (
        "COMPLETE"
        if counts["AMBIGUOUS"] == 0 and counts["INCOMPLETE"] == 0
        else "AMBIGUOUS" if counts["AMBIGUOUS"] else "INCOMPLETE"
    )
    return {
        "schema_version": "1.0",
        "status": status,
        "evidence_package": str(verified.root),
        "package_id": verified.manifest["package_id"],
        "certified_count": counts["CERTIFIED"],
        "ranked_count": counts["RANKED"],
        "ambiguous_count": counts["AMBIGUOUS"],
        "incomplete_count": counts["INCOMPLETE"],
    }


def _select_endpoint_results(
    results: tuple[AttributionResult, ...],
    endpoint_ref: str,
    *,
    session_request_ref: str | None,
) -> tuple[AttributionResult, ...]:
    selected = tuple(
        result
        for result in results
        if result.final_decision
        in {AttributionDecision.CERTIFIED, AttributionDecision.RANKED}
        and result.endpoint_ref == endpoint_ref
    )
    if not selected:
        raise LogLinkedVerificationError("endpoint has no resolved attribution")
    if session_request_ref is None:
        return selected
    session = tuple(
        result for result in selected if result.request_ref == session_request_ref
    )
    if len(session) != 1:
        raise LogLinkedVerificationError(
            "selected endpoint session has no resolved attribution"
        )
    return session


def export_endpoint_bundle(
    *,
    evidence_root: Path,
    endpoint_ref: str,
    bundle_root: Path,
    session_request_ref: str | None = None,
) -> Mapping[str, object]:
    package = LogLinkedEvidencePackage.verify(evidence_root / "package")
    profile = load_workflow_profile(evidence_root / "workflow_profile.json")
    result_path = package.root / str(
        package.manifest["attribution_results_file"]
    )
    results = tuple(
        AttributionResult(_parse_json(line + b"\n", "attribution result"))
        for line in _read_regular(result_path).splitlines()
        if line
    )
    selected = _select_endpoint_results(
        results,
        endpoint_ref,
        session_request_ref=session_request_ref,
    )
    units, seals = _load_state_units(package.root / "observer" / "state-units")
    refs = {
        _generation_ref_for_result(result, units)
        for result in selected
    }
    if len(refs) != 1:
        raise LogLinkedVerificationError(
            "endpoint spans multiple sequence generations; select a session"
        )
    latest = max(
        selected,
        key=lambda result: (
            result.cutoff_apply_call_id,
            result.request_ref,
        ),
    )
    selected_ref = next(iter(refs))
    seal = next(
        (
            item
            for item in seals
            if item.ref == selected_ref
        ),
        None,
    )
    if seal is None:
        raise LogLinkedVerificationError("correlated generation seal is unavailable")
    materialized = EndpointStateMaterializer().materialize(
        result=latest,
        generation_seal=seal,
        state_units=units,
    )
    attribution_digest = _sha256(latest.to_bytes())
    package_manifest_digest = _sha256(_read_regular(package.manifest_path))
    exported = EndpointBundleExporter().export(
        bundle_root,
        EndpointBundleExportInputs(
            run_id="run-" + str(package.manifest["package_id"])[4:20],
            attribution_result=latest,
            materialized_state=materialized,
            evidence_package_manifest_digest=package_manifest_digest,
            attribution_receipt_digest=attribution_digest,
            model_profile_id=profile.rehosting.model_profile_id,
            model_digest=profile.rehosting.model_digest,
            runtime_profile_digest=profile.rehosting.runtime_profile_digest,
            n_ctx=profile.rehosting.n_ctx,
            n_seq_max=profile.rehosting.n_seq_max,
            request_refs=tuple(sorted(result.request_ref for result in selected)),
            runtime_metadata=profile.rehosting.runtime_metadata,
        ),
    )
    return {
        "schema_version": "1.0",
        "status": "COMPLETE",
        "endpoint_ref": exported.endpoint_ref,
        "bundle_id": exported.bundle_id,
        "bundle_root": str(exported.root),
    }


def _generation_ref_for_result(
    result: object,
    units: tuple[StoredStateUnit, ...],
) -> SequenceGenerationRef:
    chain = getattr(result, "state_unit_chain_digests", None)
    if not isinstance(chain, tuple) or not chain:
        raise LogLinkedVerificationError(
            "attribution result has no verified state-unit chain"
        )
    by_digest = {unit.manifest_digest: unit for unit in units}
    if len(by_digest) != len(units):
        raise LogLinkedVerificationError("state-unit digest inventory is duplicated")
    try:
        matched = tuple(by_digest[digest] for digest in chain)
    except KeyError as error:
        raise LogLinkedVerificationError(
            "attribution result references an unavailable state unit"
        ) from error
    refs = {unit.ref for unit in matched}
    if len(refs) != 1:
        raise LogLinkedVerificationError(
            "correlation result crosses multiple sequence generations"
        )
    return next(iter(refs))


def _load_state_units(
    root: Path,
) -> tuple[tuple[StoredStateUnit, ...], tuple[GenerationSeal, ...]]:
    if root.is_symlink() or not root.is_dir():
        raise LogLinkedVerificationError("state-unit root is unavailable")
    units: list[StoredStateUnit] = []
    seals: list[GenerationSeal] = []
    for manifest_path in sorted(root.rglob("state_unit_manifest.json")):
        manifest_bytes = _read_regular(manifest_path)
        document = parse_state_unit_manifest(manifest_bytes)
        payload_path = manifest_path.with_name("runtime_state_payload.bin")
        payload = _read_regular(payload_path)
        if (
            len(payload) != document["payload_size_bytes"]
            or _sha256(payload) != document["payload_digest"]
        ):
            raise LogLinkedVerificationError("state-unit payload differs")
        unit_name = manifest_path.parent.name
        if not unit_name.startswith("unit_"):
            raise LogLinkedVerificationError("state-unit directory is invalid")
        ref = _ref_from(document)
        units.append(
            StoredStateUnit(
                root=manifest_path.parent,
                payload_path=payload_path,
                manifest_path=manifest_path,
                unit_number=int(unit_name.removeprefix("unit_")),
                ref=ref,
                unit_id=str(document["unit_id"]),
                payload_digest=str(document["payload_digest"]),
                manifest_digest=_sha256(manifest_bytes),
            )
        )
    for seal_path in sorted(root.rglob("generation_seal.json")):
        document = _parse_json(_read_regular(seal_path), "generation seal")
        expected = {
            "schema_version",
            "runtime_instance_id",
            "seq_id",
            "sequence_generation",
            "state",
            "reason_codes",
            "unit_count",
            "last_unit_digest",
            "opened_apply_call_id",
            "closed_apply_call_id",
        }
        _exact(document, expected, "generation seal")
        seals.append(
            GenerationSeal(
                ref=_ref_from(document),
                state=str(document["state"]),
                reason_codes=tuple(document["reason_codes"]),
                unit_count=int(document["unit_count"]),
                last_unit_digest=(
                    None
                    if document["last_unit_digest"] is None
                    else str(document["last_unit_digest"])
                ),
                opened_apply_call_id=int(document["opened_apply_call_id"]),
                closed_apply_call_id=int(document["closed_apply_call_id"]),
            )
        )
    return tuple(units), tuple(seals)


def _ref_from(document: Mapping[str, object]) -> SequenceGenerationRef:
    return SequenceGenerationRef(
        str(document["runtime_instance_id"]),
        int(document["seq_id"]),
        int(document["sequence_generation"]),
    )


def _clock_ticks_per_second() -> int:
    try:
        value = os.sysconf("SC_CLK_TCK")
    except (AttributeError, OSError, ValueError) as error:
        raise LogLinkedCommandError(
            "runtime clock-tick frequency is unavailable"
        ) from error
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise LogLinkedCommandError("runtime clock-tick frequency is invalid")
    return value


def _runtime_clock_receipt(
    *,
    runtime_start_time_ticks: int,
    wall_time_utc_ns: int,
    monotonic_ns: int,
    clock_ticks_per_second: int,
) -> dict[str, object]:
    values = (
        runtime_start_time_ticks,
        wall_time_utc_ns,
        monotonic_ns,
        clock_ticks_per_second,
    )
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise LogLinkedCommandError("runtime clock sample is invalid")
    if (
        runtime_start_time_ticks <= 0
        or wall_time_utc_ns < 0
        or monotonic_ns < 0
        or clock_ticks_per_second <= 0
    ):
        raise LogLinkedCommandError("runtime clock sample is invalid")
    runtime_start_monotonic_ns = (
        runtime_start_time_ticks * 1_000_000_000 // clock_ticks_per_second
    )
    if runtime_start_monotonic_ns > monotonic_ns:
        raise LogLinkedCommandError("runtime start time is later than the clock sample")
    origin = wall_time_utc_ns - (monotonic_ns - runtime_start_monotonic_ns)
    if origin < 0:
        raise LogLinkedCommandError("runtime wall-clock origin is invalid")
    return {
        "schema_version": "1.0",
        "wall_time_utc_ns": wall_time_utc_ns,
        "monotonic_ns": monotonic_ns,
        "wall_to_monotonic_offset_ns": monotonic_ns - wall_time_utc_ns,
        "clock_ticks_per_second": clock_ticks_per_second,
        "runtime_start_time_ticks": runtime_start_time_ticks,
        "runtime_start_monotonic_ns": runtime_start_monotonic_ns,
        "application_timestamp_origin_utc_ns": origin,
        "application_timestamp_format": "LLAMA_CPP_ELAPSED_M_S_MS_US",
        "application_timestamp_origin_method": "PROCESS_START_TICKS_ESTIMATE",
        "process_start_tick_uncertainty_ns": (
            1_000_000_000 + clock_ticks_per_second - 1
        )
        // clock_ticks_per_second,
    }


def _sample_runtime_clock(runtime_start_time_ticks: int) -> dict[str, object]:
    return _runtime_clock_receipt(
        runtime_start_time_ticks=runtime_start_time_ticks,
        wall_time_utc_ns=time.time_ns(),
        monotonic_ns=time.monotonic_ns(),
        clock_ticks_per_second=_clock_ticks_per_second(),
    )


def _process_arguments(pid: int, expected_start_ticks: int) -> tuple[str, ...]:
    _verify_start_ticks(pid, expected_start_ticks)
    payload = _read_regular(Path(f"/proc/{pid}/cmdline"))
    values = tuple(
        item.decode("utf-8") for item in payload.rstrip(b"\0").split(b"\0")
    )
    if not values or any(not value for value in values):
        raise LogLinkedCommandError("runtime launch arguments are unavailable")
    return values


def _sha256_proc_executable(pid: int) -> str:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise LogLinkedCommandError("runtime executable PID is invalid")
    proc_executable = Path(f"/proc/{pid}/exe")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_BINARY", 0)
    try:
        descriptor = os.open(proc_executable, flags)
    except OSError as error:
        raise LogLinkedCommandError(
            "runtime executable cannot be pinned through procfs"
        ) from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > 1024 * 1024 * 1024
        ):
            raise LogLinkedCommandError("runtime executable inode is invalid")
        digest = hashlib.sha256()
        total = 0
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            total += len(block)
            if total > before.st_size:
                raise LogLinkedCommandError(
                    "runtime executable grew while hashing"
                )
            digest.update(block)
        after = os.fstat(descriptor)
        if (
            total != before.st_size
            or (before.st_dev, before.st_ino, before.st_size)
            != (after.st_dev, after.st_ino, after.st_size)
        ):
            raise LogLinkedCommandError(
                "runtime executable changed while hashing"
            )
        return "sha256:" + digest.hexdigest()
    except OSError as error:
        raise LogLinkedCommandError(
            "runtime executable cannot be read through procfs"
        ) from error
    finally:
        os.close(descriptor)


def _process_identity(runtime: RuntimeInputs) -> ExecutableIdentity:
    _verify_start_ticks(runtime.pid, runtime.start_time_ticks)
    proc_executable = Path(f"/proc/{runtime.pid}/exe")
    try:
        actual_display = os.readlink(proc_executable)
    except OSError as error:
        raise LogLinkedCommandError("runtime executable identity is unavailable") from error
    if not actual_display.startswith("/"):
        raise LogLinkedCommandError("runtime executable identity is invalid")
    digest = _sha256_proc_executable(runtime.pid)
    _verify_start_ticks(runtime.pid, runtime.start_time_ticks)
    # Do not execute an externally owned runtime during read-only identity
    # verification. Its version label is launch metadata bound to the PID,
    # start ticks, and procfs-pinned executable digest checks above.
    # Re-execution can also fail when the live process uses a private dynamic-
    # library search path. The approved path remains the portable identity
    # label because procfs appends " (deleted)" after an in-place app cutover.
    return ExecutableIdentity(
        path=str(runtime.executable), digest=digest, version=runtime.version
    )


def _verify_start_ticks(pid: int, expected: int) -> None:
    stat_bytes = _read_regular(Path(f"/proc/{pid}/stat"))
    try:
        actual = int(stat_bytes.decode("ascii").rsplit(") ", 1)[1].split()[19])
    except (UnicodeError, ValueError, IndexError) as error:
        raise LogLinkedCommandError("runtime process identity is invalid") from error
    if actual != expected:
        raise LogLinkedCommandError("runtime process identity changed")


def _send_canary(url: str, canary: str, authorization: str) -> str:
    body = canonical_json(
        {
            "messages": [{"role": "user", "content": canary}],
            "max_tokens": 1,
            "temperature": 0,
            "stream": False,
        }
    )
    request = urllib.request.Request(url, data=body, method="POST")
    request.add_header("Content-Type", "application/json")
    if not authorization.startswith("LARIS-NO-AUTHORIZATION-"):
        request.add_header("Authorization", authorization)
    try:
        with urllib.request.urlopen(request, timeout=30.0) as response:
            payload = response.read(4 * 1024 * 1024)
    except (OSError, urllib.error.URLError) as error:
        raise LogLinkedCommandError("canary request failed") from error
    try:
        document = json.loads(payload)
        content = document["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, json.JSONDecodeError):
        content = payload.decode("utf-8", errors="strict")[:256]
    if not isinstance(content, str) or not content:
        return "LARIS-CANARY-EMPTY-RESPONSE"
    return content[:256]


def _read_secret(path: Path) -> str:
    payload = _read_regular(path)
    try:
        value = payload.decode("utf-8").strip()
    except UnicodeError as error:
        raise LogLinkedCommandError("authorization file is not UTF-8") from error
    if not value or "\0" in value or "\n" in value:
        raise LogLinkedCommandError("authorization file is invalid")
    return value


def _read_mapping(path: Path, *, maximum_bytes: int) -> dict[str, object]:
    payload = _read_regular(path)
    if len(payload) > maximum_bytes:
        raise LogLinkedCommandError("configuration file is too large")
    try:
        value = yaml.safe_load(payload)
    except yaml.YAMLError as error:
        raise LogLinkedCommandError("configuration YAML is invalid") from error
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise LogLinkedCommandError("configuration must be an object")
    return value


def _parse_json(payload: bytes, label: str) -> dict[str, object]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise LogLinkedVerificationError(f"{label} repeats a field")
            result[key] = value
        return result

    try:
        value = json.loads(payload, object_pairs_hook=reject_duplicates)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise LogLinkedVerificationError(f"{label} is invalid") from error
    if not isinstance(value, dict):
        raise LogLinkedVerificationError(f"{label} must be an object")
    return value


def _regular_path(path: Path, label: str) -> Path:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise LogLinkedCommandError(f"{label} is unavailable") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise LogLinkedCommandError(f"{label} is not a regular file")
    return path


def _read_regular(path: Path) -> bytes:
    source = _regular_path(path, str(path))
    before = source.stat()
    payload = source.read_bytes()
    after = source.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise LogLinkedVerificationError("evidence changed while reading")
    return payload


def _read_append_only_snapshot(path: Path, *, maximum_bytes: int) -> bytes:
    if (
        isinstance(maximum_bytes, bool)
        or not isinstance(maximum_bytes, int)
        or maximum_bytes <= 0
    ):
        raise LogLinkedVerificationError("active log size limit is invalid")
    source = _regular_path(path, str(path))
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(source, flags)
    except OSError as error:
        raise LogLinkedVerificationError("active log cannot be opened") from error

    def read_prefix(size: int) -> bytes:
        os.lseek(descriptor, 0, os.SEEK_SET)
        blocks: list[bytes] = []
        remaining = size
        while remaining:
            block = os.read(descriptor, min(remaining, 1024 * 1024))
            if not block:
                raise LogLinkedVerificationError(
                    "active log was truncated while snapshotting"
                )
            blocks.append(block)
            remaining -= len(block)
        return b"".join(blocks)

    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > maximum_bytes:
            raise LogLinkedVerificationError("active log is invalid or oversized")
        first = read_prefix(before.st_size)
        second = read_prefix(before.st_size)
        after_descriptor = os.fstat(descriptor)
        after_path = source.lstat()
        if (
            stat.S_ISLNK(after_path.st_mode)
            or not stat.S_ISREG(after_path.st_mode)
            or (before.st_dev, before.st_ino)
            != (after_descriptor.st_dev, after_descriptor.st_ino)
            or (before.st_dev, before.st_ino)
            != (after_path.st_dev, after_path.st_ino)
            or after_descriptor.st_size < before.st_size
            or after_path.st_size < before.st_size
            or first != second
        ):
            raise LogLinkedVerificationError(
                "active log prefix changed while snapshotting"
            )
        return second
    finally:
        os.close(descriptor)


def _write_new(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as error:
        raise LogLinkedVerificationError("evidence file cannot be created") from error
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise LogLinkedVerificationError("evidence write made no progress")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _make_private(path, directory=False)


def _make_private(path: Path, *, directory: bool) -> None:
    if os.name == "posix":
        os.chmod(path, 0o700 if directory else 0o600, follow_symlinks=False)


def _exact(value: Mapping[str, object], fields: set[str], label: str) -> None:
    if set(value) != fields:
        raise LogLinkedCommandError(f"{label} field set is invalid")


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise LogLinkedCommandError(f"{label} must be an object")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or "\0" in value:
        raise LogLinkedCommandError(f"{label} is invalid")
    return value


def _safe_id(value: object, label: str) -> str:
    text = _text(value, label)
    if _SAFE_ID.fullmatch(text) is None:
        raise LogLinkedCommandError(f"{label} is invalid")
    return text


def _runtime_instance_id(value: object, label: str) -> str:
    text = _text(value, label)
    if _RUNTIME_INSTANCE_ID.fullmatch(text) is None:
        raise LogLinkedCommandError(f"{label} is invalid")
    return text


def _digest(value: object, label: str) -> str:
    text = _text(value, label)
    if _DIGEST.fullmatch(text) is None:
        raise LogLinkedCommandError(f"{label} is invalid")
    return text


def _text_tuple(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise LogLinkedCommandError(f"{label} must be a non-empty list")
    result = tuple(_text(item, label) for item in value)
    if len(set(result)) != len(result):
        raise LogLinkedCommandError(f"{label} contains a duplicate")
    return result


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise LogLinkedCommandError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise LogLinkedCommandError(f"{label} must be non-negative")
    return value


def _positive_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise LogLinkedCommandError(f"{label} must be positive")
    return float(value)


def _absolute_path(value: object, label: str) -> Path:
    path = Path(_text(value, label))
    if not path.is_absolute():
        raise LogLinkedCommandError(f"{label} must be absolute")
    return path


def _http_url(value: object) -> str:
    text = _text(value, "canary_url")
    if not text.startswith(("http://", "https://")):
        raise LogLinkedCommandError("canary_url must be HTTP or HTTPS")
    return text


def _sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


__all__ = [
    "LogLinkedWorkflowProfile",
    "ObservationSession",
    "correlate_observation",
    "export_endpoint_bundle",
    "load_workflow_profile",
    "run_preflight",
]
