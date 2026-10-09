from __future__ import annotations

import threading
import time
import uuid
import re
from collections import deque
from collections.abc import Mapping

from .binding_ledger import BindingLedger
from laris_onpremise.core.records import (
    SCHEMA_VERSION,
    JsonValue,
    LeaseEvent,
    LeaseRecord,
    LeaseState,
    RequestIdentity,
)
from .sequence_lease import (
    LeaseTransitionError,
    require_state,
    require_transition,
)


class QueueWaitTimeout(TimeoutError):
    """Raised when a queued request cannot obtain a Lease by its deadline."""


class QueueRequestError(RuntimeError):
    """Raised when a request cannot enter or remain in the Lease queue."""


_SHA256_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")


def _is_nonnegative_integer(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def _is_positive_integer(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value > 0


class SequenceLeaseController:
    """Single-writer request queue and Sequence Lease state controller."""

    def __init__(
        self,
        *,
        ledger: BindingLedger,
        controller_instance_id: str,
        runtime_instance_id: str,
        context_ref: str,
        context_ref_source: str,
        slot_count: int,
        allow_capture_skip: bool = False,
    ) -> None:
        if slot_count <= 0:
            raise ValueError("slot_count must be positive")
        self.ledger = ledger
        self.controller_instance_id = controller_instance_id
        self.runtime_instance_id = runtime_instance_id
        self.context_ref = context_ref
        self.context_ref_source = context_ref_source
        self.slot_count = slot_count
        self.allow_capture_skip = allow_capture_skip

        self._condition = threading.Condition(threading.RLock())
        self._queue: deque[str] = deque()
        self._requests: dict[str, RequestIdentity] = {}
        self._request_to_lease: dict[str, str] = {}
        self._held_by_slot: dict[int, str] = {}
        self._available_slots = set(range(slot_count))
        self._quarantined_slots: set[int] = set()
        self._cancelled_requests: set[str] = set()
        self._dispatch_authorized: set[str] = set()

    @staticmethod
    def _event(
        event_type: str,
        *,
        lease: LeaseRecord | None = None,
        request: RequestIdentity | None = None,
        runtime_instance_id: str | None = None,
        details: Mapping[str, JsonValue] | None = None,
        monotonic_ns: int | None = None,
    ) -> LeaseEvent:
        if lease is not None:
            request = lease.request
            runtime_instance_id = lease.runtime_instance_id
        return LeaseEvent(
            schema_version=SCHEMA_VERSION,
            event_id=f"evt_{uuid.uuid4().hex}",
            event_type=event_type,
            lease_id=None if lease is None else lease.lease_id,
            request_ref=None if request is None else request.request_ref,
            runtime_instance_id=runtime_instance_id,
            wall_time_ns=time.time_ns(),
            monotonic_ns=(
                time.monotonic_ns() if monotonic_ns is None else monotonic_ns
            ),
            details={} if details is None else details,
        )

    def enqueue(self, request: RequestIdentity) -> None:
        with self._condition:
            if request.request_ref in self._requests:
                raise QueueRequestError(
                    f"duplicate request_ref: {request.request_ref}"
                )
            self.ledger.append(
                self._event(
                    "request_received",
                    request=request,
                    details={
                        "user_hmac": request.user_hmac,
                        "identity_key_id": request.identity_key_id,
                        "auth_receipt_id": request.auth_receipt_id,
                        "session_ref": request.session_ref,
                        "connection_ref": request.connection_ref,
                        "retry_of": request.retry_of,
                    },
                )
            )
            self.ledger.append(
                self._event(
                    "request_queued",
                    request=request,
                    details={"reason": "waiting_for_sequence_lease"},
                )
            )
            self._requests[request.request_ref] = request
            self._queue.append(request.request_ref)
            self._condition.notify_all()

    def get_request(self, request_ref: str) -> RequestIdentity:
        with self._condition:
            try:
                return self._requests[request_ref]
            except KeyError as error:
                raise KeyError(request_ref) from error

    def get_lease(self, lease_id: str) -> LeaseRecord:
        return self.ledger.get_lease(lease_id)

    def lease_for_request(self, request_ref: str) -> LeaseRecord | None:
        with self._condition:
            lease_id = self._request_to_lease.get(request_ref)
        return None if lease_id is None else self.ledger.get_lease(lease_id)

    def _reserve_next_locked(self) -> LeaseRecord | None:
        if not self._queue or not self._available_slots:
            return None
        while self._queue and self._queue[0] in self._cancelled_requests:
            self._queue.popleft()
        if not self._queue:
            return None
        request_ref = self._queue[0]
        request = self._requests[request_ref]
        expected_slot_id = min(self._available_slots)
        lease = self.ledger.reserve_lease(
            lease_id=f"lease_{uuid.uuid4().hex}",
            request=request,
            runtime_instance_id=self.runtime_instance_id,
            context_ref=self.context_ref,
            context_ref_source=self.context_ref_source,
            expected_slot_id=expected_slot_id,
            expected_seq_id=expected_slot_id,
            slot_count=self.slot_count,
        )
        self._queue.popleft()
        self._available_slots.remove(expected_slot_id)
        self._held_by_slot[expected_slot_id] = lease.lease_id
        self._request_to_lease[request_ref] = lease.lease_id
        self._condition.notify_all()
        return lease

    def reserve_next(self) -> LeaseRecord | None:
        with self._condition:
            return self._reserve_next_locked()

    def wait_for_lease(
        self, request_ref: str, *, deadline: float | None
    ) -> LeaseRecord:
        with self._condition:
            if request_ref not in self._requests:
                raise KeyError(request_ref)
            while True:
                lease_id = self._request_to_lease.get(request_ref)
                if lease_id is not None:
                    return self.ledger.get_lease(lease_id)
                if request_ref in self._cancelled_requests:
                    raise QueueWaitTimeout(
                        f"request was removed from queue: {request_ref}"
                    )
                self._reserve_next_locked()
                lease_id = self._request_to_lease.get(request_ref)
                if lease_id is not None:
                    return self.ledger.get_lease(lease_id)
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    self._cancel_queued_locked(
                        request_ref, reason="queue_deadline_exceeded"
                    )
                    raise QueueWaitTimeout(
                        f"request did not receive a Lease: {request_ref}"
                    )
                self._condition.wait(timeout=remaining)

    def _cancel_queued_locked(self, request_ref: str, *, reason: str) -> None:
        if request_ref in self._request_to_lease:
            raise QueueRequestError("cannot cancel a request after Lease reservation")
        if request_ref in self._cancelled_requests:
            return
        try:
            self._queue.remove(request_ref)
        except ValueError:
            pass
        request = self._requests[request_ref]
        self.ledger.append(
            self._event(
                "request_queue_cancelled",
                request=request,
                details={"reason": reason},
            )
        )
        self._cancelled_requests.add(request_ref)
        self._condition.notify_all()

    def cancel_queued_request(self, request_ref: str, *, reason: str) -> None:
        with self._condition:
            self._cancel_queued_locked(request_ref, reason=reason)

    def authorize_dispatch(self, lease_id: str, evidence_ref: str) -> LeaseRecord:
        with self._condition:
            lease = self.ledger.get_lease(lease_id)
            require_state(lease, LeaseState.RESERVED)
            if lease_id in self._dispatch_authorized:
                raise LeaseTransitionError("dispatch is already authorized")
            updated = self.ledger.transition_lease(
                lease_id,
                state=LeaseState.RESERVED,
                event=self._event(
                    "dispatch_authorized",
                    lease=lease,
                    details={"evidence_ref": evidence_ref},
                ),
            )
            self._dispatch_authorized.add(lease_id)
            return updated

    def mark_submitted(
        self,
        lease_id: str,
        evidence_ref: str,
        *,
        monotonic_ns: int | None = None,
    ) -> LeaseRecord:
        with self._condition:
            lease = self.ledger.get_lease(lease_id)
            require_state(lease, LeaseState.RESERVED)
            if lease_id not in self._dispatch_authorized:
                raise LeaseTransitionError(
                    "dispatch must be authorized before upstream submission"
                )
            require_transition(lease, LeaseState.SUBMITTED)
            return self.ledger.transition_lease(
                lease_id,
                state=LeaseState.SUBMITTED,
                event=self._event(
                    "upstream_request_sent",
                    lease=lease,
                    details={"evidence_ref": evidence_ref},
                    monotonic_ns=monotonic_ns,
                ),
            )

    def confirm_active(
        self,
        lease_id: str,
        *,
        response_confirmed_slot_id: int,
        evidence_ref: str,
    ) -> LeaseRecord:
        with self._condition:
            lease = self.ledger.get_lease(lease_id)
            require_state(lease, LeaseState.SUBMITTED)
            if response_confirmed_slot_id != lease.expected_slot_id:
                self._quarantine_locked(
                    lease,
                    reason="response_slot_mismatch",
                    evidence_ref=evidence_ref,
                )
                raise LeaseTransitionError(
                    "slot mismatch between Lease and upstream response"
                )
            require_transition(lease, LeaseState.ACTIVE)
            return self.ledger.confirm_slot_assignment(
                lease_id,
                response_confirmed_slot_id=response_confirmed_slot_id,
                evidence_ref=evidence_ref,
            )

    def begin_quiescing(
        self, lease_id: str, *, reason: str, evidence_ref: str
    ) -> LeaseRecord:
        with self._condition:
            lease = self.ledger.get_lease(lease_id)
            require_state(lease, LeaseState.ACTIVE)
            require_transition(lease, LeaseState.QUIESCING)
            return self.ledger.transition_lease(
                lease_id,
                state=LeaseState.QUIESCING,
                event=self._event(
                    "terminal_response_received",
                    lease=lease,
                    details={"reason": reason, "evidence_ref": evidence_ref},
                ),
            )

    def confirm_slot_idle(self, lease_id: str, evidence_ref: str) -> LeaseRecord:
        with self._condition:
            lease = self.ledger.get_lease(lease_id)
            require_state(lease, LeaseState.QUIESCING)
            if lease.backend_released:
                raise LeaseTransitionError("slot idle was already confirmed")
            return self.ledger.transition_lease(
                lease_id,
                state=LeaseState.QUIESCING,
                backend_released=True,
                event=self._event(
                    "slot_idle_confirmed",
                    lease=lease,
                    details={"evidence_ref": evidence_ref},
                ),
            )

    def begin_capture(self, lease_id: str, capture_id: str) -> LeaseRecord:
        with self._condition:
            lease = self.ledger.get_lease(lease_id)
            require_state(lease, LeaseState.QUIESCING)
            if not lease.backend_released:
                raise LeaseTransitionError(
                    "slot idle must be confirmed before runtime-state capture"
                )
            require_transition(lease, LeaseState.CHECKPOINTING)
            return self.ledger.transition_lease(
                lease_id,
                state=LeaseState.CHECKPOINTING,
                capture_id=capture_id,
                event=self._event(
                    "runtime_state_capture_started",
                    lease=lease,
                    details={"capture_id": capture_id},
                ),
            )

    def _has_lease_event(self, lease_id: str, event_type: str) -> bool:
        return any(
            stored.event.lease_id == lease_id
            and stored.event.event_type == event_type
            for stored in self.ledger.read_events()
        )

    def record_acquisition_requested(
        self,
        lease_id: str,
        *,
        capture_id: str,
        observed_seq_id: int,
        evidence_ref: str,
    ) -> LeaseRecord:
        with self._condition:
            lease = self.ledger.get_lease(lease_id)
            require_state(lease, LeaseState.CHECKPOINTING)
            if lease.capture_id != capture_id:
                self._quarantine_locked(
                    lease,
                    reason="stale_runtime_state_acquisition_request",
                    evidence_ref=evidence_ref,
                )
                raise LeaseTransitionError(
                    "acquisition request does not match active capture"
                )
            if not _is_nonnegative_integer(observed_seq_id):
                self._quarantine_locked(
                    lease,
                    reason="invalid_runtime_state_sequence_identifier",
                    evidence_ref=evidence_ref,
                )
                raise LeaseTransitionError(
                    "observed_seq_id must be a non-negative integer"
                )
            if observed_seq_id != lease.expected_seq_id:
                self._quarantine_locked(
                    lease,
                    reason="runtime_state_sequence_mismatch",
                    evidence_ref=evidence_ref,
                )
                raise LeaseTransitionError(
                    "observed sequence does not match the active Lease"
                )
            if self._has_lease_event(
                lease_id, "runtime_state_acquisition_requested"
            ):
                self._quarantine_locked(
                    lease,
                    reason="duplicate_runtime_state_acquisition_request",
                    evidence_ref=evidence_ref,
                )
                raise LeaseTransitionError("acquisition was already requested")
            return self.ledger.transition_lease(
                lease_id,
                state=LeaseState.CHECKPOINTING,
                observed_seq_id=observed_seq_id,
                event=self._event(
                    "runtime_state_acquisition_requested",
                    lease=lease,
                    details={
                        "capture_id": capture_id,
                        "observed_seq_id": observed_seq_id,
                        "evidence_ref": evidence_ref,
                    },
                ),
            )

    def record_acquisition_completed(
        self,
        lease_id: str,
        *,
        capture_id: str,
        request_ref: str,
        runtime_instance_id: str,
        slot_id: int,
        slot_epoch: int | None,
        observed_seq_id: int,
        digest: str,
        size_bytes: int,
        evidence_ref: str,
    ) -> None:
        with self._condition:
            lease = self.ledger.get_lease(lease_id)
            require_state(lease, LeaseState.CHECKPOINTING)
            if lease.capture_id != capture_id:
                self._quarantine_locked(
                    lease,
                    reason="stale_runtime_state_acquisition_completion",
                    evidence_ref=evidence_ref,
                )
                raise LeaseTransitionError(
                    "acquisition completion does not match active capture"
                )
            if not self._has_lease_event(
                lease_id, "runtime_state_acquisition_requested"
            ):
                self._quarantine_locked(
                    lease,
                    reason="acquisition_completion_without_request",
                    evidence_ref=evidence_ref,
                )
                raise LeaseTransitionError(
                    "acquisition completion requires a request event"
                )
            if self._has_lease_event(
                lease_id, "runtime_state_acquisition_completed"
            ):
                self._quarantine_locked(
                    lease,
                    reason="duplicate_runtime_state_acquisition_completion",
                    evidence_ref=evidence_ref,
                )
                raise LeaseTransitionError("acquisition was already completed")
            if (
                not _is_nonnegative_integer(slot_id)
                or not _is_positive_integer(slot_epoch)
                or not _is_nonnegative_integer(observed_seq_id)
            ):
                self._quarantine_locked(
                    lease,
                    reason="invalid_runtime_state_binding_identifier",
                    evidence_ref=evidence_ref,
                )
                raise LeaseTransitionError(
                    "slot_id, slot_epoch and observed_seq_id must be integers"
                )
            binding_matches = (
                request_ref == lease.request.request_ref
                and runtime_instance_id == lease.runtime_instance_id
                and slot_id == lease.expected_slot_id
                and slot_epoch == lease.slot_epoch
                and observed_seq_id == lease.observed_seq_id
            )
            if not binding_matches:
                self._quarantine_locked(
                    lease,
                    reason="runtime_state_acquisition_binding_mismatch",
                    evidence_ref=evidence_ref,
                )
                raise LeaseTransitionError(
                    "acquisition completion does not match the Lease binding"
                )
            if (
                not isinstance(digest, str)
                or _SHA256_DIGEST.fullmatch(digest) is None
                or isinstance(size_bytes, bool)
                or not isinstance(size_bytes, int)
                or size_bytes <= 0
            ):
                self._quarantine_locked(
                    lease,
                    reason="invalid_runtime_state_acquisition_completion",
                    evidence_ref=evidence_ref,
                )
                raise LeaseTransitionError("acquisition completion is invalid")
            self.ledger.append(
                self._event(
                    "runtime_state_acquisition_completed",
                    lease=lease,
                    details={
                        "capture_id": capture_id,
                        "request_ref": request_ref,
                        "runtime_instance_id": runtime_instance_id,
                        "slot_id": slot_id,
                        "slot_epoch": slot_epoch,
                        "observed_seq_id": observed_seq_id,
                        "payload_digest": digest,
                        "payload_size_bytes": size_bytes,
                        "evidence_ref": evidence_ref,
                    },
                )
            )

    def record_acquisition_copied(
        self,
        lease_id: str,
        *,
        capture_id: str,
        evidence_ref: str,
    ) -> LeaseRecord:
        with self._condition:
            lease = self.ledger.get_lease(lease_id)
            require_state(lease, LeaseState.CHECKPOINTING)
            if lease.capture_id != capture_id:
                self._quarantine_locked(
                    lease,
                    reason="stale_runtime_state_verified_copy",
                    evidence_ref=evidence_ref,
                )
                raise LeaseTransitionError(
                    "verified copy does not match active capture"
                )
            if not self._has_lease_event(
                lease_id, "runtime_state_acquisition_requested"
            ):
                self._quarantine_locked(
                    lease,
                    reason="verified_copy_without_acquisition_request",
                    evidence_ref=evidence_ref,
                )
                raise LeaseTransitionError(
                    "verified copy requires an acquisition request"
                )
            if not lease.holds_sequence:
                self._quarantine_locked(
                    lease,
                    reason="duplicate_runtime_state_verified_copy",
                    evidence_ref=evidence_ref,
                )
                raise LeaseTransitionError("verified copy was already recorded")
            if self._held_by_slot.get(lease.expected_slot_id) != lease_id:
                raise LeaseTransitionError(
                    "verified copy Lease does not own its runtime slot"
                )
            updated = self.ledger.transition_lease(
                lease_id,
                state=LeaseState.CHECKPOINTING,
                holds_sequence=False,
                event=self._event(
                    "runtime_state_acquisition_copied",
                    lease=lease,
                    details={
                        "capture_id": capture_id,
                        "evidence_ref": evidence_ref,
                    },
                ),
            )
            self._held_by_slot.pop(lease.expected_slot_id, None)
            if lease.expected_slot_id not in self._quarantined_slots:
                self._available_slots.add(lease.expected_slot_id)
            self._condition.notify_all()
            return updated

    def commit_runtime_state_payload(
        self,
        lease_id: str,
        *,
        capture_id: str,
        digest: str,
        size_bytes: int,
        evidence_ref: str,
    ) -> LeaseRecord:
        with self._condition:
            lease = self.ledger.get_lease(lease_id)
            require_state(lease, LeaseState.CHECKPOINTING)
            if lease.capture_id != capture_id:
                self._quarantine_locked(
                    lease,
                    reason="stale_runtime_state_payload_commit",
                    evidence_ref=evidence_ref,
                )
                raise LeaseTransitionError(
                    "payload commit does not match active capture"
                )
            requested = self._has_lease_event(
                lease_id, "runtime_state_acquisition_requested"
            )
            completed_events = [
                stored.event
                for stored in self.ledger.read_events()
                if stored.event.lease_id == lease_id
                and stored.event.event_type
                == "runtime_state_acquisition_completed"
            ]
            if not requested or len(completed_events) != 1:
                self._quarantine_locked(
                    lease,
                    reason="payload_without_acquisition_completion",
                    evidence_ref=evidence_ref,
                )
                raise LeaseTransitionError(
                    "payload commit requires exactly one acquisition completion"
                )
            completed = completed_events[0]
            if (
                completed.details.get("capture_id") != capture_id
                or completed.details.get("payload_digest") != digest
                or completed.details.get("payload_size_bytes") != size_bytes
                or completed.details.get("request_ref")
                != lease.request.request_ref
                or completed.details.get("runtime_instance_id")
                != lease.runtime_instance_id
                or completed.details.get("slot_id") != lease.expected_slot_id
                or completed.details.get("slot_epoch") != lease.slot_epoch
                or completed.details.get("observed_seq_id")
                != lease.observed_seq_id
            ):
                self._quarantine_locked(
                    lease,
                    reason="payload_acquisition_evidence_mismatch",
                    evidence_ref=evidence_ref,
                )
                raise LeaseTransitionError(
                    "payload does not match acquisition completion"
                )
            require_transition(lease, LeaseState.CHECKPOINTED)
            return self.ledger.transition_lease(
                lease_id,
                state=LeaseState.CHECKPOINTED,
                event=self._event(
                    "runtime_state_payload_committed",
                    lease=lease,
                    details={
                        "capture_id": capture_id,
                        "payload_digest": digest,
                        "payload_size_bytes": size_bytes,
                        "evidence_ref": evidence_ref,
                        "context_ref": lease.context_ref,
                        "expected_slot_id": lease.expected_slot_id,
                        "response_confirmed_slot_id": (
                            lease.response_confirmed_slot_id
                        ),
                        "slot_epoch": lease.slot_epoch,
                    },
                ),
            )

    def mark_capture_skipped(self, lease_id: str, *, reason: str) -> LeaseRecord:
        with self._condition:
            if not self.allow_capture_skip:
                raise LeaseTransitionError(
                    "capture_skipped is disabled in the required-capture profile"
                )
            lease = self.ledger.get_lease(lease_id)
            require_state(lease, LeaseState.QUIESCING)
            if not lease.backend_released:
                raise LeaseTransitionError(
                    "slot idle must be confirmed before capture decision"
                )
            require_transition(lease, LeaseState.CAPTURE_SKIPPED)
            return self.ledger.transition_lease(
                lease_id,
                state=LeaseState.CAPTURE_SKIPPED,
                capture_skipped_reason=reason,
                event=self._event(
                    "capture_skipped",
                    lease=lease,
                    details={"reason": reason},
                ),
            )

    def release(self, lease_id: str) -> LeaseRecord:
        with self._condition:
            lease = self.ledger.get_lease(lease_id)
            if lease.state is LeaseState.ACTIVE:
                raise LeaseTransitionError(
                    "Lease must pass through QUIESCING and payload commit before release"
                )
            require_state(
                lease, LeaseState.CHECKPOINTED, LeaseState.CAPTURE_SKIPPED
            )
            if not lease.backend_released:
                raise LeaseTransitionError("slot idle must precede Lease release")
            require_transition(lease, LeaseState.RELEASED)
            updated = self.ledger.transition_lease(
                lease_id,
                state=LeaseState.RELEASED,
                holds_sequence=False,
                event=self._event("lease_released", lease=lease),
            )
            if lease.holds_sequence:
                if self._held_by_slot.get(lease.expected_slot_id) == lease_id:
                    self._held_by_slot.pop(lease.expected_slot_id, None)
                    if lease.expected_slot_id not in self._quarantined_slots:
                        self._available_slots.add(lease.expected_slot_id)
            self._condition.notify_all()
            return updated

    def _quarantine_locked(
        self, lease: LeaseRecord, *, reason: str, evidence_ref: str | None
    ) -> LeaseRecord:
        require_transition(lease, LeaseState.QUARANTINED)
        resolved_evidence_ref = (
            f"controller-policy:{reason}"
            if evidence_ref is None
            else evidence_ref
        )
        details: dict[str, JsonValue] = {
            "reason": reason,
            "evidence_ref": resolved_evidence_ref,
        }
        if not lease.holds_sequence:
            updated = self.ledger.transition_lease(
                lease.lease_id,
                state=LeaseState.QUARANTINED,
                holds_sequence=False,
                event=self._event(
                    "runtime_state_post_copy_failed",
                    lease=lease,
                    details=details,
                ),
            )
            self._condition.notify_all()
            return updated
        updated = self.ledger.transition_lease(
            lease.lease_id,
            state=LeaseState.QUARANTINED,
            holds_sequence=True,
            event=self._event(
                "slot_quarantined", lease=lease, details=details
            ),
        )
        self._available_slots.discard(lease.expected_slot_id)
        self._quarantined_slots.add(lease.expected_slot_id)
        self._condition.notify_all()
        return updated

    def quarantine(
        self, lease_id: str, *, reason: str, evidence_ref: str | None = None
    ) -> LeaseRecord:
        with self._condition:
            return self._quarantine_locked(
                self.ledger.get_lease(lease_id),
                reason=reason,
                evidence_ref=evidence_ref,
            )

    @property
    def quarantined_slots(self) -> frozenset[int]:
        with self._condition:
            return frozenset(self._quarantined_slots)

    def recover_controller(self) -> tuple[str, ...]:
        with self._condition:
            affected = self.ledger.recover_controller_state(
                controller_instance_id=self.controller_instance_id,
                runtime_instance_id=self.runtime_instance_id,
            )
            for lease_id in affected:
                lease = self.ledger.get_lease(lease_id)
                self._available_slots.discard(lease.expected_slot_id)
                self._quarantined_slots.add(lease.expected_slot_id)
                self._held_by_slot[lease.expected_slot_id] = lease_id
            self._condition.notify_all()
            return affected

    def handle_runtime_restart(
        self, *, new_runtime_instance_id: str, evidence_ref: str
    ) -> tuple[str, ...]:
        with self._condition:
            old_runtime_instance_id = self.runtime_instance_id
            affected = self.ledger.handle_runtime_restart(
                controller_instance_id=self.controller_instance_id,
                old_runtime_instance_id=old_runtime_instance_id,
                new_runtime_instance_id=new_runtime_instance_id,
                evidence_ref=evidence_ref,
            )
            self.runtime_instance_id = new_runtime_instance_id
            self._held_by_slot.clear()
            self._quarantined_slots.clear()
            self._available_slots = set(range(self.slot_count))
            self._dispatch_authorized.clear()
            self._condition.notify_all()
            return affected
