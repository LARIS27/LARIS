from __future__ import annotations

from collections.abc import Callable

from .binding_ledger import BindingLedger
from laris_onpremise.core.records import LeaseRecord, LeaseState

from laris_onpremise.kv_state.acquisition.contracts import (
    RuntimeStateAcquirer,
    RuntimeStateCaptureRequest,
    RuntimeStatePayloadCommit,
    validate_payload_binding,
)


class AcquisitionCoordinationError(RuntimeError):
    """Raised when a Lease cannot authorize one selective acquisition."""


class AcquisitionCoordinator:
    """Join authenticated Lease lineage to anonymous observer metadata."""

    def __init__(
        self,
        *,
        ledger: BindingLedger,
        acquirer: RuntimeStateAcquirer,
        runtime_pid: int,
        runtime_start_time_ticks: int,
    ) -> None:
        if not isinstance(ledger, BindingLedger):
            raise AcquisitionCoordinationError("ledger must be a BindingLedger")
        if not callable(getattr(acquirer, "capture", None)):
            raise AcquisitionCoordinationError("acquirer must implement capture")
        _require_positive_int(runtime_pid, "runtime_pid")
        _require_positive_int(
            runtime_start_time_ticks, "runtime_start_time_ticks"
        )
        self.ledger = ledger
        self.acquirer = acquirer
        self.runtime_pid = runtime_pid
        self.runtime_start_time_ticks = runtime_start_time_ticks

    def capture(
        self,
        lease: LeaseRecord,
        *,
        on_copy_sealed: Callable[[LeaseRecord], None] | None = None,
    ) -> RuntimeStatePayloadCommit:
        if not isinstance(lease, LeaseRecord):
            raise AcquisitionCoordinationError("lease must be a LeaseRecord")
        try:
            current = self.ledger.get_lease(lease.lease_id)
        except KeyError as error:
            raise AcquisitionCoordinationError(
                "Lease is not present in the binding ledger"
            ) from error
        if current != lease:
            raise AcquisitionCoordinationError(
                "Lease record is stale relative to the binding ledger"
            )
        if lease.state is not LeaseState.CHECKPOINTING:
            raise AcquisitionCoordinationError(
                "Lease must be CHECKPOINTING before selective acquisition"
            )
        if not lease.holds_sequence or not lease.backend_released:
            raise AcquisitionCoordinationError(
                "CHECKPOINTING Lease must still hold an idle sequence"
            )
        if lease.capture_id is None:
            raise AcquisitionCoordinationError(
                "CHECKPOINTING Lease has no capture_id"
            )
        if lease.observed_seq_id is None:
            raise AcquisitionCoordinationError(
                "CHECKPOINTING Lease has no observed_seq_id"
            )
        if lease.slot_epoch is None:
            raise AcquisitionCoordinationError(
                "CHECKPOINTING Lease has no slot_epoch"
            )
        if lease.response_confirmed_slot_id is None:
            raise AcquisitionCoordinationError(
                "CHECKPOINTING Lease has no confirmed slot assignment"
            )
        if lease.observed_seq_id != lease.expected_seq_id:
            raise AcquisitionCoordinationError(
                "observed sequence does not match the authorized Lease"
            )

        events = tuple(
            stored.event
            for stored in self.ledger.read_events()
            if stored.event.lease_id == lease.lease_id
        )
        sent = tuple(
            event for event in events if event.event_type == "upstream_request_sent"
        )
        capture_started = tuple(
            event
            for event in events
            if event.event_type == "runtime_state_capture_started"
        )
        acquisition_requested = tuple(
            event
            for event in events
            if event.event_type == "runtime_state_acquisition_requested"
        )
        if len(sent) != 1:
            raise AcquisitionCoordinationError(
                "Lease must have exactly one upstream_request_sent event"
            )
        if len(capture_started) != 1:
            raise AcquisitionCoordinationError(
                "Lease must have exactly one runtime_state_capture_started event"
            )
        if len(acquisition_requested) != 1:
            raise AcquisitionCoordinationError(
                "Lease must have exactly one runtime_state_acquisition_requested event"
            )

        started_event = capture_started[0]
        requested_event = acquisition_requested[0]
        if started_event.details.get("capture_id") != lease.capture_id:
            raise AcquisitionCoordinationError(
                "capture-start event does not match the active capture_id"
            )
        if requested_event.details.get("capture_id") != lease.capture_id:
            raise AcquisitionCoordinationError(
                "acquisition-request event does not match the active capture_id"
            )
        if requested_event.details.get("observed_seq_id") != lease.observed_seq_id:
            raise AcquisitionCoordinationError(
                "acquisition-request event does not match observed_seq_id"
            )
        interval_start = sent[0].monotonic_ns
        interval_end = requested_event.monotonic_ns
        if not (
            interval_start < started_event.monotonic_ns <= interval_end
        ):
            raise AcquisitionCoordinationError(
                "capture events are not ordered after upstream submission"
            )

        request = RuntimeStateCaptureRequest(
            capture_id=lease.capture_id,
            lease_id=lease.lease_id,
            request_ref=lease.request.request_ref,
            runtime_instance_id=lease.runtime_instance_id,
            runtime_pid=self.runtime_pid,
            runtime_start_time_ticks=self.runtime_start_time_ticks,
            slot_id=lease.response_confirmed_slot_id,
            slot_epoch=lease.slot_epoch,
            observed_seq_id=lease.observed_seq_id,
            interval_start_monotonic_ns=interval_start,
            interval_end_monotonic_ns=interval_end,
        )
        if on_copy_sealed is None:
            commit = self.acquirer.capture(request)
        else:
            commit = self.acquirer.capture(
                request,
                on_copy_sealed=lambda: on_copy_sealed(lease),
            )
        validate_payload_binding(commit, request)
        return commit


def _require_positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise AcquisitionCoordinationError(f"{field} must be a positive integer")
    return value


__all__ = ["AcquisitionCoordinationError", "AcquisitionCoordinator"]
