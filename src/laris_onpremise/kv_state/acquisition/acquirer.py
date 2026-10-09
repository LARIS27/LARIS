from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import re
import struct
import time
from typing import Protocol

from .contracts import (
    SIMPLE_KV_MEMORY_FAMILY,
    SIMPLE_KV_PAYLOAD_FORMAT,
    RuntimeStateCaptureRequest,
    RuntimeStateCopySealedCallback,
    RuntimeStatePayloadCommit,
    validate_payload_binding,
)
from .layout import SelectedCell, SimpleKvLayout, plan_selected_read_runs
from .payload import encode_simple_kv_payload_parts_from_runs
from .reader import AddressReader, ProcessIdentityError, ProcessMemoryReader
from laris_onpremise.kv_state.evidence.store import RuntimeStatePayloadStore


class AcquisitionError(RuntimeError):
    """Base error for fail-closed selective runtime-state acquisition."""


class AcquisitionStabilityError(AcquisitionError):
    """Raised when a coherent target-scoped snapshot cannot be proven."""


class AcquisitionTargetConflict(AcquisitionStabilityError):
    """Raised when target ownership changes during one Lease capture."""


class AcquisitionObserverIntegrityError(AcquisitionStabilityError):
    """Raised when observer/process evidence cannot authorize one Lease."""


class AcquisitionBindingError(AcquisitionObserverIntegrityError):
    """Raised when runtime observation does not match the capture request."""


@dataclass(frozen=True)
class RuntimeStateTargetMarker:
    observed_seq_id: int
    stream_index: int
    target_decode_count: int
    target_apply_watermark: int
    target_ownership_generation: int
    target_cell_map_digest: str
    layout_profile_digest: str
    layout_topology_digest: str
    kv_backing_provenance_digest: str
    observer_loss_count: int
    parse_failure_count: int
    unmatched_call_count: int
    unsupported_mutation_count: int

    def __init__(
        self,
        observed_seq_id: int,
        stream_index: int,
        target_decode_count: int,
        target_apply_watermark: int,
        target_ownership_generation: int,
        target_cell_map_digest: str,
        layout_profile_digest: str,
        layout_topology_digest: str,
        kv_backing_provenance_digest: str,
        observer_loss_count: int = 0,
        parse_failure_count: int = 0,
        unmatched_call_count: int = 0,
        unsupported_mutation_count: int = 0,
    ) -> None:
        object.__setattr__(self, "observed_seq_id", observed_seq_id)
        object.__setattr__(self, "stream_index", stream_index)
        object.__setattr__(self, "target_decode_count", target_decode_count)
        object.__setattr__(self, "target_apply_watermark", target_apply_watermark)
        object.__setattr__(
            self, "target_ownership_generation", target_ownership_generation
        )
        object.__setattr__(self, "target_cell_map_digest", target_cell_map_digest)
        object.__setattr__(self, "layout_profile_digest", layout_profile_digest)
        object.__setattr__(self, "layout_topology_digest", layout_topology_digest)
        object.__setattr__(
            self,
            "kv_backing_provenance_digest",
            kv_backing_provenance_digest,
        )
        object.__setattr__(self, "observer_loss_count", observer_loss_count)
        object.__setattr__(self, "parse_failure_count", parse_failure_count)
        object.__setattr__(self, "unmatched_call_count", unmatched_call_count)
        object.__setattr__(
            self, "unsupported_mutation_count", unsupported_mutation_count
        )


@dataclass(frozen=True)
class ObservedKvBackingRange:
    backing_ref: str
    address_start: int
    address_end: int
    provenance_digest: str


@dataclass(frozen=True)
class RuntimeStateAcquisitionView:
    runtime_instance_id: str
    runtime_pid: int
    runtime_start_time_ticks: int
    observed_seq_id: int
    snapshot_id: str
    snapshot_cut_monotonic_ns: int
    target_marker: RuntimeStateTargetMarker
    layout: SimpleKvLayout
    cells: tuple[SelectedCell, ...]
    kv_backing_ranges: tuple[ObservedKvBackingRange, ...]
    acquisition_evidence_ref: str
    observer_loss_count: int
    parse_failure_count: int
    unmatched_call_count: int
    unsupported_mutation_count: int


class RuntimeStateAcquisitionSource(Protocol):
    def resolve(
        self, request: RuntimeStateCaptureRequest
    ) -> RuntimeStateAcquisitionView: ...

    def target_marker(
        self,
        request: RuntimeStateCaptureRequest,
        view: RuntimeStateAcquisitionView,
    ) -> RuntimeStateTargetMarker: ...


ReaderFactory = Callable[[int, int], AddressReader]


class SelectiveRuntimeStateAcquirer:
    """Acquire only the observer-authorized K/V spans for one request."""

    def __init__(
        self,
        *,
        observation_source: RuntimeStateAcquisitionSource,
        store: RuntimeStatePayloadStore,
        reader_factory: ReaderFactory | None = None,
        phase_sink: Callable[[str, int, int], None] | None = None,
    ) -> None:
        self.observation_source = observation_source
        self.store = store
        self.phase_sink = phase_sink
        self.reader_factory = reader_factory or (
            lambda pid, start_ticks: ProcessMemoryReader(
                pid=pid, expected_start_time_ticks=start_ticks
            )
        )

    def _record_phase(self, phase: str, started_ns: int, byte_count: int = 0) -> None:
        if self.phase_sink is not None:
            self.phase_sink(phase, time.monotonic_ns() - started_ns, byte_count)

    def capture(
        self,
        request: RuntimeStateCaptureRequest,
        *,
        on_copy_sealed: RuntimeStateCopySealedCallback | None = None,
    ) -> RuntimeStatePayloadCommit:
        started_ns = time.monotonic_ns() if self.phase_sink is not None else 0
        view = self.observation_source.resolve(request)
        self._record_phase("snapshot_resolve", started_ns)
        _validate_view_binding(view, request)
        _validate_observer_health(view)
        spans = plan_selected_read_runs(view.layout, view.cells)
        backing_provenance_digest = _validate_backing_ranges(
            view.kv_backing_ranges, spans
        )

        baseline = view.target_marker
        _validate_target_marker(baseline)
        try:
            reader = self.reader_factory(
                request.runtime_pid, request.runtime_start_time_ticks
            )
        except ProcessIdentityError as error:
            raise AcquisitionObserverIntegrityError(str(error)) from error
        try:
            try:
                read_trace = tuple((span.address, span.size) for span in spans)
                started_ns = time.monotonic_ns() if self.phase_sink is not None else 0
                read_values = reader.pread_many_exact(read_trace)
                self._record_phase(
                    "memory_read", started_ns, sum(size for _, size in read_trace)
                )
            except ProcessIdentityError as error:
                raise AcquisitionObserverIntegrityError(str(error)) from error
            if len(read_values) != len(spans):
                raise AcquisitionStabilityError(
                    "address reader returned the wrong read-round cardinality"
                )
            values = dict(zip(spans, read_values, strict=True))
            started_ns = time.monotonic_ns() if self.phase_sink is not None else 0
            after = self.observation_source.target_marker(request, view)
            self._record_phase("marker_after", started_ns)
            _validate_target_marker(after)
            if after != baseline:
                raise AcquisitionTargetConflict(
                    "target runtime state changed during acquisition"
                )
        finally:
            close = getattr(reader, "close", None)
            if callable(close):
                close()

        expected_trace = tuple((span.address, span.size) for span in spans)
        try:
            actual_trace = tuple(reader.access_trace)
        except (AttributeError, TypeError) as error:
            raise AcquisitionStabilityError(
                "address reader did not expose a sealed access trace"
            ) from error
        if actual_trace != expected_trace:
            raise AcquisitionStabilityError(
                "actual process-memory read trace does not equal the plan once"
            )
        if on_copy_sealed is not None:
            on_copy_sealed()

        started_ns = time.monotonic_ns() if self.phase_sink is not None else 0
        payload_parts = encode_simple_kv_payload_parts_from_runs(
            view.layout,
            view.cells,
            values,
            planned_runs=spans,
        )
        payload_size = sum(len(part) for part in payload_parts)
        self._record_phase("payload_encode", started_ns, payload_size)
        positions = tuple(cell.position for cell in view.cells)
        cell_indices = tuple(cell.cell_index for cell in view.cells)
        position_range = (min(positions), max(positions))
        kv_cell_ranges = _compress_ranges(cell_indices)
        next_position = max(positions) + 1
        manifest = {
            "schema_version": "1.1",
            "capture_id": request.capture_id,
            "lease_id": request.lease_id,
            "request_ref": request.request_ref,
            "runtime_instance_id": request.runtime_instance_id,
            "runtime_pid": request.runtime_pid,
            "runtime_start_time_ticks": request.runtime_start_time_ticks,
            "slot_id": request.slot_id,
            "slot_epoch": request.slot_epoch,
            "observed_seq_id": request.observed_seq_id,
            "observed_stream_index": view.cells[0].stream_index,
            "snapshot_id": view.snapshot_id,
            "snapshot_cut_monotonic_ns": view.snapshot_cut_monotonic_ns,
            "target_marker_before_digest": _digest_target_marker(baseline),
            "target_marker_after_digest": _digest_target_marker(after),
            "layout_profile_digest": view.layout.profile_digest,
            "kv_backing_provenance_digest": backing_provenance_digest,
            "kv_backing_range_count": len(view.kv_backing_ranges),
            "memory_family": SIMPLE_KV_MEMORY_FAMILY,
            "payload_format": SIMPLE_KV_PAYLOAD_FORMAT,
            "position_range": list(position_range),
            "kv_cell_ranges": [list(item) for item in kv_cell_ranges],
            "next_position": next_position,
            "planned_read_span_count": len(spans),
            "executed_read_count": len(actual_trace),
            "read_rounds": 1,
            "read_byte_count": sum(size for _, size in actual_trace),
            "read_size_range": [
                min(span.size for span in spans),
                max(span.size for span in spans),
            ],
            "read_trace_digest": _digest_access_trace(actual_trace),
            "acquisition_evidence_ref": view.acquisition_evidence_ref,
            "contains_prompt_text": False,
            "contains_response_text": False,
            "contains_explicit_token_history": False,
        }
        started_ns = time.monotonic_ns() if self.phase_sink is not None else 0
        payload_path, _, size, digest = self.store.commit_parts(
            request_ref=request.request_ref,
            payload_parts=payload_parts,
            manifest=manifest,
            phase_sink=self.phase_sink,
        )
        self._record_phase("payload_commit", started_ns, size)
        commit = RuntimeStatePayloadCommit(
            capture_id=request.capture_id,
            lease_id=request.lease_id,
            request_ref=request.request_ref,
            runtime_instance_id=request.runtime_instance_id,
            slot_id=request.slot_id,
            slot_epoch=request.slot_epoch,
            observed_seq_id=request.observed_seq_id,
            memory_family=SIMPLE_KV_MEMORY_FAMILY,
            payload_format=SIMPLE_KV_PAYLOAD_FORMAT,
            payload_path=payload_path,
            payload_size_bytes=size,
            payload_digest=digest,
            position_range=position_range,
            kv_cell_ranges=kv_cell_ranges,
            next_position=next_position,
            acquisition_evidence_ref=view.acquisition_evidence_ref,
        )
        validate_payload_binding(commit, request)
        return commit


def _validate_view_binding(
    view: RuntimeStateAcquisitionView,
    request: RuntimeStateCaptureRequest,
) -> None:
    for field in (
        "runtime_instance_id",
        "runtime_pid",
        "runtime_start_time_ticks",
        "observed_seq_id",
    ):
        if getattr(view, field) != getattr(request, field):
            raise AcquisitionBindingError(
                f"observation {field} does not match capture request"
            )
    if not view.acquisition_evidence_ref:
        raise AcquisitionBindingError("acquisition_evidence_ref is missing")
    if not view.cells:
        raise AcquisitionBindingError("observation contains no selected cells")
    if any(cell.seq_id != request.observed_seq_id for cell in view.cells):
        raise AcquisitionBindingError(
            "selected cell sequence does not match observed_seq_id"
        )
    if not isinstance(view.snapshot_id, str) or not view.snapshot_id.strip():
        raise AcquisitionBindingError("snapshot_id is missing")
    if (
        isinstance(view.snapshot_cut_monotonic_ns, bool)
        or not isinstance(view.snapshot_cut_monotonic_ns, int)
        or view.snapshot_cut_monotonic_ns < request.interval_end_monotonic_ns
    ):
        raise AcquisitionBindingError(
            "snapshot cut must not precede the capture interval end"
        )
    _validate_target_marker(view.target_marker)


def _validate_observer_health(view: RuntimeStateAcquisitionView) -> None:
    for field in (
        "observer_loss_count",
        "parse_failure_count",
        "unmatched_call_count",
        "unsupported_mutation_count",
    ):
        value = getattr(view, field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise AcquisitionObserverIntegrityError(f"{field} is invalid")
        if value:
            raise AcquisitionObserverIntegrityError(f"{field} is nonzero")
        if value != getattr(view.target_marker, field):
            raise AcquisitionObserverIntegrityError(
                f"{field} disagrees with the atomic snapshot marker"
            )


def _validate_target_marker(marker: RuntimeStateTargetMarker) -> None:
    if not isinstance(marker, RuntimeStateTargetMarker):
        raise AcquisitionObserverIntegrityError("target marker has an invalid type")
    for field in (
        "observed_seq_id",
        "stream_index",
        "target_decode_count",
        "target_apply_watermark",
        "target_ownership_generation",
        "observer_loss_count",
        "parse_failure_count",
        "unmatched_call_count",
        "unsupported_mutation_count",
    ):
        value = getattr(marker, field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise AcquisitionObserverIntegrityError(
                "target marker contains an invalid counter"
            )
        if field in {
            "observer_loss_count",
            "parse_failure_count",
            "unmatched_call_count",
            "unsupported_mutation_count",
        } and value:
            raise AcquisitionObserverIntegrityError(
                f"target marker {field} is nonzero"
            )
    for field in (
        "target_cell_map_digest",
        "layout_profile_digest",
        "layout_topology_digest",
        "kv_backing_provenance_digest",
    ):
        if _SHA256_DIGEST.fullmatch(getattr(marker, field)) is None:
            raise AcquisitionObserverIntegrityError(
                f"target marker {field} is invalid"
            )


_SHA256_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")


def _validate_backing_ranges(
    ranges: tuple[ObservedKvBackingRange, ...], spans
) -> str:
    if not isinstance(ranges, tuple) or not ranges:
        raise AcquisitionBindingError(
            "observer-confirmed KV backing ranges are missing"
        )
    previous_end = -1
    digest = hashlib.sha256(b"LARIS-KV-BACKING-RANGES-V1\x00")
    for item in ranges:
        if not isinstance(item, ObservedKvBackingRange):
            raise AcquisitionBindingError("KV backing range has an invalid type")
        if not item.backing_ref:
            raise AcquisitionBindingError("KV backing range ref is missing")
        if (
            isinstance(item.address_start, bool)
            or isinstance(item.address_end, bool)
            or not isinstance(item.address_start, int)
            or not isinstance(item.address_end, int)
            or item.address_start < 0
            or item.address_end <= item.address_start
        ):
            raise AcquisitionBindingError("KV backing range is invalid")
        if item.address_start < previous_end:
            raise AcquisitionBindingError("KV backing ranges overlap or are unsorted")
        if _SHA256_DIGEST.fullmatch(item.provenance_digest) is None:
            raise AcquisitionBindingError(
                "KV backing provenance digest is invalid"
            )
        previous_end = item.address_end
        ref = item.backing_ref.encode("utf-8")
        digest.update(struct.pack("<I", len(ref)))
        digest.update(ref)
        digest.update(struct.pack("<QQ", item.address_start, item.address_end))
        digest.update(bytes.fromhex(item.provenance_digest.removeprefix("sha256:")))
    for span in spans:
        span_end = span.address + span.size
        matches = [
            item
            for item in ranges
            if item.address_start <= span.address and span_end <= item.address_end
        ]
        if len(matches) != 1:
            raise AcquisitionBindingError(
                "planned read span is not contained in one confirmed KV backing range"
            )
    return "sha256:" + digest.hexdigest()


def _digest_target_marker(marker: RuntimeStateTargetMarker) -> str:
    digest = hashlib.sha256(b"LARIS-RUNTIME-TARGET-MARKER-V1\x00")
    digest.update(
        struct.pack(
            "<QQQQQ",
            marker.observed_seq_id,
            marker.stream_index,
            marker.target_decode_count,
            marker.target_apply_watermark,
            marker.target_ownership_generation,
        )
    )
    for value in (
        marker.target_cell_map_digest,
        marker.layout_profile_digest,
        marker.layout_topology_digest,
        marker.kv_backing_provenance_digest,
    ):
        digest.update(bytes.fromhex(value.removeprefix("sha256:")))
    digest.update(
        struct.pack(
            "<QQQQ",
            marker.observer_loss_count,
            marker.parse_failure_count,
            marker.unmatched_call_count,
            marker.unsupported_mutation_count,
        )
    )
    return "sha256:" + digest.hexdigest()


def _digest_access_trace(trace: tuple[tuple[int, int], ...]) -> str:
    digest = hashlib.sha256(b"LARIS-PROCESS-MEMORY-READ-TRACE-V1\x00")
    for ordinal, (address, size) in enumerate(trace):
        digest.update(struct.pack("<IQQ", ordinal, address, size))
    return "sha256:" + digest.hexdigest()


def _compress_ranges(values: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    ranges: list[tuple[int, int]] = []
    start = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append((start, previous))
        start = previous = value
    ranges.append((start, previous))
    return tuple(ranges)
