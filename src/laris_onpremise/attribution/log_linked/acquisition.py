from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import json
import struct
from typing import Protocol

from laris_onpremise.kv_state.acquisition.acquirer import (
    AcquisitionError,
    ObservedKvBackingRange,
    RuntimeStateTargetMarker,
    _validate_backing_ranges,
    _validate_target_marker,
)
from laris_onpremise.kv_state.acquisition.contracts import RuntimeStateCaptureRequest
from laris_onpremise.kv_state.acquisition.layout import (
    ReadRun,
    SelectedCell,
    SimpleKvLayout,
    plan_selected_read_runs,
)
from laris_onpremise.kv_state.acquisition.payload import (
    encode_simple_kv_payload_from_runs,
    inspect_simple_kv_payload,
)
from laris_onpremise.kv_state.acquisition.reader import (
    AddressReader,
    ProcessIdentityError,
    ProcessMemoryReader,
)
from laris_onpremise.kv_state.observation.runtime_state_contract import (
    RuntimeStateSnapshot,
    TokenCellEvent,
)
from laris_onpremise.kv_state.observation.runtime_state_repository import (
    RuntimeStateClientObservationSource,
    RuntimeStateObservationError,
    _latest_cell_owners,
)
from laris_onpremise.kv_state.observation.runtime_state_service import (
    NativeCapturedUnitDescriptor,
    NativeStreamVersionSnapshot,
    RuntimeStateTopologySummary,
)
from laris_onpremise.kv_state.observation.runtime_state_native_units import (
    NativeUnitFragmentRecord,
    NativeUnitScheduleRecord,
    NativeUnitSpoolError,
    iter_native_unit_records,
)
from laris_onpremise.kv_state.observation.runtime_state_profile import (
    RuntimeStateObserverProfile,
)

from .generation import UnitRequest
from .state_units import (
    StateUnitManifest,
    StateUnitStore,
    StoredStateUnit,
)


class LogLinkedAcquisitionError(RuntimeError):
    """Raised when one sequence state unit cannot be captured coherently."""


@dataclass(frozen=True)
class LogLinkedStreamVersionToken:
    stream_index: int
    apply_call_id: int
    mutation_generation: int
    backing_generation: int
    layout_generation: int

    def __post_init__(self) -> None:
        for field in (
            "stream_index",
            "apply_call_id",
            "mutation_generation",
            "backing_generation",
            "layout_generation",
        ):
            value = getattr(self, field)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise LogLinkedAcquisitionError(
                    f"stream version {field} is invalid"
                )
        if self.apply_call_id == 0:
            raise LogLinkedAcquisitionError(
                "stream version apply_call_id must be positive"
            )


@dataclass(frozen=True)
class LogLinkedStabilityCut:
    marker: RuntimeStateTargetMarker
    version: LogLinkedStreamVersionToken


@dataclass(frozen=True)
class LogLinkedAcquisitionView:
    runtime_instance_id: str
    runtime_pid: int
    runtime_start_time_ticks: int
    apply_call_id: int
    apply_monotonic_ns: int
    layout: SimpleKvLayout
    cells: tuple[SelectedCell, ...]
    target_marker: RuntimeStateTargetMarker
    stream_version: LogLinkedStreamVersionToken
    kv_backing_ranges: tuple[ObservedKvBackingRange, ...]
    observer_loss_count: int
    parse_failure_count: int
    unmatched_call_count: int
    unsupported_mutation_count: int


@dataclass(frozen=True)
class CopiedStateUnit:
    """Immutable live-memory copy ready for offline encoding and persistence."""

    request: UnitRequest
    view: LogLinkedAcquisitionView
    marker: RuntimeStateTargetMarker
    stream_version: LogLinkedStreamVersionToken
    backing_digest: str
    runs: tuple[ReadRun, ...]
    values: tuple[bytes, ...]


@dataclass(frozen=True)
class PreparedStateUnit:
    """Offline-encoded state unit ready for ordered durable publication."""

    request: UnitRequest
    manifest: StateUnitManifest
    payload: bytes
    manifest_digest: str


class LogLinkedObservationSource(Protocol):
    def resolve(self, request: UnitRequest) -> LogLinkedAcquisitionView: ...

    def stability_cut(
        self, request: UnitRequest, view: LogLinkedAcquisitionView
    ) -> LogLinkedStabilityCut: ...


class RuntimeStateUnitObservationSource:
    """Select BASE/DELTA cells from the existing validated snapshot service."""

    def __init__(self, repository: RuntimeStateClientObservationSource) -> None:
        if not isinstance(repository, RuntimeStateClientObservationSource):
            raise LogLinkedAcquisitionError(
                "repository must be a RuntimeStateClientObservationSource"
            )
        self.repository = repository

    def resolve(self, request: UnitRequest) -> LogLinkedAcquisitionView:
        capture_request = _snapshot_request(request)
        try:
            cut = self.repository.log_linked_cut(capture_request)
            snapshot = cut.snapshot
            cells = _select_state_unit_cells(snapshot, request)
            backing_ranges = self.repository.backing_ranges(snapshot)
            marker = self.repository.marker_for_cells(
                snapshot,
                request.ref.seq_id,
                cells,
                backing_ranges,
            )
            version = _stream_version_token(
                snapshot,
                cut.stream_version,
                request.ref.seq_id,
            )
        except (AcquisitionError, RuntimeStateObservationError) as error:
            raise LogLinkedAcquisitionError(str(error)) from error
        if marker.target_apply_watermark != request.apply_call_id:
            raise LogLinkedAcquisitionError(
                "completed apply is no longer the current sequence cut"
            )
        return LogLinkedAcquisitionView(
            runtime_instance_id=snapshot.runtime_instance_id,
            runtime_pid=snapshot.runtime_pid,
            runtime_start_time_ticks=snapshot.runtime_start_time_ticks,
            apply_call_id=request.apply_call_id,
            apply_monotonic_ns=request.monotonic_ns,
            layout=snapshot.layout,
            cells=cells,
            target_marker=marker,
            stream_version=version,
            kv_backing_ranges=backing_ranges,
            observer_loss_count=snapshot.observer_loss_count,
            parse_failure_count=snapshot.parse_failure_count,
            unmatched_call_count=snapshot.unmatched_call_count,
            unsupported_mutation_count=snapshot.unsupported_mutation_count,
        )

    def stability_cut(
        self, request: UnitRequest, view: LogLinkedAcquisitionView
    ) -> LogLinkedStabilityCut:
        try:
            cut = self.repository.log_linked_cut(_snapshot_request(request))
            snapshot = cut.snapshot
            backing_ranges = self.repository.backing_ranges(snapshot)
            return LogLinkedStabilityCut(
                marker=self.repository.marker_for_cells(
                    snapshot,
                    request.ref.seq_id,
                    view.cells,
                    backing_ranges,
                ),
                version=_stream_version_token(
                    snapshot,
                    cut.stream_version,
                    request.ref.seq_id,
                ),
            )
        except (AcquisitionError, RuntimeStateObservationError) as error:
            raise LogLinkedAcquisitionError(str(error)) from error


ReaderFactory = Callable[[int, int], AddressReader]


class LogLinkedStateUnitAcquirer:
    """Read each Observer-authorized KV span once for one state unit."""

    def __init__(
        self,
        *,
        store: StateUnitStore,
        observation_source: LogLinkedObservationSource | None = None,
        observer_profile: RuntimeStateObserverProfile | None = None,
        native_topology: RuntimeStateTopologySummary | None = None,
        reader_factory: ReaderFactory | None = None,
    ) -> None:
        if observation_source is not None and (
            not callable(getattr(observation_source, "resolve", None))
            or not callable(getattr(observation_source, "stability_cut", None))
        ):
            raise LogLinkedAcquisitionError("observation_source is invalid")
        if observer_profile is not None and not isinstance(
            observer_profile, RuntimeStateObserverProfile
        ):
            raise LogLinkedAcquisitionError("observer_profile is invalid")
        if native_topology is not None and not isinstance(
            native_topology, RuntimeStateTopologySummary
        ):
            raise LogLinkedAcquisitionError("native_topology is invalid")
        if observation_source is None and observer_profile is None:
            raise LogLinkedAcquisitionError(
                "observer_profile is required for native-unit acquisition"
            )
        if not isinstance(store, StateUnitStore):
            raise LogLinkedAcquisitionError("store must be a StateUnitStore")
        self.observation_source = observation_source
        self.observer_profile = observer_profile
        self.native_topology = native_topology
        self.store = store
        self.reader_factory = reader_factory or (
            lambda pid, start_ticks: ProcessMemoryReader(
                pid=pid, expected_start_time_ticks=start_ticks
            )
        )

    def capture(self, request: UnitRequest) -> StoredStateUnit:
        return self.commit_prepared(self.prepare(self.copy(request)))

    def copy(self, request: UnitRequest) -> CopiedStateUnit:
        if not isinstance(request, UnitRequest):
            raise LogLinkedAcquisitionError("request must be a UnitRequest")
        if self.observation_source is None:
            raise LogLinkedAcquisitionError(
                "live-memory acquisition is unavailable in native-unit mode"
            )
        view = self.observation_source.resolve(request)
        _validate_view(view, request)
        spans = plan_selected_read_runs(view.layout, view.cells)
        try:
            backing_digest = _validate_backing_ranges(
                view.kv_backing_ranges, spans
            )
            baseline = self.observation_source.stability_cut(request, view)
            _validate_stability_cut(baseline)
        except AcquisitionError as error:
            raise LogLinkedAcquisitionError(str(error)) from error
        if (
            baseline.marker != view.target_marker
            or baseline.version != view.stream_version
        ):
            raise LogLinkedAcquisitionError(
                "runtime state changed after the completed-apply cut"
            )

        try:
            reader = self.reader_factory(
                request.runtime_pid, request.runtime_start_time_ticks
            )
        except ProcessIdentityError as error:
            raise LogLinkedAcquisitionError(str(error)) from error
        try:
            read_trace = tuple((span.address, span.size) for span in spans)
            try:
                values_read = reader.pread_many_exact(read_trace)
            except ProcessIdentityError as error:
                raise LogLinkedAcquisitionError(str(error)) from error
            if len(values_read) != len(spans):
                raise LogLinkedAcquisitionError(
                    "address reader returned the wrong read cardinality"
                )
            values = dict(zip(spans, values_read, strict=True))
            try:
                after = self.observation_source.stability_cut(request, view)
                _validate_stability_cut(after)
            except AcquisitionError as error:
                raise LogLinkedAcquisitionError(str(error)) from error
            if after != baseline:
                raise LogLinkedAcquisitionError(
                    "target runtime state changed during state-unit acquisition"
                )
        finally:
            close = getattr(reader, "close", None)
            if callable(close):
                close()

        try:
            actual_trace = tuple(reader.access_trace)
        except (AttributeError, TypeError) as error:
            raise LogLinkedAcquisitionError(
                "address reader did not expose a sealed access trace"
            ) from error
        if actual_trace != read_trace:
            raise LogLinkedAcquisitionError(
                "selected spans were not read exactly once"
            )

        return CopiedStateUnit(
            request=request,
            view=view,
            marker=baseline.marker,
            stream_version=baseline.version,
            backing_digest=backing_digest,
            runs=spans,
            values=tuple(values[span] for span in spans),
        )

    def prepare(self, copied: CopiedStateUnit) -> PreparedStateUnit:
        if not isinstance(copied, CopiedStateUnit):
            raise LogLinkedAcquisitionError("copied state unit is invalid")
        request = copied.request
        view = copied.view
        baseline = copied.marker
        stream_version = copied.stream_version
        backing_digest = copied.backing_digest
        values = dict(zip(copied.runs, copied.values, strict=True))

        payload = encode_simple_kv_payload_from_runs(
            view.layout, view.cells, values
        )
        summary = inspect_simple_kv_payload(payload)
        if summary.source_seq_ids != (request.ref.seq_id,):
            raise LogLinkedAcquisitionError(
                "state-unit payload contains another sequence"
            )
        payload_digest = "sha256:" + hashlib.sha256(payload).hexdigest()
        unit_id = _unit_id(request, payload_digest)
        positions = tuple(sorted({cell.position for cell in view.cells}))
        cell_indexes = tuple(sorted({cell.cell_index for cell in view.cells}))
        manifest = StateUnitManifest(
            {
                "schema_version": "1.0",
                "unit_id": unit_id,
                "unit_kind": request.unit_kind.value,
                "runtime_instance_id": request.ref.runtime_instance_id,
                "runtime_pid": request.runtime_pid,
                "runtime_start_time_ticks": request.runtime_start_time_ticks,
                "seq_id": request.ref.seq_id,
                "sequence_generation": request.ref.sequence_generation,
                "apply_call_id": request.apply_call_id,
                "apply_monotonic_ns": request.monotonic_ns,
                "previous_unit_digest": request.previous_unit_digest,
                "base_unit_digest": request.base_unit_digest,
                "position_ranges": [
                    list(item) for item in _compress_ranges(positions)
                ],
                "kv_cell_ranges": [
                    list(item) for item in _compress_ranges(cell_indexes)
                ],
                "stream_indexes": sorted(
                    {cell.stream_index for cell in view.cells}
                ),
                "mutation_kinds": list(request.mutation_kinds),
                "layout_profile_digest": view.layout.profile_digest,
                "layout_topology_digest": baseline.layout_topology_digest,
                "kv_backing_provenance_digest": backing_digest,
                "observer_health": {
                    "observer_loss_count": view.observer_loss_count,
                    "parse_failure_count": view.parse_failure_count,
                    "unmatched_call_count": view.unmatched_call_count,
                    "unsupported_mutation_count": (
                        view.unsupported_mutation_count
                    ),
                },
                "acquisition_version": {
                    "stream_index": stream_version.stream_index,
                    "stream_apply_call_id": stream_version.apply_call_id,
                    "mutation_generation": (
                        stream_version.mutation_generation
                    ),
                    "backing_generation": (
                        stream_version.backing_generation
                    ),
                    "layout_generation": stream_version.layout_generation,
                    "target_cell_map_digest": (
                        baseline.target_cell_map_digest
                    ),
                    "state": "READY",
                },
                "payload_file": "runtime_state_payload.bin",
                "payload_size_bytes": len(payload),
                "payload_digest": payload_digest,
                "contains_prompt_text": False,
                "contains_response_text": False,
                "contains_explicit_token_history": False,
                "state": "COMPLETE",
            }
        )
        return PreparedStateUnit(
            request=request,
            manifest=manifest,
            payload=payload,
            manifest_digest=(
                "sha256:" + hashlib.sha256(manifest.to_bytes()).hexdigest()
            ),
        )

    def prepare_captured(
        self,
        request: UnitRequest,
        captured: NativeCapturedUnitDescriptor,
        captured_payload: bytes,
    ) -> PreparedStateUnit:
        if not isinstance(request, UnitRequest):
            raise LogLinkedAcquisitionError("request must be a UnitRequest")
        if not isinstance(captured, NativeCapturedUnitDescriptor):
            raise LogLinkedAcquisitionError(
                "captured native-unit descriptor is invalid"
            )
        if self.observer_profile is None:
            raise LogLinkedAcquisitionError(
                "observer profile is unavailable for native-unit acquisition"
            )
        expected = {
            "runtime_instance_id": request.ref.runtime_instance_id,
            "apply_call_id": request.apply_call_id,
            "seq_id": request.ref.seq_id,
            "writer_monotonic_ns": request.monotonic_ns,
        }
        for field, value in expected.items():
            if getattr(captured, field) != value:
                raise LogLinkedAcquisitionError(
                    f"captured native-unit {field} differs from the request"
                )
        stream_indexes = {cell.stream_index for cell in request.token_cells}
        if stream_indexes != {captured.stream_index}:
            raise LogLinkedAcquisitionError(
                "captured native-unit stream differs from the request"
            )
        request_cells = tuple(
            sorted(
                (cell.token_position, cell.cell_index)
                for cell in request.token_cells
            )
        )
        if request_cells != captured.token_cells:
            raise LogLinkedAcquisitionError(
                "captured native-unit cell plan differs from the request"
            )
        try:
            records = tuple(
                iter_native_unit_records(
                    captured_payload,
                    descriptor=captured,
                )
            )
        except NativeUnitSpoolError as error:
            raise LogLinkedAcquisitionError(str(error)) from error
        if any(isinstance(item, NativeUnitScheduleRecord) for item in records):
            (
                payload,
                layout_topology_digest,
                backing_digest,
                target_cell_map_digest,
                n_layer,
            ) = _encode_compact_native_capture(
                profile=self.observer_profile,
                topology=self.native_topology,
                request=request,
                captured=captured,
                records=records,
            )
        else:
            (
                payload,
                layout_topology_digest,
                backing_digest,
                target_cell_map_digest,
                n_layer,
            ) = _encode_native_capture(
                profile=self.observer_profile,
                topology=self.native_topology,
                request=request,
                captured=captured,
                records=tuple(
                    item
                    for item in records
                    if isinstance(item, NativeUnitFragmentRecord)
                ),
            )
        summary = inspect_simple_kv_payload(payload)
        if (
            summary.source_seq_ids != (request.ref.seq_id,)
            or summary.nonempty_streams != (captured.stream_index,)
            or summary.cell_count != len(captured.token_cells)
        ):
            raise LogLinkedAcquisitionError(
                "native state-unit payload identity differs"
            )
        payload_digest = "sha256:" + hashlib.sha256(payload).hexdigest()
        unit_id = _unit_id(request, payload_digest)
        positions = tuple(item[0] for item in captured.token_cells)
        cell_indexes = tuple(sorted(item[1] for item in captured.token_cells))
        manifest = StateUnitManifest(
            {
                "schema_version": "1.0",
                "unit_id": unit_id,
                "unit_kind": request.unit_kind.value,
                "runtime_instance_id": request.ref.runtime_instance_id,
                "runtime_pid": request.runtime_pid,
                "runtime_start_time_ticks": request.runtime_start_time_ticks,
                "seq_id": request.ref.seq_id,
                "sequence_generation": request.ref.sequence_generation,
                "apply_call_id": request.apply_call_id,
                "apply_monotonic_ns": request.monotonic_ns,
                "previous_unit_digest": request.previous_unit_digest,
                "base_unit_digest": request.base_unit_digest,
                "position_ranges": [
                    list(item) for item in _compress_ranges(positions)
                ],
                "kv_cell_ranges": [
                    list(item) for item in _compress_ranges(cell_indexes)
                ],
                "stream_indexes": [captured.stream_index],
                "mutation_kinds": list(request.mutation_kinds),
                "layout_profile_digest": self.observer_profile.profile_digest,
                "layout_topology_digest": layout_topology_digest,
                "kv_backing_provenance_digest": backing_digest,
                "observer_health": {
                    "observer_loss_count": 0,
                    "parse_failure_count": 0,
                    "unmatched_call_count": 0,
                    "unsupported_mutation_count": 0,
                },
                "acquisition_version": {
                    "stream_index": captured.stream_index,
                    "stream_apply_call_id": captured.apply_call_id,
                    "mutation_generation": 0,
                    "backing_generation": 2,
                    "layout_generation": (
                        2 * self.observer_profile.n_stream * n_layer
                    ),
                    "target_cell_map_digest": target_cell_map_digest,
                    "state": "READY",
                },
                "payload_file": "runtime_state_payload.bin",
                "payload_size_bytes": len(payload),
                "payload_digest": payload_digest,
                "contains_prompt_text": False,
                "contains_response_text": False,
                "contains_explicit_token_history": False,
                "state": "COMPLETE",
            }
        )
        return PreparedStateUnit(
            request=request,
            manifest=manifest,
            payload=payload,
            manifest_digest=(
                "sha256:" + hashlib.sha256(manifest.to_bytes()).hexdigest()
            ),
        )

    def commit_prepared(self, prepared: PreparedStateUnit) -> StoredStateUnit:
        if not isinstance(prepared, PreparedStateUnit):
            raise LogLinkedAcquisitionError("prepared state unit is invalid")
        stored = self.store.commit(
            manifest=prepared.manifest,
            payload=prepared.payload,
        )
        if stored.manifest_digest != prepared.manifest_digest:
            raise LogLinkedAcquisitionError(
                "published state-unit manifest digest differs"
            )
        return stored

    def persist(self, copied: CopiedStateUnit) -> StoredStateUnit:
        return self.commit_prepared(self.prepare(copied))


def _validate_view(view: object, request: UnitRequest) -> None:
    if not isinstance(view, LogLinkedAcquisitionView):
        raise LogLinkedAcquisitionError("observation view is invalid")
    expected = {
        "runtime_instance_id": request.ref.runtime_instance_id,
        "runtime_pid": request.runtime_pid,
        "runtime_start_time_ticks": request.runtime_start_time_ticks,
        "apply_call_id": request.apply_call_id,
        "apply_monotonic_ns": request.monotonic_ns,
    }
    for field, value in expected.items():
        if getattr(view, field) != value:
            raise LogLinkedAcquisitionError(
                f"observation {field} differs from the unit request"
            )
    if not isinstance(view.layout, SimpleKvLayout) or not view.cells:
        raise LogLinkedAcquisitionError("observation layout or cells are missing")
    if any(cell.seq_id != request.ref.seq_id for cell in view.cells):
        raise LogLinkedAcquisitionError(
            "observation contains a different sequence"
        )
    request_cells = {
        (cell.stream_index, cell.token_position, cell.cell_index)
        for cell in request.token_cells
    }
    view_cells = {
        (cell.stream_index, cell.position, cell.cell_index)
        for cell in view.cells
    }
    if not request_cells.issubset(view_cells):
        raise LogLinkedAcquisitionError(
            "observation does not cover the requested sequence changes"
        )
    for field in (
        "observer_loss_count",
        "parse_failure_count",
        "unmatched_call_count",
        "unsupported_mutation_count",
    ):
        value = getattr(view, field)
        if isinstance(value, bool) or not isinstance(value, int) or value != 0:
            raise LogLinkedAcquisitionError(f"observation {field} must be zero")
        if value != getattr(view.target_marker, field):
            raise LogLinkedAcquisitionError(
                f"observation {field} differs from the target marker"
            )
    if view.target_marker.observed_seq_id != request.ref.seq_id:
        raise LogLinkedAcquisitionError("target marker sequence differs")
    if view.target_marker.target_apply_watermark != request.apply_call_id:
        raise LogLinkedAcquisitionError("target marker apply differs")
    if view.target_marker.layout_profile_digest != view.layout.profile_digest:
        raise LogLinkedAcquisitionError("target marker profile differs")
    _validate_stream_version(view.stream_version)
    if view.stream_version.stream_index != view.target_marker.stream_index:
        raise LogLinkedAcquisitionError(
            "stream version target differs from the marker"
        )
    if view.stream_version.apply_call_id != request.apply_call_id:
        raise LogLinkedAcquisitionError(
            "stream version apply differs from the unit request"
        )


def _validate_stream_version(value: object) -> None:
    if not isinstance(value, LogLinkedStreamVersionToken):
        raise LogLinkedAcquisitionError("stream version token is invalid")


def _validate_stability_cut(value: object) -> None:
    if not isinstance(value, LogLinkedStabilityCut):
        raise LogLinkedAcquisitionError("stability cut is invalid")
    try:
        _validate_target_marker(value.marker)
    except AcquisitionError as error:
        raise LogLinkedAcquisitionError(str(error)) from error
    _validate_stream_version(value.version)
    if value.marker.stream_index != value.version.stream_index:
        raise LogLinkedAcquisitionError(
            "stability cut marker and stream version differ"
        )


def _stream_version_token(
    snapshot: RuntimeStateSnapshot,
    version: NativeStreamVersionSnapshot,
    observed_seq_id: int,
) -> LogLinkedStreamVersionToken:
    if not isinstance(version, NativeStreamVersionSnapshot):
        raise LogLinkedAcquisitionError(
            "native stream version is unavailable"
        )
    stream_index = dict(snapshot.seq_to_stream).get(observed_seq_id)
    if (
        stream_index is None
        or stream_index >= version.active_stream_count
    ):
        raise LogLinkedAcquisitionError(
            "native target stream version is unavailable"
        )
    if version.stream_states[stream_index] != "READY":
        raise LogLinkedAcquisitionError(
            "native target stream has an active writer"
        )
    return LogLinkedStreamVersionToken(
        stream_index=stream_index,
        apply_call_id=version.stream_apply_call_ids[stream_index],
        mutation_generation=version.mutation_generation,
        backing_generation=version.backing_generation,
        layout_generation=version.layout_generation,
    )


def _snapshot_request(request: UnitRequest) -> RuntimeStateCaptureRequest:
    return RuntimeStateCaptureRequest(
        capture_id=(
            f"log-linked-{request.ref.seq_id}-"
            f"{request.ref.sequence_generation}-{request.apply_call_id}"
        ),
        lease_id="log-linked-observer",
        request_ref="unattributed-sequence-state-unit",
        runtime_instance_id=request.ref.runtime_instance_id,
        runtime_pid=request.runtime_pid,
        runtime_start_time_ticks=request.runtime_start_time_ticks,
        slot_id=request.ref.seq_id,
        slot_epoch=request.ref.sequence_generation,
        observed_seq_id=request.ref.seq_id,
        interval_start_monotonic_ns=0,
        interval_end_monotonic_ns=request.monotonic_ns,
    )


def _select_state_unit_cells(
    snapshot: RuntimeStateSnapshot, request: UnitRequest
) -> tuple[SelectedCell, ...]:
    latest = _latest_cell_owners(snapshot)
    requested = {
        (cell.stream_index, cell.cell_index): cell
        for cell in request.token_cells
    }
    full_sequence = request.unit_kind.value == "BASE" or bool(
        request.mutation_kinds
    )
    if full_sequence:
        events = tuple(
            event for event in latest.values()
            if event.seq_id == request.ref.seq_id
        )
    else:
        events_list: list[TokenCellEvent] = []
        for key, expected in requested.items():
            event = latest.get(key)
            if event is None:
                raise LogLinkedAcquisitionError(
                    "changed state-unit cell is no longer present"
                )
            if (
                event.seq_id != expected.seq_id
                or event.token_position != expected.token_position
                or event.apply_call_id != expected.ownership_generation
            ):
                raise LogLinkedAcquisitionError(
                    "changed state-unit cell ownership differs"
                )
            events_list.append(event)
        events = tuple(events_list)
    if not events:
        raise LogLinkedAcquisitionError(
            "sequence has no current cells at the completed apply"
        )
    positions = [event.token_position for event in events]
    if len(set(positions)) != len(positions):
        raise LogLinkedAcquisitionError(
            "sequence state has duplicate token positions"
        )
    if full_sequence and set(positions) != set(range(max(positions) + 1)):
        raise LogLinkedAcquisitionError(
            "base state unit has a missing token position"
        )
    for expected in requested.values():
        owner = latest.get((expected.stream_index, expected.cell_index))
        if owner is None or (
            owner.seq_id,
            owner.token_position,
            owner.apply_call_id,
        ) != (
            expected.seq_id,
            expected.token_position,
            expected.ownership_generation,
        ):
            raise LogLinkedAcquisitionError(
                "completed apply cell is no longer current"
            )
    return tuple(
        sorted(
            (
                SelectedCell(
                    position=event.token_position,
                    cell_index=event.cell_index,
                    seq_id=event.seq_id,
                    stream_index=event.stream_index,
                )
                for event in events
            ),
            key=lambda cell: (cell.stream_index, cell.cell_index),
        )
    )


def _native_capture_topology(
    profile: RuntimeStateObserverProfile,
    topology: RuntimeStateTopologySummary | None,
) -> tuple[int, bool]:
    if topology is None:
        if profile.n_layer <= 0 or profile.v_trans is None:
            raise LogLinkedAcquisitionError(
                "observed native topology is required for dynamic acquisition"
            )
        return profile.n_layer, profile.v_trans
    if (
        topology.profile_digest != profile.profile_digest
        or topology.n_stream != profile.n_stream
        or topology.n_seq_max != profile.n_seq_max
    ):
        raise LogLinkedAcquisitionError(
            "observed native topology differs from the qualified profile"
        )
    if profile.has_dynamic_topology:
        if (
            topology.n_layer > profile.max_n_layer
            or topology.n_pos_per_embd > profile.max_n_pos_per_embd
        ):
            raise LogLinkedAcquisitionError(
                "observed native topology exceeds the qualified profile"
            )
    elif (
        topology.n_layer != profile.n_layer
        or topology.n_pos_per_embd != profile.n_pos_per_embd
        or topology.v_trans != profile.v_trans
    ):
        raise LogLinkedAcquisitionError(
            "observed native topology differs from the qualified profile"
        )
    return topology.n_layer, topology.v_trans


def _encode_compact_native_capture(
    *,
    profile: RuntimeStateObserverProfile,
    topology: RuntimeStateTopologySummary | None,
    request: UnitRequest,
    captured: NativeCapturedUnitDescriptor,
    records: tuple[NativeUnitFragmentRecord | NativeUnitScheduleRecord, ...],
) -> tuple[bytes, str, str, str, int]:
    n_layer, v_trans = _native_capture_topology(profile, topology)
    if not v_trans:
        raise LogLinkedAcquisitionError(
            "native compact schedules require transposed value storage"
        )
    key_groups: dict[int, list[NativeUnitFragmentRecord]] = {}
    value_groups: dict[
        int, list[NativeUnitFragmentRecord | NativeUnitScheduleRecord]
    ] = {}
    schedule_record_count = 0
    for record in records:
        if isinstance(record, NativeUnitFragmentRecord):
            destination = (
                key_groups if record.tensor_kind == "k" else value_groups
            )
            destination.setdefault(record.layer, []).append(record)
            continue
        if record.tensor_kind != 2 or record.fragments_per_tensor == 0:
            raise LogLinkedAcquisitionError(
                "native compact schedule tensor identity differs"
            )
        first_layer = record.first_flat_index // record.fragments_per_tensor
        last_layer = (
            record.first_flat_index + record.record_count - 1
        ) // record.fragments_per_tensor
        if first_layer != last_layer:
            raise LogLinkedAcquisitionError(
                "native compact schedule crosses a layer boundary"
            )
        value_groups.setdefault(first_layer, []).append(record)
        schedule_record_count += 1

    layers = tuple(range(n_layer))
    if (
        tuple(sorted(key_groups)) != layers
        or tuple(sorted(value_groups)) != layers
    ):
        raise LogLinkedAcquisitionError(
            "native fragment layout has an incomplete layer set"
        )
    cells = captured.token_cells
    encoded = bytearray(struct.pack("<I", profile.n_stream))
    for stream_index in range(profile.n_stream):
        if stream_index != captured.stream_index:
            encoded += struct.pack("<I", 0)
            continue
        encoded += struct.pack("<I", len(cells))
        for token_position, _cell_index in cells:
            encoded += struct.pack("<iIi", token_position, 1, request.ref.seq_id)
        encoded += struct.pack("<II", 1, n_layer)

        key_row_sizes: list[int] = []
        for layer in layers:
            group = tuple(
                sorted(key_groups[layer], key=lambda item: item.fragment_index)
            )
            cycles = _fragment_cycles(group, cells)
            if len(cycles) != 1:
                raise LogLinkedAcquisitionError(
                    "native key fragment layout is not canonical"
                )
            row_size = _fragment_unit_size(group)
            _validate_linear_offsets(cycles[0], unit_size=row_size)
            key_row_sizes.append(row_size)
            encoded += struct.pack("<iQ", profile.key_ggml_type_id, row_size)
            encoded += b"".join(item.payload for item in cycles[0])

        value_element_sizes: list[int] = []
        value_embedding_widths: list[int] = []
        for layer in layers:
            element_size, embedding_width, layer_payload = (
                _encode_compact_transposed_value_layer(
                    layer=layer,
                    cells=cells,
                    records=tuple(value_groups[layer]),
                )
            )
            value_element_sizes.append(element_size)
            value_embedding_widths.append(embedding_width)
            encoded += struct.pack(
                "<iII",
                profile.value_ggml_type_id,
                element_size,
                embedding_width,
            )
            encoded += layer_payload

    layout_document = {
        "profile_digest": profile.profile_digest,
        "n_stream": profile.n_stream,
        "n_layer": n_layer,
        "v_trans": True,
        "key_row_sizes": key_row_sizes,
        "value_element_or_row_sizes": value_element_sizes,
        "value_embedding_widths": value_embedding_widths,
    }
    backing_document = {
        "encoding": "native-compact-schedule-v1",
        "native_unit_id": captured.unit_id,
        "native_payload_sha256": captured.payload_sha256,
        "logical_fragment_count": captured.fragment_count,
        "compact_record_count": len(records),
        "schedule_record_count": schedule_record_count,
    }
    cell_document = {
        "apply_call_id": captured.apply_call_id,
        "seq_id": captured.seq_id,
        "stream_index": captured.stream_index,
        "token_cells": [list(item) for item in cells],
    }
    return (
        bytes(encoded),
        _document_digest(layout_document),
        _document_digest(backing_document),
        _document_digest(cell_document),
        n_layer,
    )


def _encode_compact_transposed_value_layer(
    *,
    layer: int,
    cells: tuple[tuple[int, int], ...],
    records: tuple[NativeUnitFragmentRecord | NativeUnitScheduleRecord, ...],
) -> tuple[int, int, bytes]:
    schedules = tuple(
        item for item in records if isinstance(item, NativeUnitScheduleRecord)
    )
    if not schedules or not cells:
        raise LogLinkedAcquisitionError(
            "native compact value layout is incomplete"
        )
    shape = schedules[0]
    expected_cells = tuple(
        (
            shape.first_token_position + offset,
            shape.first_cell_index + offset,
        )
        for offset in range(shape.token_count)
    )
    if expected_cells != cells:
        raise LogLinkedAcquisitionError(
            "native compact value cell ordering differs"
        )
    comparable_shape = (
        shape.fragments_per_tensor,
        shape.first_token_position,
        shape.first_cell_index,
        shape.chunks_per_dimension,
        shape.token_count,
        shape.row_size,
        shape.cell_capacity,
        shape.element_size,
        shape.embedding_width,
        shape.transposed_value,
    )
    if any(
        (
            item.fragments_per_tensor,
            item.first_token_position,
            item.first_cell_index,
            item.chunks_per_dimension,
            item.token_count,
            item.row_size,
            item.cell_capacity,
            item.element_size,
            item.embedding_width,
            item.transposed_value,
        )
        != comparable_shape
        for item in schedules
    ):
        raise LogLinkedAcquisitionError(
            "native compact value schedule shape differs"
        )
    if (
        shape.transposed_value != 1
        or shape.row_size != shape.element_size * shape.embedding_width
        or shape.fragments_per_tensor
        != shape.chunks_per_dimension * shape.embedding_width
    ):
        raise LogLinkedAcquisitionError(
            "native compact value schedule shape is invalid"
        )

    def ordering_key(
        item: NativeUnitFragmentRecord | NativeUnitScheduleRecord,
    ) -> int:
        if isinstance(item, NativeUnitScheduleRecord):
            return item.first_fragment_index
        return item.fragment_index

    ordered = tuple(sorted(records, key=ordering_key))
    fragment_base = ordering_key(ordered[0])
    flat_base = layer * shape.fragments_per_tensor
    expected_in_layer = 0
    units_per_fragment = 4096 // shape.element_size
    payload = bytearray()
    for item in ordered:
        if ordering_key(item) != fragment_base + expected_in_layer:
            raise LogLinkedAcquisitionError(
                "native compact value fragment ordering differs"
            )
        if isinstance(item, NativeUnitScheduleRecord):
            if item.first_flat_index != flat_base + expected_in_layer:
                raise LogLinkedAcquisitionError(
                    "native compact value schedule ordering differs"
                )
            expected_in_layer += item.record_count
            payload += item.packed_payload
            continue

        fragment_in_tensor = expected_in_layer
        dimension = fragment_in_tensor // shape.chunks_per_dimension
        chunk_index = fragment_in_tensor % shape.chunks_per_dimension
        cell_offset = chunk_index * units_per_fragment
        span_cell_count = min(
            shape.token_count - cell_offset,
            units_per_fragment,
        )
        expected_offset = (
            dimension * shape.element_size * shape.cell_capacity
            + (shape.first_cell_index + cell_offset) * shape.element_size
        )
        if (
            item.tensor_kind != "v"
            or item.layer != layer
            or item.token_position
            != shape.first_token_position + cell_offset
            or item.cell_index != shape.first_cell_index + cell_offset
            or item.span_cell_count != span_cell_count
            or item.tensor_byte_offset != expected_offset
            or len(item.payload) != span_cell_count * shape.element_size
        ):
            raise LogLinkedAcquisitionError(
                "native compact value tail fragment differs"
            )
        expected_in_layer += 1
        payload += item.payload
    if (
        expected_in_layer != shape.fragments_per_tensor
        or len(payload) != shape.token_count * shape.row_size
    ):
        raise LogLinkedAcquisitionError(
            "native compact value byte coverage is incomplete"
        )
    return shape.element_size, shape.embedding_width, bytes(payload)


def _encode_native_capture(
    *,
    profile: RuntimeStateObserverProfile,
    topology: RuntimeStateTopologySummary | None,
    request: UnitRequest,
    captured: NativeCapturedUnitDescriptor,
    records: tuple[NativeUnitFragmentRecord, ...],
) -> tuple[bytes, str, str, str, int]:
    n_layer, v_trans = _native_capture_topology(profile, topology)
    layers_by_kind = {
        kind: tuple(sorted({item.layer for item in records if item.tensor_kind == kind}))
        for kind in ("k", "v")
    }
    if (
        not layers_by_kind["k"]
        or layers_by_kind["k"] != layers_by_kind["v"]
        or layers_by_kind["k"]
        != tuple(range(len(layers_by_kind["k"])))
    ):
        raise LogLinkedAcquisitionError(
            "native fragment layout has an incomplete layer set"
        )
    layers = layers_by_kind["k"]
    if len(layers) != n_layer:
        raise LogLinkedAcquisitionError(
            "native fragment layout differs from the qualified topology"
        )
    cells = captured.token_cells
    grouped: dict[tuple[str, int], tuple[NativeUnitFragmentRecord, ...]] = {}
    for kind in ("k", "v"):
        for layer in layers:
            group = tuple(
                sorted(
                    (
                        item
                        for item in records
                        if item.tensor_kind == kind and item.layer == layer
                    ),
                    key=lambda item: item.fragment_index,
                )
            )
            if not group:
                raise LogLinkedAcquisitionError(
                    "native fragment layout is incomplete"
                )
            grouped[(kind, layer)] = group

    encoded = bytearray(struct.pack("<I", profile.n_stream))
    for stream_index in range(profile.n_stream):
        if stream_index != captured.stream_index:
            encoded += struct.pack("<I", 0)
            continue
        encoded += struct.pack("<I", len(cells))
        for token_position, _cell_index in cells:
            encoded += struct.pack(
                "<iIi", token_position, 1, request.ref.seq_id
            )
        encoded += struct.pack("<II", int(v_trans), len(layers))
        for layer in layers:
            group = grouped[("k", layer)]
            cycles = _fragment_cycles(group, cells)
            if len(cycles) != 1:
                raise LogLinkedAcquisitionError(
                    "native key fragment layout is not canonical"
                )
            row_size = _fragment_unit_size(group)
            _validate_linear_offsets(cycles[0], unit_size=row_size)
            encoded += struct.pack("<iQ", profile.key_ggml_type_id, row_size)
            encoded += b"".join(item.payload for item in cycles[0])
        for layer in layers:
            group = grouped[("v", layer)]
            cycles = _fragment_cycles(group, cells)
            unit_size = _fragment_unit_size(group)
            if v_trans:
                _validate_transposed_offsets(cycles, unit_size=unit_size)
                encoded += struct.pack(
                    "<iII",
                    profile.value_ggml_type_id,
                    unit_size,
                    len(cycles),
                )
                encoded += b"".join(
                    item.payload for cycle in cycles for item in cycle
                )
            else:
                if len(cycles) != 1:
                    raise LogLinkedAcquisitionError(
                        "native value fragment layout is not canonical"
                    )
                _validate_linear_offsets(cycles[0], unit_size=unit_size)
                encoded += struct.pack(
                    "<iQ", profile.value_ggml_type_id, unit_size
                )
                encoded += b"".join(item.payload for item in cycles[0])

    layout_document = {
        "profile_digest": profile.profile_digest,
        "n_stream": profile.n_stream,
        "n_layer": len(layers),
        "v_trans": v_trans,
        "key_row_sizes": [
            _fragment_unit_size(grouped[("k", layer)]) for layer in layers
        ],
        "value_element_or_row_sizes": [
            _fragment_unit_size(grouped[("v", layer)]) for layer in layers
        ],
        "value_embedding_widths": [
            len(_fragment_cycles(grouped[("v", layer)], cells))
            for layer in layers
        ],
    }
    backing_document = {
        "native_unit_id": captured.unit_id,
        "native_payload_sha256": captured.payload_sha256,
        "fragments": [
            {
                "fragment_index": item.fragment_index,
                "kind": item.tensor_kind,
                "layer": item.layer,
                "tensor_byte_offset": item.tensor_byte_offset,
                "payload_size": len(item.payload),
            }
            for item in records
        ],
    }
    cell_document = {
        "apply_call_id": captured.apply_call_id,
        "seq_id": captured.seq_id,
        "stream_index": captured.stream_index,
        "token_cells": [list(item) for item in cells],
    }
    return (
        bytes(encoded),
        _document_digest(layout_document),
        _document_digest(backing_document),
        _document_digest(cell_document),
        len(layers),
    )


def _fragment_cycles(
    records: tuple[NativeUnitFragmentRecord, ...],
    cells: tuple[tuple[int, int], ...],
) -> tuple[tuple[NativeUnitFragmentRecord, ...], ...]:
    cycles: list[tuple[NativeUnitFragmentRecord, ...]] = []
    current: list[NativeUnitFragmentRecord] = []
    cell_offset = 0
    for record in records:
        span = tuple(
            (
                record.token_position + offset,
                record.cell_index + offset,
            )
            for offset in range(record.span_cell_count)
        )
        if span != cells[cell_offset : cell_offset + len(span)]:
            raise LogLinkedAcquisitionError(
                "native fragment cell ordering differs"
            )
        current.append(record)
        cell_offset += len(span)
        if cell_offset == len(cells):
            cycles.append(tuple(current))
            current = []
            cell_offset = 0
        elif cell_offset > len(cells):
            raise LogLinkedAcquisitionError(
                "native fragment cell coverage exceeds the unit"
            )
    if current or not cycles:
        raise LogLinkedAcquisitionError(
            "native fragment cell coverage is incomplete"
        )
    return tuple(cycles)


def _fragment_unit_size(
    records: tuple[NativeUnitFragmentRecord, ...]
) -> int:
    sizes = {
        len(item.payload) // item.span_cell_count
        for item in records
        if len(item.payload) % item.span_cell_count == 0
    }
    if len(sizes) != 1 or not sizes or 0 in sizes:
        raise LogLinkedAcquisitionError(
            "native fragment element size is inconsistent"
        )
    size = next(iter(sizes))
    if any(len(item.payload) != item.span_cell_count * size for item in records):
        raise LogLinkedAcquisitionError(
            "native fragment byte coverage is inconsistent"
        )
    return size


def _validate_linear_offsets(
    records: tuple[NativeUnitFragmentRecord, ...], *, unit_size: int
) -> None:
    if any(
        item.tensor_byte_offset != item.cell_index * unit_size
        for item in records
    ):
        raise LogLinkedAcquisitionError(
            "native fragment linear tensor offsets differ"
        )


def _validate_transposed_offsets(
    cycles: tuple[tuple[NativeUnitFragmentRecord, ...], ...],
    *,
    unit_size: int,
) -> None:
    bases: list[int] = []
    maximum_cell = max(
        item.cell_index + item.span_cell_count - 1
        for cycle in cycles
        for item in cycle
    )
    for cycle in cycles:
        cycle_bases = {
            item.tensor_byte_offset - item.cell_index * unit_size
            for item in cycle
        }
        if len(cycle_bases) != 1 or min(cycle_bases) < 0:
            raise LogLinkedAcquisitionError(
                "native transposed value offsets differ"
            )
        bases.append(next(iter(cycle_bases)))
    if bases[0] != 0:
        raise LogLinkedAcquisitionError(
            "native transposed value origin differs"
        )
    if len(bases) == 1:
        return
    stride = bases[1]
    if (
        stride <= maximum_cell * unit_size
        or stride % unit_size != 0
        or bases != [dimension * stride for dimension in range(len(bases))]
    ):
        raise LogLinkedAcquisitionError(
            "native transposed value stride differs"
        )


def _document_digest(document: object) -> str:
    encoded = json.dumps(
        document,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _unit_id(request: UnitRequest, payload_digest: str) -> str:
    document = {
        "apply_call_id": request.apply_call_id,
        "base_unit_digest": request.base_unit_digest,
        "payload_digest": payload_digest,
        "previous_unit_digest": request.previous_unit_digest,
        "runtime_instance_id": request.ref.runtime_instance_id,
        "seq_id": request.ref.seq_id,
        "sequence_generation": request.ref.sequence_generation,
        "unit_kind": request.unit_kind.value,
    }
    encoded = json.dumps(
        document,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "su_" + hashlib.sha256(encoded).hexdigest()


def _compress_ranges(values: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    if not values:
        raise LogLinkedAcquisitionError("state unit has no selected values")
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


__all__ = [
    "CopiedStateUnit",
    "LogLinkedAcquisitionError",
    "LogLinkedAcquisitionView",
    "LogLinkedStabilityCut",
    "LogLinkedStreamVersionToken",
    "LogLinkedObservationSource",
    "LogLinkedStateUnitAcquirer",
    "PreparedStateUnit",
    "RuntimeStateUnitObservationSource",
]
