from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Protocol

from laris_onpremise.kv_state.observation.runtime_state_client import (
    CompletedApplyPage,
    CompletedNativeUnitPage,
    NativeCapturedUnitDescriptor,
)
from laris_onpremise.kv_state.observation.runtime_state_native_units import (
    native_unit_copied_byte_count,
)

from .acquisition import LogLinkedStateUnitAcquirer
from .generation import (
    ObservedMutation,
    ObservedTokenCell,
    ObserverLifecycleEvent,
    SequenceGenerationTracker,
)


class CompletedApplyClient(Protocol):
    def completed_applies(self, *, after_cursor: int) -> CompletedApplyPage: ...


class NativeUnitClient(Protocol):
    def completed_native_units(
        self, *, after_cursor: int
    ) -> CompletedNativeUnitPage: ...

    def read_complete_native_unit(
        self, descriptor: NativeCapturedUnitDescriptor
    ) -> bytes: ...


@dataclass(frozen=True)
class RecorderProgress:
    previous_cursor: int
    next_cursor: int
    boundary_count: int
    captured_unit_count: int
    incomplete_generation_count: int
    observation_truncated: bool


@dataclass(frozen=True)
class WriterCaptureStatistics:
    apply_count: int
    unit_count: int
    fragment_count: int
    copied_byte_count: int
    native_ring_frame_high_water_mark: int
    persistence_queue_high_water_mark: int
    incomplete_unit_count: int
    persistence_failure_count: int

    def __post_init__(self) -> None:
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(f"{field_name} must be a non-negative integer")


class StateUnitRecorder:
    """Drain Observer applies independently of the application-log channel."""

    def __init__(
        self,
        *,
        client: CompletedApplyClient,
        tracker: SequenceGenerationTracker,
        acquirer: LogLinkedStateUnitAcquirer,
        maximum_pending_persistence: int = 64,
        runtime_pid: int | None = None,
        runtime_start_time_ticks: int | None = None,
    ) -> None:
        native_mode = callable(
            getattr(client, "completed_native_units", None)
        ) and callable(getattr(client, "read_complete_native_unit", None))
        if not native_mode and not callable(
            getattr(client, "completed_applies", None)
        ):
            raise TypeError(
                "client must expose completed native units or completed applies"
            )
        if not isinstance(tracker, SequenceGenerationTracker):
            raise TypeError("tracker must be a SequenceGenerationTracker")
        operations = (
            ("prepare_captured", "commit_prepared")
            if native_mode
            else ("copy", "prepare", "commit_prepared")
        )
        for operation in operations:
            if not callable(getattr(acquirer, operation, None)):
                raise TypeError(f"acquirer must expose {operation}")
        if native_mode and (
            isinstance(runtime_pid, bool)
            or not isinstance(runtime_pid, int)
            or runtime_pid <= 0
            or isinstance(runtime_start_time_ticks, bool)
            or not isinstance(runtime_start_time_ticks, int)
            or runtime_start_time_ticks <= 0
        ):
            raise TypeError("native recorder requires the runtime process identity")
        if (
            isinstance(maximum_pending_persistence, bool)
            or not isinstance(maximum_pending_persistence, int)
            or maximum_pending_persistence <= 0
        ):
            raise TypeError("maximum_pending_persistence must be positive")
        self.client = client
        self.tracker = tracker
        self.acquirer = acquirer
        self._cursor = 0
        self._native_mode = native_mode
        self.runtime_pid = runtime_pid
        self.runtime_start_time_ticks = runtime_start_time_ticks
        self.maximum_pending_persistence = maximum_pending_persistence
        self._copy_workers: ThreadPoolExecutor | None = None
        self._persistence_worker: ThreadPoolExecutor | None = None
        self._pending_persistence: set[Future[object]] = set()
        self._closed = False
        self._observed_apply_ids: set[int] = set()
        self._observed_unit_count = 0
        self._observed_fragment_count = 0
        self._copied_byte_count = 0
        self._native_ring_frame_high_water_mark = 0
        self._persistence_queue_high_water_mark = 0
        self._incomplete_unit_count = 0
        self._persistence_failure_count = 0

    @property
    def writer_capture_statistics(self) -> WriterCaptureStatistics:
        return WriterCaptureStatistics(
            apply_count=len(self._observed_apply_ids),
            unit_count=self._observed_unit_count,
            fragment_count=self._observed_fragment_count,
            copied_byte_count=self._copied_byte_count,
            native_ring_frame_high_water_mark=(
                self._native_ring_frame_high_water_mark
            ),
            persistence_queue_high_water_mark=(
                self._persistence_queue_high_water_mark
            ),
            incomplete_unit_count=self._incomplete_unit_count,
            persistence_failure_count=self._persistence_failure_count,
        )

    def prime(self) -> int:
        """Exclude applies completed before this observation interval."""

        if self._cursor != 0:
            raise RuntimeError("state-unit recorder is already active")
        if self._native_mode:
            while True:
                page = self.client.completed_native_units(
                    after_cursor=self._cursor
                )
                self._cursor = page.next_cursor
                if not page.truncated:
                    break
        else:
            page = self.client.completed_applies(after_cursor=0)
            self._cursor = page.next_cursor
        return self._cursor

    def drain_once(self) -> RecorderProgress:
        if self._closed:
            raise RuntimeError("state-unit recorder is closed")
        self._reap_persistence()
        if self._native_mode:
            return self._drain_native_units()
        previous_cursor = self._cursor
        page = self.client.completed_applies(after_cursor=previous_cursor)
        self._cursor = page.next_cursor
        if page.truncated:
            incomplete = 0
            for boundary in page.boundaries:
                if not boundary.sequence_ids:
                    continue
                self.tracker.consume(_lifecycle_event(boundary, force_loss=True))
                incomplete += len(boundary.sequence_ids)
            return RecorderProgress(
                previous_cursor=previous_cursor,
                next_cursor=page.next_cursor,
                boundary_count=len(page.boundaries),
                captured_unit_count=0,
                incomplete_generation_count=incomplete,
                observation_truncated=True,
            )

        captured = 0
        incomplete = 0
        for event in _coalesced_lifecycle_events(page.boundaries):
            requests = self.tracker.consume(event)
            if not requests:
                continue
            workers = self._copy_pool()
            pending_copies = {
                workers.submit(self.acquirer.copy, request): request
                for request in requests
            }
            copied_units: list[tuple[object, object]] = []
            for future in as_completed(pending_copies):
                request = pending_copies[future]
                try:
                    copied = future.result()
                except Exception:
                    self.tracker.mark_incomplete(
                        request.ref, "ACQUISITION_FAILED"
                    )
                    incomplete += 1
                    continue
                copied_units.append((request, copied))

            pending_preparation = {
                workers.submit(self.acquirer.prepare, copied): request
                for request, copied in copied_units
            }
            for future in as_completed(pending_preparation):
                request = pending_preparation[future]
                try:
                    prepared = future.result()
                    manifest_digest = prepared.manifest_digest
                except Exception:
                    self.tracker.mark_incomplete(
                        request.ref, "ACQUISITION_FAILED"
                    )
                    incomplete += 1
                    continue
                self._reap_persistence()
                if (
                    len(self._pending_persistence)
                    >= self.maximum_pending_persistence
                ):
                    self.tracker.mark_incomplete(
                        request.ref, "BUFFER_EXHAUSTED"
                    )
                    incomplete += 1
                    continue
                self.tracker.acknowledge(request, unit_digest=manifest_digest)
                self._pending_persistence.add(
                    self._persistence_pool().submit(
                        self.acquirer.commit_prepared,
                        prepared,
                    )
                )
                captured += 1
        return RecorderProgress(
            previous_cursor=previous_cursor,
            next_cursor=page.next_cursor,
            boundary_count=len(page.boundaries),
            captured_unit_count=captured,
            incomplete_generation_count=incomplete,
            observation_truncated=False,
        )

    def _drain_native_units(self) -> RecorderProgress:
        previous_cursor = self._cursor
        page = self.client.completed_native_units(
            after_cursor=previous_cursor
        )
        self._cursor = page.next_cursor
        captured_count = 0
        incomplete_count = 0
        assert self.runtime_pid is not None
        assert self.runtime_start_time_ticks is not None
        for descriptor in page.units:
            self._observed_apply_ids.add(descriptor.apply_call_id)
            self._observed_unit_count += 1
            self._observed_fragment_count += descriptor.fragment_count
            self._copied_byte_count += native_unit_copied_byte_count(descriptor)
            self._native_ring_frame_high_water_mark = max(
                self._native_ring_frame_high_water_mark,
                descriptor.fragment_count + 1,
            )
            requests = self.tracker.consume(
                _native_lifecycle_event(
                    descriptor,
                    runtime_pid=self.runtime_pid,
                    runtime_start_time_ticks=self.runtime_start_time_ticks,
                )
            )
            if len(requests) > 1:
                raise RuntimeError(
                    "one native state unit produced multiple sequence requests"
                )
            if not requests:
                continue
            request = requests[0]
            try:
                payload = self.client.read_complete_native_unit(descriptor)
                prepared = self.acquirer.prepare_captured(
                    request, descriptor, payload
                )
                manifest_digest = prepared.manifest_digest
            except Exception as error:
                self.tracker.mark_incomplete(
                    request.ref, _native_failure_reason(error)
                )
                incomplete_count += 1
                self._incomplete_unit_count += 1
                continue
            self._reap_persistence()
            if (
                len(self._pending_persistence)
                >= self.maximum_pending_persistence
            ):
                self.tracker.mark_incomplete(
                    request.ref, "BUFFER_EXHAUSTED"
                )
                incomplete_count += 1
                self._incomplete_unit_count += 1
                continue
            self.tracker.acknowledge(
                request, unit_digest=manifest_digest
            )
            self._pending_persistence.add(
                self._persistence_pool().submit(
                    self.acquirer.commit_prepared,
                    prepared,
                )
            )
            self._persistence_queue_high_water_mark = max(
                self._persistence_queue_high_water_mark,
                len(self._pending_persistence),
            )
            captured_count += 1
        return RecorderProgress(
            previous_cursor=previous_cursor,
            next_cursor=page.next_cursor,
            boundary_count=len(page.units),
            captured_unit_count=captured_count,
            incomplete_generation_count=incomplete_count,
            observation_truncated=False,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._copy_workers is not None:
            self._copy_workers.shutdown(wait=True, cancel_futures=False)
        if self._persistence_worker is not None:
            self._persistence_worker.shutdown(wait=True, cancel_futures=False)
        self._reap_persistence()

    def _copy_pool(self) -> ThreadPoolExecutor:
        if self._copy_workers is None:
            self._copy_workers = ThreadPoolExecutor(
                max_workers=8,
                thread_name_prefix="laris-state-copy",
            )
        return self._copy_workers

    def _persistence_pool(self) -> ThreadPoolExecutor:
        if self._persistence_worker is None:
            self._persistence_worker = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="laris-state-persist",
            )
        return self._persistence_worker

    def _reap_persistence(self) -> None:
        completed = tuple(
            future for future in self._pending_persistence if future.done()
        )
        for future in completed:
            self._pending_persistence.remove(future)
            try:
                future.result()
            except Exception as error:
                self._persistence_failure_count += 1
                raise RuntimeError("state-unit persistence failed") from error


@dataclass
class _SequenceAggregate:
    runtime_instance_id: str
    runtime_pid: int
    runtime_start_time_ticks: int
    seq_id: int
    apply_call_id: int
    monotonic_ns: int
    cells: dict[tuple[int, int], ObservedTokenCell] = field(default_factory=dict)
    mutation_support: dict[str, bool] = field(default_factory=dict)
    event_loss: bool = False
    read_race: bool = False


def _coalesced_lifecycle_events(
    boundaries: tuple[object, ...],
) -> tuple[ObserverLifecycleEvent, ...]:
    """Keep one latest coherent cut per sequence while preserving changed cells."""

    aggregates: dict[tuple[str, int], _SequenceAggregate] = {}
    for boundary in boundaries:
        for seq_id in boundary.sequence_ids:
            key = (boundary.runtime_instance_id, seq_id)
            aggregate = aggregates.get(key)
            if aggregate is None:
                aggregate = _SequenceAggregate(
                    runtime_instance_id=boundary.runtime_instance_id,
                    runtime_pid=boundary.runtime_pid,
                    runtime_start_time_ticks=boundary.runtime_start_time_ticks,
                    seq_id=seq_id,
                    apply_call_id=boundary.apply_call_id,
                    monotonic_ns=boundary.monotonic_ns,
                )
                aggregates[key] = aggregate
            elif (
                aggregate.runtime_pid != boundary.runtime_pid
                or aggregate.runtime_start_time_ticks
                != boundary.runtime_start_time_ticks
            ):
                raise RuntimeError(
                    "completed-apply page contains multiple runtime identities"
                )
            aggregate.apply_call_id = boundary.apply_call_id
            aggregate.monotonic_ns = boundary.monotonic_ns
            for cell in boundary.token_cells:
                if cell.seq_id != seq_id:
                    continue
                aggregate.cells[(cell.stream_index, cell.cell_index)] = (
                    ObservedTokenCell(
                        seq_id=cell.seq_id,
                        stream_index=cell.stream_index,
                        token_position=cell.token_position,
                        cell_index=cell.cell_index,
                        ownership_generation=cell.ownership_generation,
                    )
                )
            supported = boundary.unsupported_mutation_count == 0
            for kind in boundary.mutation_kinds:
                canonical_kind = kind.upper()
                aggregate.mutation_support[canonical_kind] = (
                    aggregate.mutation_support.get(canonical_kind, True)
                    and supported
                )
            aggregate.event_loss = (
                aggregate.event_loss or boundary.observer_loss_count != 0
            )
            aggregate.read_race = (
                aggregate.read_race or boundary.parse_failure_count != 0
            )

    grouped: dict[
        tuple[str, int, int, int, int], list[_SequenceAggregate]
    ] = {}
    for aggregate in aggregates.values():
        group_key = (
            aggregate.runtime_instance_id,
            aggregate.runtime_pid,
            aggregate.runtime_start_time_ticks,
            aggregate.apply_call_id,
            aggregate.monotonic_ns,
        )
        grouped.setdefault(group_key, []).append(aggregate)

    events: list[ObserverLifecycleEvent] = []
    for group_key, members in grouped.items():
        runtime_instance_id, runtime_pid, start_ticks, apply_call_id, monotonic_ns = (
            group_key
        )
        sequence_ids = tuple(sorted(member.seq_id for member in members))
        mutation_members: dict[str, list[_SequenceAggregate]] = {}
        for member in members:
            for kind in member.mutation_support:
                mutation_members.setdefault(kind, []).append(member)
        events.append(
            ObserverLifecycleEvent(
                runtime_instance_id=runtime_instance_id,
                runtime_pid=runtime_pid,
                runtime_start_time_ticks=start_ticks,
                apply_call_id=apply_call_id,
                monotonic_ns=monotonic_ns,
                sequence_ids=sequence_ids,
                token_cells=tuple(
                    sorted(
                        (
                            cell
                            for member in members
                            for cell in member.cells.values()
                        ),
                        key=lambda cell: (
                            cell.seq_id,
                            cell.token_position,
                            cell.stream_index,
                            cell.cell_index,
                        ),
                    )
                ),
                mutations=tuple(
                    ObservedMutation(
                        mutation_kind=kind,
                        sequence_ids=tuple(
                            sorted(member.seq_id for member in affected)
                        ),
                        supported=all(
                            member.mutation_support[kind]
                            for member in affected
                        ),
                    )
                    for kind, affected in sorted(mutation_members.items())
                ),
                event_loss_sequence_ids=tuple(
                    sorted(member.seq_id for member in members if member.event_loss)
                ),
                read_race_sequence_ids=tuple(
                    sorted(member.seq_id for member in members if member.read_race)
                ),
            )
        )
    return tuple(
        sorted(
            events,
            key=lambda event: (
                event.monotonic_ns,
                event.apply_call_id,
                event.sequence_ids,
            ),
        )
    )


def _lifecycle_event(
    boundary: object, *, force_loss: bool = False
) -> ObserverLifecycleEvent:
    sequence_ids = boundary.sequence_ids
    mutation_kinds = tuple(
        ObservedMutation(
            mutation_kind=kind.upper(),
            sequence_ids=sequence_ids,
            supported=boundary.unsupported_mutation_count == 0,
        )
        for kind in boundary.mutation_kinds
    )
    loss_ids = (
        sequence_ids
        if force_loss or boundary.observer_loss_count
        else ()
    )
    read_race_ids = (
        sequence_ids if boundary.parse_failure_count else ()
    )
    return ObserverLifecycleEvent(
        runtime_instance_id=boundary.runtime_instance_id,
        runtime_pid=boundary.runtime_pid,
        runtime_start_time_ticks=boundary.runtime_start_time_ticks,
        apply_call_id=boundary.apply_call_id,
        monotonic_ns=boundary.monotonic_ns,
        sequence_ids=sequence_ids,
        token_cells=tuple(
            ObservedTokenCell(
                seq_id=cell.seq_id,
                stream_index=cell.stream_index,
                token_position=cell.token_position,
                cell_index=cell.cell_index,
                ownership_generation=cell.ownership_generation,
            )
            for cell in boundary.token_cells
        ),
        mutations=mutation_kinds,
        event_loss_sequence_ids=loss_ids,
        read_race_sequence_ids=read_race_ids,
    )


def _native_lifecycle_event(
    descriptor: NativeCapturedUnitDescriptor,
    *,
    runtime_pid: int,
    runtime_start_time_ticks: int,
) -> ObserverLifecycleEvent:
    return ObserverLifecycleEvent(
        runtime_instance_id=descriptor.runtime_instance_id,
        runtime_pid=runtime_pid,
        runtime_start_time_ticks=runtime_start_time_ticks,
        apply_call_id=descriptor.apply_call_id,
        monotonic_ns=descriptor.writer_monotonic_ns,
        sequence_ids=(descriptor.seq_id,),
        token_cells=tuple(
            ObservedTokenCell(
                seq_id=descriptor.seq_id,
                stream_index=descriptor.stream_index,
                token_position=token_position,
                cell_index=cell_index,
                ownership_generation=descriptor.apply_call_id,
            )
            for token_position, cell_index in descriptor.token_cells
        ),
    )


def _native_failure_reason(error: BaseException) -> str:
    text = str(error).upper()
    for reason in (
        "WRITER_CONTEXT_MISSING",
        "WRITER_CONTEXT_CONFLICT",
        "GRAPH_COMPUTE_FAILED",
        "KV_FRAGMENT_READ_FAILED",
        "KV_FRAGMENT_LOSS",
        "APPLY_UNIT_SEAL_MISSING",
        "FRAGMENT_SET_INCOMPLETE",
        "FRAGMENT_LAYOUT_MISMATCH",
        "UNSUPPORTED_MUTATION",
        "BUFFER_EXHAUSTED",
        "PERSISTENCE_FAILED",
    ):
        if reason in text:
            return reason
    if "DIGEST" in text or "PAYLOAD" in text:
        return "KV_FRAGMENT_LOSS"
    if "LAYOUT" in text or "CELL" in text or "STREAM" in text:
        return "FRAGMENT_LAYOUT_MISMATCH"
    return "ACQUISITION_FAILED"


__all__ = [
    "NativeUnitClient",
    "RecorderProgress",
    "StateUnitRecorder",
    "WriterCaptureStatistics",
]
