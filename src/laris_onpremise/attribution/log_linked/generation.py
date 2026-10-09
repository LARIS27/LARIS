from __future__ import annotations

from dataclasses import dataclass, field
import re

from .contracts import SequenceGenerationRef, StateUnitKind


class GenerationTrackingError(RuntimeError):
    """Raised when sequence-state continuity cannot be represented safely."""


_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_SAFE_REASON = re.compile(r"[A-Z][A-Z0-9_]{0,95}\Z")
_RESTART_MUTATIONS = frozenset(
    {"SEQUENCE_REMOVE", "SEQUENCE_RESET", "SEQUENCE_REUSE"}
)


def _nonnegative(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise GenerationTrackingError(f"{name} must be a non-negative integer")
    return value


def _positive(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise GenerationTrackingError(f"{name} must be a positive integer")
    return value


def _sorted_unique(name: str, values: tuple[int, ...]) -> tuple[int, ...]:
    if not isinstance(values, tuple):
        raise GenerationTrackingError(f"{name} must be a tuple")
    for value in values:
        _nonnegative(name, value)
    if values != tuple(sorted(set(values))):
        raise GenerationTrackingError(f"{name} must be sorted and unique")
    return values


@dataclass(frozen=True)
class ObservedTokenCell:
    seq_id: int
    stream_index: int
    token_position: int
    cell_index: int
    ownership_generation: int

    def __post_init__(self) -> None:
        for name in ("seq_id", "stream_index", "token_position", "cell_index"):
            _nonnegative(name, getattr(self, name))
        _positive("ownership_generation", self.ownership_generation)


@dataclass(frozen=True)
class ObservedMutation:
    mutation_kind: str
    sequence_ids: tuple[int, ...]
    supported: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.mutation_kind, str) or _SAFE_REASON.fullmatch(
            self.mutation_kind
        ) is None:
            raise GenerationTrackingError("mutation_kind is invalid")
        if not self.sequence_ids:
            raise GenerationTrackingError("mutation sequence_ids cannot be empty")
        _sorted_unique("mutation sequence_ids", self.sequence_ids)
        if not isinstance(self.supported, bool):
            raise GenerationTrackingError("mutation supported flag is invalid")


@dataclass(frozen=True)
class ObserverLifecycleEvent:
    runtime_instance_id: str
    runtime_pid: int
    runtime_start_time_ticks: int
    apply_call_id: int
    monotonic_ns: int
    sequence_ids: tuple[int, ...]
    token_cells: tuple[ObservedTokenCell, ...]
    mutations: tuple[ObservedMutation, ...] = ()
    event_loss_sequence_ids: tuple[int, ...] = ()
    read_race_sequence_ids: tuple[int, ...] = ()
    shared_membership_supported: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.runtime_instance_id, str) or not self.runtime_instance_id:
            raise GenerationTrackingError("runtime_instance_id is required")
        _positive("runtime_pid", self.runtime_pid)
        _positive("runtime_start_time_ticks", self.runtime_start_time_ticks)
        _nonnegative("apply_call_id", self.apply_call_id)
        _nonnegative("monotonic_ns", self.monotonic_ns)
        if not self.sequence_ids:
            raise GenerationTrackingError("sequence_ids cannot be empty")
        _sorted_unique("sequence_ids", self.sequence_ids)
        if not isinstance(self.token_cells, tuple) or any(
            not isinstance(cell, ObservedTokenCell) for cell in self.token_cells
        ):
            raise GenerationTrackingError("token_cells are invalid")
        if set(cell.seq_id for cell in self.token_cells) - set(self.sequence_ids):
            raise GenerationTrackingError("token cell sequence is not in sequence_ids")
        if not isinstance(self.mutations, tuple) or any(
            not isinstance(item, ObservedMutation) for item in self.mutations
        ):
            raise GenerationTrackingError("mutations are invalid")
        for name in ("event_loss_sequence_ids", "read_race_sequence_ids"):
            values = getattr(self, name)
            _sorted_unique(name, values)
            if set(values) - set(self.sequence_ids):
                raise GenerationTrackingError(f"{name} is outside sequence_ids")
        if not isinstance(self.shared_membership_supported, bool):
            raise GenerationTrackingError("shared_membership_supported is invalid")


@dataclass(frozen=True)
class UnitRequest:
    ref: SequenceGenerationRef
    unit_kind: StateUnitKind
    runtime_pid: int
    runtime_start_time_ticks: int
    apply_call_id: int
    monotonic_ns: int
    token_cells: tuple[ObservedTokenCell, ...]
    mutation_kinds: tuple[str, ...]
    previous_unit_digest: str | None
    base_unit_digest: str | None


@dataclass(frozen=True)
class GenerationSeal:
    ref: SequenceGenerationRef
    state: str
    reason_codes: tuple[str, ...]
    unit_count: int
    last_unit_digest: str | None
    opened_apply_call_id: int
    closed_apply_call_id: int

    def __post_init__(self) -> None:
        if self.state not in {"COMPLETE", "INCOMPLETE"}:
            raise GenerationTrackingError("generation seal state is invalid")
        if not self.reason_codes or self.reason_codes != tuple(
            sorted(set(self.reason_codes))
        ):
            raise GenerationTrackingError("generation seal reasons are invalid")
        for reason in self.reason_codes:
            if _SAFE_REASON.fullmatch(reason) is None:
                raise GenerationTrackingError("generation seal reason is invalid")
        _nonnegative("unit_count", self.unit_count)
        _nonnegative("opened_apply_call_id", self.opened_apply_call_id)
        _nonnegative("closed_apply_call_id", self.closed_apply_call_id)
        if self.closed_apply_call_id < self.opened_apply_call_id:
            raise GenerationTrackingError("generation seal apply range is invalid")
        if self.last_unit_digest is not None and _DIGEST.fullmatch(
            self.last_unit_digest
        ) is None:
            raise GenerationTrackingError("last_unit_digest is invalid")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": "1.0",
            **self.ref.to_mapping(),
            "state": self.state,
            "reason_codes": list(self.reason_codes),
            "unit_count": self.unit_count,
            "last_unit_digest": self.last_unit_digest,
            "opened_apply_call_id": self.opened_apply_call_id,
            "closed_apply_call_id": self.closed_apply_call_id,
        }


@dataclass
class _ActiveGeneration:
    ref: SequenceGenerationRef
    runtime_pid: int
    runtime_start_time_ticks: int
    opened_apply_call_id: int
    last_apply_call_id: int
    unit_count: int = 0
    base_unit_digest: str | None = None
    last_unit_digest: str | None = None
    maximum_position: int | None = None
    cell_ownership_generations: dict[tuple[int, int], int] = field(
        default_factory=dict
    )
    pending: UnitRequest | None = None


class SequenceGenerationTracker:
    """Track independent, fail-closed sequence continuity generations."""

    def __init__(self) -> None:
        self._active: dict[tuple[str, int], _ActiveGeneration] = {}
        self._latest_ref: dict[tuple[str, int], SequenceGenerationRef] = {}
        self._next_generation: dict[tuple[str, int], int] = {}
        self._seals: dict[SequenceGenerationRef, GenerationSeal] = {}
        self._runtime_identity: tuple[str, int, int] | None = None

    def consume(self, event: ObserverLifecycleEvent) -> tuple[UnitRequest, ...]:
        if not isinstance(event, ObserverLifecycleEvent):
            raise GenerationTrackingError("observer lifecycle event is invalid")
        identity = (
            event.runtime_instance_id,
            event.runtime_pid,
            event.runtime_start_time_ticks,
        )
        if self._runtime_identity is not None and identity != self._runtime_identity:
            for active in tuple(self._active.values()):
                state = "INCOMPLETE" if active.pending is not None else "COMPLETE"
                reason = (
                    "MISSING_PREDECESSOR"
                    if active.pending is not None
                    else "PROCESS_IDENTITY_CHANGED"
                )
                self._close(active, state=state, reason=reason)
        self._runtime_identity = identity

        cells_by_sequence = {
            seq_id: tuple(
                sorted(
                    (cell for cell in event.token_cells if cell.seq_id == seq_id),
                    key=lambda cell: (
                        cell.token_position,
                        cell.cell_index,
                        cell.stream_index,
                    ),
                )
            )
            for seq_id in event.sequence_ids
        }
        failed: dict[int, str] = {
            seq_id: "EVENT_LOSS" for seq_id in event.event_loss_sequence_ids
        }
        failed.update(
            {seq_id: "READ_RACE" for seq_id in event.read_race_sequence_ids}
        )
        restart: set[int] = set()
        mutation_kinds: dict[int, set[str]] = {
            seq_id: set() for seq_id in event.sequence_ids
        }
        for mutation in event.mutations:
            for seq_id in mutation.sequence_ids:
                if seq_id not in mutation_kinds:
                    continue
                mutation_kinds[seq_id].add(mutation.mutation_kind)
                if not mutation.supported:
                    failed[seq_id] = "UNSUPPORTED_MUTATION"
                elif mutation.mutation_kind in _RESTART_MUTATIONS:
                    restart.add(seq_id)

        if not event.shared_membership_supported:
            cell_members: dict[tuple[int, int], set[int]] = {}
            for cell in event.token_cells:
                cell_members.setdefault(
                    (cell.stream_index, cell.cell_index), set()
                ).add(cell.seq_id)
            for members in cell_members.values():
                if len(members) > 1:
                    for seq_id in members:
                        failed[seq_id] = "UNREPRESENTABLE_SHARED_CELL"

        requests: list[UnitRequest] = []
        for seq_id in event.sequence_ids:
            key = (event.runtime_instance_id, seq_id)
            active = self._active.get(key)
            if active is not None and active.pending is not None:
                self._close(
                    active, state="INCOMPLETE", reason="MISSING_PREDECESSOR"
                )
                active = None
                failed.setdefault(seq_id, "MISSING_PREDECESSOR")

            if seq_id in restart and active is not None:
                self._close(
                    active,
                    state="COMPLETE",
                    reason="EXPLICIT_SEQUENCE_RESTART",
                )
                active = None

            cells = cells_by_sequence[seq_id]
            if seq_id in failed:
                if active is None:
                    active = self._open(event, seq_id)
                self._close(active, state="INCOMPLETE", reason=failed[seq_id])
                continue
            if not cells:
                continue

            if active is not None and self._requires_new_generation(active, cells):
                reason = (
                    "POSITION_REGRESSION"
                    if self._has_position_regression(active, cells)
                    else "OWNERSHIP_GENERATION_CHANGED"
                )
                self._close(active, state="COMPLETE", reason=reason)
                active = None
            if active is None:
                active = self._open(event, seq_id)

            kind = (
                StateUnitKind.BASE
                if active.unit_count == 0
                else StateUnitKind.DELTA
            )
            request = UnitRequest(
                ref=active.ref,
                unit_kind=kind,
                runtime_pid=event.runtime_pid,
                runtime_start_time_ticks=event.runtime_start_time_ticks,
                apply_call_id=event.apply_call_id,
                monotonic_ns=event.monotonic_ns,
                token_cells=cells,
                mutation_kinds=tuple(sorted(mutation_kinds[seq_id])),
                previous_unit_digest=(
                    active.last_unit_digest if kind is StateUnitKind.DELTA else None
                ),
                base_unit_digest=(
                    active.base_unit_digest if kind is StateUnitKind.DELTA else None
                ),
            )
            if kind is StateUnitKind.DELTA and (
                request.previous_unit_digest is None
                or request.base_unit_digest is None
            ):
                self._close(
                    active, state="INCOMPLETE", reason="MISSING_PREDECESSOR"
                )
                continue
            active.pending = request
            active.last_apply_call_id = event.apply_call_id
            requests.append(request)
        return tuple(requests)

    def acknowledge(self, request: UnitRequest, *, unit_digest: str) -> None:
        if not isinstance(request, UnitRequest):
            raise GenerationTrackingError("unit request is invalid")
        if not isinstance(unit_digest, str) or _DIGEST.fullmatch(unit_digest) is None:
            raise GenerationTrackingError("unit_digest is invalid")
        key = (request.ref.runtime_instance_id, request.ref.seq_id)
        active = self._active.get(key)
        if active is None or active.ref != request.ref or active.pending != request:
            raise GenerationTrackingError("unit request is not the pending request")
        active.unit_count += 1
        active.last_unit_digest = unit_digest
        if request.unit_kind is StateUnitKind.BASE:
            active.base_unit_digest = unit_digest
        active.maximum_position = max(cell.token_position for cell in request.token_cells)
        for cell in request.token_cells:
            active.cell_ownership_generations[
                (cell.stream_index, cell.cell_index)
            ] = cell.ownership_generation
        active.pending = None

    def mark_incomplete(
        self, ref: SequenceGenerationRef, reason: str
    ) -> GenerationSeal:
        if not isinstance(ref, SequenceGenerationRef):
            raise GenerationTrackingError("sequence generation reference is invalid")
        if not isinstance(reason, str) or _SAFE_REASON.fullmatch(reason) is None:
            raise GenerationTrackingError("incomplete reason is invalid")
        active = self._active.get((ref.runtime_instance_id, ref.seq_id))
        if active is None or active.ref != ref:
            raise GenerationTrackingError("sequence generation is not active")
        return self._close(active, state="INCOMPLETE", reason=reason)

    def active_refs(self) -> tuple[SequenceGenerationRef, ...]:
        return tuple(
            sorted(
                (active.ref for active in self._active.values()),
                key=lambda ref: (
                    ref.runtime_instance_id,
                    ref.seq_id,
                    ref.sequence_generation,
                ),
            )
        )

    def seals(self) -> tuple[GenerationSeal, ...]:
        return tuple(
            sorted(
                self._seals.values(),
                key=lambda seal: (
                    seal.ref.runtime_instance_id,
                    seal.ref.seq_id,
                    seal.ref.sequence_generation,
                ),
            )
        )

    def seal(self, ref: SequenceGenerationRef) -> GenerationSeal:
        existing = self._seals.get(ref)
        if existing is not None:
            return existing
        active = self._active.get((ref.runtime_instance_id, ref.seq_id))
        if active is None or active.ref != ref:
            raise GenerationTrackingError("sequence generation is unknown")
        if active.pending is not None:
            return self._close(
                active, state="INCOMPLETE", reason="MISSING_PREDECESSOR"
            )
        return self._close(active, state="COMPLETE", reason="MANUAL_SEAL")

    def current_ref(
        self, runtime_instance_id: str, seq_id: int
    ) -> SequenceGenerationRef | None:
        return self._latest_ref.get((runtime_instance_id, seq_id))

    def _open(
        self, event: ObserverLifecycleEvent, seq_id: int
    ) -> _ActiveGeneration:
        key = (event.runtime_instance_id, seq_id)
        number = self._next_generation.get(key, 1)
        self._next_generation[key] = number + 1
        ref = SequenceGenerationRef(event.runtime_instance_id, seq_id, number)
        active = _ActiveGeneration(
            ref=ref,
            runtime_pid=event.runtime_pid,
            runtime_start_time_ticks=event.runtime_start_time_ticks,
            opened_apply_call_id=event.apply_call_id,
            last_apply_call_id=event.apply_call_id,
        )
        self._active[key] = active
        self._latest_ref[key] = ref
        return active

    @staticmethod
    def _has_position_regression(
        active: _ActiveGeneration, cells: tuple[ObservedTokenCell, ...]
    ) -> bool:
        return active.maximum_position is not None and max(
            cell.token_position for cell in cells
        ) <= active.maximum_position

    @classmethod
    def _requires_new_generation(
        cls, active: _ActiveGeneration, cells: tuple[ObservedTokenCell, ...]
    ) -> bool:
        if cls._has_position_regression(active, cells):
            return True
        for cell in cells:
            previous = active.cell_ownership_generations.get(
                (cell.stream_index, cell.cell_index)
            )
            if previous is not None and previous != cell.ownership_generation:
                return True
        return False

    def _close(
        self, active: _ActiveGeneration, *, state: str, reason: str
    ) -> GenerationSeal:
        seal = GenerationSeal(
            ref=active.ref,
            state=state,
            reason_codes=(reason,),
            unit_count=active.unit_count,
            last_unit_digest=active.last_unit_digest,
            opened_apply_call_id=active.opened_apply_call_id,
            closed_apply_call_id=active.last_apply_call_id,
        )
        self._seals[active.ref] = seal
        key = (active.ref.runtime_instance_id, active.ref.seq_id)
        if self._active.get(key) is active:
            del self._active[key]
        return seal


__all__ = [
    "GenerationSeal",
    "GenerationTrackingError",
    "ObservedMutation",
    "ObservedTokenCell",
    "ObserverLifecycleEvent",
    "SequenceGenerationTracker",
    "UnitRequest",
]
