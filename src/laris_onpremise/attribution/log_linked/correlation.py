from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
from pathlib import Path
import re
from types import MappingProxyType
from typing import Mapping

from .contracts import (
    AttributionDecision,
    SequenceGenerationRef,
    canonical_document_bytes,
    parse_attribution_result,
    parse_state_unit_manifest,
)
from .endpoint import (
    derive_endpoint_ref,
    validate_correlation_key_id,
)
from .generation import GenerationSeal
from .llama_server_log import ApplicationEvent
from .state_units import StoredStateUnit


class CorrelationError(RuntimeError):
    """Raised when correlation inputs violate their sealed evidence contract."""


_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,191}\Z")
_RUNTIME_INSTANCE_ID = re.compile(
    r"(?:[A-Za-z0-9][A-Za-z0-9._-]{0,191}|sha256:[0-9a-f]{64})\Z"
)
GLOBAL_STRUCTURAL_RANKED_V2_CONTRACT = MappingProxyType(
    {
        "analysis_version": "global-structural-ranked-v2",
        "state_candidate_rules": (
            "COMPLETE_APPLICATION_LIFECYCLE",
            "SLOT_SEQUENCE_MAPPING",
            "EXPECTED_FINAL_POSITION_SET",
            "COMPLETE_GENERATION_SEAL",
            "GENERATION_TERMINAL_CUT",
            "MAXIMUM_MONOTONE_SEQUENCE_ASSIGNMENT",
        ),
        "endpoint_candidate_rules": (
            "COMPLETE_APPLICATION_LIFECYCLE",
            "ALLOWED_PATH_CLASS",
            "SOURCE_ORDER_INSIDE_SLOT_EPOCH",
            "CLOCK_START_FEASIBILITY",
            "TERMINAL_OUTCOME_COMPATIBILITY",
            "MAXIMUM_BIPARTITE_ASSIGNMENT",
            "QUIESCENT_LIFECYCLE_COMPONENT",
        ),
        "endpoint_ranking_objective": (
            "MAX_TERMINAL_DELTA",
            "TOTAL_TERMINAL_DELTA",
            "MAX_SOURCE_GAP",
            "TOTAL_SOURCE_GAP",
            "SOURCE_ORDER_INVERSIONS",
        ),
        "final_decision_precedence": (
            "INCOMPLETE",
            "AMBIGUOUS",
            "CERTIFIED_IF_BOTH_BINDINGS_CERTIFIED",
            "RANKED",
        ),
        "maximum_ranked_matchings": 4096,
    }
)
_MAX_MATCHINGS = int(
    GLOBAL_STRUCTURAL_RANKED_V2_CONTRACT["maximum_ranked_matchings"]
)


@dataclass(frozen=True)
class RuntimeCorrelationProfile:
    adapter_id: str
    runtime_instance_id: str
    slot_sequence_map: tuple[tuple[int, int], ...]
    wall_to_monotonic_offset_ns: int
    maximum_clock_error_ns: int
    maximum_access_delay_ns: int
    allowed_path_classes: tuple[str, ...]
    source_application_log_digest: str
    source_observer_terminal_seal_digest: str
    clock_normalization_receipt_digest: str
    maximum_access_pre_terminal_skew_ns: int = 0

    def __post_init__(self) -> None:
        _safe_id("adapter_id", self.adapter_id)
        _runtime_instance_id(self.runtime_instance_id)
        if not isinstance(self.slot_sequence_map, tuple) or not self.slot_sequence_map:
            raise CorrelationError("slot_sequence_map must not be empty")
        slots: list[int] = []
        for item in self.slot_sequence_map:
            if not isinstance(item, tuple) or len(item) != 2:
                raise CorrelationError("slot_sequence_map contains an invalid pair")
            slot_id, seq_id = item
            _nonnegative("slot_id", slot_id)
            _nonnegative("seq_id", seq_id)
            slots.append(slot_id)
        if self.slot_sequence_map != tuple(sorted(self.slot_sequence_map)):
            raise CorrelationError("slot_sequence_map must be sorted")
        if len(set(slots)) != len(slots):
            raise CorrelationError("slot_sequence_map contains a duplicate slot")
        if isinstance(self.wall_to_monotonic_offset_ns, bool) or not isinstance(
            self.wall_to_monotonic_offset_ns, int
        ):
            raise CorrelationError("wall_to_monotonic_offset_ns must be an integer")
        _nonnegative("maximum_clock_error_ns", self.maximum_clock_error_ns)
        _nonnegative("maximum_access_delay_ns", self.maximum_access_delay_ns)
        _nonnegative(
            "maximum_access_pre_terminal_skew_ns",
            self.maximum_access_pre_terminal_skew_ns,
        )
        if (
            not isinstance(self.allowed_path_classes, tuple)
            or not self.allowed_path_classes
            or self.allowed_path_classes
            != tuple(sorted(set(self.allowed_path_classes)))
        ):
            raise CorrelationError("allowed_path_classes must be sorted and unique")
        for value in self.allowed_path_classes:
            _safe_id("allowed_path_class", value)
        for name in (
            "source_application_log_digest",
            "source_observer_terminal_seal_digest",
            "clock_normalization_receipt_digest",
        ):
            _digest(name, getattr(self, name))


@dataclass(frozen=True)
class CorrelationInputs:
    application_events: tuple[ApplicationEvent, ...]
    generation_seals: tuple[GenerationSeal, ...]
    state_units: tuple[StoredStateUnit, ...]
    runtime_profile: RuntimeCorrelationProfile

    def __post_init__(self) -> None:
        if not isinstance(self.runtime_profile, RuntimeCorrelationProfile):
            raise CorrelationError("runtime_profile is invalid")
        if not isinstance(self.application_events, tuple) or any(
            not isinstance(event, ApplicationEvent)
            for event in self.application_events
        ):
            raise CorrelationError("application_events are invalid")
        orders = tuple(event.source_order for event in self.application_events)
        if orders != tuple(sorted(orders)) or len(set(orders)) != len(orders):
            raise CorrelationError("application events are not canonically ordered")
        if any(
            event.adapter_id != self.runtime_profile.adapter_id
            for event in self.application_events
        ):
            raise CorrelationError("application event adapter differs")
        if not isinstance(self.generation_seals, tuple) or any(
            not isinstance(seal, GenerationSeal) for seal in self.generation_seals
        ):
            raise CorrelationError("generation_seals are invalid")
        if any(
            seal.ref.runtime_instance_id
            != self.runtime_profile.runtime_instance_id
            for seal in self.generation_seals
        ):
            raise CorrelationError("generation seal runtime differs")
        if not isinstance(self.state_units, tuple) or any(
            not isinstance(unit, StoredStateUnit) for unit in self.state_units
        ):
            raise CorrelationError("state_units are invalid")


@dataclass(frozen=True)
class BindingAssessment:
    decision: AttributionDecision
    reason_code: str
    candidate_ids: tuple[str, ...]
    selected_candidate_id: str | None = None
    optimal_candidate_count: int | None = None
    optimal_assignment_count: int | None = None
    ranking_stage: str | None = None

    def to_mapping(self) -> dict[str, object]:
        return {
            "decision": self.decision.value,
            "reason_code": self.reason_code,
            "candidate_ids": list(self.candidate_ids),
            "candidate_count": len(self.candidate_ids),
            "selected_candidate_id": self.selected_candidate_id,
            "optimal_candidate_count": self.optimal_candidate_count,
            "optimal_assignment_count": self.optimal_assignment_count,
            "ranking_stage": self.ranking_stage,
        }


@dataclass(frozen=True, init=False)
class AttributionResult:
    _document: dict[str, object]

    def __init__(self, document: Mapping[str, object]) -> None:
        payload = canonical_document_bytes(document)
        parsed = parse_attribution_result(payload)
        object.__setattr__(self, "_document", parsed)

    def to_mapping(self) -> dict[str, object]:
        return dict(self._document)

    def to_bytes(self) -> bytes:
        return canonical_document_bytes(self._document)

    @property
    def final_decision(self) -> AttributionDecision:
        return AttributionDecision(str(self._document["final_decision"]))

    def __getattr__(self, name: str) -> object:
        if name in self._document:
            value = self._document[name]
            if name == "state_unit_chain_digests":
                return tuple(value)
            return value
        raise AttributeError(name)


@dataclass
class _EpochBuilder:
    task_id: int
    slot_id: int
    slot_epoch: int
    start_wall_ns: int
    start_source_order: int
    prompt_tokens_total: int | None
    prompt_tokens_new: int | None = None
    prompt_tokens_reused: int = 0
    generated_tokens: int | None = None
    terminal_outcome: str | None = None
    end_wall_ns: int | None = None
    end_source_order: int | None = None
    complete: bool = True


@dataclass(frozen=True)
class _StateCandidate:
    candidate_id: str
    ref: SequenceGenerationRef
    units: tuple[StoredStateUnit, ...]
    cutoff_apply_call_id: int
    cutoff_monotonic_ns: int


@dataclass(frozen=True)
class _IndexedStateCut:
    ref: SequenceGenerationRef
    units: tuple[StoredStateUnit, ...]
    cutoff_apply_call_id: int
    cutoff_monotonic_ns: int
    positions: frozenset[int]


@dataclass(frozen=True)
class _RankedAccessSupport:
    access_indexes: frozenset[int]
    optimal_assignment_count: int
    ranking_stage: str | None
    search_complete: bool


class EndpointCorrelator:
    """Produce the sole global structural attribution decision."""

    def analyze(
        self,
        inputs: CorrelationInputs,
        *,
        correlation_key: bytes,
        correlation_key_id: str,
    ) -> tuple[AttributionResult, ...]:
        """Return the canonical global structural attribution results."""
        if not isinstance(inputs, CorrelationInputs):
            raise CorrelationError("inputs must be CorrelationInputs")
        if type(correlation_key) is not bytes or len(correlation_key) != 32:
            raise CorrelationError("correlation_key must be exactly 32 bytes")
        validate_correlation_key_id(correlation_key_id)
        epochs, accesses = _reconstruct_epochs(inputs.application_events)
        access_sets, application_complete = _possible_access_assignments(
            epochs,
            accesses,
            inputs.runtime_profile,
        )
        state_sets, state_complete = _possible_state_assignments(
            epochs,
            inputs.generation_seals,
            inputs.state_units,
            inputs.runtime_profile,
        )
        ranked_support = _rank_access_assignments(
            epochs,
            accesses,
            access_sets,
            application_complete,
        )
        results: list[AttributionResult] = []
        for index, epoch in enumerate(epochs):
            results.append(
                _attribution_result_for_epoch(
                    epoch=epoch,
                    accesses=accesses,
                    access_candidates=access_sets[index],
                    application_complete=application_complete[index],
                    ranked_support=ranked_support[index],
                    state_candidates=state_sets[index],
                    state_complete=state_complete[index],
                    profile=inputs.runtime_profile,
                    correlation_key=correlation_key,
                    correlation_key_id=correlation_key_id,
                )
            )
        return tuple(results)


def _reconstruct_epochs(
    events: tuple[ApplicationEvent, ...],
) -> tuple[tuple[_EpochBuilder, ...], tuple[ApplicationEvent, ...]]:
    active: dict[int, _EpochBuilder] = {}
    epochs: list[_EpochBuilder] = []
    accesses: list[ApplicationEvent] = []
    slot_epochs: dict[int, int] = {}
    for event in events:
        if event.event_type == "ACCESS":
            accesses.append(event)
            continue
        if event.slot_id is None or event.task_id is None:
            continue
        if event.event_type == "SLOT_PROMPT":
            prior = active.pop(event.slot_id, None)
            if prior is not None:
                prior.complete = False
                epochs.append(prior)
            slot_epoch = slot_epochs.get(event.slot_id, 0) + 1
            slot_epochs[event.slot_id] = slot_epoch
            active[event.slot_id] = _EpochBuilder(
                task_id=event.task_id,
                slot_id=event.slot_id,
                slot_epoch=slot_epoch,
                start_wall_ns=event.wall_time_utc_ns,
                start_source_order=event.source_order,
                prompt_tokens_total=event.prompt_tokens_total,
            )
            continue
        current = active.get(event.slot_id)
        if current is None or current.task_id != event.task_id:
            continue
        if event.event_type == "SLOT_REUSE":
            if event.prompt_tokens_reused is None:
                current.complete = False
            else:
                current.prompt_tokens_reused = event.prompt_tokens_reused
        elif event.event_type == "SLOT_PROMPT_EVAL":
            current.prompt_tokens_new = event.prompt_tokens_new
        elif event.event_type == "SLOT_TERMINAL":
            current.generated_tokens = event.generated_tokens
            current.terminal_outcome = event.terminal_outcome
            current.end_wall_ns = event.wall_time_utc_ns
            current.end_source_order = event.source_order
            active.pop(event.slot_id, None)
            epochs.append(current)
    for current in active.values():
        current.complete = False
        epochs.append(current)
    epochs.sort(
        key=lambda item: (item.start_source_order, item.slot_id, item.slot_epoch)
    )
    accesses.sort(key=lambda item: item.source_order)
    return tuple(epochs), tuple(accesses)


def _possible_access_assignments(
    epochs: tuple[_EpochBuilder, ...],
    accesses: tuple[ApplicationEvent, ...],
    profile: RuntimeCorrelationProfile,
) -> tuple[tuple[frozenset[int], ...], tuple[bool, ...]]:
    eligible_accesses = tuple(
        index
        for index, access in enumerate(accesses)
        if access.path_class in profile.allowed_path_classes
    )
    edges: list[tuple[int, ...]] = []
    for epoch_index, epoch in enumerate(epochs):
        if (
            not epoch.complete
            or epoch.end_wall_ns is None
            or epoch.end_source_order is None
        ):
            edges.append(())
            continue
        next_same_slot_start = min(
            (
                later.start_source_order
                for later in epochs[epoch_index + 1 :]
                if later.slot_id == epoch.slot_id
            ),
            default=None,
        )
        candidates = []
        for access_index in eligible_accesses:
            access = accesses[access_index]
            if access.source_order <= epoch.start_source_order:
                continue
            if (
                next_same_slot_start is not None
                and access.source_order >= next_same_slot_start
            ):
                continue
            if (
                access.wall_time_utc_ns + profile.maximum_clock_error_ns
                < epoch.start_wall_ns
            ):
                continue
            if (
                epoch.terminal_outcome in {"CANCELLED", "ERROR"}
                and access.terminal_outcome != epoch.terminal_outcome
            ):
                continue
            candidates.append(access_index)
        edges.append(tuple(candidates))
    if not epochs:
        return (), ()
    frozen_edges = tuple(frozenset(edge) for edge in edges)
    optimum = _maximum_bipartite_matches(frozen_edges)
    possible: list[frozenset[int]] = []
    complete: list[bool] = []
    for epoch_index, edge in enumerate(frozen_edges):
        retained = frozenset(
            access_index
            for access_index in edge
            if 1
            + _maximum_bipartite_matches(
                frozen_edges,
                skip_epoch=epoch_index,
                skip_access=access_index,
            )
            == optimum
        )
        possible.append(retained)
        complete.append(
            _maximum_bipartite_matches(
                frozen_edges,
                skip_epoch=epoch_index,
            )
            < optimum
        )
    return tuple(possible), tuple(complete)


def _maximum_bipartite_matches(
    edges: tuple[frozenset[int], ...],
    *,
    skip_epoch: int | None = None,
    skip_access: int | None = None,
) -> int:
    access_to_epoch: dict[int, int] = {}

    def augment(epoch_index: int, visited: set[int]) -> bool:
        for access_index in sorted(edges[epoch_index]):
            if access_index == skip_access or access_index in visited:
                continue
            visited.add(access_index)
            previous = access_to_epoch.get(access_index)
            if previous is None or augment(previous, visited):
                access_to_epoch[access_index] = epoch_index
                return True
        return False

    matched = 0
    for epoch_index in range(len(edges)):
        if epoch_index == skip_epoch:
            continue
        if augment(epoch_index, set()):
            matched += 1
    return matched


def _state_cut_inventory(
    seals: tuple[GenerationSeal, ...],
    units: tuple[StoredStateUnit, ...],
) -> dict[int, tuple[_IndexedStateCut, ...]]:
    documents = {unit: _unit_document(unit) for unit in units}
    units_by_ref: dict[SequenceGenerationRef, list[StoredStateUnit]] = {}
    for unit in units:
        units_by_ref.setdefault(unit.ref, []).append(unit)
    indexed: dict[int, list[_IndexedStateCut]] = {}
    for seal in seals:
        if seal.state != "COMPLETE":
            continue
        generation_units = tuple(
            sorted(
                units_by_ref.get(seal.ref, ()),
                key=lambda unit: unit.unit_number,
            )
        )
        if not _valid_unit_chain(
            generation_units,
            seal,
            documents=documents,
        ):
            continue
        positions: set[int] = set()
        chain: list[StoredStateUnit] = []
        for unit in generation_units:
            document = documents[unit]
            current = _expanded_positions(document["position_ranges"])
            mutation_kinds = document["mutation_kinds"]
            if not isinstance(mutation_kinds, list):
                raise CorrelationError("state-unit mutation kinds are invalid")
            if not chain or mutation_kinds:
                positions = set(current)
            else:
                positions.update(current)
            chain.append(unit)
            apply_call_id = int(document["apply_call_id"])
            if not (
                seal.opened_apply_call_id
                <= apply_call_id
                <= seal.closed_apply_call_id
            ):
                continue
            if (
                apply_call_id != seal.closed_apply_call_id
                or not hmac.compare_digest(
                    unit.manifest_digest,
                    str(seal.last_unit_digest),
                )
            ):
                continue
            indexed.setdefault(seal.ref.seq_id, []).append(
                _IndexedStateCut(
                    ref=seal.ref,
                    units=tuple(chain),
                    cutoff_apply_call_id=apply_call_id,
                    cutoff_monotonic_ns=int(document["apply_monotonic_ns"]),
                    positions=frozenset(positions),
                )
            )
    return {
        seq_id: tuple(
            sorted(
                cuts,
                key=lambda item: (
                    item.cutoff_apply_call_id,
                    item.ref.sequence_generation,
                ),
            )
        )
        for seq_id, cuts in indexed.items()
    }


def _raw_state_candidates(
    epoch: _EpochBuilder,
    inventory: Mapping[int, tuple[_IndexedStateCut, ...]],
    profile: RuntimeCorrelationProfile,
) -> tuple[tuple[_StateCandidate, ...], bool]:
    mapping = dict(profile.slot_sequence_map)
    seq_id = mapping.get(epoch.slot_id)
    if seq_id is None or epoch.end_wall_ns is None or not epoch.complete:
        return (), False
    if (
        epoch.prompt_tokens_total is None
        or epoch.generated_tokens is None
    ):
        return (), False
    prompt_new = epoch.prompt_tokens_new
    if prompt_new is None:
        prompt_new = epoch.prompt_tokens_total - epoch.prompt_tokens_reused
    if (
        prompt_new < 0
        or prompt_new + epoch.prompt_tokens_reused
        != epoch.prompt_tokens_total
    ):
        return (), False
    expected_final_positions = frozenset(
        range(epoch.prompt_tokens_total + epoch.generated_tokens)
    )
    candidates: list[_StateCandidate] = []
    for cutoff in inventory.get(seq_id, ()):
        if cutoff.positions != expected_final_positions:
            continue
        candidate_id = _candidate_id(
            epoch,
            cutoff.ref,
            cutoff.cutoff_apply_call_id,
            tuple(unit.manifest_digest for unit in cutoff.units),
        )
        candidates.append(
            _StateCandidate(
                candidate_id=candidate_id,
                ref=cutoff.ref,
                units=cutoff.units,
                cutoff_apply_call_id=cutoff.cutoff_apply_call_id,
                cutoff_monotonic_ns=cutoff.cutoff_monotonic_ns,
            )
        )
    candidates.sort(
        key=lambda item: (item.cutoff_apply_call_id, item.candidate_id)
    )
    return tuple(candidates), True


def _possible_state_assignments(
    epochs: tuple[_EpochBuilder, ...],
    seals: tuple[GenerationSeal, ...],
    units: tuple[StoredStateUnit, ...],
    profile: RuntimeCorrelationProfile,
) -> tuple[tuple[tuple[_StateCandidate, ...], ...], tuple[bool, ...]]:
    raw: list[tuple[_StateCandidate, ...]] = []
    structurally_complete: list[bool] = []
    mapping = dict(profile.slot_sequence_map)
    inventory = _state_cut_inventory(seals, units)
    for epoch in epochs:
        candidates, complete = _raw_state_candidates(
            epoch,
            inventory,
            profile,
        )
        raw.append(candidates)
        structurally_complete.append(complete)

    possible: list[tuple[_StateCandidate, ...]] = [() for _ in epochs]
    complete_by_epoch = list(structurally_complete)
    grouped: dict[int, list[int]] = {}
    for index, epoch in enumerate(epochs):
        seq_id = mapping.get(epoch.slot_id)
        if seq_id is None:
            complete_by_epoch[index] = False
            continue
        grouped.setdefault(seq_id, []).append(index)

    for epoch_indexes in grouped.values():
        apply_ids = tuple(
            sorted(
                {
                    candidate.cutoff_apply_call_id
                    for epoch_index in epoch_indexes
                    for candidate in raw[epoch_index]
                }
            )
        )
        apply_index = {apply_id: index for index, apply_id in enumerate(apply_ids)}
        edges = tuple(
            frozenset(
                apply_index[candidate.cutoff_apply_call_id]
                for candidate in raw[epoch_index]
            )
            for epoch_index in epoch_indexes
        )
        optimum = _maximum_monotone_matches(edges, len(apply_ids))
        prefix = _monotone_prefix_scores(edges, len(apply_ids))
        suffix = _monotone_suffix_scores(edges, len(apply_ids))
        for local_epoch, epoch_index in enumerate(epoch_indexes):
            retained: list[_StateCandidate] = []
            for candidate in raw[epoch_index]:
                local_apply = apply_index[candidate.cutoff_apply_call_id]
                if local_apply not in edges[local_epoch]:
                    continue
                if (
                    prefix[local_epoch][local_apply]
                    + 1
                    + suffix[local_epoch + 1][local_apply + 1]
                    == optimum
                ):
                    retained.append(candidate)
            possible[epoch_index] = tuple(retained)
            without_epoch = _maximum_monotone_matches(
                edges,
                len(apply_ids),
                skip_epoch=local_epoch,
            )
            if without_epoch == optimum:
                complete_by_epoch[epoch_index] = False

    return tuple(possible), tuple(complete_by_epoch)


def _maximum_monotone_matches(
    edges: tuple[frozenset[int], ...],
    candidate_count: int,
    *,
    skip_epoch: int | None = None,
) -> int:
    selected = tuple(
        edge for index, edge in enumerate(edges) if index != skip_epoch
    )
    return _monotone_prefix_scores(selected, candidate_count)[-1][-1]


def _monotone_prefix_scores(
    edges: tuple[frozenset[int], ...], candidate_count: int
) -> tuple[tuple[int, ...], ...]:
    scores = [[0] * (candidate_count + 1) for _ in range(len(edges) + 1)]
    for epoch_index, edge in enumerate(edges, start=1):
        for candidate_index in range(1, candidate_count + 1):
            best = max(
                scores[epoch_index - 1][candidate_index],
                scores[epoch_index][candidate_index - 1],
            )
            if candidate_index - 1 in edge:
                best = max(
                    best,
                    scores[epoch_index - 1][candidate_index - 1] + 1,
                )
            scores[epoch_index][candidate_index] = best
    return tuple(tuple(row) for row in scores)


def _monotone_suffix_scores(
    edges: tuple[frozenset[int], ...], candidate_count: int
) -> tuple[tuple[int, ...], ...]:
    scores = [[0] * (candidate_count + 1) for _ in range(len(edges) + 1)]
    for epoch_index in range(len(edges) - 1, -1, -1):
        edge = edges[epoch_index]
        for candidate_index in range(candidate_count - 1, -1, -1):
            best = max(
                scores[epoch_index + 1][candidate_index],
                scores[epoch_index][candidate_index + 1],
            )
            if candidate_index in edge:
                best = max(
                    best,
                    scores[epoch_index + 1][candidate_index + 1] + 1,
                )
            scores[epoch_index][candidate_index] = best
    return tuple(tuple(row) for row in scores)


_RANKING_STAGES = tuple(
    str(stage)
    for stage in GLOBAL_STRUCTURAL_RANKED_V2_CONTRACT[
        "endpoint_ranking_objective"
    ]
)


def _rank_access_assignments(
    epochs: tuple[_EpochBuilder, ...],
    accesses: tuple[ApplicationEvent, ...],
    edges: tuple[frozenset[int], ...],
    complete: tuple[bool, ...],
) -> tuple[_RankedAccessSupport, ...]:
    unresolved = _RankedAccessSupport(
        access_indexes=frozenset(),
        optimal_assignment_count=0,
        ranking_stage=None,
        search_complete=True,
    )
    result = [unresolved for _ in epochs]
    component_edges = _quiescent_component_edges(epochs, accesses, edges)
    for epoch_indexes, access_indexes in _bipartite_components(component_edges):
        if (
            len(epoch_indexes) != len(access_indexes)
            or any(not complete[index] for index in epoch_indexes)
        ):
            continue
        assignments, search_complete = _enumerate_access_assignments(
            epoch_indexes,
            component_edges,
        )
        if not assignments or not search_complete:
            support = frozenset(
                access_index
                for epoch_index in epoch_indexes
                for access_index in component_edges[epoch_index]
            )
            for epoch_index in epoch_indexes:
                result[epoch_index] = _RankedAccessSupport(
                    access_indexes=component_edges[epoch_index] & support,
                    optimal_assignment_count=0,
                    ranking_stage=None,
                    search_complete=search_complete,
                )
            continue
        costs = tuple(
            _access_assignment_cost(assignment, epochs, accesses)
            for assignment in assignments
        )
        best = min(costs)
        optimal = tuple(
            assignment
            for assignment, cost in zip(assignments, costs, strict=True)
            if cost == best
        )
        for epoch_index in epoch_indexes:
            optimal_support = frozenset(
                assignment[epoch_index] for assignment in optimal
            )
            ranking_stage = (
                "LIFECYCLE_COMPONENT"
                if len(component_edges[epoch_index]) == 1
                and len(edges[epoch_index]) > 1
                else _ranking_stage_for_epoch(
                    epoch_index,
                    assignments,
                    costs,
                )
            )
            result[epoch_index] = _RankedAccessSupport(
                access_indexes=optimal_support,
                optimal_assignment_count=len(optimal),
                ranking_stage=ranking_stage,
                search_complete=True,
            )
    return tuple(result)


def _quiescent_component_edges(
    epochs: tuple[_EpochBuilder, ...],
    accesses: tuple[ApplicationEvent, ...],
    edges: tuple[frozenset[int], ...],
) -> tuple[frozenset[int], ...]:
    """Partition ranking at lifecycle-quiescent source-order boundaries."""
    components: list[dict[str, object]] = []
    for epoch_index in sorted(
        range(len(epochs)),
        key=lambda index: (
            epochs[index].start_source_order,
            epochs[index].slot_id,
            epochs[index].slot_epoch,
        ),
    ):
        epoch = epochs[epoch_index]
        if epoch.end_source_order is None:
            continue
        if (
            not components
            or epoch.start_source_order > int(components[-1]["maximum_end_order"])
        ):
            components.append(
                {
                    "start_order": epoch.start_source_order,
                    "maximum_end_order": epoch.end_source_order,
                    "epoch_indexes": [epoch_index],
                }
            )
        else:
            component = components[-1]
            component["maximum_end_order"] = max(
                int(component["maximum_end_order"]),
                epoch.end_source_order,
            )
            epoch_indexes = component["epoch_indexes"]
            assert isinstance(epoch_indexes, list)
            epoch_indexes.append(epoch_index)

    epoch_components: dict[int, int] = {}
    for component_index, component in enumerate(components):
        epoch_indexes = component["epoch_indexes"]
        assert isinstance(epoch_indexes, list)
        for epoch_index in epoch_indexes:
            epoch_components[int(epoch_index)] = component_index

    access_components: dict[int, int] = {}
    for access_index, access in enumerate(accesses):
        eligible = tuple(
            component_index
            for component_index, component in enumerate(components)
            if int(component["start_order"]) < access.source_order
        )
        if eligible:
            access_components[access_index] = eligible[-1]

    return tuple(
        frozenset(
            access_index
            for access_index in edge
            if access_components.get(access_index) == epoch_components.get(epoch_index)
        )
        for epoch_index, edge in enumerate(edges)
    )


def _bipartite_components(
    edges: tuple[frozenset[int], ...],
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    access_to_epochs: dict[int, set[int]] = {}
    for epoch_index, access_indexes in enumerate(edges):
        for access_index in access_indexes:
            access_to_epochs.setdefault(access_index, set()).add(epoch_index)
    unseen = {index for index, edge in enumerate(edges) if edge}
    components: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    while unseen:
        pending_epochs = [min(unseen)]
        component_epochs: set[int] = set()
        component_accesses: set[int] = set()
        while pending_epochs:
            epoch_index = pending_epochs.pop()
            if epoch_index in component_epochs:
                continue
            component_epochs.add(epoch_index)
            unseen.discard(epoch_index)
            for access_index in edges[epoch_index]:
                if access_index in component_accesses:
                    continue
                component_accesses.add(access_index)
                pending_epochs.extend(access_to_epochs.get(access_index, ()))
        components.append(
            (tuple(sorted(component_epochs)), tuple(sorted(component_accesses)))
        )
    return tuple(components)


def _enumerate_access_assignments(
    epoch_indexes: tuple[int, ...],
    edges: tuple[frozenset[int], ...],
) -> tuple[tuple[dict[int, int], ...], bool]:
    assignments: list[dict[int, int]] = []
    ordered_epochs = tuple(
        sorted(epoch_indexes, key=lambda index: (len(edges[index]), index))
    )

    def visit(position: int, selected: dict[int, int], used: set[int]) -> None:
        if len(assignments) > _MAX_MATCHINGS:
            return
        if position == len(ordered_epochs):
            assignments.append(dict(selected))
            return
        epoch_index = ordered_epochs[position]
        for access_index in sorted(edges[epoch_index] - used):
            selected[epoch_index] = access_index
            used.add(access_index)
            visit(position + 1, selected, used)
            used.remove(access_index)
            selected.pop(epoch_index)

    visit(0, {}, set())
    if len(assignments) > _MAX_MATCHINGS:
        return tuple(assignments[:_MAX_MATCHINGS]), False
    return tuple(assignments), True


def _access_assignment_cost(
    assignment: Mapping[int, int],
    epochs: tuple[_EpochBuilder, ...],
    accesses: tuple[ApplicationEvent, ...],
) -> tuple[int, int, int, int, int]:
    terminal_deltas = tuple(
        abs(
            accesses[access_index].wall_time_utc_ns
            - int(epochs[epoch_index].end_wall_ns)
        )
        for epoch_index, access_index in assignment.items()
    )
    source_gaps = tuple(
        abs(
            accesses[access_index].source_order
            - int(epochs[epoch_index].end_source_order)
        )
        for epoch_index, access_index in assignment.items()
    )
    ordered_epochs = tuple(
        sorted(
            assignment,
            key=lambda index: (
                int(epochs[index].end_source_order),
                index,
            ),
        )
    )
    assigned_orders = tuple(
        accesses[assignment[index]].source_order for index in ordered_epochs
    )
    inversions = sum(
        assigned_orders[left] > assigned_orders[right]
        for left in range(len(assigned_orders))
        for right in range(left + 1, len(assigned_orders))
    )
    return (
        max(terminal_deltas, default=0),
        sum(terminal_deltas),
        max(source_gaps, default=0),
        sum(source_gaps),
        inversions,
    )


def _ranking_stage_for_epoch(
    epoch_index: int,
    assignments: tuple[dict[int, int], ...],
    costs: tuple[tuple[int, int, int, int, int], ...],
) -> str | None:
    for length, stage in enumerate(_RANKING_STAGES, start=1):
        best_prefix = min(cost[:length] for cost in costs)
        support = {
            assignment[epoch_index]
            for assignment, cost in zip(assignments, costs, strict=True)
            if cost[:length] == best_prefix
        }
        if len(support) == 1:
            return stage
    return None


def _endpoint_candidate_id(
    access: ApplicationEvent,
    profile: RuntimeCorrelationProfile,
    correlation_key: bytes,
) -> str:
    endpoint_ref = derive_endpoint_ref(
        correlation_key=correlation_key,
        endpoint_type=str(access.endpoint_type),
        endpoint_value=str(access.endpoint_value),
    )
    material = {
        "endpoint_ref": endpoint_ref,
        "source_application_log_digest": profile.source_application_log_digest,
        "source_order": access.source_order,
    }
    return "epc_" + hashlib.sha256(canonical_document_bytes(material)).hexdigest()


def _attribution_result_for_epoch(
    *,
    epoch: _EpochBuilder,
    accesses: tuple[ApplicationEvent, ...],
    access_candidates: frozenset[int],
    application_complete: bool,
    ranked_support: _RankedAccessSupport,
    state_candidates: tuple[_StateCandidate, ...],
    state_complete: bool,
    profile: RuntimeCorrelationProfile,
    correlation_key: bytes,
    correlation_key_id: str,
) -> AttributionResult:
    state_ids = tuple(candidate.candidate_id for candidate in state_candidates)
    if not epoch.complete or not state_complete or not state_candidates:
        state_binding = BindingAssessment(
            decision=AttributionDecision.INCOMPLETE,
            reason_code="STATE_EVIDENCE_INCOMPLETE",
            candidate_ids=(),
        )
        selected_state = None
    elif len(state_candidates) == 1:
        selected_state = state_candidates[0]
        state_binding = BindingAssessment(
            decision=AttributionDecision.CERTIFIED,
            reason_code="CERTIFIED_STATE_BINDING",
            candidate_ids=state_ids,
            selected_candidate_id=state_ids[0],
            optimal_candidate_count=1,
        )
    else:
        selected_state = None
        state_binding = BindingAssessment(
            decision=AttributionDecision.AMBIGUOUS,
            reason_code="MULTIPLE_STATE_BINDINGS",
            candidate_ids=state_ids,
            optimal_candidate_count=len(state_ids),
        )

    access_ids = {
        index: _endpoint_candidate_id(accesses[index], profile, correlation_key)
        for index in access_candidates
    }
    selected_access_index: int | None = None
    if not epoch.complete or not application_complete or not access_candidates:
        endpoint_binding = BindingAssessment(
            decision=AttributionDecision.INCOMPLETE,
            reason_code="ENDPOINT_EVIDENCE_INCOMPLETE",
            candidate_ids=(),
        )
    elif len(access_candidates) == 1:
        selected_access_index = next(iter(access_candidates))
        selected_id = access_ids[selected_access_index]
        endpoint_binding = BindingAssessment(
            decision=AttributionDecision.CERTIFIED,
            reason_code="CERTIFIED_ENDPOINT_BINDING",
            candidate_ids=tuple(sorted(access_ids.values())),
            selected_candidate_id=selected_id,
            optimal_candidate_count=1,
            optimal_assignment_count=1,
        )
    elif not ranked_support.search_complete:
        endpoint_binding = BindingAssessment(
            decision=AttributionDecision.AMBIGUOUS,
            reason_code="RANKING_SEARCH_LIMIT_REACHED",
            candidate_ids=tuple(sorted(access_ids.values())),
        )
    elif len(ranked_support.access_indexes) == 1:
        selected_access_index = next(iter(ranked_support.access_indexes))
        selected_id = access_ids[selected_access_index]
        endpoint_binding = BindingAssessment(
            decision=AttributionDecision.RANKED,
            reason_code="GLOBAL_RANKED_ENDPOINT_BINDING",
            candidate_ids=tuple(sorted(access_ids.values())),
            selected_candidate_id=selected_id,
            optimal_candidate_count=1,
            optimal_assignment_count=ranked_support.optimal_assignment_count,
            ranking_stage=ranked_support.ranking_stage,
        )
    else:
        endpoint_binding = BindingAssessment(
            decision=AttributionDecision.AMBIGUOUS,
            reason_code="MULTIPLE_GLOBAL_OPTIMA",
            candidate_ids=tuple(sorted(access_ids.values())),
            optimal_candidate_count=len(ranked_support.access_indexes),
            optimal_assignment_count=ranked_support.optimal_assignment_count,
        )

    final_decision = _final_decision(state_binding, endpoint_binding)
    selected_access = (
        accesses[selected_access_index] if selected_access_index is not None else None
    )
    endpoint_ref = (
        derive_endpoint_ref(
            correlation_key=correlation_key,
            endpoint_type=str(selected_access.endpoint_type),
            endpoint_value=str(selected_access.endpoint_value),
        )
        if selected_access is not None
        else None
    )
    return AttributionResult(
        {
            "schema_version": "1.0",
            "analysis_version": GLOBAL_STRUCTURAL_RANKED_V2_CONTRACT[
                "analysis_version"
            ],
            "adapter_id": profile.adapter_id,
            "runtime_instance_id": profile.runtime_instance_id,
            "request_ref": _request_ref(epoch, profile.adapter_id),
            "task_id": epoch.task_id,
            "slot_id": epoch.slot_id,
            "slot_epoch": epoch.slot_epoch,
            "state_binding": state_binding.to_mapping(),
            "endpoint_binding": endpoint_binding.to_mapping(),
            "final_decision": final_decision.value,
            "endpoint_type": (
                selected_access.endpoint_type
                if selected_access is not None
                else None
            ),
            "endpoint_ref": endpoint_ref,
            "correlation_key_id": correlation_key_id,
            "cutoff_apply_call_id": (
                selected_state.cutoff_apply_call_id
                if selected_state is not None
                else None
            ),
            "cutoff_monotonic_ns": (
                selected_state.cutoff_monotonic_ns
                if selected_state is not None
                else None
            ),
            "state_unit_chain_digests": (
                [unit.manifest_digest for unit in selected_state.units]
                if selected_state is not None
                else []
            ),
            "source_application_log_digest": profile.source_application_log_digest,
            "source_observer_terminal_seal_digest": (
                profile.source_observer_terminal_seal_digest
            ),
            "clock_normalization_receipt_digest": (
                profile.clock_normalization_receipt_digest
            ),
        }
    )


def _final_decision(
    state_binding: BindingAssessment,
    endpoint_binding: BindingAssessment,
) -> AttributionDecision:
    decisions = {state_binding.decision, endpoint_binding.decision}
    if AttributionDecision.INCOMPLETE in decisions:
        return AttributionDecision.INCOMPLETE
    if AttributionDecision.AMBIGUOUS in decisions:
        return AttributionDecision.AMBIGUOUS
    if decisions == {AttributionDecision.CERTIFIED}:
        return AttributionDecision.CERTIFIED
    return AttributionDecision.RANKED


def _valid_unit_chain(
    units: tuple[StoredStateUnit, ...],
    seal: GenerationSeal,
    *,
    documents: Mapping[StoredStateUnit, Mapping[str, object]] | None = None,
) -> bool:
    if len(units) != seal.unit_count or not units:
        return False
    if tuple(unit.unit_number for unit in units) != tuple(range(1, len(units) + 1)):
        return False
    previous: str | None = None
    base: str | None = None
    for number, unit in enumerate(units, start=1):
        document = (
            _unit_document(unit) if documents is None else documents[unit]
        )
        if number == 1:
            if document["unit_kind"] != "BASE":
                return False
            base = unit.manifest_digest
        elif (
            document["unit_kind"] != "DELTA"
            or document["previous_unit_digest"] != previous
            or document["base_unit_digest"] != base
        ):
            return False
        previous = unit.manifest_digest
    return hmac.compare_digest(str(seal.last_unit_digest), str(previous))


def _unit_document(unit: StoredStateUnit) -> dict[str, object]:
    path = unit.manifest_path
    if not isinstance(path, Path) or path.is_symlink() or not path.is_file():
        raise CorrelationError("state-unit manifest is unavailable")
    payload = path.read_bytes()
    digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    if not hmac.compare_digest(digest, unit.manifest_digest):
        raise CorrelationError("state-unit manifest digest differs")
    document = parse_state_unit_manifest(payload)
    if (
        document["unit_id"] != unit.unit_id
        or document["runtime_instance_id"] != unit.ref.runtime_instance_id
        or document["seq_id"] != unit.ref.seq_id
        or document["sequence_generation"] != unit.ref.sequence_generation
    ):
        raise CorrelationError("stored state-unit binding differs")
    payload_path = unit.payload_path
    if (
        not isinstance(payload_path, Path)
        or payload_path.is_symlink()
        or not payload_path.is_file()
    ):
        raise CorrelationError("state-unit payload is unavailable")
    state_payload = payload_path.read_bytes()
    payload_digest = "sha256:" + hashlib.sha256(state_payload).hexdigest()
    if (
        not hmac.compare_digest(payload_digest, unit.payload_digest)
        or not hmac.compare_digest(
            payload_digest, str(document["payload_digest"])
        )
        or len(state_payload) != document["payload_size_bytes"]
    ):
        raise CorrelationError("state-unit payload digest differs")
    return document


def _expanded_positions(value: object) -> frozenset[int]:
    if not isinstance(value, list):
        raise CorrelationError("state-unit position ranges are invalid")
    positions: set[int] = set()
    for item in value:
        if (
            not isinstance(item, list)
            or len(item) != 2
            or any(isinstance(part, bool) or not isinstance(part, int) for part in item)
        ):
            raise CorrelationError("state-unit position ranges are invalid")
        start, end = item
        if start < 0 or end < start:
            raise CorrelationError("state-unit position ranges are invalid")
        positions.update(range(start, end + 1))
    return frozenset(positions)


def _materialized_positions(
    units: tuple[StoredStateUnit, ...],
) -> frozenset[int]:
    positions: set[int] = set()
    for number, unit in enumerate(units):
        document = _unit_document(unit)
        current = _expanded_positions(document["position_ranges"])
        mutation_kinds = document["mutation_kinds"]
        if not isinstance(mutation_kinds, list):
            raise CorrelationError("state-unit mutation kinds are invalid")
        if number == 0 or mutation_kinds:
            positions = set(current)
        else:
            positions.update(current)
    return frozenset(positions)


def _request_ref(epoch: _EpochBuilder, adapter_id: str) -> str:
    value = (
        f"{adapter_id}:{epoch.task_id}:{epoch.slot_id}:"
        f"{epoch.slot_epoch}:{epoch.start_wall_ns}"
    )
    return "request_" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:32]


def _candidate_id(
    epoch: _EpochBuilder,
    ref: SequenceGenerationRef,
    cutoff: int,
    digests: tuple[str, ...],
) -> str:
    value = {
        "cutoff": cutoff,
        "digests": list(digests),
        "generation": ref.sequence_generation,
        "seq_id": ref.seq_id,
        "slot_epoch": epoch.slot_epoch,
    }
    return "state_" + hashlib.sha256(canonical_document_bytes(value)).hexdigest()[:40]


def _safe_id(name: str, value: object) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise CorrelationError(f"{name} is invalid")
    return value


def _runtime_instance_id(value: object) -> str:
    if not isinstance(value, str) or _RUNTIME_INSTANCE_ID.fullmatch(value) is None:
        raise CorrelationError("runtime_instance_id is invalid")
    return value


def _digest(name: str, value: object) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise CorrelationError(f"{name} is invalid")
    return value


def _nonnegative(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CorrelationError(f"{name} must be non-negative")
    return value


__all__ = [
    "AttributionDecision",
    "AttributionResult",
    "BindingAssessment",
    "CorrelationError",
    "CorrelationInputs",
    "EndpointCorrelator",
    "RuntimeCorrelationProfile",
]
