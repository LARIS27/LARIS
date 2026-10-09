from __future__ import annotations

from typing import Final

from laris_onpremise.core.records import LeaseRecord, LeaseState


class LeaseTransitionError(RuntimeError):
    """Raised when a Sequence Lease transition violates the state machine."""


ALLOWED_TRANSITIONS: Final[dict[LeaseState, frozenset[LeaseState]]] = {
    LeaseState.RESERVED: frozenset(
        {LeaseState.SUBMITTED, LeaseState.UNKNOWN, LeaseState.QUARANTINED}
    ),
    LeaseState.SUBMITTED: frozenset(
        {LeaseState.ACTIVE, LeaseState.UNKNOWN, LeaseState.QUARANTINED}
    ),
    LeaseState.ACTIVE: frozenset(
        {LeaseState.QUIESCING, LeaseState.UNKNOWN, LeaseState.QUARANTINED}
    ),
    LeaseState.QUIESCING: frozenset(
        {
            LeaseState.CHECKPOINTING,
            LeaseState.CAPTURE_SKIPPED,
            LeaseState.UNKNOWN,
            LeaseState.QUARANTINED,
        }
    ),
    LeaseState.CHECKPOINTING: frozenset(
        {LeaseState.CHECKPOINTED, LeaseState.UNKNOWN, LeaseState.QUARANTINED}
    ),
    LeaseState.CHECKPOINTED: frozenset(
        {LeaseState.RELEASED, LeaseState.UNKNOWN, LeaseState.QUARANTINED}
    ),
    LeaseState.CAPTURE_SKIPPED: frozenset(
        {LeaseState.RELEASED, LeaseState.UNKNOWN, LeaseState.QUARANTINED}
    ),
    LeaseState.RELEASED: frozenset(),
    LeaseState.UNKNOWN: frozenset(),
    LeaseState.QUARANTINED: frozenset(),
}


def require_state(lease: LeaseRecord, *expected: LeaseState) -> None:
    if lease.state not in expected:
        names = ", ".join(state.value for state in expected)
        raise LeaseTransitionError(
            f"Lease {lease.lease_id} must be in {names}; current state is {lease.state.value}"
        )


def require_transition(lease: LeaseRecord, target: LeaseState) -> None:
    if target not in ALLOWED_TRANSITIONS[lease.state]:
        raise LeaseTransitionError(
            f"illegal Lease transition: {lease.state.value} -> {target.value}"
        )
