from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Callable, Iterable

from .contract import (
    CONTROL_ATTACHED,
    CONTROL_DETACH,
    CONTROL_DETACHED,
    CONTROL_DRAIN_READY,
    CONTROL_STOP,
    ControlMessage,
)


_DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
_SAFE_BASENAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")

FAILURE_REASON_CODES = (
    "control_protocol_error",
    "observer_not_ready",
    "admission_closed",
    "preflight_failed",
    "attach_failed",
    "wire_protocol_error",
    "target_identity_mismatch",
    "drain_timeout",
    "quiescence_failed",
    "detach_failed",
    "accounting_mismatch",
    "native_exit_failed",
    "termination_failed",
    "serialization_failed",
    "publication_failed",
)


class LifecycleState(str, Enum):
    PREPARED = "PREPARED"
    ATTACHED = "ATTACHED"
    OBSERVER_READY = "OBSERVER_READY"
    STOP = "STOP"
    DRAIN_READY = "DRAIN_READY"
    TARGET_STOPPED = "TARGET_STOPPED"
    DETACH = "DETACH"
    DETACHED = "DETACHED"
    TARGET_TERMINATED = "TARGET_TERMINATED"
    TARGET_RETAINED = "TARGET_RETAINED"
    SESSION_PUBLISHED = "SESSION_PUBLISHED"
    OBSERVER_FAILED = "OBSERVER_FAILED"


class LifecycleError(RuntimeError):
    """Raised when the runtime observer lifecycle must fail closed."""

    def __init__(self, message: str, *, reason_code: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


@dataclass(frozen=True)
class DrainSnapshot:
    entry_attempt_count: int
    return_probe_count: int
    ringbuf_reserve_fail_count: int
    bpf_ringbuf_submit_count: int
    native_ring_received_count: int
    native_wire_written_count: int
    inflight_count: int
    anomaly_count: int

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")

    @property
    def healthy(self) -> bool:
        return (
            self.inflight_count == 0
            and self.anomaly_count == 0
            and self.ringbuf_reserve_fail_count == 0
            and self.entry_attempt_count == self.return_probe_count
            and self.return_probe_count == self.bpf_ringbuf_submit_count
            and self.bpf_ringbuf_submit_count == self.native_ring_received_count
            and self.native_ring_received_count == self.native_wire_written_count
        )

    def to_mapping(self) -> dict[str, int]:
        return asdict(self)


class LifecycleStateMachine:
    """Enforce the frozen bidirectional observer lifecycle order."""

    def __init__(self, *, run_nonce: str) -> None:
        self._run_nonce = _require_nonce(run_nonce)
        self._state = LifecycleState.PREPARED
        self._history = [self._state]
        self._next_native_sequence = 1
        self._next_supervisor_sequence = 1
        self._target_stopped_digest: str | None = None

    @property
    def state(self) -> LifecycleState:
        return self._state

    @property
    def history(self) -> tuple[LifecycleState, ...]:
        return tuple(self._history)

    def _transition(self, expected: LifecycleState, target: LifecycleState) -> None:
        if self._state is LifecycleState.OBSERVER_FAILED:
            raise LifecycleError(
                "observer lifecycle is already terminal",
                reason_code="control_protocol_error",
            )
        if self._state is not expected:
            raise LifecycleError(
                f"invalid lifecycle transition from {self._state.value}",
                reason_code="control_protocol_error",
            )
        self._state = target
        self._history.append(target)

    def _check_nonce(self, run_nonce: str) -> None:
        if run_nonce != self._run_nonce:
            raise LifecycleError(
                "run nonce does not match the prepared lifecycle",
                reason_code="control_protocol_error",
            )

    def accept_native_control(self, message: ControlMessage) -> None:
        expected = {
            LifecycleState.PREPARED: (CONTROL_ATTACHED, LifecycleState.ATTACHED),
            LifecycleState.STOP: (CONTROL_DRAIN_READY, LifecycleState.DRAIN_READY),
            LifecycleState.DETACH: (CONTROL_DETACHED, LifecycleState.DETACHED),
        }.get(self._state)
        if (
            expected is None
            or message.type != expected[0]
            or message.sequence != self._next_native_sequence
            or message.status != 0
        ):
            raise LifecycleError(
                "native control type, sequence, or status is invalid",
                reason_code="control_protocol_error",
            )
        prior = self._state
        self._next_native_sequence += 1
        self._transition(prior, expected[1])

    def mark_observer_ready(self, run_nonce: str) -> None:
        self._check_nonce(run_nonce)
        self._transition(LifecycleState.ATTACHED, LifecycleState.OBSERVER_READY)

    def require_request_admission(self, run_nonce: str) -> None:
        self._check_nonce(run_nonce)
        if self._state is LifecycleState.OBSERVER_READY:
            return
        reason = (
            "observer_not_ready"
            if self._state in {LifecycleState.PREPARED, LifecycleState.ATTACHED}
            else "admission_closed"
        )
        raise LifecycleError(
            "request admission requires an active OBSERVER_READY state",
            reason_code=reason,
        )

    def command_stop(self, run_nonce: str) -> ControlMessage:
        self._check_nonce(run_nonce)
        self._transition(LifecycleState.OBSERVER_READY, LifecycleState.STOP)
        message = ControlMessage(CONTROL_STOP, self._next_supervisor_sequence, 0)
        self._next_supervisor_sequence += 1
        return message

    def mark_target_stopped(self, run_nonce: str, marker_digest: str) -> None:
        self._check_nonce(run_nonce)
        self._target_stopped_digest = _require_digest(marker_digest)
        self._transition(LifecycleState.DRAIN_READY, LifecycleState.TARGET_STOPPED)

    def command_detach(
        self,
        run_nonce: str,
        target_stopped_digest: str,
    ) -> ControlMessage:
        self._check_nonce(run_nonce)
        digest = _require_digest(target_stopped_digest)
        if digest != self._target_stopped_digest:
            raise LifecycleError(
                "TARGET_STOPPED digest does not match the verified marker",
                reason_code="control_protocol_error",
            )
        self._transition(LifecycleState.TARGET_STOPPED, LifecycleState.DETACH)
        message = ControlMessage(CONTROL_DETACH, self._next_supervisor_sequence, 0)
        self._next_supervisor_sequence += 1
        return message

    def command_detach_retained(self, run_nonce: str) -> ControlMessage:
        """Detach after drain without stopping an externally owned target."""

        self._check_nonce(run_nonce)
        self._transition(LifecycleState.DRAIN_READY, LifecycleState.DETACH)
        message = ControlMessage(CONTROL_DETACH, self._next_supervisor_sequence, 0)
        self._next_supervisor_sequence += 1
        return message

    def mark_target_terminated(self, run_nonce: str) -> None:
        self._check_nonce(run_nonce)
        self._transition(LifecycleState.DETACHED, LifecycleState.TARGET_TERMINATED)

    def mark_target_retained(self, run_nonce: str) -> None:
        self._check_nonce(run_nonce)
        self._transition(LifecycleState.DETACHED, LifecycleState.TARGET_RETAINED)

    def mark_session_published(self, run_nonce: str) -> None:
        self._check_nonce(run_nonce)
        if self._state is LifecycleState.OBSERVER_FAILED:
            raise LifecycleError(
                "observer lifecycle is already terminal",
                reason_code="control_protocol_error",
            )
        if self._state not in {
            LifecycleState.TARGET_TERMINATED,
            LifecycleState.TARGET_RETAINED,
        }:
            raise LifecycleError(
                f"invalid lifecycle transition from {self._state.value}",
                reason_code="control_protocol_error",
            )
        self._state = LifecycleState.SESSION_PUBLISHED
        self._history.append(self._state)

    def fail(self, reason_code: str) -> None:
        _require_reason_code(reason_code)
        if self._state in {
            LifecycleState.SESSION_PUBLISHED,
            LifecycleState.OBSERVER_FAILED,
        }:
            raise LifecycleError(
                "observer lifecycle is already terminal",
                reason_code="control_protocol_error",
            )
        self._state = LifecycleState.OBSERVER_FAILED
        self._history.append(self._state)


def wait_for_drain(
    read_snapshot: Callable[[], DrainSnapshot],
    *,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    interval_seconds: float = 0.01,
    stable_samples: int = 3,
    timeout_seconds: float = 30.0,
) -> DrainSnapshot:
    """Wait for three stable healthy snapshots while probes remain attached."""
    if interval_seconds <= 0 or stable_samples <= 0 or timeout_seconds <= 0:
        raise ValueError("drain timing parameters must be positive")
    started = monotonic()
    previous: DrainSnapshot | None = None
    stable_count = 0
    while monotonic() - started < timeout_seconds:
        current = read_snapshot()
        if current.healthy:
            stable_count = stable_count + 1 if current == previous else 1
            if stable_count >= stable_samples:
                return current
            previous = current
        else:
            previous = None
            stable_count = 0
        sleep(interval_seconds)
    raise LifecycleError("observer drain timeout", reason_code="drain_timeout")


def canonical_lifecycle_bytes(value: dict[str, object]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def build_ready_marker(
    *,
    run_id: str,
    run_nonce: str,
    runtime_instance_id: str,
    oracle_instance_id: str,
    probe_profile_digest: str,
    expected_target_manifest_digest: str,
    observed_attach_manifest_digest: str,
    attached_at_monotonic_ns: int,
    ready_at_monotonic_ns: int,
    target_elf_digest: str,
) -> dict[str, object]:
    marker = {
        "schema_version": "1.0",
        "marker_type": "OBSERVER_READY",
        "run_id": _require_text(run_id, "run_id"),
        "run_nonce": _require_nonce(run_nonce),
        "runtime_instance_id": _require_digest(runtime_instance_id),
        "oracle_instance_id": _require_digest(oracle_instance_id),
        "probe_profile_digest": _require_digest(probe_profile_digest),
        "expected_target_manifest_digest": _require_digest(
            expected_target_manifest_digest
        ),
        "observed_attach_manifest_digest": _require_digest(
            observed_attach_manifest_digest
        ),
        "attached_at_monotonic_ns": _require_nonnegative_integer(
            attached_at_monotonic_ns,
            "attached_at_monotonic_ns",
        ),
        "ready_at_monotonic_ns": _require_nonnegative_integer(
            ready_at_monotonic_ns,
            "ready_at_monotonic_ns",
        ),
        "target_elf_digest": _require_digest(target_elf_digest),
    }
    if marker["ready_at_monotonic_ns"] < marker["attached_at_monotonic_ns"]:
        raise LifecycleError(
            "ready timestamp precedes attach timestamp",
            reason_code="attach_failed",
        )
    return marker


def build_drain_ready_marker(
    *,
    run_id: str,
    run_nonce: str,
    runtime_instance_id: str,
    oracle_instance_id: str,
    observed_at_monotonic_ns: int,
    counters: DrainSnapshot,
) -> dict[str, object]:
    if not isinstance(counters, DrainSnapshot) or not counters.healthy:
        raise LifecycleError(
            "OBSERVER_DRAIN_READY requires healthy counters",
            reason_code="accounting_mismatch",
        )
    return {
        "schema_version": "1.0",
        "marker_type": "OBSERVER_DRAIN_READY",
        "run_id": _require_text(run_id, "run_id"),
        "run_nonce": _require_nonce(run_nonce),
        "runtime_instance_id": _require_digest(runtime_instance_id),
        "oracle_instance_id": _require_digest(oracle_instance_id),
        "observed_at_monotonic_ns": _require_nonnegative_integer(
            observed_at_monotonic_ns,
            "observed_at_monotonic_ns",
        ),
        "counters": counters.to_mapping(),
    }


def build_target_stopped_marker(
    *,
    run_id: str,
    run_nonce: str,
    runtime_instance_id: str,
    driver_completion_digest: str,
    stopped_at_monotonic_ns: int,
    stopped_thread_count: int,
) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "marker_type": "TARGET_STOPPED",
        "run_id": _require_text(run_id, "run_id"),
        "run_nonce": _require_nonce(run_nonce),
        "runtime_instance_id": _require_digest(runtime_instance_id),
        "driver_completion_digest": _require_digest(driver_completion_digest),
        "stopped_at_monotonic_ns": _require_nonnegative_integer(
            stopped_at_monotonic_ns,
            "stopped_at_monotonic_ns",
        ),
        "stopped_thread_count": _require_positive_integer(
            stopped_thread_count,
            "stopped_thread_count",
        ),
    }


def build_observer_detached_marker(
    *,
    run_id: str,
    run_nonce: str,
    runtime_instance_id: str,
    oracle_instance_id: str,
    observed_quiescent_manifest_digest: str,
    detached_at_monotonic_ns: int,
    counters: DrainSnapshot,
) -> dict[str, object]:
    if not isinstance(counters, DrainSnapshot) or not counters.healthy:
        raise LifecycleError(
            "OBSERVER_DETACHED requires healthy counters",
            reason_code="accounting_mismatch",
        )
    return {
        "schema_version": "1.0",
        "marker_type": "OBSERVER_DETACHED",
        "run_id": _require_text(run_id, "run_id"),
        "run_nonce": _require_nonce(run_nonce),
        "runtime_instance_id": _require_digest(runtime_instance_id),
        "oracle_instance_id": _require_digest(oracle_instance_id),
        "observed_quiescent_manifest_digest": _require_digest(
            observed_quiescent_manifest_digest
        ),
        "detached_at_monotonic_ns": _require_nonnegative_integer(
            detached_at_monotonic_ns,
            "detached_at_monotonic_ns",
        ),
        "counters": counters.to_mapping(),
    }


def build_observer_detached_retained_marker(
    *,
    run_id: str,
    run_nonce: str,
    runtime_instance_id: str,
    oracle_instance_id: str,
    detached_at_monotonic_ns: int,
    counters: DrainSnapshot,
) -> dict[str, object]:
    if not isinstance(counters, DrainSnapshot) or not counters.healthy:
        raise LifecycleError(
            "OBSERVER_DETACHED requires healthy counters",
            reason_code="accounting_mismatch",
        )
    return {
        "schema_version": "1.1",
        "marker_type": "OBSERVER_DETACHED",
        "runtime_mode": "ATTACHED",
        "runtime_ownership": "external",
        "run_id": _require_text(run_id, "run_id"),
        "run_nonce": _require_nonce(run_nonce),
        "runtime_instance_id": _require_digest(runtime_instance_id),
        "oracle_instance_id": _require_digest(oracle_instance_id),
        "detached_at_monotonic_ns": _require_nonnegative_integer(
            detached_at_monotonic_ns,
            "detached_at_monotonic_ns",
        ),
        "counters": counters.to_mapping(),
    }


def build_failure_marker(
    *,
    run_id: str,
    run_nonce: str,
    runtime_instance_id: str | None,
    oracle_instance_id: str | None,
    failed_state: LifecycleState,
    reason_code: str,
    observed_at_monotonic_ns: int,
    counters: DrainSnapshot | None,
    partial_artifacts: Iterable[tuple[str, str]],
) -> dict[str, object]:
    reason = _require_reason_code(reason_code)
    if not isinstance(failed_state, LifecycleState) or failed_state in {
        LifecycleState.SESSION_PUBLISHED,
        LifecycleState.OBSERVER_FAILED,
    }:
        raise LifecycleError(
            "failed_state must be a non-terminal lifecycle state",
            reason_code="serialization_failed",
        )
    artifacts: list[dict[str, str]] = []
    seen: set[str] = set()
    for basename, digest in partial_artifacts:
        if not _SAFE_BASENAME_PATTERN.fullmatch(basename) or basename in seen:
            raise LifecycleError(
                "partial artifact basename is invalid or duplicated",
                reason_code="serialization_failed",
            )
        seen.add(basename)
        artifacts.append({"basename": basename, "digest": _require_digest(digest)})
    artifacts.sort(key=lambda item: item["basename"])
    return {
        "schema_version": "1.0",
        "marker_type": "OBSERVER_FAILED",
        "run_id": _require_text(run_id, "run_id"),
        "run_nonce": _require_nonce(run_nonce),
        "runtime_instance_id": (
            None if runtime_instance_id is None else _require_digest(runtime_instance_id)
        ),
        "oracle_instance_id": (
            None if oracle_instance_id is None else _require_digest(oracle_instance_id)
        ),
        "failed_state": failed_state.value,
        "reason_code": reason,
        "observed_at_monotonic_ns": _require_nonnegative_integer(
            observed_at_monotonic_ns,
            "observed_at_monotonic_ns",
        ),
        "counters": None if counters is None else counters.to_mapping(),
        "partial_artifacts": artifacts,
    }


def build_target_terminated_artifact(
    *,
    run_id: str,
    run_nonce: str,
    oracle_instance_id: str,
    runtime_instance_id: str,
    host_boot_id: str,
    server_pid: int,
    server_start_time_ticks: int,
    wait_status: int,
    observed_quiescent_manifest_digest: str,
) -> dict[str, object]:
    """Build the exact Phase 1 termination record consumed by Gate 0/Gate A."""
    status = _require_nonnegative_integer(wait_status, "wait_status")
    if status != 9:
        raise LifecycleError(
            "TARGET_TERMINATED requires SIGKILL wait status 9",
            reason_code="termination_failed",
        )
    return {
        "schema_version": "1.0",
        "run_id": _require_text(run_id, "run_id"),
        "run_nonce": _require_nonce(run_nonce),
        "oracle_instance_id": _require_digest(oracle_instance_id),
        "runtime_instance_id": _require_digest(runtime_instance_id),
        "host_boot_id": _require_text(host_boot_id, "host_boot_id"),
        "server_pid": _require_positive_integer(server_pid, "server_pid"),
        "server_start_time_ticks": _require_positive_integer(
            server_start_time_ticks,
            "server_start_time_ticks",
        ),
        "pidfd_bound": True,
        "termination_signal": "SIGKILL",
        "wait_status": status,
        "observed_quiescent_manifest_digest": _require_digest(
            observed_quiescent_manifest_digest
        ),
    }


def build_target_retained_artifact(
    *,
    run_id: str,
    run_nonce: str,
    oracle_instance_id: str,
    runtime_instance_id: str,
    host_boot_id: str,
    server_pid: int,
    server_start_time_ticks: int,
    observer_detached_at_monotonic_ns: int,
    identity_verified_at_monotonic_ns: int,
    live_executable_digest: str,
    mapped_libllama_digest: str,
    observer_detached_digest: str,
) -> dict[str, object]:
    """Record that an externally owned target remained identical after detach."""

    detached_at = _require_nonnegative_integer(
        observer_detached_at_monotonic_ns,
        "observer_detached_at_monotonic_ns",
    )
    verified_at = _require_nonnegative_integer(
        identity_verified_at_monotonic_ns,
        "identity_verified_at_monotonic_ns",
    )
    if verified_at < detached_at:
        raise LifecycleError(
            "target identity verification precedes detach",
            reason_code="target_identity_mismatch",
        )
    return {
        "schema_version": "1.0",
        "run_id": _require_text(run_id, "run_id"),
        "run_nonce": _require_nonce(run_nonce),
        "oracle_instance_id": _require_digest(oracle_instance_id),
        "runtime_instance_id": _require_digest(runtime_instance_id),
        "host_boot_id": _require_text(host_boot_id, "host_boot_id"),
        "server_pid": _require_positive_integer(server_pid, "server_pid"),
        "server_start_time_ticks": _require_positive_integer(
            server_start_time_ticks,
            "server_start_time_ticks",
        ),
        "runtime_mode": "ATTACHED",
        "runtime_ownership": "external",
        "observer_detached_at_monotonic_ns": detached_at,
        "identity_verified_at_monotonic_ns": verified_at,
        "live_executable_digest": _require_digest(live_executable_digest),
        "mapped_libllama_digest": _require_digest(mapped_libllama_digest),
        "observer_detached_digest": _require_digest(observer_detached_digest),
    }


def load_lifecycle_marker(payload: bytes) -> dict[str, object]:
    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LifecycleError(
            "lifecycle marker is not strict JSON",
            reason_code="serialization_failed",
        ) from error
    if not isinstance(value, dict):
        raise LifecycleError(
            "lifecycle marker must be an object",
            reason_code="serialization_failed",
        )
    marker_type = value.get("marker_type")
    expected_by_type = {
        "OBSERVER_READY": _READY_KEYS,
        "OBSERVER_DRAIN_READY": _DRAIN_READY_KEYS,
        "TARGET_STOPPED": _TARGET_STOPPED_KEYS,
        "OBSERVER_DETACHED": _DETACHED_KEYS,
        "OBSERVER_FAILED": _FAILURE_KEYS,
    }
    expected = (
        _DETACHED_RETAINED_KEYS
        if marker_type == "OBSERVER_DETACHED"
        and value.get("schema_version") == "1.1"
        else expected_by_type.get(marker_type)
    )
    if expected is None or set(value) != expected:
        raise LifecycleError(
            "lifecycle marker fields are invalid",
            reason_code="serialization_failed",
        )
    if marker_type == "OBSERVER_READY":
        rebuilt = build_ready_marker(
            run_id=value["run_id"],
            run_nonce=value["run_nonce"],
            runtime_instance_id=value["runtime_instance_id"],
            oracle_instance_id=value["oracle_instance_id"],
            probe_profile_digest=value["probe_profile_digest"],
            expected_target_manifest_digest=value["expected_target_manifest_digest"],
            observed_attach_manifest_digest=value["observed_attach_manifest_digest"],
            attached_at_monotonic_ns=value["attached_at_monotonic_ns"],
            ready_at_monotonic_ns=value["ready_at_monotonic_ns"],
            target_elf_digest=value["target_elf_digest"],
        )
    elif marker_type == "OBSERVER_DRAIN_READY":
        rebuilt = build_drain_ready_marker(
            run_id=value["run_id"],
            run_nonce=value["run_nonce"],
            runtime_instance_id=value["runtime_instance_id"],
            oracle_instance_id=value["oracle_instance_id"],
            observed_at_monotonic_ns=value["observed_at_monotonic_ns"],
            counters=_load_counters(value["counters"]),
        )
    elif marker_type == "TARGET_STOPPED":
        rebuilt = build_target_stopped_marker(
            run_id=value["run_id"],
            run_nonce=value["run_nonce"],
            runtime_instance_id=value["runtime_instance_id"],
            driver_completion_digest=value["driver_completion_digest"],
            stopped_at_monotonic_ns=value["stopped_at_monotonic_ns"],
            stopped_thread_count=value["stopped_thread_count"],
        )
    elif marker_type == "OBSERVER_DETACHED":
        if value["schema_version"] == "1.1":
            rebuilt = build_observer_detached_retained_marker(
                run_id=value["run_id"],
                run_nonce=value["run_nonce"],
                runtime_instance_id=value["runtime_instance_id"],
                oracle_instance_id=value["oracle_instance_id"],
                detached_at_monotonic_ns=value["detached_at_monotonic_ns"],
                counters=_load_counters(value["counters"]),
            )
        else:
            rebuilt = build_observer_detached_marker(
                run_id=value["run_id"],
                run_nonce=value["run_nonce"],
                runtime_instance_id=value["runtime_instance_id"],
                oracle_instance_id=value["oracle_instance_id"],
                observed_quiescent_manifest_digest=value[
                    "observed_quiescent_manifest_digest"
                ],
                detached_at_monotonic_ns=value["detached_at_monotonic_ns"],
                counters=_load_counters(value["counters"]),
            )
    else:
        counters_value = value["counters"]
        counters = None
        if counters_value is not None:
            counters = _load_counters(counters_value)
        artifacts_value = value["partial_artifacts"]
        if not isinstance(artifacts_value, list):
            raise LifecycleError(
                "partial_artifacts must be an array",
                reason_code="serialization_failed",
            )
        partial: list[tuple[str, str]] = []
        for item in artifacts_value:
            if not isinstance(item, dict) or set(item) != {"basename", "digest"}:
                raise LifecycleError(
                    "partial artifact fields are invalid",
                    reason_code="serialization_failed",
                )
            partial.append((item["basename"], item["digest"]))
        try:
            failed_state = LifecycleState(value["failed_state"])
        except (TypeError, ValueError) as error:
            raise LifecycleError(
                "failed_state is invalid",
                reason_code="serialization_failed",
            ) from error
        rebuilt = build_failure_marker(
            run_id=value["run_id"],
            run_nonce=value["run_nonce"],
            runtime_instance_id=value["runtime_instance_id"],
            oracle_instance_id=value["oracle_instance_id"],
            failed_state=failed_state,
            reason_code=value["reason_code"],
            observed_at_monotonic_ns=value["observed_at_monotonic_ns"],
            counters=counters,
            partial_artifacts=partial,
        )
    if rebuilt != value:
        raise LifecycleError(
            "lifecycle marker is not canonical in-memory data",
            reason_code="serialization_failed",
        )
    return rebuilt


_READY_KEYS = {
    "schema_version",
    "marker_type",
    "run_id",
    "run_nonce",
    "runtime_instance_id",
    "oracle_instance_id",
    "probe_profile_digest",
    "expected_target_manifest_digest",
    "observed_attach_manifest_digest",
    "attached_at_monotonic_ns",
    "ready_at_monotonic_ns",
    "target_elf_digest",
}
_DRAIN_READY_KEYS = {
    "schema_version",
    "marker_type",
    "run_id",
    "run_nonce",
    "runtime_instance_id",
    "oracle_instance_id",
    "observed_at_monotonic_ns",
    "counters",
}
_TARGET_STOPPED_KEYS = {
    "schema_version",
    "marker_type",
    "run_id",
    "run_nonce",
    "runtime_instance_id",
    "driver_completion_digest",
    "stopped_at_monotonic_ns",
    "stopped_thread_count",
}
_DETACHED_KEYS = {
    "schema_version",
    "marker_type",
    "run_id",
    "run_nonce",
    "runtime_instance_id",
    "oracle_instance_id",
    "observed_quiescent_manifest_digest",
    "detached_at_monotonic_ns",
    "counters",
}
_DETACHED_RETAINED_KEYS = {
    "schema_version",
    "marker_type",
    "runtime_mode",
    "runtime_ownership",
    "run_id",
    "run_nonce",
    "runtime_instance_id",
    "oracle_instance_id",
    "detached_at_monotonic_ns",
    "counters",
}
_FAILURE_KEYS = {
    "schema_version",
    "marker_type",
    "run_id",
    "run_nonce",
    "runtime_instance_id",
    "oracle_instance_id",
    "failed_state",
    "reason_code",
    "observed_at_monotonic_ns",
    "counters",
    "partial_artifacts",
}


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise LifecycleError(
                f"duplicate JSON key: {key}",
                reason_code="serialization_failed",
            )
        value[key] = item
    return value


def _load_counters(value: object) -> DrainSnapshot:
    if not isinstance(value, dict) or set(value) != set(
        DrainSnapshot.__dataclass_fields__
    ):
        raise LifecycleError(
            "counter fields are invalid",
            reason_code="serialization_failed",
        )
    try:
        return DrainSnapshot(**value)
    except (TypeError, ValueError) as error:
        raise LifecycleError(
            "counter values are invalid",
            reason_code="serialization_failed",
        ) from error


def _require_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise LifecycleError(
            f"{name} must be non-empty text",
            reason_code="serialization_failed",
        )
    return value


def _require_nonce(value: object) -> str:
    text = _require_text(value, "run_nonce")
    if len(text) < 16:
        raise LifecycleError(
            "run_nonce must contain at least 16 characters",
            reason_code="serialization_failed",
        )
    return text


def _require_digest(value: object) -> str:
    if not isinstance(value, str) or _DIGEST_PATTERN.fullmatch(value) is None:
        raise LifecycleError(
            "digest must use canonical lowercase sha256 form",
            reason_code="serialization_failed",
        )
    return value


def _require_nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise LifecycleError(
            f"{name} must be a non-negative integer",
            reason_code="serialization_failed",
        )
    return value


def _require_positive_integer(value: object, name: str) -> int:
    result = _require_nonnegative_integer(value, name)
    if result == 0:
        raise LifecycleError(
            f"{name} must be a positive integer",
            reason_code="serialization_failed",
        )
    return result


def _require_reason_code(value: object) -> str:
    if not isinstance(value, str) or value not in FAILURE_REASON_CODES:
        raise LifecycleError(
            "failure reason code is not in the fixed vocabulary",
            reason_code="serialization_failed",
        )
    return value
