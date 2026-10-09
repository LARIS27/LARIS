from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final


_EVENT_TYPES: Final[frozenset[str]] = frozenset({"decode_observed"})
_EVENT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "run_id",
        "run_nonce",
        "oracle_instance_id",
        "event_counter",
        "event_type",
        "runtime_instance_id",
        "host_boot_id",
        "clock_id",
        "decode_entry_monotonic_ns",
        "decode_return_monotonic_ns",
        "wall_time_ns",
        "decode_call_id",
        "context_ref",
        "observed_seq_ids",
        "batch_n_tokens",
        "decode_return_code",
    }
)
_SESSION_V20_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "run_id",
        "run_nonce",
        "oracle_instance_id",
        "runtime_instance_id",
        "host_boot_id",
        "server_pid",
        "server_start_time_ticks",
        "live_executable_digest",
        "clock_id",
        "attached_at_monotonic_ns",
        "detached_at_monotonic_ns",
        "probe_profile_digest",
        "abi_layout_manifest_digest",
        "expected_target_manifest_digest",
        "observed_attach_manifest_digest",
        "observed_quiescent_manifest_digest",
        "supervisor_pid",
        "supervisor_start_time_ticks",
        "supervisor_source_bundle_digest",
        "supervisor_interpreter_digest",
        "native_collector_pid",
        "native_collector_start_time_ticks",
        "native_collector_digest",
        "native_dependency_manifest_digest",
        "bpf_object_digest",
        "loaded_bpf_program_tags_digest",
        "target_terminated_digest",
        "entry_attempt_count",
        "return_probe_count",
        "ringbuf_reserve_fail_count",
        "bpf_ringbuf_submit_count",
        "native_ring_received_count",
        "native_wire_written_count",
        "supervisor_wire_received_count",
        "json_serialized_count",
        "writer_failure_count",
        "final_inflight_count",
        "anomaly_count",
        "lost_event_count",
        "counter_gap_count",
        "event_count",
        "event_stream_digest",
        "session_complete",
    }
)
_SESSION_V21_KEYS: Final[frozenset[str]] = frozenset(
    {
        *(_SESSION_V20_KEYS - {"target_terminated_digest"}),
        "runtime_mode",
        "runtime_ownership",
        "terminal_target_state",
        "terminal_target_digest",
    }
)
_DIGEST_FIELDS: Final[tuple[str, ...]] = (
    "live_executable_digest",
    "probe_profile_digest",
    "abi_layout_manifest_digest",
    "expected_target_manifest_digest",
    "observed_attach_manifest_digest",
    "supervisor_source_bundle_digest",
    "supervisor_interpreter_digest",
    "native_collector_digest",
    "native_dependency_manifest_digest",
    "bpf_object_digest",
    "loaded_bpf_program_tags_digest",
    "event_stream_digest",
)
_SHA256_PATTERN: Final[re.Pattern[str]] = re.compile(r"^sha256:[0-9a-f]{64}$")
_POINTER_PATTERN: Final[re.Pattern[str]] = re.compile(r"^0x[0-9a-fA-F]+$")
_RUN_NONCE_MIN_LENGTH: Final[int] = 16


class OracleValidationError(ValueError):
    """Raised when independent runtime-oracle evidence is malformed."""


def _require_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise OracleValidationError(f"{field} must be a non-empty string")
    return value


def _require_run_nonce(value: object) -> str:
    run_nonce = _require_string(value, "run_nonce")
    if len(run_nonce) < _RUN_NONCE_MIN_LENGTH:
        raise OracleValidationError(
            "run_nonce must contain at least 16 characters"
        )
    return run_nonce


def _require_integer(value: object, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise OracleValidationError(
            f"{field} must be an integer greater than or equal to {minimum}"
        )
    return value


def _require_signed_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise OracleValidationError(f"{field} must be a signed integer")
    return value


def _require_digest(value: object, field: str) -> str:
    digest = _require_string(value, field)
    if _SHA256_PATTERN.fullmatch(digest) is None:
        raise OracleValidationError(
            f"{field} must be a lowercase sha256 digest"
        )
    return digest


def _require_exact_fields(
    value: dict[str, object],
    expected: frozenset[str],
    kind: str,
) -> None:
    extra = set(value) - expected
    if extra:
        raise OracleValidationError(f"unexpected {kind} fields: {sorted(extra)}")
    missing = expected - set(value)
    if missing:
        raise OracleValidationError(f"missing {kind} fields: {sorted(missing)}")


def _decode_json_line(raw: str, line_number: int) -> dict[str, object]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise OracleValidationError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(raw, object_pairs_hook=reject_duplicates)
    except OracleValidationError:
        raise
    except json.JSONDecodeError as error:
        raise OracleValidationError(
            f"invalid JSON at line {line_number}"
        ) from error
    if not isinstance(value, dict):
        raise OracleValidationError(
            f"oracle event at line {line_number} must be an object"
        )
    return value


@dataclass(frozen=True)
class RuntimeOracleSession:
    schema_version: str
    run_id: str
    run_nonce: str
    oracle_instance_id: str
    runtime_instance_id: str
    host_boot_id: str
    server_pid: int
    server_start_time_ticks: int
    live_executable_digest: str
    clock_id: str
    attached_at_monotonic_ns: int
    detached_at_monotonic_ns: int
    probe_profile_digest: str
    abi_layout_manifest_digest: str
    expected_target_manifest_digest: str
    observed_attach_manifest_digest: str
    observed_quiescent_manifest_digest: str | None
    supervisor_pid: int
    supervisor_start_time_ticks: int
    supervisor_source_bundle_digest: str
    supervisor_interpreter_digest: str
    native_collector_pid: int
    native_collector_start_time_ticks: int
    native_collector_digest: str
    native_dependency_manifest_digest: str
    bpf_object_digest: str
    loaded_bpf_program_tags_digest: str
    target_terminated_digest: str | None
    entry_attempt_count: int
    return_probe_count: int
    ringbuf_reserve_fail_count: int
    bpf_ringbuf_submit_count: int
    native_ring_received_count: int
    native_wire_written_count: int
    supervisor_wire_received_count: int
    json_serialized_count: int
    writer_failure_count: int
    final_inflight_count: int
    anomaly_count: int
    lost_event_count: int
    counter_gap_count: int
    event_count: int
    event_stream_digest: str
    session_complete: bool
    runtime_mode: str = "MANAGED_LEGACY"
    runtime_ownership: str = "laris"
    terminal_target_state: str = "terminated"
    terminal_target_digest: str | None = None

    def to_mapping(self) -> dict[str, object]:
        if self.schema_version == "2.0":
            return {field: getattr(self, field) for field in _SESSION_V20_KEYS}
        if self.schema_version == "2.1":
            return {field: getattr(self, field) for field in _SESSION_V21_KEYS}
        raise OracleValidationError("unsupported oracle session schema version")

    @classmethod
    def from_mapping(cls, value: dict[str, object]) -> RuntimeOracleSession:
        schema_version = value.get("schema_version")
        if schema_version == "2.0":
            _require_exact_fields(value, _SESSION_V20_KEYS, "oracle session")
            runtime_mode = "MANAGED_LEGACY"
            runtime_ownership = "laris"
            terminal_target_state = "terminated"
            terminal_target_digest = _require_digest(
                value["target_terminated_digest"],
                "target_terminated_digest",
            )
            observed_quiescent_digest: str | None = _require_digest(
                value["observed_quiescent_manifest_digest"],
                "observed_quiescent_manifest_digest",
            )
            target_terminated_digest: str | None = terminal_target_digest
        elif schema_version == "2.1":
            _require_exact_fields(value, _SESSION_V21_KEYS, "oracle session")
            runtime_mode = _require_string(value["runtime_mode"], "runtime_mode")
            runtime_ownership = _require_string(
                value["runtime_ownership"], "runtime_ownership"
            )
            terminal_target_state = _require_string(
                value["terminal_target_state"], "terminal_target_state"
            )
            terminal_target_digest = _require_digest(
                value["terminal_target_digest"], "terminal_target_digest"
            )
            raw_quiescent = value["observed_quiescent_manifest_digest"]
            observed_quiescent_digest = (
                None
                if raw_quiescent is None
                else _require_digest(
                    raw_quiescent, "observed_quiescent_manifest_digest"
                )
            )
            target_terminated_digest = (
                terminal_target_digest
                if terminal_target_state == "terminated"
                else None
            )
            expected = {
                "MANAGED_LEGACY": ("laris", "terminated"),
                "ATTACHED": ("external", "retained"),
            }.get(runtime_mode)
            if expected != (runtime_ownership, terminal_target_state):
                raise OracleValidationError(
                    "runtime ownership and terminal target state do not match"
                )
            if (
                runtime_mode == "MANAGED_LEGACY"
                and observed_quiescent_digest is None
            ) or (
                runtime_mode == "ATTACHED"
                and observed_quiescent_digest is not None
            ):
                raise OracleValidationError(
                    "runtime mode and observed quiescent evidence do not match"
                )
        else:
            raise OracleValidationError("unsupported oracle session schema version")
        complete = value["session_complete"]
        if not isinstance(complete, bool):
            raise OracleValidationError("session_complete must be a boolean")

        strings = {
            field: _require_string(value[field], field)
            for field in (
                "schema_version",
                "run_id",
                "oracle_instance_id",
                "runtime_instance_id",
                "host_boot_id",
                "clock_id",
            )
        }
        strings["run_nonce"] = _require_run_nonce(value["run_nonce"])
        digests: dict[str, str | None] = {
            field: _require_digest(value[field], field) for field in _DIGEST_FIELDS
        }
        digests["observed_quiescent_manifest_digest"] = observed_quiescent_digest
        digests["target_terminated_digest"] = target_terminated_digest
        positive_integers = {
            field: _require_integer(value[field], field, 1)
            for field in (
                "server_pid",
                "server_start_time_ticks",
                "supervisor_pid",
                "supervisor_start_time_ticks",
                "native_collector_pid",
                "native_collector_start_time_ticks",
            )
        }
        nonnegative_integers = {
            field: _require_integer(value[field], field)
            for field in (
                "attached_at_monotonic_ns",
                "detached_at_monotonic_ns",
                "entry_attempt_count",
                "return_probe_count",
                "ringbuf_reserve_fail_count",
                "bpf_ringbuf_submit_count",
                "native_ring_received_count",
                "native_wire_written_count",
                "supervisor_wire_received_count",
                "json_serialized_count",
                "writer_failure_count",
                "final_inflight_count",
                "anomaly_count",
                "lost_event_count",
                "counter_gap_count",
                "event_count",
            )
        }
        session = cls(
            **strings,
            **digests,
            **positive_integers,
            **nonnegative_integers,
            session_complete=complete,
            runtime_mode=runtime_mode,
            runtime_ownership=runtime_ownership,
            terminal_target_state=terminal_target_state,
            terminal_target_digest=terminal_target_digest,
        )
        if session.clock_id != "CLOCK_MONOTONIC":
            raise OracleValidationError("unsupported oracle session clock_id")
        if session.detached_at_monotonic_ns < session.attached_at_monotonic_ns:
            raise OracleValidationError("oracle session time range is reversed")
        return session


@dataclass(frozen=True)
class RuntimeOracleEvent:
    schema_version: str
    run_id: str
    run_nonce: str
    oracle_instance_id: str
    event_counter: int
    event_type: str
    runtime_instance_id: str
    host_boot_id: str
    clock_id: str
    decode_entry_monotonic_ns: int
    decode_return_monotonic_ns: int
    wall_time_ns: int
    decode_call_id: str
    context_ref: str
    observed_seq_ids: tuple[int, ...]
    batch_n_tokens: int
    decode_return_code: int

    def __post_init__(self) -> None:
        if self.schema_version != "2.0":
            raise OracleValidationError("unsupported oracle event schema version")
        for field, value in (
            ("run_id", self.run_id),
            ("oracle_instance_id", self.oracle_instance_id),
            ("runtime_instance_id", self.runtime_instance_id),
            ("host_boot_id", self.host_boot_id),
            ("clock_id", self.clock_id),
            ("decode_call_id", self.decode_call_id),
            ("context_ref", self.context_ref),
        ):
            _require_string(value, field)
        _require_run_nonce(self.run_nonce)
        if self.clock_id != "CLOCK_MONOTONIC":
            raise OracleValidationError("unsupported oracle event clock_id")
        _require_integer(self.event_counter, "event_counter", 1)
        expected_decode_call_id = (
            f"{self.oracle_instance_id}:{self.event_counter}"
        )
        if self.decode_call_id != expected_decode_call_id:
            raise OracleValidationError(
                "decode_call_id must equal oracle_instance_id:event_counter"
            )
        _require_integer(
            self.decode_entry_monotonic_ns,
            "decode_entry_monotonic_ns",
        )
        _require_integer(
            self.decode_return_monotonic_ns,
            "decode_return_monotonic_ns",
        )
        _require_integer(self.wall_time_ns, "wall_time_ns")
        _require_integer(self.batch_n_tokens, "batch_n_tokens", 1)
        _require_signed_integer(self.decode_return_code, "decode_return_code")
        if self.decode_return_monotonic_ns < self.decode_entry_monotonic_ns:
            raise OracleValidationError("oracle decode time range is reversed")
        if self.event_type not in _EVENT_TYPES:
            raise OracleValidationError("unsupported oracle event type")
        if _POINTER_PATTERN.fullmatch(self.context_ref) is not None:
            raise OracleValidationError(
                "context_ref must be opaque and cannot expose a raw pointer"
            )
        if tuple(sorted(set(self.observed_seq_ids))) != self.observed_seq_ids:
            raise OracleValidationError(
                "observed_seq_ids must be sorted and unique"
            )
        if not self.observed_seq_ids:
            raise OracleValidationError("decode_observed requires observed_seq_ids")
        if any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in self.observed_seq_ids
        ):
            raise OracleValidationError(
                "observed_seq_ids must contain non-negative integers"
            )

    @property
    def monotonic_ns(self) -> int:
        """Return the decode-entry timestamp for legacy evaluator ordering."""
        return self.decode_entry_monotonic_ns

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "run_nonce": self.run_nonce,
            "oracle_instance_id": self.oracle_instance_id,
            "event_counter": self.event_counter,
            "event_type": self.event_type,
            "runtime_instance_id": self.runtime_instance_id,
            "host_boot_id": self.host_boot_id,
            "clock_id": self.clock_id,
            "decode_entry_monotonic_ns": self.decode_entry_monotonic_ns,
            "decode_return_monotonic_ns": self.decode_return_monotonic_ns,
            "wall_time_ns": self.wall_time_ns,
            "decode_call_id": self.decode_call_id,
            "context_ref": self.context_ref,
            "observed_seq_ids": list(self.observed_seq_ids),
            "batch_n_tokens": self.batch_n_tokens,
            "decode_return_code": self.decode_return_code,
        }

    @classmethod
    def from_mapping(cls, value: dict[str, object]) -> RuntimeOracleEvent:
        _require_exact_fields(value, _EVENT_KEYS, "oracle event")
        observed = value["observed_seq_ids"]
        if not isinstance(observed, list):
            raise OracleValidationError("observed_seq_ids must be an array")
        return cls(
            schema_version=_require_string(value["schema_version"], "schema_version"),
            run_id=_require_string(value["run_id"], "run_id"),
            run_nonce=_require_run_nonce(value["run_nonce"]),
            oracle_instance_id=_require_string(
                value["oracle_instance_id"], "oracle_instance_id"
            ),
            event_counter=_require_integer(value["event_counter"], "event_counter", 1),
            event_type=_require_string(value["event_type"], "event_type"),
            runtime_instance_id=_require_string(
                value["runtime_instance_id"], "runtime_instance_id"
            ),
            host_boot_id=_require_string(value["host_boot_id"], "host_boot_id"),
            clock_id=_require_string(value["clock_id"], "clock_id"),
            decode_entry_monotonic_ns=_require_integer(
                value["decode_entry_monotonic_ns"],
                "decode_entry_monotonic_ns",
            ),
            decode_return_monotonic_ns=_require_integer(
                value["decode_return_monotonic_ns"],
                "decode_return_monotonic_ns",
            ),
            wall_time_ns=_require_integer(value["wall_time_ns"], "wall_time_ns"),
            decode_call_id=_require_string(value["decode_call_id"], "decode_call_id"),
            context_ref=_require_string(value["context_ref"], "context_ref"),
            observed_seq_ids=tuple(observed),
            batch_n_tokens=_require_integer(
                value["batch_n_tokens"], "batch_n_tokens", 1
            ),
            decode_return_code=_require_signed_integer(
                value["decode_return_code"], "decode_return_code"
            ),
        )


def runtime_oracle_events_digest(
    events: tuple[RuntimeOracleEvent, ...],
) -> str:
    """Digest the canonical JSONL representation of runtime oracle events."""
    digest = hashlib.sha256()
    for event in events:
        payload = json.dumps(
            event.to_mapping(),
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        digest.update(payload)
        digest.update(b"\n")
    return "sha256:" + digest.hexdigest()


def load_runtime_oracle(path: Path) -> tuple[RuntimeOracleEvent, ...]:
    events: list[RuntimeOracleEvent] = []
    previous_counter = 0
    with path.open("r", encoding="utf-8", newline="") as stream:
        for line_number, raw in enumerate(stream, start=1):
            if not raw.strip():
                continue
            event = RuntimeOracleEvent.from_mapping(
                _decode_json_line(raw, line_number)
            )
            if event.event_counter != previous_counter + 1:
                raise OracleValidationError(
                    "event_counter must be continuous from 1"
                )
            previous_counter = event.event_counter
            events.append(event)
    return tuple(events)


def load_runtime_oracle_session(path: Path) -> RuntimeOracleSession:
    raw = path.read_text(encoding="utf-8")
    value = _decode_json_line(raw, 1)
    return RuntimeOracleSession.from_mapping(value)
