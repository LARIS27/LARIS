from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


SIMPLE_KV_MEMORY_FAMILY = "simple_kv"
SIMPLE_KV_PAYLOAD_FORMAT = "llama-simple-kv-seq-payload-v1"

_SHA256_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")

RuntimeStateCopySealedCallback = Callable[[], None]


class RuntimeStateContractError(ValueError):
    """Raised when runtime-state acquisition data violates its contract."""


class RuntimeStateBindingError(RuntimeStateContractError):
    """Raised when a payload commit does not belong to its capture request."""


def _require_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeStateContractError(f"{name} must be a non-empty string")
    return value


def _require_positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeStateContractError(f"{name} must be a positive integer")
    return value


def _require_nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeStateContractError(f"{name} must be a non-negative integer")
    return value


def _validate_inclusive_pair(
    name: str, value: object
) -> tuple[int, int]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise RuntimeStateContractError(f"{name} must be an inclusive integer pair")
    start, end = value
    _require_nonnegative_int(f"{name}[0]", start)
    _require_nonnegative_int(f"{name}[1]", end)
    if start > end:
        raise RuntimeStateContractError(f"{name} start must not exceed end")
    return start, end


@dataclass(frozen=True)
class RuntimeStateCaptureRequest:
    capture_id: str
    lease_id: str
    request_ref: str
    runtime_instance_id: str
    runtime_pid: int
    runtime_start_time_ticks: int
    slot_id: int
    slot_epoch: int
    observed_seq_id: int
    interval_start_monotonic_ns: int
    interval_end_monotonic_ns: int

    def __post_init__(self) -> None:
        for name in (
            "capture_id",
            "lease_id",
            "request_ref",
            "runtime_instance_id",
        ):
            _require_text(name, getattr(self, name))
        _require_positive_int("runtime_pid", self.runtime_pid)
        _require_positive_int(
            "runtime_start_time_ticks", self.runtime_start_time_ticks
        )
        _require_nonnegative_int("slot_id", self.slot_id)
        _require_positive_int("slot_epoch", self.slot_epoch)
        _require_nonnegative_int("observed_seq_id", self.observed_seq_id)
        _require_nonnegative_int(
            "interval_start_monotonic_ns", self.interval_start_monotonic_ns
        )
        _require_nonnegative_int(
            "interval_end_monotonic_ns", self.interval_end_monotonic_ns
        )
        if self.interval_end_monotonic_ns < self.interval_start_monotonic_ns:
            raise RuntimeStateContractError(
                "capture interval end must not precede its start"
            )


@dataclass(frozen=True)
class RuntimeStatePayloadCommit:
    capture_id: str
    lease_id: str
    request_ref: str
    runtime_instance_id: str
    slot_id: int
    slot_epoch: int
    observed_seq_id: int
    memory_family: str
    payload_format: str
    payload_path: Path
    payload_size_bytes: int
    payload_digest: str
    position_range: tuple[int, int]
    kv_cell_ranges: tuple[tuple[int, int], ...]
    next_position: int
    acquisition_evidence_ref: str

    def __post_init__(self) -> None:
        for name in (
            "capture_id",
            "lease_id",
            "request_ref",
            "runtime_instance_id",
            "acquisition_evidence_ref",
        ):
            _require_text(name, getattr(self, name))
        _require_nonnegative_int("slot_id", self.slot_id)
        _require_positive_int("slot_epoch", self.slot_epoch)
        _require_nonnegative_int("observed_seq_id", self.observed_seq_id)
        if self.memory_family != SIMPLE_KV_MEMORY_FAMILY:
            raise RuntimeStateContractError(
                f"memory_family must be {SIMPLE_KV_MEMORY_FAMILY}"
            )
        if self.payload_format != SIMPLE_KV_PAYLOAD_FORMAT:
            raise RuntimeStateContractError(
                f"payload_format must be {SIMPLE_KV_PAYLOAD_FORMAT}"
            )
        if not isinstance(self.payload_path, Path):
            raise RuntimeStateContractError("payload_path must be a pathlib.Path")
        _require_positive_int("payload_size_bytes", self.payload_size_bytes)
        if not isinstance(self.payload_digest, str) or _SHA256_DIGEST.fullmatch(
            self.payload_digest
        ) is None:
            raise RuntimeStateContractError(
                "payload_digest must be lowercase sha256:<64 hex>"
            )

        position_start, position_end = _validate_inclusive_pair(
            "position_range", self.position_range
        )
        if not isinstance(self.kv_cell_ranges, tuple) or not self.kv_cell_ranges:
            raise RuntimeStateContractError(
                "kv_cell_ranges must contain at least one inclusive range"
            )
        previous_end = -1
        for index, item in enumerate(self.kv_cell_ranges):
            start, end = _validate_inclusive_pair(f"kv_cell_ranges[{index}]", item)
            if start <= previous_end:
                raise RuntimeStateContractError(
                    "kv_cell_ranges must be sorted and non-overlapping"
                )
            previous_end = end
        _require_positive_int("next_position", self.next_position)
        if self.next_position <= position_end:
            raise RuntimeStateContractError(
                "next_position must be greater than the position_range end"
            )
        if position_start >= self.next_position:
            raise RuntimeStateContractError(
                "position_range must begin before next_position"
            )


class RuntimeStateAcquirer(Protocol):
    def capture(
        self,
        request: RuntimeStateCaptureRequest,
        *,
        on_copy_sealed: RuntimeStateCopySealedCallback | None = None,
    ) -> RuntimeStatePayloadCommit: ...


def validate_payload_binding(
    commit: RuntimeStatePayloadCommit,
    request: RuntimeStateCaptureRequest,
) -> None:
    """Fail closed unless a commit exactly matches its authorized capture."""

    for field in (
        "capture_id",
        "lease_id",
        "request_ref",
        "runtime_instance_id",
        "slot_id",
        "slot_epoch",
        "observed_seq_id",
    ):
        if getattr(commit, field) != getattr(request, field):
            raise RuntimeStateBindingError(
                f"payload {field} does not match the capture request"
            )
