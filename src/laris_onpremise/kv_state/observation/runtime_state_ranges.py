"""Compact native token-cell range evidence for Proxy snapshot capture."""

from __future__ import annotations

from dataclasses import dataclass

from .runtime_state_contract import TokenCellEvent

MAX_APPLY_TOKEN_COUNT = 2048
MAX_STREAM_COUNT = 8
UINT32_MAX = (1 << 32) - 1
UINT64_MAX = (1 << 64) - 1


class TokenCellRangeError(ValueError):
    """Raised when compact token-cell evidence is not exact and bounded."""


def _require_uint(value: object, label: str, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value > maximum
    ):
        raise TokenCellRangeError(f"{label} is out of range")
    return value


def _checked_last(first: int, count: int, label: str) -> int:
    first = _require_uint(first, label, UINT64_MAX)
    last = first + count - 1
    if last > UINT64_MAX:
        raise TokenCellRangeError(f"{label} range overflows uint64")
    return last


@dataclass(frozen=True)
class NativeTokenCellRange:
    apply_call_id: int
    tgid: int
    tid: int
    monotonic_ns: int
    seq_id: int
    stream_index: int
    first_token_position: int
    first_cell_index: int
    token_count: int

    def __post_init__(self) -> None:
        _require_uint(self.apply_call_id, "apply_call_id", UINT64_MAX)
        if self.apply_call_id == 0:
            raise TokenCellRangeError("apply_call_id must be positive")
        _require_uint(self.tgid, "tgid", UINT32_MAX)
        _require_uint(self.tid, "tid", UINT32_MAX)
        if self.tgid == 0 or self.tid == 0:
            raise TokenCellRangeError("process identifiers must be positive")
        _require_uint(self.monotonic_ns, "monotonic_ns", UINT64_MAX)
        _require_uint(self.seq_id, "seq_id", UINT32_MAX)
        _require_uint(self.stream_index, "stream_index", UINT32_MAX)
        if self.stream_index >= MAX_STREAM_COUNT:
            raise TokenCellRangeError("stream_index is out of range")
        _require_uint(self.token_count, "token_count", UINT32_MAX)
        if not 1 <= self.token_count <= MAX_APPLY_TOKEN_COUNT:
            raise TokenCellRangeError("token_count is out of range")
        _checked_last(
            self.first_token_position,
            self.token_count,
            "token_position",
        )
        _checked_last(self.first_cell_index, self.token_count, "cell_index")

    @property
    def last_token_position(self) -> int:
        return self.first_token_position + self.token_count - 1

    @property
    def last_cell_index(self) -> int:
        return self.first_cell_index + self.token_count - 1


def _reject_mixed(
    token_cells: tuple[TokenCellEvent, ...],
    token_ranges: tuple[NativeTokenCellRange, ...],
) -> None:
    if token_cells and token_ranges:
        raise TokenCellRangeError("mixed legacy and compact token events")


def expand_token_cell_range(
    item: NativeTokenCellRange,
) -> tuple[TokenCellEvent, ...]:
    if not isinstance(item, NativeTokenCellRange):
        raise TokenCellRangeError("token range is invalid")
    return tuple(
        TokenCellEvent(
            apply_call_id=item.apply_call_id,
            tgid=item.tgid,
            tid=item.tid,
            monotonic_ns=item.monotonic_ns,
            seq_id=item.seq_id,
            stream_index=item.stream_index,
            token_position=item.first_token_position + offset,
            cell_index=item.first_cell_index + offset,
        )
        for offset in range(item.token_count)
    )


def materialize_token_cells(
    token_cells: tuple[TokenCellEvent, ...],
    token_ranges: tuple[NativeTokenCellRange, ...],
) -> tuple[TokenCellEvent, ...]:
    _reject_mixed(token_cells, token_ranges)
    if token_cells:
        return token_cells
    return tuple(
        cell
        for item in token_ranges
        for cell in expand_token_cell_range(item)
    )


def logical_token_cell_count(
    token_cells: tuple[TokenCellEvent, ...],
    token_ranges: tuple[NativeTokenCellRange, ...],
) -> int:
    _reject_mixed(token_cells, token_ranges)
    return len(token_cells) + sum(item.token_count for item in token_ranges)


def select_token_cells(
    token_cells: tuple[TokenCellEvent, ...],
    token_ranges: tuple[NativeTokenCellRange, ...],
    *,
    seq_id: int | None = None,
    apply_call_id: int | None = None,
    not_before_ns: int | None = None,
    not_after_ns: int | None = None,
) -> tuple[TokenCellEvent, ...]:
    _reject_mixed(token_cells, token_ranges)
    selected_cells = tuple(
        item
        for item in token_cells
        if (seq_id is None or item.seq_id == seq_id)
        and (apply_call_id is None or item.apply_call_id == apply_call_id)
        and (not_before_ns is None or item.monotonic_ns >= not_before_ns)
        and (not_after_ns is None or item.monotonic_ns <= not_after_ns)
    )
    selected_ranges = tuple(
        item
        for item in token_ranges
        if (seq_id is None or item.seq_id == seq_id)
        and (apply_call_id is None or item.apply_call_id == apply_call_id)
        and (not_before_ns is None or item.monotonic_ns >= not_before_ns)
        and (not_after_ns is None or item.monotonic_ns <= not_after_ns)
    )
    return materialize_token_cells(selected_cells, selected_ranges)


def contains_token_cell(
    token_cells: tuple[TokenCellEvent, ...],
    token_ranges: tuple[NativeTokenCellRange, ...],
    candidate: TokenCellEvent,
) -> bool:
    _reject_mixed(token_cells, token_ranges)
    if token_cells:
        return candidate in token_cells
    for item in token_ranges:
        if (
            candidate.apply_call_id != item.apply_call_id
            or candidate.tgid != item.tgid
            or candidate.tid != item.tid
            or candidate.monotonic_ns != item.monotonic_ns
            or candidate.seq_id != item.seq_id
            or candidate.stream_index != item.stream_index
        ):
            continue
        offset = candidate.token_position - item.first_token_position
        if (
            0 <= offset < item.token_count
            and candidate.cell_index == item.first_cell_index + offset
        ):
            return True
    return False
