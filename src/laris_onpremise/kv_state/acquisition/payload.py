from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import struct

from .layout import (
    ReadRun,
    ReadSpan,
    SelectedCell,
    SimpleKvLayout,
    TensorRowLayout,
    plan_selected_read_runs,
    plan_selected_rows,
)


class PayloadValidationError(ValueError):
    """Raised when a Simple KV payload is incomplete or malformed."""


@dataclass(frozen=True)
class SimpleKvPayloadSummary:
    n_stream: int
    nonempty_streams: tuple[int, ...]
    positions: tuple[int, ...]
    source_seq_ids: tuple[int, ...]
    cell_count: int
    v_trans: bool | None
    n_layer: int | None
    key_ggml_type_ids: tuple[int, ...]
    key_row_sizes: tuple[int, ...]
    value_ggml_type_ids: tuple[int, ...]
    value_row_sizes: tuple[int, ...]
    value_element_sizes: tuple[int, ...]
    value_embedding_widths: tuple[int, ...]
    explicit_token_count: None = None


def encode_simple_kv_payload(
    layout: SimpleKvLayout,
    cells: tuple[SelectedCell, ...],
    row_bytes: Mapping[ReadSpan, bytes],
) -> bytes:
    """Encode target rows as the raw Simple KV state_write payload."""

    spans = plan_selected_rows(layout, cells)
    expected = set(spans)
    supplied = set(row_bytes)
    if supplied != expected:
        missing = len(expected - supplied)
        extra = len(supplied - expected)
        raise PayloadValidationError(
            f"row mapping mismatch: {missing} missing, {extra} extra"
        )
    for span in spans:
        value = row_bytes[span]
        if not isinstance(value, bytes) or len(value) != span.size:
            raise PayloadValidationError(
                f"row byte length mismatch at ordinal {span.serialization_ordinal}"
            )

    stream_index = cells[0].stream_index
    tensors = {
        (tensor.kind, tensor.layer): tensor
        for tensor in layout.tensors
        if tensor.stream_index == stream_index
    }
    layers = tuple(sorted(layer for kind, layer in tensors if kind == "k"))
    by_tensor: dict[tuple[str, int], list[ReadSpan]] = {}
    for span in spans:
        by_tensor.setdefault((span.kind, span.layer), []).append(span)

    payload = bytearray(struct.pack("<I", layout.n_stream))
    for current_stream in range(layout.n_stream):
        if current_stream != stream_index:
            payload += struct.pack("<I", 0)
            continue

        payload += struct.pack("<I", len(cells))
        for cell in cells:
            payload += struct.pack("<iIi", cell.position, 1, cell.seq_id)
        payload += struct.pack("<II", int(layout.v_trans), len(layers))

        for layer in layers:
            tensor = tensors[("k", layer)]
            payload += struct.pack("<iQ", tensor.ggml_type_id, tensor.row_size)
            payload += _join_rows(by_tensor[("k", layer)], row_bytes)

        for layer in layers:
            tensor = tensors[("v", layer)]
            if layout.v_trans:
                payload += struct.pack(
                    "<iII",
                    tensor.ggml_type_id,
                    tensor.element_size,
                    tensor.embedding_width,
                )
            else:
                payload += struct.pack(
                    "<iQ", tensor.ggml_type_id, tensor.row_size
                )
            payload += _join_rows(by_tensor[("v", layer)], row_bytes)

    return bytes(payload)


def encode_simple_kv_payload_from_runs(
    layout: SimpleKvLayout,
    cells: tuple[SelectedCell, ...],
    run_bytes: Mapping[ReadRun, bytes],
    *,
    planned_runs: tuple[ReadRun, ...] | None = None,
) -> bytes:
    """Encode the canonical payload from coalesced selective read runs."""

    return b"".join(
        encode_simple_kv_payload_parts_from_runs(
            layout,
            cells,
            run_bytes,
            planned_runs=planned_runs,
        )
    )


def encode_simple_kv_payload_parts_from_runs(
    layout: SimpleKvLayout,
    cells: tuple[SelectedCell, ...],
    run_bytes: Mapping[ReadRun, bytes],
    *,
    planned_runs: tuple[ReadRun, ...] | None = None,
) -> tuple[bytes, ...]:
    """Return canonical payload chunks without assembling a second full copy."""

    if planned_runs is None:
        runs = plan_selected_read_runs(layout, cells)
    else:
        if not isinstance(planned_runs, tuple):
            raise PayloadValidationError("planned read runs must be a tuple")
        runs = planned_runs
    expected = set(runs)
    if len(expected) != len(runs):
        raise PayloadValidationError("planned read runs contain duplicates")
    supplied = set(run_bytes)
    if supplied != expected:
        missing = len(expected - supplied)
        extra = len(supplied - expected)
        raise PayloadValidationError(
            f"read run mapping mismatch: {missing} missing, {extra} extra"
        )
    for ordinal, run in enumerate(runs):
        if (
            not isinstance(run, ReadRun)
            or run.serialization_ordinal != ordinal
        ):
            raise PayloadValidationError(
                "planned read run serialization order is invalid"
            )
        value = run_bytes[run]
        if not isinstance(value, bytes) or len(value) != run.size:
            raise PayloadValidationError(
                "read run byte length mismatch at ordinal "
                f"{run.serialization_ordinal}"
            )

    stream_index = cells[0].stream_index
    tensors = {
        (tensor.kind, tensor.layer): tensor
        for tensor in layout.tensors
        if tensor.stream_index == stream_index
    }
    layers = tuple(sorted(layer for kind, layer in tensors if kind == "k"))
    by_tensor: dict[tuple[str, int], list[ReadRun]] = {}
    for run in runs:
        by_tensor.setdefault((run.kind, run.layer), []).append(run)

    parts: list[bytes] = []
    header = bytearray(struct.pack("<I", layout.n_stream))
    for current_stream in range(layout.n_stream):
        if current_stream != stream_index:
            header += struct.pack("<I", 0)
            continue

        header += struct.pack("<I", len(cells))
        for cell in cells:
            header += struct.pack("<iIi", cell.position, 1, cell.seq_id)
        header += struct.pack("<II", int(layout.v_trans), len(layers))

        for layer in layers:
            tensor = tensors[("k", layer)]
            header += struct.pack("<iQ", tensor.ggml_type_id, tensor.row_size)
            parts.append(bytes(header))
            header.clear()
            parts.extend(run_bytes[run] for run in by_tensor[("k", layer)])

        for layer in layers:
            tensor = tensors[("v", layer)]
            if layout.v_trans:
                header += struct.pack(
                    "<iII",
                    tensor.ggml_type_id,
                    tensor.element_size,
                    tensor.embedding_width,
                )
            else:
                header += struct.pack(
                    "<iQ", tensor.ggml_type_id, tensor.row_size
                )
            parts.append(bytes(header))
            header.clear()
            parts.extend(run_bytes[run] for run in by_tensor[("v", layer)])

    if header:
        parts.append(bytes(header))
    return tuple(parts)


def inspect_simple_kv_payload(payload: bytes) -> SimpleKvPayloadSummary:
    """Parse the explicit-token-ID-history-free KV payload without llama.cpp."""

    if not isinstance(payload, bytes):
        raise PayloadValidationError("payload must be bytes")
    cursor = _Cursor(payload)
    n_stream = cursor.u32()
    if n_stream <= 0 or n_stream > 1024:
        raise PayloadValidationError("payload n_stream is invalid")

    nonempty_streams: list[int] = []
    positions: list[int] = []
    source_seq_ids: set[int] = set()
    total_cells = 0
    observed_v_trans: bool | None = None
    observed_n_layer: int | None = None
    key_ggml_type_ids: list[int] = []
    key_row_sizes: list[int] = []
    value_ggml_type_ids: list[int] = []
    value_row_sizes: list[int] = []
    value_element_sizes: list[int] = []
    value_embedding_widths: list[int] = []

    for stream_index in range(n_stream):
        cell_count = cursor.u32()
        if cell_count > 1_000_000:
            raise PayloadValidationError("payload cell_count is invalid")
        if cell_count == 0:
            continue
        nonempty_streams.append(stream_index)
        total_cells += cell_count
        for _ in range(cell_count):
            positions.append(cursor.i32())
            n_seq_id = cursor.u32()
            if n_seq_id != 1:
                raise PayloadValidationError(
                    "target payload cell must contain exactly one source seq_id"
                )
            source_seq_ids.add(cursor.i32())

        v_trans_raw = cursor.u32()
        if v_trans_raw not in {0, 1}:
            raise PayloadValidationError("payload v_trans is invalid")
        v_trans = bool(v_trans_raw)
        n_layer = cursor.u32()
        if n_layer <= 0 or n_layer > 4096:
            raise PayloadValidationError("payload n_layer is invalid")
        if observed_v_trans is not None and observed_v_trans != v_trans:
            raise PayloadValidationError("payload streams disagree on v_trans")
        if observed_n_layer is not None and observed_n_layer != n_layer:
            raise PayloadValidationError("payload streams disagree on n_layer")
        observed_v_trans = v_trans
        observed_n_layer = n_layer

        for _ in range(n_layer):
            key_ggml_type_ids.append(cursor.i32())
            row_size = cursor.u64()
            _require_positive_dimension(row_size, "K row_size")
            key_row_sizes.append(row_size)
            cursor.take(row_size * cell_count)

        for _ in range(n_layer):
            value_ggml_type_ids.append(cursor.i32())
            if v_trans:
                element_size = cursor.u32()
                embedding_width = cursor.u32()
                _require_positive_dimension(element_size, "V element_size")
                _require_positive_dimension(embedding_width, "V embedding_width")
                value_element_sizes.append(element_size)
                value_embedding_widths.append(embedding_width)
                value_row_sizes.append(element_size * embedding_width)
                cursor.take(element_size * embedding_width * cell_count)
            else:
                row_size = cursor.u64()
                _require_positive_dimension(row_size, "V row_size")
                value_row_sizes.append(row_size)
                value_element_sizes.append(0)
                value_embedding_widths.append(0)
                cursor.take(row_size * cell_count)

    if cursor.remaining:
        raise PayloadValidationError("payload contains trailing bytes")
    return SimpleKvPayloadSummary(
        n_stream=n_stream,
        nonempty_streams=tuple(nonempty_streams),
        positions=tuple(positions),
        source_seq_ids=tuple(sorted(source_seq_ids)),
        cell_count=total_cells,
        v_trans=observed_v_trans,
        n_layer=observed_n_layer,
        key_ggml_type_ids=tuple(key_ggml_type_ids),
        key_row_sizes=tuple(key_row_sizes),
        value_ggml_type_ids=tuple(value_ggml_type_ids),
        value_row_sizes=tuple(value_row_sizes),
        value_element_sizes=tuple(value_element_sizes),
        value_embedding_widths=tuple(value_embedding_widths),
    )


def _join_rows(
    spans: list[ReadSpan], row_bytes: Mapping[ReadSpan, bytes]
) -> bytes:
    return b"".join(row_bytes[span] for span in spans)


def _join_runs(
    runs: list[ReadRun], run_bytes: Mapping[ReadRun, bytes]
) -> bytes:
    return b"".join(run_bytes[run] for run in runs)


def _require_positive_dimension(value: int, name: str) -> None:
    if value <= 0 or value > (1 << 32):
        raise PayloadValidationError(f"payload {name} is invalid")


class _Cursor:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.offset = 0

    @property
    def remaining(self) -> int:
        return len(self.payload) - self.offset

    def take(self, size: int) -> bytes:
        if size < 0 or size > self.remaining:
            raise PayloadValidationError("payload is truncated")
        start = self.offset
        self.offset += size
        return self.payload[start : start + size]

    def u32(self) -> int:
        return struct.unpack("<I", self.take(4))[0]

    def i32(self) -> int:
        return struct.unpack("<i", self.take(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.take(8))[0]
