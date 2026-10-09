from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
import struct
from typing import Final


BULK_PAYLOAD_BYTES: Final[int] = 65_536
BULK_FLAG_FIRST: Final[int] = 1
BULK_FLAG_FINAL: Final[int] = 2
_BULK_FLAG_MASK: Final[int] = BULK_FLAG_FIRST | BULK_FLAG_FINAL
_MAX_FRAGMENT_BYTES: Final[int] = 4096
_MAX_STREAM_COUNT: Final[int] = 8
_MAX_U32: Final[int] = (1 << 32) - 1
_MAX_U64: Final[int] = (1 << 64) - 1
_TENSOR_KIND_K: Final[int] = 1
_TENSOR_KIND_V: Final[int] = 2
_BULK_PREFIX: Final[struct.Struct] = struct.Struct("<QQQQQQ" + "I" * 20)
BULK_PREFIX_BYTES: Final[int] = _BULK_PREFIX.size


class NativeBulkDecodeError(ValueError):
    pass


@dataclass(frozen=True)
class NativeBulkFrame:
    apply_call_id: int
    writer_monotonic_ns: int
    plan_copy_byte_count: int
    frame_payload_offset: int
    first_token_position: int
    first_cell_index: int
    first_fragment_index: int
    fragment_count: int
    frame_first_fragment_index: int
    frame_fragment_count: int
    payload_size: int
    seq_id: int
    stream_index: int
    tensor_kind: int
    layer: int
    expected_n_layer: int
    token_count: int
    row_size: int
    cell_capacity: int
    element_size: int
    embedding_width: int
    chunks_per_dimension: int
    fragments_per_tensor: int
    transposed_value: int
    flags: int
    reserved: int
    payload: memoryview


@dataclass(frozen=True)
class NativeBulkFragmentView:
    apply_call_id: int
    fragment_index: int
    seq_id: int
    stream_index: int
    tensor_kind: str
    token_position: int
    cell_index: int
    layer: int
    payload_size: int
    tensor_byte_offset: int
    writer_monotonic_ns: int
    span_cell_count: int
    payload: memoryview


def _fail(message: str) -> None:
    raise NativeBulkDecodeError(message)


def _fragment_layout(
    frame: NativeBulkFrame,
    fragment_offset: int,
) -> tuple[int, int, int, int, int, int]:
    unit_size = frame.element_size if frame.transposed_value else frame.row_size
    units_per_fragment = _MAX_FRAGMENT_BYTES // unit_size
    if units_per_fragment == 0 or not 0 <= fragment_offset < frame.fragment_count:
        _fail("bulk fragment shape is invalid")
    chunk_index = fragment_offset % frame.chunks_per_dimension
    dimension = (
        fragment_offset // frame.chunks_per_dimension
        if frame.transposed_value
        else 0
    )
    cell_offset = chunk_index * units_per_fragment
    if cell_offset >= frame.token_count:
        _fail("bulk fragment cell range is invalid")
    span_cell_count = min(
        frame.token_count - cell_offset,
        units_per_fragment,
    )
    fragment_payload_size = span_cell_count * unit_size
    if frame.transposed_value:
        packed_offset = (
            dimension * frame.token_count * frame.element_size
            + cell_offset * frame.element_size
        )
        tensor_byte_offset = (
            dimension * frame.cell_capacity * frame.element_size
            + (frame.first_cell_index + cell_offset) * frame.element_size
        )
    else:
        packed_offset = cell_offset * frame.row_size
        tensor_byte_offset = (frame.first_cell_index + cell_offset) * frame.row_size
    if (
        fragment_payload_size <= 0
        or fragment_payload_size > _MAX_FRAGMENT_BYTES
        or packed_offset > frame.plan_copy_byte_count
        or fragment_payload_size
        > frame.plan_copy_byte_count - packed_offset
    ):
        _fail("bulk fragment payload range is invalid")
    return (
        packed_offset,
        fragment_payload_size,
        cell_offset,
        span_cell_count,
        dimension,
        tensor_byte_offset,
    )


def _packed_prefix_size(frame: NativeBulkFrame, fragment_offset: int) -> int:
    unit_size = frame.element_size if frame.transposed_value else frame.row_size
    units_per_fragment = _MAX_FRAGMENT_BYTES // unit_size
    if units_per_fragment == 0 or not 0 <= fragment_offset <= frame.fragment_count:
        _fail("bulk fragment shape is invalid")
    if frame.transposed_value:
        dimension, chunk_index = divmod(
            fragment_offset,
            frame.chunks_per_dimension,
        )
        if dimension > frame.embedding_width:
            _fail("bulk fragment shape is invalid")
        completed_cells = min(
            chunk_index * units_per_fragment,
            frame.token_count,
        )
        return (
            dimension * frame.token_count + completed_cells
        ) * frame.element_size
    completed_cells = min(
        fragment_offset * units_per_fragment,
        frame.token_count,
    )
    return completed_cells * frame.row_size


def _validate_frame(frame: NativeBulkFrame) -> None:
    if (
        frame.apply_call_id == 0
        or frame.writer_monotonic_ns == 0
        or frame.plan_copy_byte_count == 0
        or frame.fragment_count == 0
        or frame.frame_fragment_count == 0
        or frame.payload_size == 0
        or frame.payload_size > BULK_PAYLOAD_BYTES
        or frame.payload.nbytes != frame.payload_size
        or frame.stream_index >= _MAX_STREAM_COUNT
        or frame.tensor_kind not in {_TENSOR_KIND_K, _TENSOR_KIND_V}
        or frame.expected_n_layer == 0
        or frame.layer >= frame.expected_n_layer
        or frame.token_count == 0
        or frame.row_size == 0
        or frame.cell_capacity == 0
        or frame.element_size == 0
        or frame.embedding_width == 0
        or frame.chunks_per_dimension == 0
        or frame.fragments_per_tensor == 0
        or frame.reserved != 0
        or frame.flags & ~_BULK_FLAG_MASK
        or frame.first_cell_index >= frame.cell_capacity
        or frame.token_count > frame.cell_capacity - frame.first_cell_index
        or frame.row_size != frame.element_size * frame.embedding_width
        or frame.first_fragment_index > _MAX_U32 - frame.fragment_count
        or frame.first_token_position > _MAX_U64 - frame.token_count
        or (frame.tensor_kind == _TENSOR_KIND_K and frame.transposed_value != 0)
        or (frame.tensor_kind == _TENSOR_KIND_V and frame.transposed_value > 1)
    ):
        _fail("bulk frame metadata is invalid")
    derived_fragments_per_tensor = frame.chunks_per_dimension * (
        frame.embedding_width if frame.transposed_value else 1
    )
    unit_size = frame.element_size if frame.transposed_value else frame.row_size
    units_per_fragment = _MAX_FRAGMENT_BYTES // unit_size
    canonical_chunks_per_dimension = (
        (frame.token_count + units_per_fragment - 1) // units_per_fragment
        if units_per_fragment
        else 0
    )
    if (
        units_per_fragment == 0
        or frame.chunks_per_dimension != canonical_chunks_per_dimension
        or derived_fragments_per_tensor != frame.fragments_per_tensor
        or frame.fragment_count != derived_fragments_per_tensor
        or frame.plan_copy_byte_count != frame.token_count * frame.row_size
        or frame.frame_first_fragment_index < frame.first_fragment_index
    ):
        _fail("bulk plan shape is invalid")
    first_offset = frame.frame_first_fragment_index - frame.first_fragment_index
    if (
        first_offset >= frame.fragment_count
        or frame.frame_fragment_count > frame.fragment_count - first_offset
    ):
        _fail("bulk frame fragment range is invalid")
    expected_payload_offset = _packed_prefix_size(frame, first_offset)
    payload_end = _packed_prefix_size(
        frame,
        first_offset + frame.frame_fragment_count,
    )
    accumulated_payload_size = payload_end - expected_payload_offset
    expected_flags = (
        BULK_FLAG_FIRST if first_offset == 0 else 0
    ) | (
        BULK_FLAG_FINAL
        if first_offset + frame.frame_fragment_count == frame.fragment_count
        else 0
    )
    if (
        frame.frame_payload_offset != expected_payload_offset
        or frame.payload_size != accumulated_payload_size
        or frame.flags != expected_flags
        or frame.frame_payload_offset > frame.plan_copy_byte_count
        or frame.payload_size
        > frame.plan_copy_byte_count - frame.frame_payload_offset
    ):
        _fail("bulk frame window is non-canonical")


def decode_native_bulk_payload(payload: bytes) -> NativeBulkFrame:
    if not isinstance(payload, bytes) or len(payload) <= BULK_PREFIX_BYTES:
        _fail("bulk payload is truncated")
    values = _BULK_PREFIX.unpack_from(payload)
    frame = NativeBulkFrame(
        *values,
        payload=memoryview(payload)[BULK_PREFIX_BYTES:],
    )
    _validate_frame(frame)
    if len(payload) != BULK_PREFIX_BYTES + frame.payload_size:
        _fail("bulk payload length conflicts")
    return frame


def iter_native_bulk_fragment_views(
    frame: NativeBulkFrame,
) -> Iterator[NativeBulkFragmentView]:
    if not isinstance(frame, NativeBulkFrame):
        _fail("bulk frame is invalid")
    _validate_frame(frame)
    first_offset = frame.frame_first_fragment_index - frame.first_fragment_index
    for relative_index in range(frame.frame_fragment_count):
        fragment_offset = first_offset + relative_index
        (
            packed_offset,
            fragment_payload_size,
            cell_offset,
            span_cell_count,
            _dimension,
            tensor_byte_offset,
        ) = _fragment_layout(frame, fragment_offset)
        frame_relative_offset = packed_offset - frame.frame_payload_offset
        payload_end = frame_relative_offset + fragment_payload_size
        if frame_relative_offset < 0 or payload_end > frame.payload.nbytes:
            _fail("bulk fragment falls outside the frame payload")
        yield NativeBulkFragmentView(
            apply_call_id=frame.apply_call_id,
            fragment_index=frame.first_fragment_index + fragment_offset,
            seq_id=frame.seq_id,
            stream_index=frame.stream_index,
            tensor_kind="k" if frame.tensor_kind == _TENSOR_KIND_K else "v",
            token_position=frame.first_token_position + cell_offset,
            cell_index=frame.first_cell_index + cell_offset,
            layer=frame.layer,
            payload_size=fragment_payload_size,
            tensor_byte_offset=tensor_byte_offset,
            writer_monotonic_ns=frame.writer_monotonic_ns,
            span_cell_count=span_cell_count,
            payload=frame.payload[frame_relative_offset:payload_end],
        )


__all__ = [
    "BULK_FLAG_FINAL",
    "BULK_FLAG_FIRST",
    "BULK_PAYLOAD_BYTES",
    "BULK_PREFIX_BYTES",
    "NativeBulkDecodeError",
    "NativeBulkFragmentView",
    "NativeBulkFrame",
    "decode_native_bulk_payload",
    "iter_native_bulk_fragment_views",
]
