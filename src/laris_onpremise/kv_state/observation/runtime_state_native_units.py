from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
import re
import stat
import struct
import tempfile
import threading
from collections.abc import Iterator
from typing import Final, Protocol

from .runtime_state_bulk import (
    BULK_FLAG_FINAL,
    BULK_FLAG_FIRST,
    NativeBulkDecodeError,
    NativeBulkFrame,
    decode_native_bulk_payload,
    iter_native_bulk_fragment_views,
)


_UNIT_MAGIC: Final[bytes] = b"LNU1"
_UNIT_VERSION: Final[int] = 2
_LEGACY_UNIT_VERSION: Final[int] = 1
_UNIT_HEADER: Final[struct.Struct] = struct.Struct("<4sHH")
_FRAGMENT_HEADER: Final[struct.Struct] = struct.Struct("<QIIIIQQIIQQQ")
_FRAGMENT_BODY: Final[struct.Struct] = struct.Struct("<IIIIQQIIQQQ")
_COMPACT_RECORD_HEADER: Final[struct.Struct] = struct.Struct("<4sI")
_COMPACT_FRAGMENT_TAG: Final[bytes] = b"FRG2"
_COMPACT_SCHEDULE_TAG: Final[bytes] = b"SCH2"
_COMPACT_BULK_TAG: Final[bytes] = b"BLK2"
_COMPACT_BULK: Final[struct.Struct] = struct.Struct("<QQ" + "I" * 9)
_SCHEDULE_BATCH: Final[struct.Struct] = struct.Struct(
    "<IIQIIIIIIQQIIIIIIIIQ"
)
_MAX_FRAGMENT_BYTES: Final[int] = 4096
_MAX_BATCH_FRAGMENT_BYTES: Final[int] = 128
_MAX_READ_BYTES: Final[int] = 1024 * 1024
_WRITE_BUFFER_BYTES: Final[int] = 64 * 1024
_SUPPORTED_STREAM_COUNTS: Final[frozenset[int]] = frozenset({4, 8})
_MAX_STREAM_COUNT: Final[int] = max(_SUPPORTED_STREAM_COUNTS)
_UNIT_ID: Final[re.Pattern[str]] = re.compile(r"unit_[0-9a-f]{64}")
_GROUP_NAME: Final[re.Pattern[str]] = re.compile(
    r"group_[0-9]{20}_[0-9a-f]{16}\.json"
)


class NativeUnitSpoolError(RuntimeError):
    pass


class NativeUnitFragment(Protocol):
    apply_call_id: int
    fragment_index: int
    seq_id: int
    stream_index: int
    token_position: int
    cell_index: int
    tensor_kind: str
    layer: int
    span_cell_count: int
    tensor_byte_offset: int
    writer_monotonic_ns: int
    payload: bytes


class NativeUnitSeal(Protocol):
    apply_call_id: int
    expected_fragment_count: int
    expected_token_cell_count: int
    stream_mask: int
    writer_state: str
    copy_byte_count: int


@dataclass(frozen=True)
class NativeCapturedUnitDescriptor:
    cursor: int
    unit_id: str
    runtime_instance_id: str
    apply_call_id: int
    seq_id: int
    stream_index: int
    writer_monotonic_ns: int
    token_cells: tuple[tuple[int, int], ...]
    fragment_count: int
    payload_size: int
    payload_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "cursor",
            "apply_call_id",
            "seq_id",
            "stream_index",
            "writer_monotonic_ns",
            "fragment_count",
            "payload_size",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise NativeUnitSpoolError(f"{name} is invalid")
        if self.cursor == 0 or self.apply_call_id == 0:
            raise NativeUnitSpoolError("descriptor identity is invalid")
        if self.stream_index >= _MAX_STREAM_COUNT or self.fragment_count == 0:
            raise NativeUnitSpoolError("descriptor shape is invalid")
        if self.payload_size < _UNIT_HEADER.size:
            raise NativeUnitSpoolError("descriptor payload size is invalid")
        if not _UNIT_ID.fullmatch(self.unit_id):
            raise NativeUnitSpoolError("unit_id is invalid")
        if not _valid_runtime_id(self.runtime_instance_id):
            raise NativeUnitSpoolError("runtime_instance_id is invalid")
        if not re.fullmatch(r"[0-9a-f]{64}", self.payload_sha256):
            raise NativeUnitSpoolError("payload_sha256 is invalid")
        if not self.token_cells:
            raise NativeUnitSpoolError("token_cells must not be empty")
        prior: tuple[int, int] | None = None
        for item in self.token_cells:
            if (
                not isinstance(item, tuple)
                or len(item) != 2
                or any(
                    not isinstance(value, int)
                    or isinstance(value, bool)
                    or value < 0
                    for value in item
                )
                or (prior is not None and item <= prior)
            ):
                raise NativeUnitSpoolError("token_cells are not canonical")
            prior = item


@dataclass(frozen=True)
class CompletedNativeUnitPage:
    next_cursor: int
    truncated: bool
    units: tuple[NativeCapturedUnitDescriptor, ...]


@dataclass(frozen=True)
class NativeUnitFragmentRecord:
    fragment_index: int
    tensor_kind: str
    layer: int
    token_position: int
    cell_index: int
    span_cell_count: int
    tensor_byte_offset: int
    payload: bytes


@dataclass(frozen=True)
class NativeUnitScheduleRecord:
    record_count: int
    payload_size: int
    apply_call_id: int
    first_fragment_index: int
    first_flat_index: int
    seq_id: int
    stream_index: int
    tensor_kind: int
    fragments_per_tensor: int
    first_token_position: int
    first_cell_index: int
    chunks_per_dimension: int
    token_count: int
    row_size: int
    cell_capacity: int
    element_size: int
    embedding_width: int
    transposed_value: int
    reserved: int
    writer_monotonic_ns: int
    packed_payload: bytes


def _schedule_payload_prefix_size(
    fragment_count: int,
    *,
    chunks_per_dimension: int,
    token_count: int,
    units_per_fragment: int,
    element_size: int,
) -> int:
    cycles, remainder = divmod(fragment_count, chunks_per_dimension)
    return (
        cycles * token_count * element_size
        + remainder * units_per_fragment * element_size
    )


def _decode_schedule_batch(payload: bytes) -> NativeUnitScheduleRecord:
    if not isinstance(payload, bytes) or len(payload) <= _SCHEDULE_BATCH.size:
        raise NativeUnitSpoolError("native-unit schedule batch is truncated")
    values = _SCHEDULE_BATCH.unpack_from(payload)
    batch = NativeUnitScheduleRecord(
        *values,
        packed_payload=payload[_SCHEDULE_BATCH.size :],
    )
    maximum_u32 = (1 << 32) - 1
    maximum_u64 = (1 << 64) - 1
    if (
        batch.record_count == 0
        or batch.record_count > 32
        or batch.payload_size == 0
        or batch.payload_size > _MAX_FRAGMENT_BYTES
        or batch.payload_size != len(batch.packed_payload)
        or batch.apply_call_id == 0
        or batch.stream_index >= _MAX_STREAM_COUNT
        or batch.tensor_kind != 2
        or batch.fragments_per_tensor == 0
        or batch.chunks_per_dimension == 0
        or batch.token_count == 0
        or batch.row_size == 0
        or batch.cell_capacity == 0
        or batch.element_size == 0
        or batch.embedding_width == 0
        or batch.transposed_value != 1
        or batch.reserved != 0
        or batch.writer_monotonic_ns == 0
        or batch.row_size != batch.element_size * batch.embedding_width
        or batch.fragments_per_tensor
        != batch.chunks_per_dimension * batch.embedding_width
        or batch.first_cell_index >= batch.cell_capacity
        or batch.token_count
        > batch.cell_capacity - batch.first_cell_index
        or batch.first_fragment_index
        > maximum_u32 - batch.record_count
        or batch.first_flat_index > maximum_u32 - batch.record_count
        or batch.first_token_position > maximum_u64 - batch.token_count
    ):
        raise NativeUnitSpoolError("native-unit schedule batch metadata is invalid")
    units_per_fragment = _MAX_FRAGMENT_BYTES // batch.element_size
    if (
        units_per_fragment == 0
        or batch.chunks_per_dimension
        != (batch.token_count + units_per_fragment - 1)
        // units_per_fragment
    ):
        raise NativeUnitSpoolError("native-unit schedule batch shape is invalid")
    expected_payload_size = _schedule_payload_prefix_size(
        batch.first_flat_index + batch.record_count,
        chunks_per_dimension=batch.chunks_per_dimension,
        token_count=batch.token_count,
        units_per_fragment=units_per_fragment,
        element_size=batch.element_size,
    ) - _schedule_payload_prefix_size(
        batch.first_flat_index,
        chunks_per_dimension=batch.chunks_per_dimension,
        token_count=batch.token_count,
        units_per_fragment=units_per_fragment,
        element_size=batch.element_size,
    )
    if expected_payload_size != batch.payload_size:
        raise NativeUnitSpoolError("native-unit schedule batch bytes are invalid")
    return batch


def _iter_schedule_batch_records(
    batch: NativeUnitScheduleRecord,
) -> Iterator[NativeUnitFragmentRecord]:
    units_per_fragment = _MAX_FRAGMENT_BYTES // batch.element_size
    payload_offset = 0
    for record_index in range(batch.record_count):
        flat_index = batch.first_flat_index + record_index
        layer = flat_index // batch.fragments_per_tensor
        fragment_in_tensor = flat_index % batch.fragments_per_tensor
        dimension = fragment_in_tensor // batch.chunks_per_dimension
        chunk_index = fragment_in_tensor % batch.chunks_per_dimension
        cell_offset = chunk_index * units_per_fragment
        span_cell_count = min(
            batch.token_count - cell_offset,
            units_per_fragment,
        )
        fragment_payload_size = span_cell_count * batch.element_size
        payload_end = payload_offset + fragment_payload_size
        if (
            dimension >= batch.embedding_width
            or span_cell_count <= 0
            or fragment_payload_size > _MAX_BATCH_FRAGMENT_BYTES
            or payload_end > len(batch.packed_payload)
        ):
            raise NativeUnitSpoolError("native-unit schedule record is invalid")
        cell_index = batch.first_cell_index + cell_offset
        yield NativeUnitFragmentRecord(
            fragment_index=batch.first_fragment_index + record_index,
            tensor_kind="v",
            layer=layer,
            token_position=batch.first_token_position + cell_offset,
            cell_index=cell_index,
            span_cell_count=span_cell_count,
            tensor_byte_offset=(
                dimension * batch.element_size * batch.cell_capacity
                + cell_index * batch.element_size
            ),
            payload=batch.packed_payload[payload_offset:payload_end],
        )
        payload_offset = payload_end
    if payload_offset != len(batch.packed_payload):
        raise NativeUnitSpoolError("native-unit schedule payload has trailing bytes")


def _compact_bulk_window(
    *,
    first_fragment_index: int,
    frame_first_fragment_index: int,
    frame_fragment_count: int,
    tensor_kind: int,
    token_count: int,
    element_size: int,
    embedding_width: int,
    transposed_value: int,
) -> tuple[int, int, int, int]:
    if (
        tensor_kind not in {1, 2}
        or token_count <= 0
        or element_size <= 0
        or embedding_width <= 0
        or frame_fragment_count <= 0
        or (tensor_kind == 1 and transposed_value != 0)
        or (tensor_kind == 2 and transposed_value not in {0, 1})
    ):
        raise ValueError("invalid compact bulk shape")
    row_size = element_size * embedding_width
    unit_size = element_size if transposed_value else row_size
    units_per_fragment = _MAX_FRAGMENT_BYTES // unit_size
    if units_per_fragment == 0:
        raise ValueError("invalid compact bulk unit")
    chunks_per_dimension = (
        token_count + units_per_fragment - 1
    ) // units_per_fragment
    plan_fragment_count = chunks_per_dimension * (
        embedding_width if transposed_value else 1
    )
    first_offset = frame_first_fragment_index - first_fragment_index
    if (
        first_offset < 0
        or first_offset >= plan_fragment_count
        or frame_fragment_count > plan_fragment_count - first_offset
    ):
        raise ValueError("invalid compact bulk range")
    frame_payload_offset: int | None = None
    frame_payload_size = 0
    for fragment_offset in range(
        first_offset,
        first_offset + frame_fragment_count,
    ):
        chunk_index = fragment_offset % chunks_per_dimension
        dimension = (
            fragment_offset // chunks_per_dimension
            if transposed_value
            else 0
        )
        cell_offset = chunk_index * units_per_fragment
        span_cell_count = min(
            token_count - cell_offset,
            units_per_fragment,
        )
        if span_cell_count <= 0:
            raise ValueError("invalid compact bulk span")
        fragment_size = span_cell_count * unit_size
        packed_offset = (
            dimension * token_count * element_size
            + cell_offset * element_size
            if transposed_value
            else cell_offset * row_size
        )
        if frame_payload_offset is None:
            frame_payload_offset = packed_offset
        elif packed_offset != frame_payload_offset + frame_payload_size:
            raise ValueError("non-contiguous compact bulk span")
        frame_payload_size += fragment_size
    if frame_payload_offset is None or frame_payload_size <= 0:
        raise ValueError("empty compact bulk window")
    return (
        plan_fragment_count,
        chunks_per_dimension,
        frame_payload_offset,
        frame_payload_size,
    )


def _bulk_plan_identity(frame: NativeBulkFrame) -> tuple[int, ...]:
    return (
        frame.apply_call_id,
        frame.writer_monotonic_ns,
        frame.plan_copy_byte_count,
        frame.first_token_position,
        frame.first_cell_index,
        frame.first_fragment_index,
        frame.fragment_count,
        frame.seq_id,
        frame.stream_index,
        frame.tensor_kind,
        frame.layer,
        frame.expected_n_layer,
        frame.token_count,
        frame.row_size,
        frame.cell_capacity,
        frame.element_size,
        frame.embedding_width,
        frame.chunks_per_dimension,
        frame.fragments_per_tensor,
        frame.transposed_value,
    )


def parse_native_unit_payload(
    payload: bytes,
    *,
    descriptor: NativeCapturedUnitDescriptor,
) -> tuple[NativeUnitFragmentRecord, ...]:
    return tuple(iter_native_unit_payload(payload, descriptor=descriptor))


def iter_native_unit_payload(
    payload: bytes,
    *,
    descriptor: NativeCapturedUnitDescriptor,
) -> Iterator[NativeUnitFragmentRecord]:
    for record in iter_native_unit_records(payload, descriptor=descriptor):
        if isinstance(record, NativeUnitScheduleRecord):
            yield from _iter_schedule_batch_records(record)
        elif isinstance(record, NativeBulkFrame):
            for view in iter_native_bulk_fragment_views(record):
                yield NativeUnitFragmentRecord(
                    fragment_index=view.fragment_index,
                    tensor_kind=view.tensor_kind,
                    layer=view.layer,
                    token_position=view.token_position,
                    cell_index=view.cell_index,
                    span_cell_count=view.span_cell_count,
                    tensor_byte_offset=view.tensor_byte_offset,
                    payload=bytes(view.payload),
                )
        else:
            yield record


def iter_native_unit_records(
    payload: bytes,
    *,
    descriptor: NativeCapturedUnitDescriptor,
) -> Iterator[
    NativeUnitFragmentRecord | NativeUnitScheduleRecord | NativeBulkFrame
]:
    if not isinstance(payload, bytes):
        raise NativeUnitSpoolError("native-unit payload must be bytes")
    if not isinstance(descriptor, NativeCapturedUnitDescriptor):
        raise NativeUnitSpoolError("native-unit descriptor is invalid")
    if (
        len(payload) != descriptor.payload_size
        or hashlib.sha256(payload).hexdigest() != descriptor.payload_sha256
    ):
        raise NativeUnitSpoolError("native-unit payload digest conflicts")
    if len(payload) < _UNIT_HEADER.size:
        raise NativeUnitSpoolError("native-unit payload is truncated")
    magic, version, reserved = _UNIT_HEADER.unpack_from(payload)
    if (
        magic != _UNIT_MAGIC
        or version not in {_LEGACY_UNIT_VERSION, _UNIT_VERSION}
        or reserved != 0
    ):
        raise NativeUnitSpoolError("native-unit payload header is invalid")
    if version == _LEGACY_UNIT_VERSION:
        yield from _iter_legacy_unit_payload(
            payload,
            descriptor=descriptor,
        )
    else:
        yield from _iter_compact_unit_records(
            payload,
            descriptor=descriptor,
        )


def _iter_legacy_unit_payload(
    payload: bytes,
    *,
    descriptor: NativeCapturedUnitDescriptor,
) -> Iterator[NativeUnitFragmentRecord]:
    cursor = _UNIT_HEADER.size
    record_count = 0
    prior_index: int | None = None
    spans: set[tuple[int, int, int]] = set()
    while cursor < len(payload):
        if len(payload) - cursor < _FRAGMENT_HEADER.size:
            raise NativeUnitSpoolError("native-unit fragment header is truncated")
        (
            apply_call_id,
            fragment_index,
            seq_id,
            stream_index,
            tensor_kind_raw,
            token_position,
            cell_index,
            layer,
            payload_size,
            tensor_byte_offset,
            writer_monotonic_ns,
            span_cell_count,
        ) = _FRAGMENT_HEADER.unpack_from(payload, cursor)
        cursor += _FRAGMENT_HEADER.size
        if payload_size == 0 or payload_size > 4096 or payload_size > len(payload) - cursor:
            raise NativeUnitSpoolError("native-unit fragment payload is invalid")
        if (
            apply_call_id != descriptor.apply_call_id
            or seq_id != descriptor.seq_id
            or stream_index != descriptor.stream_index
            or writer_monotonic_ns != descriptor.writer_monotonic_ns
            or tensor_kind_raw not in {1, 2}
            or span_cell_count == 0
            or (prior_index is not None and fragment_index <= prior_index)
        ):
            raise NativeUnitSpoolError("native-unit fragment identity conflicts")
        fragment_payload = payload[cursor : cursor + payload_size]
        cursor += payload_size
        yield NativeUnitFragmentRecord(
            fragment_index=fragment_index,
            tensor_kind="k" if tensor_kind_raw == 1 else "v",
            layer=layer,
            token_position=token_position,
            cell_index=cell_index,
            span_cell_count=span_cell_count,
            tensor_byte_offset=tensor_byte_offset,
            payload=fragment_payload,
        )
        spans.add((token_position, cell_index, span_cell_count))
        prior_index = fragment_index
        record_count += 1
    if record_count != descriptor.fragment_count:
        raise NativeUnitSpoolError("native-unit fragment count conflicts")
    if _expand_spans(spans) != descriptor.token_cells:
        raise NativeUnitSpoolError("native-unit fragment cell plan conflicts")


def _iter_compact_unit_records(
    payload: bytes,
    *,
    descriptor: NativeCapturedUnitDescriptor,
) -> Iterator[
    NativeUnitFragmentRecord | NativeUnitScheduleRecord | NativeBulkFrame
]:
    cursor = _UNIT_HEADER.size
    logical_count = 0
    prior_index: int | None = None
    spans: set[tuple[int, int, int]] = set()
    while cursor < len(payload):
        if len(payload) - cursor < _COMPACT_RECORD_HEADER.size:
            raise NativeUnitSpoolError("native-unit compact record is truncated")
        tag, record_count = _COMPACT_RECORD_HEADER.unpack_from(payload, cursor)
        if tag == _COMPACT_FRAGMENT_TAG:
            if record_count != 1 or len(payload) - cursor < _FRAGMENT_HEADER.size:
                raise NativeUnitSpoolError("native-unit compact fragment is invalid")
            values = _FRAGMENT_BODY.unpack_from(
                payload, cursor + _COMPACT_RECORD_HEADER.size
            )
            payload_size = values[7]
            record_end = cursor + _FRAGMENT_HEADER.size + payload_size
            if (
                payload_size == 0
                or payload_size > _MAX_FRAGMENT_BYTES
                or record_end > len(payload)
            ):
                raise NativeUnitSpoolError("native-unit compact fragment is invalid")
            (
                fragment_index,
                seq_id,
                stream_index,
                tensor_kind_raw,
                token_position,
                cell_index,
                layer,
                _payload_size,
                tensor_byte_offset,
                writer_monotonic_ns,
                span_cell_count,
            ) = values
            if (
                seq_id != descriptor.seq_id
                or stream_index != descriptor.stream_index
                or writer_monotonic_ns != descriptor.writer_monotonic_ns
                or tensor_kind_raw not in {1, 2}
                or span_cell_count == 0
                or (prior_index is not None and fragment_index <= prior_index)
            ):
                raise NativeUnitSpoolError("native-unit fragment identity conflicts")
            fragment_payload = payload[
                cursor + _FRAGMENT_HEADER.size : record_end
            ]
            yield NativeUnitFragmentRecord(
                fragment_index=fragment_index,
                tensor_kind="k" if tensor_kind_raw == 1 else "v",
                layer=layer,
                token_position=token_position,
                cell_index=cell_index,
                span_cell_count=span_cell_count,
                tensor_byte_offset=tensor_byte_offset,
                payload=fragment_payload,
            )
            spans.add((token_position, cell_index, span_cell_count))
            prior_index = fragment_index
            logical_count += 1
            cursor = record_end
            continue
        if tag == _COMPACT_BULK_TAG:
            if record_count == 0:
                raise NativeUnitSpoolError("native-unit compact bulk is invalid")
            metadata_size = _FRAGMENT_HEADER.size * record_count
            compact_metadata_size = (
                _COMPACT_RECORD_HEADER.size + _COMPACT_BULK.size
            )
            if (
                metadata_size < compact_metadata_size
                or metadata_size > len(payload) - cursor
            ):
                raise NativeUnitSpoolError(
                    "native-unit compact bulk is truncated"
                )
            (
                first_token_position,
                first_cell_index,
                first_fragment_index,
                frame_first_fragment_index,
                tensor_kind,
                layer,
                token_count,
                cell_capacity,
                element_size,
                embedding_width,
                transposed_value,
            ) = _COMPACT_BULK.unpack_from(
                payload,
                cursor + _COMPACT_RECORD_HEADER.size,
            )
            if any(
                payload[
                    cursor + compact_metadata_size : cursor + metadata_size
                ]
            ):
                raise NativeUnitSpoolError(
                    "native-unit compact bulk padding is invalid"
                )
            try:
                (
                    plan_fragment_count,
                    chunks_per_dimension,
                    frame_payload_offset,
                    frame_payload_size,
                ) = _compact_bulk_window(
                    first_fragment_index=first_fragment_index,
                    frame_first_fragment_index=frame_first_fragment_index,
                    frame_fragment_count=record_count,
                    tensor_kind=tensor_kind,
                    token_count=token_count,
                    element_size=element_size,
                    embedding_width=embedding_width,
                    transposed_value=transposed_value,
                )
            except (ValueError, ZeroDivisionError) as error:
                raise NativeUnitSpoolError(
                    "native-unit compact bulk metadata is invalid"
                ) from error
            record_end = cursor + metadata_size + frame_payload_size
            if record_end > len(payload):
                raise NativeUnitSpoolError(
                    "native-unit compact bulk payload is truncated"
                )
            flags = (
                BULK_FLAG_FIRST
                if frame_first_fragment_index == first_fragment_index
                else 0
            ) | (
                BULK_FLAG_FINAL
                if frame_first_fragment_index + record_count
                == first_fragment_index + plan_fragment_count
                else 0
            )
            row_size = element_size * embedding_width
            if (
                first_cell_index >= cell_capacity
                or token_count > cell_capacity - first_cell_index
            ):
                raise NativeUnitSpoolError(
                    "native-unit compact bulk cell range is invalid"
                )
            frame = NativeBulkFrame(
                apply_call_id=descriptor.apply_call_id,
                writer_monotonic_ns=descriptor.writer_monotonic_ns,
                plan_copy_byte_count=token_count * row_size,
                frame_payload_offset=frame_payload_offset,
                first_token_position=first_token_position,
                first_cell_index=first_cell_index,
                first_fragment_index=first_fragment_index,
                fragment_count=plan_fragment_count,
                frame_first_fragment_index=frame_first_fragment_index,
                frame_fragment_count=record_count,
                payload_size=frame_payload_size,
                seq_id=descriptor.seq_id,
                stream_index=descriptor.stream_index,
                tensor_kind=tensor_kind,
                layer=layer,
                expected_n_layer=layer + 1,
                token_count=token_count,
                row_size=row_size,
                cell_capacity=cell_capacity,
                element_size=element_size,
                embedding_width=embedding_width,
                chunks_per_dimension=chunks_per_dimension,
                fragments_per_tensor=plan_fragment_count,
                transposed_value=transposed_value,
                flags=flags,
                reserved=0,
                payload=memoryview(payload)[
                    cursor + metadata_size : record_end
                ],
            )
            if prior_index is not None and (
                frame_first_fragment_index != prior_index + 1
            ):
                raise NativeUnitSpoolError(
                    "native-unit compact bulk identity conflicts"
                )
            yield frame
            spans.add(
                (
                    first_token_position,
                    first_cell_index,
                    token_count,
                )
            )
            prior_index = frame_first_fragment_index + record_count - 1
            logical_count += record_count
            cursor = record_end
            continue
        if tag != _COMPACT_SCHEDULE_TAG or record_count < 2:
            raise NativeUnitSpoolError("native-unit compact record type is invalid")
        metadata_size = _FRAGMENT_HEADER.size * record_count
        if (
            metadata_size < _COMPACT_RECORD_HEADER.size + _SCHEDULE_BATCH.size
            or metadata_size > len(payload) - cursor
        ):
            raise NativeUnitSpoolError("native-unit compact schedule is truncated")
        schedule_start = cursor + _COMPACT_RECORD_HEADER.size
        schedule_end = schedule_start + _SCHEDULE_BATCH.size
        schedule_prefix = payload[schedule_start:schedule_end]
        unpacked = _SCHEDULE_BATCH.unpack(schedule_prefix)
        packed_size = unpacked[1]
        record_end = cursor + metadata_size + packed_size
        if record_end > len(payload) or any(payload[schedule_end : cursor + metadata_size]):
            raise NativeUnitSpoolError("native-unit compact schedule padding is invalid")
        batch = _decode_schedule_batch(
            schedule_prefix + payload[cursor + metadata_size : record_end]
        )
        if (
            batch.record_count != record_count
            or batch.apply_call_id != descriptor.apply_call_id
            or batch.seq_id != descriptor.seq_id
            or batch.stream_index != descriptor.stream_index
            or batch.writer_monotonic_ns != descriptor.writer_monotonic_ns
            or (
                prior_index is not None
                and batch.first_fragment_index != prior_index + 1
            )
        ):
            raise NativeUnitSpoolError("native-unit compact schedule identity conflicts")
        yield batch
        prior_index = (
            batch.first_fragment_index + batch.record_count - 1
        )
        spans.add(
            (
                batch.first_token_position,
                batch.first_cell_index,
                batch.token_count,
            )
        )
        logical_count += batch.record_count
        cursor = record_end
    if logical_count != descriptor.fragment_count:
        raise NativeUnitSpoolError("native-unit fragment count conflicts")
    if _expand_spans(spans) != descriptor.token_cells:
        raise NativeUnitSpoolError("native-unit fragment cell plan conflicts")


def native_unit_copied_byte_count(
    descriptor: NativeCapturedUnitDescriptor,
) -> int:
    """Return only the writer-copied K/V bytes, excluding spool headers."""

    if not isinstance(descriptor, NativeCapturedUnitDescriptor):
        raise NativeUnitSpoolError("native-unit descriptor is invalid")
    copied = (
        descriptor.payload_size
        - _UNIT_HEADER.size
        - descriptor.fragment_count * _FRAGMENT_HEADER.size
    )
    if copied <= 0:
        raise NativeUnitSpoolError("native-unit copied byte count is invalid")
    return copied


def native_unit_schedule_payload_summary(payload: bytes) -> tuple[int, int]:
    batch = _decode_schedule_batch(payload)
    return batch.record_count, batch.payload_size


@dataclass
class _ActiveStream:
    path: Path
    fd: int
    digest: object
    seq_id: int
    writer_monotonic_ns: int
    size: int = 0
    fragment_count: int = 0
    spans: set[tuple[int, int, int]] | None = None
    pending_write: bytearray = field(default_factory=bytearray)


@dataclass
class _ActiveApply:
    streams: dict[int, _ActiveStream]
    next_fragment_index: int = 0
    fragment_count: int = 0
    copied_bytes: int = 0
    transport_kind: str | None = None
    open_bulk_plan: tuple[int, ...] | None = None


class NativeUnitSpool:
    def __init__(
        self,
        *,
        frame_dir: Path,
        runtime_instance_id: str,
        max_active_applies: int = 64,
        page_size: int = 128,
        active_stream_count: int = 4,
    ) -> None:
        if not isinstance(frame_dir, Path):
            raise NativeUnitSpoolError("frame_dir must be a pathlib.Path")
        if not _valid_runtime_id(runtime_instance_id):
            raise NativeUnitSpoolError("runtime_instance_id is invalid")
        if (
            not isinstance(max_active_applies, int)
            or isinstance(max_active_applies, bool)
            or max_active_applies <= 0
        ):
            raise NativeUnitSpoolError("max_active_applies is invalid")
        if (
            not isinstance(page_size, int)
            or isinstance(page_size, bool)
            or page_size <= 0
            or page_size > 1024
        ):
            raise NativeUnitSpoolError("page_size is invalid")
        if active_stream_count not in _SUPPORTED_STREAM_COUNTS:
            raise NativeUnitSpoolError("active_stream_count is invalid")
        _secure_directory(frame_dir, create=False)
        self.root = frame_dir / "native-units"
        _secure_directory(self.root, create=False)
        self.runtime_instance_id = runtime_instance_id
        self.max_active_applies = max_active_applies
        self.page_size = page_size
        self.active_stream_count = active_stream_count
        self._active: dict[int, _ActiveApply] = {}
        self._sealed_apply_ids: set[int] = set()
        self._descriptors: list[NativeCapturedUnitDescriptor] = []
        self._descriptor_by_id: dict[str, NativeCapturedUnitDescriptor] = {}
        self._session_frame_count = 0
        self._lock = threading.RLock()
        if self.root.exists():
            self._recover()

    def consume_fragment(self, fragment: NativeUnitFragment) -> None:
        with self._lock:
            self._consume_fragment(fragment)

    def _consume_fragment(self, fragment: NativeUnitFragment) -> None:
        _validate_fragment(fragment, self.active_stream_count)
        self._consume_fragment_record(
            _encode_fragment(fragment),
            (
                fragment.apply_call_id,
                fragment.fragment_index,
                fragment.seq_id,
                fragment.stream_index,
                1 if fragment.tensor_kind == "k" else 2,
                fragment.token_position,
                fragment.cell_index,
                fragment.layer,
                len(fragment.payload),
                fragment.tensor_byte_offset,
                fragment.writer_monotonic_ns,
                fragment.span_cell_count,
            ),
        )

    def consume_fragment_payload(self, payload: bytes) -> None:
        with self._lock:
            self._consume_fragment_payload(payload)

    def consume_fragment_payloads(self, payloads: tuple[bytes, ...]) -> None:
        if not isinstance(payloads, tuple) or not payloads:
            raise NativeUnitSpoolError("native-unit fragment batch is invalid")
        with self._lock:
            for payload in payloads:
                self._consume_fragment_payload(payload)

    def consume_bulk_payload(self, payload: bytes) -> tuple[int, int]:
        with self._lock:
            return self._consume_bulk_payload(payload)

    def consume_bulk_payloads(
        self,
        payloads: tuple[bytes, ...],
    ) -> tuple[int, int]:
        if not isinstance(payloads, tuple) or not payloads:
            raise NativeUnitSpoolError("native-unit bulk batch is invalid")
        logical_fragments = 0
        copied_bytes = 0
        with self._lock:
            for payload in payloads:
                fragments, copied = self._consume_bulk_payload(payload)
                logical_fragments += fragments
                copied_bytes += copied
        return logical_fragments, copied_bytes

    def _consume_bulk_payload(self, payload: bytes) -> tuple[int, int]:
        try:
            frame = decode_native_bulk_payload(payload)
        except NativeBulkDecodeError as error:
            raise NativeUnitSpoolError("native-unit bulk frame is invalid") from error
        if frame.stream_index >= self.active_stream_count:
            raise NativeUnitSpoolError("native-unit bulk stream is invalid")
        if frame.apply_call_id in self._sealed_apply_ids:
            raise NativeUnitSpoolError("apply is already sealed")
        active = self._active.get(frame.apply_call_id)
        if active is None:
            if len(self._active) >= self.max_active_applies:
                raise NativeUnitSpoolError("BUFFER_EXHAUSTED")
            active = _ActiveApply(streams={})
            self._active[frame.apply_call_id] = active
        if active.transport_kind not in {None, "bulk"}:
            self._abort_apply(frame.apply_call_id)
            raise NativeUnitSpoolError(
                "mixed legacy and bulk records are forbidden"
            )
        active.transport_kind = "bulk"
        if frame.frame_first_fragment_index < active.next_fragment_index:
            raise NativeUnitSpoolError("fragment index is duplicated")
        if frame.frame_first_fragment_index != active.next_fragment_index:
            raise NativeUnitSpoolError("fragment range is not contiguous")
        plan_identity = _bulk_plan_identity(frame)
        if frame.flags & BULK_FLAG_FIRST:
            if active.open_bulk_plan is not None:
                self._abort_apply(frame.apply_call_id)
                raise NativeUnitSpoolError("bulk plan range overlaps")
            active.open_bulk_plan = plan_identity
        elif active.open_bulk_plan != plan_identity:
            self._abort_apply(frame.apply_call_id)
            raise NativeUnitSpoolError("bulk plan continuation conflicts")
        stream = active.streams.get(frame.stream_index)
        if stream is None:
            stream = self._new_stream_values(
                apply_call_id=frame.apply_call_id,
                seq_id=frame.seq_id,
                stream_index=frame.stream_index,
                writer_monotonic_ns=frame.writer_monotonic_ns,
            )
            active.streams[frame.stream_index] = stream
        if (
            stream.seq_id != frame.seq_id
            or stream.writer_monotonic_ns != frame.writer_monotonic_ns
        ):
            self._abort_apply(frame.apply_call_id)
            raise NativeUnitSpoolError("fragment lifecycle conflicts")
        metadata_size = _FRAGMENT_HEADER.size * frame.frame_fragment_count
        compact_metadata_size = (
            _COMPACT_RECORD_HEADER.size + _COMPACT_BULK.size
        )
        padding_size = metadata_size - compact_metadata_size
        if padding_size < 0:
            self._abort_apply(frame.apply_call_id)
            raise NativeUnitSpoolError("native-unit compact bulk is invalid")
        compact_header = (
            _COMPACT_RECORD_HEADER.pack(
                _COMPACT_BULK_TAG,
                frame.frame_fragment_count,
            )
            + _COMPACT_BULK.pack(
                frame.first_token_position,
                frame.first_cell_index,
                frame.first_fragment_index,
                frame.frame_first_fragment_index,
                frame.tensor_kind,
                frame.layer,
                frame.token_count,
                frame.cell_capacity,
                frame.element_size,
                frame.embedding_width,
                frame.transposed_value,
            )
            + bytes(padding_size)
        )
        try:
            stream.pending_write.extend(compact_header)
            stream.pending_write.extend(frame.payload)
            if len(stream.pending_write) >= _WRITE_BUFFER_BYTES:
                _flush_stream(stream)
        except OSError as error:
            self._abort_apply(frame.apply_call_id)
            raise NativeUnitSpoolError("PERSISTENCE_FAILED") from error
        stream.size += len(compact_header) + frame.payload_size
        stream.fragment_count += frame.frame_fragment_count
        assert stream.spans is not None
        stream.spans.add(
            (
                frame.first_token_position,
                frame.first_cell_index,
                frame.token_count,
            )
        )
        active.next_fragment_index += frame.frame_fragment_count
        active.fragment_count += frame.frame_fragment_count
        active.copied_bytes += frame.payload_size
        self._session_frame_count += frame.frame_fragment_count
        if frame.flags & BULK_FLAG_FINAL:
            active.open_bulk_plan = None
        return frame.frame_fragment_count, frame.payload_size

    def consume_schedule_payload(self, payload: bytes) -> tuple[int, int]:
        with self._lock:
            return self._consume_schedule_payload(payload)

    def consume_schedule_payloads(
        self,
        payloads: tuple[bytes, ...],
    ) -> tuple[int, int]:
        if not isinstance(payloads, tuple) or not payloads:
            raise NativeUnitSpoolError("native-unit schedule batch is invalid")
        logical_fragments = 0
        copied_bytes = 0
        with self._lock:
            for payload in payloads:
                fragments, copied = self._consume_schedule_payload(payload)
                logical_fragments += fragments
                copied_bytes += copied
        return logical_fragments, copied_bytes

    def _consume_schedule_payload(self, payload: bytes) -> tuple[int, int]:
        batch = _decode_schedule_batch(payload)
        if batch.stream_index >= self.active_stream_count:
            raise NativeUnitSpoolError("native-unit schedule stream is invalid")
        if batch.record_count == 1:
            record = next(_iter_schedule_batch_records(batch))
            self._consume_fragment_payload(
                _FRAGMENT_HEADER.pack(
                    batch.apply_call_id,
                    record.fragment_index,
                    batch.seq_id,
                    batch.stream_index,
                    2,
                    record.token_position,
                    record.cell_index,
                    record.layer,
                    len(record.payload),
                    record.tensor_byte_offset,
                    batch.writer_monotonic_ns,
                    record.span_cell_count,
                )
                + record.payload
            )
            return 1, batch.payload_size
        if batch.apply_call_id in self._sealed_apply_ids:
            raise NativeUnitSpoolError("apply is already sealed")
        active = self._active.get(batch.apply_call_id)
        if active is None:
            if len(self._active) >= self.max_active_applies:
                raise NativeUnitSpoolError("BUFFER_EXHAUSTED")
            active = _ActiveApply(streams={})
            self._active[batch.apply_call_id] = active
        if active.transport_kind not in {None, "legacy"}:
            self._abort_apply(batch.apply_call_id)
            raise NativeUnitSpoolError(
                "mixed legacy and bulk records are forbidden"
            )
        active.transport_kind = "legacy"
        if batch.first_fragment_index != active.next_fragment_index:
            raise NativeUnitSpoolError("fragment range is not contiguous")
        stream = active.streams.get(batch.stream_index)
        if stream is None:
            stream = self._new_stream_values(
                apply_call_id=batch.apply_call_id,
                seq_id=batch.seq_id,
                stream_index=batch.stream_index,
                writer_monotonic_ns=batch.writer_monotonic_ns,
            )
            active.streams[batch.stream_index] = stream
        if (
            stream.seq_id != batch.seq_id
            or stream.writer_monotonic_ns != batch.writer_monotonic_ns
        ):
            self._abort_apply(batch.apply_call_id)
            raise NativeUnitSpoolError("fragment lifecycle conflicts")
        metadata_size = _FRAGMENT_HEADER.size * batch.record_count
        padding_size = (
            metadata_size
            - _COMPACT_RECORD_HEADER.size
            - _SCHEDULE_BATCH.size
        )
        if padding_size < 0:
            raise NativeUnitSpoolError("native-unit compact schedule is invalid")
        compact_record = (
            _COMPACT_RECORD_HEADER.pack(
                _COMPACT_SCHEDULE_TAG,
                batch.record_count,
            )
            + payload[: _SCHEDULE_BATCH.size]
            + bytes(padding_size)
            + batch.packed_payload
        )
        try:
            stream.pending_write.extend(compact_record)
            if len(stream.pending_write) >= _WRITE_BUFFER_BYTES:
                _flush_stream(stream)
        except OSError as error:
            self._abort_apply(batch.apply_call_id)
            raise NativeUnitSpoolError("PERSISTENCE_FAILED") from error
        stream.size += len(compact_record)
        stream.fragment_count += batch.record_count
        assert stream.spans is not None
        stream.spans.add(
            (
                batch.first_token_position,
                batch.first_cell_index,
                batch.token_count,
            )
        )
        active.next_fragment_index += batch.record_count
        active.fragment_count += batch.record_count
        active.copied_bytes += batch.payload_size
        self._session_frame_count += batch.record_count
        return batch.record_count, batch.payload_size

    def _consume_fragment_payload(self, payload: bytes) -> None:
        if (
            not isinstance(payload, bytes)
            or len(payload) <= _FRAGMENT_HEADER.size
        ):
            raise NativeUnitSpoolError("native-unit fragment is invalid")
        values = _FRAGMENT_HEADER.unpack_from(payload)
        if (
            values[0] == 0
            or values[3] >= self.active_stream_count
            or values[4] not in {1, 2}
            or values[8] == 0
            or values[8] > 4096
            or values[8] != len(payload) - _FRAGMENT_HEADER.size
            or values[11] == 0
        ):
            raise NativeUnitSpoolError("native-unit fragment is invalid")
        self._consume_fragment_record(payload, values)

    def _consume_fragment_record(
        self,
        record: bytes,
        values: tuple[int, ...],
    ) -> None:
        (
            apply_call_id,
            fragment_index,
            seq_id,
            stream_index,
            _tensor_kind,
            token_position,
            cell_index,
            _layer,
            payload_size,
            _tensor_byte_offset,
            writer_monotonic_ns,
            span_cell_count,
        ) = values
        if apply_call_id in self._sealed_apply_ids:
            raise NativeUnitSpoolError("apply is already sealed")
        active = self._active.get(apply_call_id)
        if active is None:
            if len(self._active) >= self.max_active_applies:
                raise NativeUnitSpoolError("BUFFER_EXHAUSTED")
            active = _ActiveApply(streams={})
            self._active[apply_call_id] = active
        if active.transport_kind not in {None, "legacy"}:
            self._abort_apply(apply_call_id)
            raise NativeUnitSpoolError(
                "mixed legacy and bulk records are forbidden"
            )
        active.transport_kind = "legacy"
        if fragment_index < active.next_fragment_index:
            raise NativeUnitSpoolError("fragment index is duplicated")
        if fragment_index != active.next_fragment_index:
            raise NativeUnitSpoolError("fragment range is not contiguous")
        stream = active.streams.get(stream_index)
        if stream is None:
            stream = self._new_stream_values(
                apply_call_id=apply_call_id,
                seq_id=seq_id,
                stream_index=stream_index,
                writer_monotonic_ns=writer_monotonic_ns,
            )
            active.streams[stream_index] = stream
        if (
            stream.seq_id != seq_id
            or stream.writer_monotonic_ns != writer_monotonic_ns
        ):
            self._abort_apply(apply_call_id)
            raise NativeUnitSpoolError("fragment lifecycle conflicts")
        try:
            compact_record = (
                _COMPACT_RECORD_HEADER.pack(_COMPACT_FRAGMENT_TAG, 1)
                + record[8:]
            )
            stream.pending_write.extend(compact_record)
            if len(stream.pending_write) >= _WRITE_BUFFER_BYTES:
                _flush_stream(stream)
        except OSError as error:
            self._abort_apply(apply_call_id)
            raise NativeUnitSpoolError("PERSISTENCE_FAILED") from error
        stream.size += len(compact_record)
        stream.fragment_count += 1
        assert stream.spans is not None
        stream.spans.add(
            (
                token_position,
                cell_index,
                span_cell_count,
            )
        )
        active.next_fragment_index += 1
        active.fragment_count += 1
        active.copied_bytes += payload_size
        self._session_frame_count += 1

    def consume_seal(
        self,
        seal: NativeUnitSeal,
    ) -> tuple[NativeCapturedUnitDescriptor, ...]:
        with self._lock:
            return self._consume_seal(seal)

    def _consume_seal(
        self,
        seal: NativeUnitSeal,
    ) -> tuple[NativeCapturedUnitDescriptor, ...]:
        _validate_seal(seal, self.active_stream_count)
        if seal.apply_call_id in self._sealed_apply_ids:
            raise NativeUnitSpoolError("apply is already sealed")
        active = self._active.get(seal.apply_call_id)
        if active is None:
            raise NativeUnitSpoolError("state-unit fragment set is incomplete")
        if active.open_bulk_plan is not None:
            self._abort_apply(seal.apply_call_id)
            raise NativeUnitSpoolError("state-unit fragment set is incomplete")
        if (
            active.fragment_count != seal.expected_fragment_count
            or active.next_fragment_index != seal.expected_fragment_count
        ):
            self._abort_apply(seal.apply_call_id)
            raise NativeUnitSpoolError("state-unit fragment set is incomplete")
        if active.copied_bytes != seal.copy_byte_count:
            self._abort_apply(seal.apply_call_id)
            raise NativeUnitSpoolError("state-unit copied byte count conflicts")
        observed_mask = 0
        token_cells_by_stream: dict[int, tuple[tuple[int, int], ...]] = {}
        for stream_index, stream in active.streams.items():
            observed_mask |= 1 << stream_index
            assert stream.spans is not None
            token_cells_by_stream[stream_index] = _expand_spans(stream.spans)
        if observed_mask != seal.stream_mask:
            self._abort_apply(seal.apply_call_id)
            raise NativeUnitSpoolError("state-unit stream mask conflicts")
        if sum(map(len, token_cells_by_stream.values())) != (
            seal.expected_token_cell_count
        ):
            self._abort_apply(seal.apply_call_id)
            raise NativeUnitSpoolError("state-unit token-cell count conflicts")

        first_cursor = (
            self._descriptors[-1].cursor + 1 if self._descriptors else 1
        )
        pending: list[tuple[NativeCapturedUnitDescriptor, Path, Path]] = []
        try:
            for ordinal, stream_index in enumerate(sorted(active.streams)):
                stream = active.streams[stream_index]
                _flush_stream(stream)
                os.fsync(stream.fd)
                os.close(stream.fd)
                stream.fd = -1
                payload_sha256 = stream.digest.hexdigest()
                identity = {
                    "runtime_instance_id": self.runtime_instance_id,
                    "apply_call_id": seal.apply_call_id,
                    "seq_id": stream.seq_id,
                    "stream_index": stream_index,
                    "writer_monotonic_ns": stream.writer_monotonic_ns,
                    "token_cells": token_cells_by_stream[stream_index],
                    "fragment_count": stream.fragment_count,
                    "payload_size": stream.size,
                    "payload_sha256": payload_sha256,
                }
                unit_id = "unit_" + hashlib.sha256(
                    _canonical_json(identity)
                ).hexdigest()
                descriptor = NativeCapturedUnitDescriptor(
                    cursor=first_cursor + ordinal,
                    unit_id=unit_id,
                    runtime_instance_id=self.runtime_instance_id,
                    apply_call_id=seal.apply_call_id,
                    seq_id=stream.seq_id,
                    stream_index=stream_index,
                    writer_monotonic_ns=stream.writer_monotonic_ns,
                    token_cells=token_cells_by_stream[stream_index],
                    fragment_count=stream.fragment_count,
                    payload_size=stream.size,
                    payload_sha256=payload_sha256,
                )
                final_path = self.root / f"{unit_id}.unit"
                if final_path.exists() or final_path.is_symlink():
                    raise NativeUnitSpoolError("immutable unit already exists")
                os.replace(stream.path, final_path)
                pending.append((descriptor, stream.path, final_path))

            manifest = {
                "schema_version": "1.0",
                "runtime_instance_id": self.runtime_instance_id,
                "descriptors": [_descriptor_document(item[0]) for item in pending],
            }
            manifest_bytes = _canonical_json(manifest) + b"\n"
            manifest_name = (
                f"group_{seal.apply_call_id:020d}_"
                f"{hashlib.sha256(manifest_bytes).hexdigest()[:16]}.json"
            )
            _publish_manifest(self.root, manifest_name, manifest_bytes)
            _fsync_directory(self.root)
        except (OSError, NativeUnitSpoolError) as error:
            self._abort_apply(seal.apply_call_id)
            for _descriptor, _partial_path, final_path in pending:
                try:
                    final_path.unlink(missing_ok=True)
                except OSError:
                    pass
            if isinstance(error, NativeUnitSpoolError):
                raise
            raise NativeUnitSpoolError("PERSISTENCE_FAILED") from error

        descriptors = tuple(item[0] for item in pending)
        self._descriptors.extend(descriptors)
        self._descriptor_by_id.update(
            (descriptor.unit_id, descriptor) for descriptor in descriptors
        )
        self._sealed_apply_ids.add(seal.apply_call_id)
        self._active.pop(seal.apply_call_id, None)
        self._session_frame_count += 1
        return descriptors

    def completed(self, *, after_cursor: int) -> CompletedNativeUnitPage:
        with self._lock:
            return self._completed(after_cursor=after_cursor)

    def _completed(self, *, after_cursor: int) -> CompletedNativeUnitPage:
        if (
            not isinstance(after_cursor, int)
            or isinstance(after_cursor, bool)
            or after_cursor < 0
        ):
            raise NativeUnitSpoolError("after_cursor is invalid")
        remaining = tuple(
            item for item in self._descriptors if item.cursor > after_cursor
        )
        selected = remaining[: self.page_size]
        next_cursor = selected[-1].cursor if selected else after_cursor
        return CompletedNativeUnitPage(
            next_cursor=next_cursor,
            truncated=len(remaining) > len(selected),
            units=selected,
        )

    def read(self, *, unit_id: str, offset: int, size: int) -> bytes:
        with self._lock:
            return self._read(unit_id=unit_id, offset=offset, size=size)

    def _read(self, *, unit_id: str, offset: int, size: int) -> bytes:
        if not isinstance(unit_id, str) or not _UNIT_ID.fullmatch(unit_id):
            raise NativeUnitSpoolError("unit_id is invalid")
        if (
            not isinstance(offset, int)
            or isinstance(offset, bool)
            or offset < 0
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size <= 0
        ):
            raise NativeUnitSpoolError("native-unit byte range is invalid")
        if size > _MAX_READ_BYTES:
            raise NativeUnitSpoolError("native-unit reads are capped at 1 MiB")
        descriptor = self._descriptor_by_id.get(unit_id)
        if descriptor is None:
            raise NativeUnitSpoolError("native unit is unknown")
        if offset > descriptor.payload_size - size:
            raise NativeUnitSpoolError("native-unit byte range exceeds payload")
        path = self.root / f"{unit_id}.unit"
        fd = _open_regular(path)
        try:
            os.lseek(fd, offset, os.SEEK_SET)
            payload = _read_exact(fd, size)
        finally:
            os.close(fd)
        return payload

    def descriptor(self, unit_id: str) -> NativeCapturedUnitDescriptor:
        with self._lock:
            if not isinstance(unit_id, str) or not _UNIT_ID.fullmatch(unit_id):
                raise NativeUnitSpoolError("unit_id is invalid")
            descriptor = self._descriptor_by_id.get(unit_id)
            if descriptor is None:
                raise NativeUnitSpoolError("native unit is unknown")
            return descriptor

    @property
    def session_frame_count(self) -> int:
        with self._lock:
            return self._session_frame_count

    @property
    def active_apply_count(self) -> int:
        with self._lock:
            return len(self._active)

    @property
    def published_frame_count(self) -> int:
        with self._lock:
            return sum(
                descriptor.fragment_count for descriptor in self._descriptors
            ) + len(self._sealed_apply_ids)

    def _new_stream(self, fragment: NativeUnitFragment) -> _ActiveStream:
        return self._new_stream_values(
            apply_call_id=fragment.apply_call_id,
            seq_id=fragment.seq_id,
            stream_index=fragment.stream_index,
            writer_monotonic_ns=fragment.writer_monotonic_ns,
        )

    def _new_stream_values(
        self,
        *,
        apply_call_id: int,
        seq_id: int,
        stream_index: int,
        writer_monotonic_ns: int,
    ) -> _ActiveStream:
        _secure_directory(self.root.parent)
        _secure_directory(self.root)
        fd, raw_path = tempfile.mkstemp(
            prefix=f".apply-{apply_call_id:020d}-s{stream_index}-",
            suffix=".partial",
            dir=self.root,
        )
        path = Path(raw_path)
        try:
            os.chmod(path, 0o600)
            header = _UNIT_HEADER.pack(_UNIT_MAGIC, _UNIT_VERSION, 0)
            _write_all(fd, header)
        except OSError:
            os.close(fd)
            path.unlink(missing_ok=True)
            raise
        digest = hashlib.sha256()
        digest.update(header)
        return _ActiveStream(
            path=path,
            fd=fd,
            digest=digest,
            seq_id=seq_id,
            writer_monotonic_ns=writer_monotonic_ns,
            size=len(header),
            spans=set(),
        )

    def _abort_apply(self, apply_call_id: int) -> None:
        active = self._active.pop(apply_call_id, None)
        if active is None:
            return
        for stream in active.streams.values():
            if stream.fd >= 0:
                try:
                    os.close(stream.fd)
                except OSError:
                    pass
            try:
                stream.path.unlink(missing_ok=True)
            except OSError:
                pass

    def _recover(self) -> None:
        recovered: list[NativeCapturedUnitDescriptor] = []
        for path in sorted(self.root.glob("group_*.json")):
            if not _GROUP_NAME.fullmatch(path.name):
                continue
            payload = _read_regular_bytes(path, 8 * 1024 * 1024)
            try:
                document = json.loads(payload)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise NativeUnitSpoolError("native-unit manifest is invalid") from error
            if payload != _canonical_json(document) + b"\n":
                raise NativeUnitSpoolError("native-unit manifest is not canonical")
            if (
                not isinstance(document, dict)
                or set(document)
                != {"schema_version", "runtime_instance_id", "descriptors"}
                or document["schema_version"] != "1.0"
                or document["runtime_instance_id"] != self.runtime_instance_id
                or not isinstance(document["descriptors"], list)
                or not document["descriptors"]
            ):
                raise NativeUnitSpoolError("native-unit manifest contract is invalid")
            for item in document["descriptors"]:
                descriptor = _descriptor_from_document(item)
                if descriptor.stream_index >= self.active_stream_count:
                    raise NativeUnitSpoolError(
                        "native-unit descriptor exceeds the active stream count"
                    )
                unit_path = self.root / f"{descriptor.unit_id}.unit"
                unit_payload = _read_regular_bytes(
                    unit_path, descriptor.payload_size
                )
                if (
                    len(unit_payload) != descriptor.payload_size
                    or hashlib.sha256(unit_payload).hexdigest()
                    != descriptor.payload_sha256
                ):
                    raise NativeUnitSpoolError("native-unit evidence digest conflicts")
                recovered.append(descriptor)
        recovered.sort(key=lambda item: item.cursor)
        if len({item.cursor for item in recovered}) != len(recovered):
            raise NativeUnitSpoolError("native-unit cursor is duplicated")
        if len({item.unit_id for item in recovered}) != len(recovered):
            raise NativeUnitSpoolError("native-unit identity is duplicated")
        self._descriptors = recovered
        self._descriptor_by_id = {item.unit_id: item for item in recovered}
        self._sealed_apply_ids = {item.apply_call_id for item in recovered}


def _valid_runtime_id(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(
        r"[A-Za-z0-9._:-]{1,256}", value
    ) is not None


def _validate_fragment(fragment: object, active_stream_count: int) -> None:
    required = (
        "apply_call_id",
        "fragment_index",
        "seq_id",
        "stream_index",
        "token_position",
        "cell_index",
        "tensor_kind",
        "layer",
        "span_cell_count",
        "tensor_byte_offset",
        "writer_monotonic_ns",
        "payload",
    )
    if any(not hasattr(fragment, name) for name in required):
        raise NativeUnitSpoolError("native-unit fragment is invalid")
    for name in required:
        if name in {"tensor_kind", "payload"}:
            continue
        value = getattr(fragment, name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise NativeUnitSpoolError("native-unit fragment is invalid")
    if (
        fragment.apply_call_id == 0
        or fragment.stream_index >= active_stream_count
        or fragment.tensor_kind not in {"k", "v"}
        or fragment.span_cell_count == 0
        or not isinstance(fragment.payload, bytes)
        or not fragment.payload
        or len(fragment.payload) > 4096
    ):
        raise NativeUnitSpoolError("native-unit fragment is invalid")


def _validate_seal(seal: object, active_stream_count: int) -> None:
    required = (
        "apply_call_id",
        "expected_fragment_count",
        "expected_token_cell_count",
        "stream_mask",
        "writer_state",
        "copy_byte_count",
    )
    if any(not hasattr(seal, name) for name in required):
        raise NativeUnitSpoolError("native-unit seal is invalid")
    if (
        seal.writer_state != "COMPLETE"
        or any(
            not isinstance(getattr(seal, name), int)
            or isinstance(getattr(seal, name), bool)
            or getattr(seal, name) <= 0
            for name in (
                "apply_call_id",
                "expected_fragment_count",
                "expected_token_cell_count",
                "stream_mask",
                "copy_byte_count",
            )
        )
        or seal.stream_mask & ~((1 << active_stream_count) - 1)
    ):
        raise NativeUnitSpoolError("native-unit seal is invalid")


def _encode_fragment(fragment: NativeUnitFragment) -> bytes:
    header = _FRAGMENT_HEADER.pack(
        fragment.apply_call_id,
        fragment.fragment_index,
        fragment.seq_id,
        fragment.stream_index,
        1 if fragment.tensor_kind == "k" else 2,
        fragment.token_position,
        fragment.cell_index,
        fragment.layer,
        len(fragment.payload),
        fragment.tensor_byte_offset,
        fragment.writer_monotonic_ns,
        fragment.span_cell_count,
    )
    return header + fragment.payload


def _expand_spans(
    spans: set[tuple[int, int, int]],
) -> tuple[tuple[int, int], ...]:
    expanded: set[tuple[int, int]] = set()
    for token_position, cell_index, count in spans:
        for offset in range(count):
            expanded.add((token_position + offset, cell_index + offset))
    return tuple(sorted(expanded))


def _descriptor_document(
    descriptor: NativeCapturedUnitDescriptor,
) -> dict[str, object]:
    document = asdict(descriptor)
    document["token_cells"] = [list(item) for item in descriptor.token_cells]
    return document


def _descriptor_from_document(document: object) -> NativeCapturedUnitDescriptor:
    expected = {
        "cursor",
        "unit_id",
        "runtime_instance_id",
        "apply_call_id",
        "seq_id",
        "stream_index",
        "writer_monotonic_ns",
        "token_cells",
        "fragment_count",
        "payload_size",
        "payload_sha256",
    }
    if not isinstance(document, dict) or set(document) != expected:
        raise NativeUnitSpoolError("native-unit descriptor is invalid")
    token_cells = document["token_cells"]
    if not isinstance(token_cells, list):
        raise NativeUnitSpoolError("native-unit descriptor token cells are invalid")
    return NativeCapturedUnitDescriptor(
        **{
            **document,
            "token_cells": tuple(
                tuple(item) if isinstance(item, list) else item
                for item in token_cells
            ),
        }
    )


def _canonical_json(document: object) -> bytes:
    return json.dumps(
        document,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _secure_directory(path: Path, *, create: bool = True) -> None:
    if path.is_symlink():
        raise NativeUnitSpoolError("native-unit spool path must not be a symlink")
    if not path.exists():
        if not create:
            return
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
    metadata = path.lstat()
    if not stat.S_ISDIR(metadata.st_mode):
        raise NativeUnitSpoolError("native-unit spool path is not a directory")
    try:
        os.chmod(path, 0o700)
    except OSError as error:
        raise NativeUnitSpoolError("native-unit spool permissions failed") from error


def _write_all(fd: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError("short native-unit write")
        view = view[written:]


def _flush_stream(stream: _ActiveStream) -> None:
    if not stream.pending_write:
        return
    _write_all(stream.fd, stream.pending_write)
    stream.digest.update(stream.pending_write)
    stream.pending_write.clear()


def _read_exact(fd: int, size: int) -> bytes:
    result = bytearray()
    while len(result) < size:
        chunk = os.read(fd, size - len(result))
        if not chunk:
            raise NativeUnitSpoolError("native-unit evidence ended early")
        result.extend(chunk)
    return bytes(result)


def _open_regular(path: Path) -> int:
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
    except OSError as error:
        raise NativeUnitSpoolError("native-unit evidence is unavailable") from error
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise NativeUnitSpoolError("native-unit evidence is not a regular file")
    except BaseException:
        os.close(fd)
        raise
    return fd


def _read_regular_bytes(path: Path, limit: int) -> bytes:
    fd = _open_regular(path)
    try:
        size = os.fstat(fd).st_size
        if size < 0 or size > limit:
            raise NativeUnitSpoolError("native-unit evidence size is invalid")
        return _read_exact(fd, size)
    finally:
        os.close(fd)


def _publish_manifest(root: Path, name: str, payload: bytes) -> None:
    if not _GROUP_NAME.fullmatch(name):
        raise NativeUnitSpoolError("native-unit manifest name is invalid")
    destination = root / name
    if destination.exists() or destination.is_symlink():
        raise NativeUnitSpoolError("native-unit manifest already exists")
    fd, raw_path = tempfile.mkstemp(
        prefix=".group-", suffix=".partial", dir=root
    )
    temporary = Path(raw_path)
    try:
        os.chmod(temporary, 0o600)
        _write_all(fd, payload)
        os.fsync(fd)
        os.close(fd)
        fd = -1
        os.replace(temporary, destination)
    finally:
        if fd >= 0:
            os.close(fd)
        temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


__all__ = [
    "CompletedNativeUnitPage",
    "NativeCapturedUnitDescriptor",
    "NativeUnitFragmentRecord",
    "NativeUnitSpool",
    "NativeUnitSpoolError",
    "NativeUnitScheduleRecord",
    "iter_native_unit_payload",
    "iter_native_unit_records",
    "native_unit_schedule_payload_summary",
    "parse_native_unit_payload",
    "native_unit_copied_byte_count",
]
