from __future__ import annotations

import struct
from dataclasses import dataclass


DATA_MAGIC = 0x444F524C
CONTROL_MAGIC = 0x434F524C
PROTOCOL_VERSION = 1

FRAME_DECODE_EVENT = 1
FRAME_TERMINAL_TRAILER = 2

CONTROL_ATTACHED = 1
CONTROL_STOP = 2
CONTROL_DRAIN_READY = 3
CONTROL_DETACH = 4
CONTROL_DETACHED = 5

DATA_HEADER_SIZE = 24
DECODE_EVENT_SIZE = 56
TERMINAL_TRAILER_SIZE = 64
CONTROL_SIZE = 16
MAX_DATA_FRAME_SIZE = 88
INT32_MAX = 2**31 - 1

_DATA_HEADER = struct.Struct("<IHHIIQ")
_DECODE_EVENT = struct.Struct("<QIIQQIIQiI")
_TERMINAL_TRAILER = struct.Struct("<QQQQQQQQ")
_CONTROL = struct.Struct("<IHHIi")


class WireContractError(ValueError):
    """Raised when bytes or values violate observer wire v1."""


@dataclass(frozen=True)
class DecodeEvent:
    call_id: int
    tgid: int
    tid: int
    entry_monotonic_ns: int
    return_monotonic_ns: int
    n_tokens: int
    context_ordinal: int
    sequence_bitmap: int
    decode_return_code: int
    anomaly_flags: int


@dataclass(frozen=True)
class TerminalTrailer:
    entry_attempt_count: int
    return_probe_count: int
    ringbuf_reserve_fail_count: int
    bpf_ringbuf_submit_count: int
    native_ring_received_count: int
    native_wire_written_count: int
    final_inflight_count: int
    anomaly_count: int


@dataclass(frozen=True)
class ControlMessage:
    type: int
    sequence: int
    status: int = 0


@dataclass(frozen=True)
class ParsedDataStream:
    events: tuple[DecodeEvent, ...]
    trailer: TerminalTrailer


def _require_uint(name: str, value: int, bits: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**bits:
        raise WireContractError(f"{name} must be an unsigned {bits}-bit integer")


def _require_int32(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not -(2**31) <= value <= INT32_MAX:
        raise WireContractError(f"{name} must be a signed 32-bit integer")


def validate_controlled_profile_event(event: DecodeEvent) -> None:
    """Validate one decoded event against the current controlled profile."""
    if event.call_id == 0:
        raise WireContractError("call_id must be positive")
    if event.tgid == 0 or event.tgid > INT32_MAX:
        raise WireContractError("tgid must be in 1..INT32_MAX")
    if event.tid == 0 or event.tid > INT32_MAX:
        raise WireContractError("tid must be in 1..INT32_MAX")
    if event.entry_monotonic_ns > event.return_monotonic_ns:
        raise WireContractError("entry monotonic time exceeds return monotonic time")
    if not 1 <= event.n_tokens <= 128:
        raise WireContractError("n_tokens must be in 1..128")
    if event.context_ordinal != 1:
        raise WireContractError("context_ordinal must be 1")
    if event.sequence_bitmap == 0 or event.sequence_bitmap & ~0xFF:
        raise WireContractError("sequence_bitmap must use nonzero slots 0..7")
    if event.sequence_bitmap.bit_count() > event.n_tokens:
        raise WireContractError("sequence_bitmap has more slots than n_tokens")
    if event.anomaly_flags != 0:
        raise WireContractError("anomaly_flags must be zero in v1")


def _validate_event_encoding(event: DecodeEvent) -> None:
    _require_uint("call_id", event.call_id, 64)
    _require_uint("tgid", event.tgid, 32)
    _require_uint("tid", event.tid, 32)
    _require_uint("entry_monotonic_ns", event.entry_monotonic_ns, 64)
    _require_uint("return_monotonic_ns", event.return_monotonic_ns, 64)
    _require_uint("n_tokens", event.n_tokens, 32)
    _require_uint("context_ordinal", event.context_ordinal, 32)
    _require_uint("sequence_bitmap", event.sequence_bitmap, 64)
    _require_int32("decode_return_code", event.decode_return_code)
    _require_uint("anomaly_flags", event.anomaly_flags, 32)


def _validate_trailer_encoding(trailer: TerminalTrailer) -> None:
    for name in trailer.__dataclass_fields__:
        _require_uint(name, getattr(trailer, name), 64)


def _encode_header(frame_type: int, payload_length: int, frame_sequence: int) -> bytes:
    _require_uint("frame_sequence", frame_sequence, 64)
    if frame_sequence == 0:
        raise WireContractError("frame_sequence must be nonzero")
    return _DATA_HEADER.pack(
        DATA_MAGIC,
        PROTOCOL_VERSION,
        frame_type,
        payload_length,
        0,
        frame_sequence,
    )


def encode_decode_event_frame(event: DecodeEvent, frame_sequence: int) -> bytes:
    """Encode one decode event with an explicit little-endian header."""
    _validate_event_encoding(event)
    payload = _DECODE_EVENT.pack(
        event.call_id,
        event.tgid,
        event.tid,
        event.entry_monotonic_ns,
        event.return_monotonic_ns,
        event.n_tokens,
        event.context_ordinal,
        event.sequence_bitmap,
        event.decode_return_code,
        event.anomaly_flags,
    )
    return _encode_header(FRAME_DECODE_EVENT, DECODE_EVENT_SIZE, frame_sequence) + payload


def encode_terminal_trailer_frame(trailer: TerminalTrailer, frame_sequence: int) -> bytes:
    """Encode one terminal trailer without applying evaluator health policy."""
    _validate_trailer_encoding(trailer)
    payload = _TERMINAL_TRAILER.pack(
        trailer.entry_attempt_count,
        trailer.return_probe_count,
        trailer.ringbuf_reserve_fail_count,
        trailer.bpf_ringbuf_submit_count,
        trailer.native_ring_received_count,
        trailer.native_wire_written_count,
        trailer.final_inflight_count,
        trailer.anomaly_count,
    )
    return _encode_header(FRAME_TERMINAL_TRAILER, TERMINAL_TRAILER_SIZE, frame_sequence) + payload


def _decode_header(data: bytes, offset: int) -> tuple[int, int, int]:
    if len(data) - offset < DATA_HEADER_SIZE:
        raise WireContractError("truncated data header")
    magic, version, frame_type, payload_length, reserved, sequence = _DATA_HEADER.unpack_from(data, offset)
    if magic != DATA_MAGIC:
        raise WireContractError("bad data magic")
    if version != PROTOCOL_VERSION:
        raise WireContractError("bad protocol version")
    if frame_type not in (FRAME_DECODE_EVENT, FRAME_TERMINAL_TRAILER):
        raise WireContractError("unknown frame type")
    expected_length = DECODE_EVENT_SIZE if frame_type == FRAME_DECODE_EVENT else TERMINAL_TRAILER_SIZE
    if payload_length != expected_length:
        raise WireContractError("bad payload length")
    if reserved != 0:
        raise WireContractError("reserved field must be zero")
    if sequence == 0:
        raise WireContractError("frame sequence must be nonzero")
    return frame_type, payload_length, sequence


def _decode_event(data: bytes, offset: int) -> DecodeEvent:
    event = DecodeEvent(*_DECODE_EVENT.unpack_from(data, offset))
    validate_controlled_profile_event(event)
    return event


def _decode_trailer(data: bytes, offset: int) -> TerminalTrailer:
    return TerminalTrailer(*_TERMINAL_TRAILER.unpack_from(data, offset))


def parse_data_stream(data: bytes) -> ParsedDataStream:
    """Parse a complete v1 stream and enforce sequencing and terminal rules."""
    if not isinstance(data, bytes):
        raise TypeError("data must be bytes")
    offset = 0
    expected_sequence = 1
    events: list[DecodeEvent] = []
    call_ids: set[int] = set()
    trailer: TerminalTrailer | None = None
    while offset < len(data):
        frame_type, payload_length, sequence = _decode_header(data, offset)
        if sequence != expected_sequence:
            raise WireContractError(
                f"frame sequence mismatch: expected {expected_sequence}, got {sequence}"
            )
        frame_end = offset + DATA_HEADER_SIZE + payload_length
        if frame_end > len(data):
            raise WireContractError("truncated data frame")
        payload_offset = offset + DATA_HEADER_SIZE
        if frame_type == FRAME_DECODE_EVENT:
            if trailer is not None:
                raise WireContractError("event after terminal trailer")
            event = _decode_event(data, payload_offset)
            if event.call_id in call_ids:
                raise WireContractError("duplicate call_id")
            call_ids.add(event.call_id)
            events.append(event)
        else:
            if trailer is not None:
                raise WireContractError("duplicate terminal trailer")
            trailer = _decode_trailer(data, payload_offset)
            if frame_end != len(data):
                raise WireContractError("trailing bytes after terminal trailer")
        offset = frame_end
        expected_sequence = sequence + 1
    if trailer is None:
        raise WireContractError("exactly one terminal trailer is required")
    return ParsedDataStream(tuple(events), trailer)


def encode_control(message: ControlMessage) -> bytes:
    """Encode one v1 control message."""
    if message.type not in {
        CONTROL_ATTACHED,
        CONTROL_STOP,
        CONTROL_DRAIN_READY,
        CONTROL_DETACH,
        CONTROL_DETACHED,
    }:
        raise WireContractError("unknown control type")
    _require_uint("control sequence", message.sequence, 32)
    if message.sequence == 0:
        raise WireContractError("control sequence must be nonzero")
    _require_int32("control status", message.status)
    if message.status != 0:
        raise WireContractError("control status must be zero in v1")
    return _CONTROL.pack(CONTROL_MAGIC, PROTOCOL_VERSION, message.type, message.sequence, message.status)


def decode_control(data: bytes) -> ControlMessage:
    """Decode and structurally validate exactly one v1 control message."""
    if not isinstance(data, bytes):
        raise TypeError("data must be bytes")
    if len(data) != CONTROL_SIZE:
        raise WireContractError("control message must be exactly 16 bytes")
    magic, version, control_type, sequence, status = _CONTROL.unpack(data)
    if magic != CONTROL_MAGIC:
        raise WireContractError("bad control magic")
    if version != PROTOCOL_VERSION:
        raise WireContractError("bad protocol version")
    message = ControlMessage(control_type, sequence, status)
    encode_control(message)
    return message
