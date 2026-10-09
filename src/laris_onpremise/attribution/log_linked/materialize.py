from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
from pathlib import Path
import struct

from laris_onpremise.kv_state.acquisition.payload import inspect_simple_kv_payload

from .contracts import AttributionDecision, parse_state_unit_manifest
from .correlation import AttributionResult
from .generation import GenerationSeal
from .state_units import StoredStateUnit


class MaterializationError(RuntimeError):
    """Raised when a correlated state-unit chain cannot be reconstructed."""


@dataclass(frozen=True)
class MaterializedEndpointState:
    payload: bytes
    payload_digest: str
    source_unit_digests: tuple[str, ...]
    source_seq_id: int
    source_stream_index: int
    positions: tuple[int, ...]
    n_stream: int
    n_layer: int
    v_trans: bool
    key_ggml_type_id: int
    value_ggml_type_id: int

    @property
    def position_range(self) -> tuple[int, int]:
        return self.positions[0], self.positions[-1]

    @property
    def next_position(self) -> int:
        return self.positions[-1] + 1


@dataclass(frozen=True)
class _CellState:
    seq_id: int
    key_rows: tuple[bytes, ...]
    value_rows: tuple[bytes, ...]


@dataclass(frozen=True)
class _DecodedPayload:
    n_stream: int
    stream_index: int
    v_trans: bool
    n_layer: int
    key_types: tuple[int, ...]
    key_row_sizes: tuple[int, ...]
    value_types: tuple[int, ...]
    value_row_sizes: tuple[int, ...]
    value_element_sizes: tuple[int, ...]
    value_embedding_widths: tuple[int, ...]
    cells: dict[int, _CellState]

    @property
    def layout_signature(self) -> tuple[object, ...]:
        return (
            self.n_stream,
            self.stream_index,
            self.v_trans,
            self.n_layer,
            self.key_types,
            self.key_row_sizes,
            self.value_types,
            self.value_row_sizes,
            self.value_element_sizes,
            self.value_embedding_widths,
        )


class EndpointStateMaterializer:
    """Reconstruct one dense Simple-KV sequence at a resolved cutoff."""

    def materialize(
        self,
        *,
        result: AttributionResult,
        generation_seal: GenerationSeal,
        state_units: tuple[StoredStateUnit, ...],
    ) -> MaterializedEndpointState:
        if not isinstance(result, AttributionResult):
            raise MaterializationError("attribution result is invalid")
        if result.final_decision not in {
            AttributionDecision.CERTIFIED,
            AttributionDecision.RANKED,
        }:
            raise MaterializationError(
                "only a resolved attribution can materialize"
            )
        if not isinstance(generation_seal, GenerationSeal) or generation_seal.state != "COMPLETE":
            raise MaterializationError("sequence generation is not complete")
        if not isinstance(state_units, tuple) or not state_units:
            raise MaterializationError("state-unit chain is unavailable")

        by_digest = {unit.manifest_digest: unit for unit in state_units}
        requested_chain = tuple(result.state_unit_chain_digests)
        if len(by_digest) != len(state_units) or any(
            digest not in by_digest for digest in requested_chain
        ):
            raise MaterializationError("state-unit chain is unavailable")
        ordered = tuple(by_digest[digest] for digest in requested_chain)
        if not ordered:
            raise MaterializationError("state-unit chain is empty")

        state: dict[int, _CellState] = {}
        base_digest: str | None = None
        prior_digest: str | None = None
        baseline: _DecodedPayload | None = None
        for index, unit in enumerate(ordered):
            if unit.ref != generation_seal.ref:
                raise MaterializationError("state-unit generation differs")
            manifest_bytes = _read_unit_file(unit.manifest_path)
            observed_manifest_digest = _digest(manifest_bytes)
            if not hmac.compare_digest(observed_manifest_digest, unit.manifest_digest):
                raise MaterializationError("state-unit manifest digest differs")
            try:
                manifest = parse_state_unit_manifest(manifest_bytes)
            except ValueError as error:
                raise MaterializationError("state-unit manifest is invalid") from error
            if manifest["state"] != "COMPLETE":
                raise MaterializationError("state-unit chain contains an incomplete unit")
            if index == 0:
                if manifest["unit_kind"] != "BASE":
                    raise MaterializationError("state-unit chain does not begin with BASE")
                base_digest = unit.manifest_digest
                if manifest["previous_unit_digest"] is not None or manifest["base_unit_digest"] is not None:
                    raise MaterializationError("BASE chain binding is invalid")
            else:
                if (
                    manifest["unit_kind"] != "DELTA"
                    or manifest["previous_unit_digest"] != prior_digest
                    or manifest["base_unit_digest"] != base_digest
                ):
                    raise MaterializationError("state-unit chain order is invalid")
            payload = _read_unit_file(unit.payload_path)
            if (
                len(payload) != manifest["payload_size_bytes"]
                or not hmac.compare_digest(_digest(payload), str(manifest["payload_digest"]))
                or not hmac.compare_digest(_digest(payload), unit.payload_digest)
            ):
                raise MaterializationError("state-unit payload differs")
            decoded = _decode(payload)
            if baseline is None:
                baseline = decoded
            elif decoded.layout_signature != baseline.layout_signature:
                raise MaterializationError("state-unit payload layouts differ")
            if set(decoded.cells) != _expand_ranges(manifest["position_ranges"]):
                raise MaterializationError("state-unit positions differ from manifest")
            if index > 0 and manifest["mutation_kinds"]:
                state = dict(decoded.cells)
            else:
                state.update(decoded.cells)
            prior_digest = unit.manifest_digest

        if baseline is None or result.cutoff_apply_call_id != parse_state_unit_manifest(
            _read_unit_file(ordered[-1].manifest_path)
        )["apply_call_id"]:
            raise MaterializationError("state-unit chain does not reach the attribution cutoff")
        positions = tuple(sorted(state))
        if not positions or positions != tuple(range(positions[-1] + 1)):
            raise MaterializationError("materialized positions are not dense from zero")
        seq_ids = {cell.seq_id for cell in state.values()}
        if len(seq_ids) != 1:
            raise MaterializationError("materialized payload contains multiple sequences")
        payload = _encode(baseline, state)
        summary = inspect_simple_kv_payload(payload)
        if summary.source_seq_ids != tuple(seq_ids) or summary.positions != positions:
            raise MaterializationError("materialized payload verification failed")
        return MaterializedEndpointState(
            payload=payload,
            payload_digest=_digest(payload),
            source_unit_digests=requested_chain,
            source_seq_id=next(iter(seq_ids)),
            source_stream_index=baseline.stream_index,
            positions=positions,
            n_stream=baseline.n_stream,
            n_layer=baseline.n_layer,
            v_trans=baseline.v_trans,
            key_ggml_type_id=baseline.key_types[0],
            value_ggml_type_id=baseline.value_types[0],
        )


class _Cursor:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.offset = 0

    def take(self, size: int) -> bytes:
        if size < 0 or self.offset + size > len(self.payload):
            raise MaterializationError("state-unit payload is truncated")
        value = self.payload[self.offset : self.offset + size]
        self.offset += size
        return value

    def u32(self) -> int:
        return struct.unpack("<I", self.take(4))[0]

    def i32(self) -> int:
        return struct.unpack("<i", self.take(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.take(8))[0]


def _decode(payload: bytes) -> _DecodedPayload:
    try:
        inspect_simple_kv_payload(payload)
    except ValueError as error:
        raise MaterializationError("state-unit payload is invalid") from error
    cursor = _Cursor(payload)
    n_stream = cursor.u32()
    decoded_stream: _DecodedPayload | None = None
    for stream_index in range(n_stream):
        cell_count = cursor.u32()
        if cell_count == 0:
            continue
        if decoded_stream is not None:
            raise MaterializationError("state-unit payload has multiple streams")
        cell_metadata: list[tuple[int, int]] = []
        for _ in range(cell_count):
            position = cursor.i32()
            if cursor.u32() != 1:
                raise MaterializationError("state-unit cell membership is unsupported")
            cell_metadata.append((position, cursor.i32()))
        v_trans = bool(cursor.u32())
        n_layer = cursor.u32()
        key_types: list[int] = []
        key_sizes: list[int] = []
        key_rows: list[list[bytes]] = []
        for _ in range(n_layer):
            key_types.append(cursor.i32())
            row_size = cursor.u64()
            key_sizes.append(row_size)
            key_rows.append([cursor.take(row_size) for _ in range(cell_count)])
        value_types: list[int] = []
        value_sizes: list[int] = []
        element_sizes: list[int] = []
        embedding_widths: list[int] = []
        value_rows: list[list[bytes]] = []
        for _ in range(n_layer):
            value_types.append(cursor.i32())
            if v_trans:
                element_size = cursor.u32()
                embedding_width = cursor.u32()
                row_size = element_size * embedding_width
                elements = [
                    [cursor.take(element_size) for _ in range(cell_count)]
                    for _ in range(embedding_width)
                ]
                rows = [
                    b"".join(
                        elements[element_index][cell_index]
                        for element_index in range(embedding_width)
                    )
                    for cell_index in range(cell_count)
                ]
            else:
                element_size = 0
                embedding_width = 0
                row_size = cursor.u64()
                rows = [cursor.take(row_size) for _ in range(cell_count)]
            element_sizes.append(element_size)
            embedding_widths.append(embedding_width)
            value_sizes.append(row_size)
            value_rows.append(rows)
        cells: dict[int, _CellState] = {}
        for cell_index, (position, seq_id) in enumerate(cell_metadata):
            if position in cells:
                raise MaterializationError("state-unit payload repeats a position")
            cells[position] = _CellState(
                seq_id=seq_id,
                key_rows=tuple(rows[cell_index] for rows in key_rows),
                value_rows=tuple(rows[cell_index] for rows in value_rows),
            )
        decoded_stream = _DecodedPayload(
            n_stream=n_stream,
            stream_index=stream_index,
            v_trans=v_trans,
            n_layer=n_layer,
            key_types=tuple(key_types),
            key_row_sizes=tuple(key_sizes),
            value_types=tuple(value_types),
            value_row_sizes=tuple(value_sizes),
            value_element_sizes=tuple(element_sizes),
            value_embedding_widths=tuple(embedding_widths),
            cells=cells,
        )
    if decoded_stream is None or cursor.offset != len(payload):
        raise MaterializationError("state-unit payload has no selected stream")
    return decoded_stream


def _encode(layout: _DecodedPayload, state: dict[int, _CellState]) -> bytes:
    positions = sorted(state)
    payload = bytearray(struct.pack("<I", layout.n_stream))
    for stream_index in range(layout.n_stream):
        if stream_index != layout.stream_index:
            payload += struct.pack("<I", 0)
            continue
        payload += struct.pack("<I", len(positions))
        for position in positions:
            payload += struct.pack("<iIi", position, 1, state[position].seq_id)
        payload += struct.pack("<II", int(layout.v_trans), layout.n_layer)
        for layer in range(layout.n_layer):
            payload += struct.pack(
                "<iQ", layout.key_types[layer], layout.key_row_sizes[layer]
            )
            payload += b"".join(state[position].key_rows[layer] for position in positions)
        for layer in range(layout.n_layer):
            if layout.v_trans:
                payload += struct.pack(
                    "<iII",
                    layout.value_types[layer],
                    layout.value_element_sizes[layer],
                    layout.value_embedding_widths[layer],
                )
                element_size = layout.value_element_sizes[layer]
                for element_index in range(layout.value_embedding_widths[layer]):
                    start = element_index * element_size
                    end = start + element_size
                    payload += b"".join(
                        state[position].value_rows[layer][start:end]
                        for position in positions
                    )
            else:
                payload += struct.pack(
                    "<iQ", layout.value_types[layer], layout.value_row_sizes[layer]
                )
                payload += b"".join(
                    state[position].value_rows[layer] for position in positions
                )
    return bytes(payload)


def _expand_ranges(value: object) -> set[int]:
    if not isinstance(value, list):
        raise MaterializationError("state-unit position ranges are invalid")
    result: set[int] = set()
    for item in value:
        if not isinstance(item, list) or len(item) != 2:
            raise MaterializationError("state-unit position ranges are invalid")
        result.update(range(int(item[0]), int(item[1]) + 1))
    return result


def _read_unit_file(path: Path) -> bytes:
    try:
        if path.is_symlink() or not path.is_file():
            raise MaterializationError("state-unit file is unavailable")
        return path.read_bytes()
    except OSError as error:
        raise MaterializationError("state-unit file is unavailable") from error


def _digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


__all__ = [
    "EndpointStateMaterializer",
    "MaterializationError",
    "MaterializedEndpointState",
]
