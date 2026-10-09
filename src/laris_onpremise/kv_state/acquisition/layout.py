from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal


class LayoutValidationError(ValueError):
    """Raised when an observed Simple KV layout is unsafe to acquire."""


_SHA256_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class SelectedCell:
    position: int
    cell_index: int
    seq_id: int
    stream_index: int


@dataclass(frozen=True)
class TensorRowLayout:
    kind: Literal["k", "v"]
    layer: int
    stream_index: int
    tensor_address: int
    row_size: int
    cell_capacity: int
    ggml_type_id: int
    element_size: int
    embedding_width: int


@dataclass(frozen=True)
class SimpleKvLayout:
    profile_digest: str
    n_stream: int
    n_ctx: int
    n_seq_max: int
    n_layer: int
    v_trans: bool
    tensors: tuple[TensorRowLayout, ...]
    n_pos_per_embd: int = 1


@dataclass(frozen=True)
class ReadSpan:
    serialization_ordinal: int
    kind: Literal["k", "v"]
    layer: int
    stream_index: int
    cell_index: int
    element_index: int | None
    address: int
    size: int


@dataclass(frozen=True)
class ReadRun:
    serialization_ordinal: int
    kind: Literal["k", "v"]
    layer: int
    stream_index: int
    first_cell_index: int
    cell_count: int
    element_index: int | None
    address: int
    size: int


def plan_selected_rows(
    layout: SimpleKvLayout,
    cells: tuple[SelectedCell, ...],
) -> tuple[ReadSpan, ...]:
    """Plan exactly the target rows in llama_kv_cache::state_write order."""

    _validate_layout_header(layout)
    stream_index = _validate_cells(layout, cells)
    tensors = _selected_tensor_map(layout, stream_index)
    layers = _validate_tensor_topology(tensors, layout.n_layer)
    cell_capacity = _validate_tensor_shapes(layout, tensors, layers)
    for cell in cells:
        if cell.cell_index >= cell_capacity:
            raise LayoutValidationError("cell_index exceeds tensor capacity")

    spans: list[ReadSpan] = []

    def append_span(
        tensor: TensorRowLayout,
        cell: SelectedCell,
        *,
        element_index: int | None,
        address: int,
        size: int,
    ) -> None:
        spans.append(
            ReadSpan(
                serialization_ordinal=len(spans),
                kind=tensor.kind,
                layer=tensor.layer,
                stream_index=stream_index,
                cell_index=cell.cell_index,
                element_index=element_index,
                address=address,
                size=size,
            )
        )

    for layer in layers:
        tensor = tensors[("k", layer)]
        for cell in cells:
            append_span(
                tensor,
                cell,
                element_index=None,
                address=tensor.tensor_address + cell.cell_index * tensor.row_size,
                size=tensor.row_size,
            )

    for layer in layers:
        tensor = tensors[("v", layer)]
        if not layout.v_trans:
            for cell in cells:
                append_span(
                    tensor,
                    cell,
                    element_index=None,
                    address=(
                        tensor.tensor_address + cell.cell_index * tensor.row_size
                    ),
                    size=tensor.row_size,
                )
            continue

        for element_index in range(tensor.embedding_width):
            for cell in cells:
                append_span(
                    tensor,
                    cell,
                    element_index=element_index,
                    address=(
                        tensor.tensor_address
                        + (
                            cell.cell_index
                            + element_index * tensor.cell_capacity
                        )
                        * tensor.element_size
                    ),
                    size=tensor.element_size,
                )

    return tuple(spans)


def plan_selected_read_runs(
    layout: SimpleKvLayout,
    cells: tuple[SelectedCell, ...],
) -> tuple[ReadRun, ...]:
    """Plan only selected bytes while coalescing adjacent physical cells."""

    _validate_layout_header(layout)
    stream_index = _validate_cells(layout, cells)
    tensors = _selected_tensor_map(layout, stream_index)
    layers = _validate_tensor_topology(tensors, layout.n_layer)
    cell_capacity = _validate_tensor_shapes(layout, tensors, layers)
    for cell in cells:
        if cell.cell_index >= cell_capacity:
            raise LayoutValidationError("cell_index exceeds tensor capacity")

    cell_runs = _contiguous_cell_runs(cells)
    runs: list[ReadRun] = []

    def append_run(
        tensor: TensorRowLayout,
        *,
        first_cell_index: int,
        cell_count: int,
        element_index: int | None,
        address: int,
        size: int,
    ) -> None:
        runs.append(
            ReadRun(
                serialization_ordinal=len(runs),
                kind=tensor.kind,
                layer=tensor.layer,
                stream_index=stream_index,
                first_cell_index=first_cell_index,
                cell_count=cell_count,
                element_index=element_index,
                address=address,
                size=size,
            )
        )

    for layer in layers:
        tensor = tensors[("k", layer)]
        for first_cell_index, cell_count in cell_runs:
            append_run(
                tensor,
                first_cell_index=first_cell_index,
                cell_count=cell_count,
                element_index=None,
                address=tensor.tensor_address + first_cell_index * tensor.row_size,
                size=cell_count * tensor.row_size,
            )

    for layer in layers:
        tensor = tensors[("v", layer)]
        if not layout.v_trans:
            for first_cell_index, cell_count in cell_runs:
                append_run(
                    tensor,
                    first_cell_index=first_cell_index,
                    cell_count=cell_count,
                    element_index=None,
                    address=(
                        tensor.tensor_address + first_cell_index * tensor.row_size
                    ),
                    size=cell_count * tensor.row_size,
                )
            continue

        for element_index in range(tensor.embedding_width):
            for first_cell_index, cell_count in cell_runs:
                append_run(
                    tensor,
                    first_cell_index=first_cell_index,
                    cell_count=cell_count,
                    element_index=element_index,
                    address=(
                        tensor.tensor_address
                        + (
                            first_cell_index
                            + element_index * tensor.cell_capacity
                        )
                        * tensor.element_size
                    ),
                    size=cell_count * tensor.element_size,
                )

    return tuple(runs)


def _contiguous_cell_runs(
    cells: tuple[SelectedCell, ...],
) -> tuple[tuple[int, int], ...]:
    first = cells[0].cell_index
    previous = first
    count = 1
    runs: list[tuple[int, int]] = []
    for cell in cells[1:]:
        if cell.cell_index == previous + 1:
            count += 1
        else:
            runs.append((first, count))
            first = cell.cell_index
            count = 1
        previous = cell.cell_index
    runs.append((first, count))
    return tuple(runs)


def _validate_layout_header(layout: SimpleKvLayout) -> None:
    if (
        not isinstance(layout.profile_digest, str)
        or _SHA256_DIGEST.fullmatch(layout.profile_digest) is None
    ):
        raise LayoutValidationError(
            "profile_digest must be lowercase sha256:<64 hex>"
        )
    _require_integer("n_stream", layout.n_stream, minimum=1)
    _require_integer("n_ctx", layout.n_ctx, minimum=1)
    _require_integer("n_seq_max", layout.n_seq_max, minimum=1)
    _require_integer("n_layer", layout.n_layer, minimum=1)
    if layout.n_stream != layout.n_seq_max:
        raise LayoutValidationError(
            "n_stream must equal n_seq_max for the pinned Simple KV profile"
        )
    if layout.n_ctx % layout.n_stream:
        raise LayoutValidationError("n_ctx must divide evenly across n_stream")
    if layout.n_pos_per_embd != 1:
        raise LayoutValidationError("n_pos_per_embd must equal 1")
    if not isinstance(layout.v_trans, bool):
        raise LayoutValidationError("v_trans must be a boolean")


def _validate_cells(
    layout: SimpleKvLayout, cells: tuple[SelectedCell, ...]
) -> int:
    if not isinstance(cells, tuple) or not cells:
        raise LayoutValidationError("at least one selected cell is required")
    if any(not isinstance(cell, SelectedCell) for cell in cells):
        raise LayoutValidationError("cells must contain SelectedCell values")
    for cell in cells:
        _require_integer("position", cell.position, minimum=0)
        if cell.position > 0x7FFFFFFF:
            raise LayoutValidationError("position is outside int32 range")
        if cell.position >= layout.n_ctx:
            raise LayoutValidationError("position is outside n_ctx")
        _require_integer("cell_index", cell.cell_index, minimum=0)
        _require_integer("seq_id", cell.seq_id, minimum=0)
        _require_integer("stream_index", cell.stream_index, minimum=0)
    if any(cell.seq_id < 0 or cell.seq_id >= layout.n_seq_max for cell in cells):
        raise LayoutValidationError("seq_id is outside n_seq_max")
    if any(
        cell.stream_index < 0 or cell.stream_index >= layout.n_stream
        for cell in cells
    ):
        raise LayoutValidationError("stream_index is outside n_stream")

    seq_ids = {cell.seq_id for cell in cells}
    if len(seq_ids) != 1:
        raise LayoutValidationError("selected cells must belong to one sequence")
    stream_indices = {cell.stream_index for cell in cells}
    if len(stream_indices) != 1:
        raise LayoutValidationError("selected cells must belong to one stream")

    cell_indices = [cell.cell_index for cell in cells]
    if len(set(cell_indices)) != len(cell_indices):
        raise LayoutValidationError("duplicate cell_index is not allowed")
    if cell_indices != sorted(cell_indices):
        raise LayoutValidationError("cells must be in physical cell order")
    positions = [cell.position for cell in cells]
    if len(set(positions)) != len(positions):
        raise LayoutValidationError("duplicate position is not allowed")
    return next(iter(stream_indices))


def _selected_tensor_map(
    layout: SimpleKvLayout, stream_index: int
) -> dict[tuple[str, int], TensorRowLayout]:
    selected: dict[tuple[str, int], TensorRowLayout] = {}
    for tensor in layout.tensors:
        if tensor.stream_index < 0 or tensor.stream_index >= layout.n_stream:
            raise LayoutValidationError("tensor stream_index is outside n_stream")
        if tensor.kind not in {"k", "v"}:
            raise LayoutValidationError("tensor kind must be k or v")
        if tensor.stream_index != stream_index:
            continue
        key = (tensor.kind, tensor.layer)
        if key in selected:
            raise LayoutValidationError("duplicate tensor layer entry")
        selected[key] = tensor
    return selected


def _validate_tensor_topology(
    tensors: dict[tuple[str, int], TensorRowLayout],
    expected_n_layer: int,
) -> tuple[int, ...]:
    k_layers = {layer for kind, layer in tensors if kind == "k"}
    v_layers = {layer for kind, layer in tensors if kind == "v"}
    if not k_layers or k_layers != v_layers:
        raise LayoutValidationError("K/V layer topology is incomplete")
    layers = tuple(sorted(k_layers))
    if layers != tuple(range(expected_n_layer)):
        raise LayoutValidationError(
            "K/V layer topology does not match expected n_layer"
        )
    return layers


def _validate_tensor_shapes(
    layout: SimpleKvLayout,
    tensors: dict[tuple[str, int], TensorRowLayout],
    layers: tuple[int, ...],
) -> int:
    capacities: set[int] = set()
    storage_intervals: list[tuple[int, int, str, int]] = []
    for layer in layers:
        for kind in ("k", "v"):
            tensor = tensors[(kind, layer)]
            _require_integer("tensor layer", tensor.layer, minimum=0)
            _require_integer(
                "tensor_address", tensor.tensor_address, minimum=0
            )
            _require_integer("tensor row_size", tensor.row_size, minimum=1)
            _require_integer(
                "tensor cell_capacity", tensor.cell_capacity, minimum=1
            )
            _require_integer("ggml_type_id", tensor.ggml_type_id, minimum=0)
            _require_integer(
                "tensor element_size", tensor.element_size, minimum=1
            )
            _require_integer(
                "tensor embedding_width", tensor.embedding_width, minimum=1
            )
            if (
                kind == "v"
                and layout.v_trans
                and tensor.row_size
                != tensor.element_size * tensor.embedding_width
            ):
                raise LayoutValidationError(
                    "transposed V row_size does not match element layout"
                )
            storage_size = tensor.row_size * tensor.cell_capacity
            storage_end = tensor.tensor_address + storage_size
            if storage_end > 1 << 64:
                raise LayoutValidationError("tensor storage exceeds uint64 range")
            storage_intervals.append(
                (tensor.tensor_address, storage_end, kind, layer)
            )
            capacities.add(tensor.cell_capacity)
    if len(capacities) != 1:
        raise LayoutValidationError("tensor cell capacities are inconsistent")
    cell_capacity = next(iter(capacities))
    if cell_capacity != layout.n_ctx // layout.n_stream:
        raise LayoutValidationError(
            "tensor cell_capacity does not match n_ctx per stream"
        )
    storage_intervals.sort()
    for previous, current in zip(storage_intervals, storage_intervals[1:]):
        if current[0] < previous[1]:
            raise LayoutValidationError(
                "tensor storage intervals overlap"
            )
    return cell_capacity


def _require_integer(name: str, value: object, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise LayoutValidationError(
            f"{name} must be an integer of at least {minimum}"
        )
    return value
