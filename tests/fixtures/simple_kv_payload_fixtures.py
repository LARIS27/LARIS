from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from laris_onpremise.acquisition.layout import (
    ReadSpan,
    SelectedCell,
    SimpleKvLayout,
    TensorRowLayout,
)


def two_layer_layout(
    *,
    n_stream: int = 4,
    cell_capacity: int = 8,
    v_trans: bool = False,
    profile_digest: str = "sha256:" + "a" * 64,
) -> SimpleKvLayout:
    tensors: list[TensorRowLayout] = []
    for stream_index in range(n_stream):
        stream_base = 0x100000 + stream_index * 0x10000
        for layer in range(2):
            tensors.append(
                TensorRowLayout(
                    kind="k",
                    layer=layer,
                    stream_index=stream_index,
                    tensor_address=stream_base + layer * 0x2000,
                    row_size=4,
                    cell_capacity=cell_capacity,
                    ggml_type_id=1,
                    element_size=2,
                    embedding_width=2,
                )
            )
            tensors.append(
                TensorRowLayout(
                    kind="v",
                    layer=layer,
                    stream_index=stream_index,
                    tensor_address=stream_base + layer * 0x2000 + 0x1000,
                    row_size=6,
                    cell_capacity=cell_capacity,
                    ggml_type_id=1,
                    element_size=2,
                    embedding_width=3,
                )
            )
    return SimpleKvLayout(
        profile_digest=profile_digest,
        n_stream=n_stream,
        n_ctx=n_stream * cell_capacity,
        n_seq_max=n_stream,
        n_layer=2,
        v_trans=v_trans,
        tensors=tuple(tensors),
        n_pos_per_embd=1,
    )


def selected_cells(
    *,
    seq_id: int = 2,
    stream_index: int = 1,
) -> tuple[SelectedCell, ...]:
    return (
        SelectedCell(
            position=3,
            cell_index=1,
            seq_id=seq_id,
            stream_index=stream_index,
        ),
        SelectedCell(
            position=9,
            cell_index=6,
            seq_id=seq_id,
            stream_index=stream_index,
        ),
    )


def span_bytes(span: ReadSpan, *, salt: int = 0) -> bytes:
    return bytes(
        ((span.address + offset + salt) & 0xFF)
        for offset in range(span.size)
    )


def rows_for(spans: Iterable[ReadSpan]) -> dict[ReadSpan, bytes]:
    return {span: span_bytes(span) for span in spans}


@dataclass
class RecordingAddressReader:
    backing: dict[tuple[int, int], bytes]
    mutate_on_second_round: bool = False
    reads: list[tuple[int, int]] | None = None
    closed: bool = False
    read_rounds: int = 0

    def __post_init__(self) -> None:
        if self.reads is None:
            self.reads = []

    def pread_exact(self, address: int, size: int) -> bytes:
        assert self.reads is not None
        self.reads.append((address, size))
        value = self.backing[(address, size)]
        if self.mutate_on_second_round and self.reads.count((address, size)) > 1:
            return bytes([value[0] ^ 0xFF]) + value[1:]
        return value

    def pread_many_exact(
        self, spans: tuple[tuple[int, int], ...]
    ) -> tuple[bytes, ...]:
        self.read_rounds += 1
        return tuple(self.pread_exact(address, size) for address, size in spans)

    @property
    def access_trace(self) -> tuple[tuple[int, int], ...]:
        assert self.reads is not None
        return tuple(self.reads)

    def close(self) -> None:
        self.closed = True


def backing_for(spans: Iterable[ReadSpan]) -> dict[tuple[int, int], bytes]:
    return {
        (span.address, span.size): span_bytes(span)
        for span in spans
    }
