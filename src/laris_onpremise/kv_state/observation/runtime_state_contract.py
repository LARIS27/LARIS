from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Literal, Protocol

from laris_onpremise.kv_state.acquisition.layout import SimpleKvLayout, TensorRowLayout


RUNTIME_STATE_FRAME_SCHEMA_VERSION = "1.0"
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_BUILD_ID = re.compile(r"[0-9a-f]{40}\Z")


class RuntimeStateFrameError(ValueError):
    """Raised when an observer frame is partial, malformed, or tampered."""


def _require_exact_fields(
    value: object, expected: frozenset[str], field: str
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise RuntimeStateFrameError(f"{field} must be an object")
    actual = set(value)
    if actual != expected:
        raise RuntimeStateFrameError(
            f"{field} fields differ: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )
    return value


def _require_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeStateFrameError(f"{field} must be a non-empty string")
    return value


def _require_nonnegative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeStateFrameError(f"{field} must be a non-negative integer")
    return value


def _require_positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeStateFrameError(f"{field} must be a positive integer")
    return value


def _require_bool(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise RuntimeStateFrameError(f"{field} must be a boolean")
    return value


def _require_digest(value: object, field: str) -> str:
    text = _require_text(value, field)
    if _SHA256.fullmatch(text) is None:
        raise RuntimeStateFrameError(
            f"{field} must be lowercase sha256:<64 hex>"
        )
    return text


def _require_array(value: object, field: str) -> list[object]:
    if not isinstance(value, list):
        raise RuntimeStateFrameError(f"{field} must be an array")
    return value


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise RuntimeStateFrameError("frame is not canonical JSON data") from error


def _frame_digest(document_without_digest: dict[str, object]) -> str:
    return "sha256:" + hashlib.sha256(
        _canonical_json(document_without_digest)
    ).hexdigest()


@dataclass(frozen=True)
class KvBackingEvent:
    event_id: int
    tgid: int
    tid: int
    monotonic_ns: int
    base_address: int
    size_bytes: int

    def __post_init__(self) -> None:
        _require_nonnegative_int(self.event_id, "event_id")
        _require_positive_int(self.tgid, "tgid")
        _require_positive_int(self.tid, "tid")
        _require_nonnegative_int(self.monotonic_ns, "monotonic_ns")
        _require_nonnegative_int(self.base_address, "base_address")
        _require_positive_int(self.size_bytes, "size_bytes")


@dataclass(frozen=True)
class TokenCellEvent:
    apply_call_id: int
    tgid: int
    tid: int
    monotonic_ns: int
    seq_id: int
    stream_index: int
    token_position: int
    cell_index: int

    def __post_init__(self) -> None:
        _require_nonnegative_int(self.apply_call_id, "apply_call_id")
        _require_positive_int(self.tgid, "tgid")
        _require_positive_int(self.tid, "tid")
        _require_nonnegative_int(self.monotonic_ns, "monotonic_ns")
        _require_nonnegative_int(self.seq_id, "seq_id")
        _require_nonnegative_int(self.stream_index, "stream_index")
        _require_nonnegative_int(self.token_position, "token_position")
        _require_nonnegative_int(self.cell_index, "cell_index")


@dataclass(frozen=True)
class RuntimeStateSnapshot:
    frame_id: str
    frame_sequence: int
    runtime_instance_id: str
    runtime_pid: int
    runtime_start_time_ticks: int
    runtime_binary_sha256: str
    mapped_libllama_sha256: str
    mapped_libllama_elf_build_id: str
    profile_digest: str
    memory_family: str
    layout: SimpleKvLayout
    observed_n_stream: int
    seq_to_stream: tuple[tuple[int, int], ...]
    backing_events: tuple[KvBackingEvent, ...]
    token_cells: tuple[TokenCellEvent, ...]
    decode_counts: tuple[tuple[int, int], ...]
    mapping_mutation_count: int
    mutation_counters: tuple[tuple[str, int], ...]
    observer_loss_count: int
    parse_failure_count: int
    unmatched_call_count: int
    unsupported_mutation_count: int
    window_start_monotonic_ns: int
    window_end_monotonic_ns: int
    cutoff_monotonic_ns: int

    def __post_init__(self) -> None:
        _require_text(self.frame_id, "frame_id")
        _require_nonnegative_int(self.frame_sequence, "frame_sequence")
        _require_text(self.runtime_instance_id, "runtime_instance_id")
        _require_positive_int(self.runtime_pid, "runtime_pid")
        _require_positive_int(
            self.runtime_start_time_ticks, "runtime_start_time_ticks"
        )
        _require_digest(self.runtime_binary_sha256, "runtime_binary_sha256")
        _require_digest(
            self.mapped_libllama_sha256, "mapped_libllama_sha256"
        )
        if _BUILD_ID.fullmatch(
            _require_text(
                self.mapped_libllama_elf_build_id,
                "mapped_libllama_elf_build_id",
            )
        ) is None:
            raise RuntimeStateFrameError(
                "mapped_libllama_elf_build_id must be 40 lowercase hex characters"
            )
        _require_digest(self.profile_digest, "profile_digest")
        _require_text(self.memory_family, "memory_family")
        if not isinstance(self.layout, SimpleKvLayout):
            raise RuntimeStateFrameError("layout must be a SimpleKvLayout")
        _require_positive_int(self.observed_n_stream, "observed_n_stream")
        _validate_pairs(self.seq_to_stream, "seq_to_stream", unique_first=True)
        if not isinstance(self.backing_events, tuple) or any(
            not isinstance(item, KvBackingEvent) for item in self.backing_events
        ):
            raise RuntimeStateFrameError(
                "backing_events must contain KvBackingEvent values"
            )
        if not self.backing_events:
            raise RuntimeStateFrameError("backing_events must not be empty")
        if not isinstance(self.token_cells, tuple) or any(
            not isinstance(item, TokenCellEvent) for item in self.token_cells
        ):
            raise RuntimeStateFrameError(
                "token_cells must contain TokenCellEvent values"
            )
        _validate_pairs(self.decode_counts, "decode_counts", unique_first=True)
        _require_nonnegative_int(
            self.mapping_mutation_count, "mapping_mutation_count"
        )
        _validate_named_counters(self.mutation_counters)
        for field in (
            "observer_loss_count",
            "parse_failure_count",
            "unmatched_call_count",
            "unsupported_mutation_count",
            "window_start_monotonic_ns",
            "window_end_monotonic_ns",
            "cutoff_monotonic_ns",
        ):
            _require_nonnegative_int(getattr(self, field), field)
        if self.window_end_monotonic_ns < self.window_start_monotonic_ns:
            raise RuntimeStateFrameError(
                "window_end_monotonic_ns must not precede the window start"
            )
        if self.cutoff_monotonic_ns < self.window_end_monotonic_ns:
            raise RuntimeStateFrameError(
                "cutoff_monotonic_ns must not precede the window end"
            )


class RuntimeStateObservationSource(Protocol):
    def snapshot(
        self,
        *,
        not_before_ns: int,
        not_after_ns: int,
        observed_seq_id: int,
    ) -> RuntimeStateSnapshot: ...


def _validate_pairs(
    value: object, field: str, *, unique_first: bool
) -> None:
    if not isinstance(value, tuple):
        raise RuntimeStateFrameError(f"{field} must be a tuple")
    first_values: list[int] = []
    for index, item in enumerate(value):
        if not isinstance(item, tuple) or len(item) != 2:
            raise RuntimeStateFrameError(f"{field}[{index}] must be a pair")
        first_values.append(_require_nonnegative_int(item[0], f"{field}[{index}][0]"))
        _require_nonnegative_int(item[1], f"{field}[{index}][1]")
    if unique_first and len(set(first_values)) != len(first_values):
        raise RuntimeStateFrameError(f"{field} contains a duplicate key")


def _validate_named_counters(value: object) -> None:
    if not isinstance(value, tuple):
        raise RuntimeStateFrameError("mutation_counters must be a tuple")
    names: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, tuple) or len(item) != 2:
            raise RuntimeStateFrameError(
                f"mutation_counters[{index}] must be a pair"
            )
        names.append(_require_text(item[0], f"mutation_counters[{index}][0]"))
        _require_nonnegative_int(item[1], f"mutation_counters[{index}][1]")
    if len(set(names)) != len(names):
        raise RuntimeStateFrameError("mutation_counters contains a duplicate key")


def _layout_to_document(layout: SimpleKvLayout) -> dict[str, object]:
    return {
        "profile_digest": layout.profile_digest,
        "n_stream": layout.n_stream,
        "n_ctx": layout.n_ctx,
        "n_seq_max": layout.n_seq_max,
        "n_layer": layout.n_layer,
        "n_pos_per_embd": layout.n_pos_per_embd,
        "v_trans": layout.v_trans,
        "tensors": [
            {
                "kind": tensor.kind,
                "layer": tensor.layer,
                "stream_index": tensor.stream_index,
                "tensor_address": tensor.tensor_address,
                "row_size": tensor.row_size,
                "cell_capacity": tensor.cell_capacity,
                "ggml_type_id": tensor.ggml_type_id,
                "element_size": tensor.element_size,
                "embedding_width": tensor.embedding_width,
            }
            for tensor in layout.tensors
        ],
    }


def _snapshot_to_document(snapshot: RuntimeStateSnapshot) -> dict[str, object]:
    return {
        "schema_version": RUNTIME_STATE_FRAME_SCHEMA_VERSION,
        "frame_id": snapshot.frame_id,
        "frame_sequence": snapshot.frame_sequence,
        "frame_state": "complete",
        "runtime_instance_id": snapshot.runtime_instance_id,
        "runtime_pid": snapshot.runtime_pid,
        "runtime_start_time_ticks": snapshot.runtime_start_time_ticks,
        "runtime_binary_sha256": snapshot.runtime_binary_sha256,
        "mapped_libllama_sha256": snapshot.mapped_libllama_sha256,
        "mapped_libllama_elf_build_id": snapshot.mapped_libllama_elf_build_id,
        "profile_digest": snapshot.profile_digest,
        "memory_family": snapshot.memory_family,
        "layout": _layout_to_document(snapshot.layout),
        "observed_n_stream": snapshot.observed_n_stream,
        "seq_to_stream": [
            {"seq_id": seq_id, "stream_index": stream_index}
            for seq_id, stream_index in snapshot.seq_to_stream
        ],
        "backing_events": [
            {
                "event_id": event.event_id,
                "tgid": event.tgid,
                "tid": event.tid,
                "monotonic_ns": event.monotonic_ns,
                "base_address": event.base_address,
                "size_bytes": event.size_bytes,
            }
            for event in snapshot.backing_events
        ],
        "token_cells": [
            {
                "apply_call_id": event.apply_call_id,
                "tgid": event.tgid,
                "tid": event.tid,
                "monotonic_ns": event.monotonic_ns,
                "seq_id": event.seq_id,
                "stream_index": event.stream_index,
                "token_position": event.token_position,
                "cell_index": event.cell_index,
            }
            for event in snapshot.token_cells
        ],
        "decode_counts": [
            {"seq_id": seq_id, "count": count}
            for seq_id, count in snapshot.decode_counts
        ],
        "mapping_mutation_count": snapshot.mapping_mutation_count,
        "mutation_counters": [
            {"mutation": name, "count": count}
            for name, count in snapshot.mutation_counters
        ],
        "observer_loss_count": snapshot.observer_loss_count,
        "parse_failure_count": snapshot.parse_failure_count,
        "unmatched_call_count": snapshot.unmatched_call_count,
        "unsupported_mutation_count": snapshot.unsupported_mutation_count,
        "window_start_monotonic_ns": snapshot.window_start_monotonic_ns,
        "window_end_monotonic_ns": snapshot.window_end_monotonic_ns,
        "cutoff_monotonic_ns": snapshot.cutoff_monotonic_ns,
    }


def runtime_state_snapshot_to_frame(
    snapshot: RuntimeStateSnapshot,
) -> dict[str, object]:
    if not isinstance(snapshot, RuntimeStateSnapshot):
        raise RuntimeStateFrameError("snapshot must be a RuntimeStateSnapshot")
    document = _snapshot_to_document(snapshot)
    document["frame_digest"] = _frame_digest(document)
    return document


_FRAME_FIELDS = frozenset(
    {
        "schema_version",
        "frame_id",
        "frame_sequence",
        "frame_state",
        "frame_digest",
        "runtime_instance_id",
        "runtime_pid",
        "runtime_start_time_ticks",
        "runtime_binary_sha256",
        "mapped_libllama_sha256",
        "mapped_libllama_elf_build_id",
        "profile_digest",
        "memory_family",
        "layout",
        "observed_n_stream",
        "seq_to_stream",
        "backing_events",
        "token_cells",
        "decode_counts",
        "mapping_mutation_count",
        "mutation_counters",
        "observer_loss_count",
        "parse_failure_count",
        "unmatched_call_count",
        "unsupported_mutation_count",
        "window_start_monotonic_ns",
        "window_end_monotonic_ns",
        "cutoff_monotonic_ns",
    }
)


def _parse_layout(value: object) -> SimpleKvLayout:
    document = _require_exact_fields(
        value,
        frozenset(
            {
                "profile_digest",
                "n_stream",
                "n_ctx",
                "n_seq_max",
                "n_layer",
                "n_pos_per_embd",
                "v_trans",
                "tensors",
            }
        ),
        "layout",
    )
    tensors: list[TensorRowLayout] = []
    for index, raw in enumerate(_require_array(document["tensors"], "layout.tensors")):
        item = _require_exact_fields(
            raw,
            frozenset(
                {
                    "kind",
                    "layer",
                    "stream_index",
                    "tensor_address",
                    "row_size",
                    "cell_capacity",
                    "ggml_type_id",
                    "element_size",
                    "embedding_width",
                }
            ),
            f"layout.tensors[{index}]",
        )
        kind = _require_text(item["kind"], f"layout.tensors[{index}].kind")
        if kind not in {"k", "v"}:
            raise RuntimeStateFrameError("tensor kind must be k or v")
        tensors.append(
            TensorRowLayout(
                kind=kind,
                layer=_require_nonnegative_int(item["layer"], "tensor.layer"),
                stream_index=_require_nonnegative_int(
                    item["stream_index"], "tensor.stream_index"
                ),
                tensor_address=_require_nonnegative_int(
                    item["tensor_address"], "tensor.tensor_address"
                ),
                row_size=_require_positive_int(item["row_size"], "tensor.row_size"),
                cell_capacity=_require_positive_int(
                    item["cell_capacity"], "tensor.cell_capacity"
                ),
                ggml_type_id=_require_nonnegative_int(
                    item["ggml_type_id"], "tensor.ggml_type_id"
                ),
                element_size=_require_positive_int(
                    item["element_size"], "tensor.element_size"
                ),
                embedding_width=_require_positive_int(
                    item["embedding_width"], "tensor.embedding_width"
                ),
            )
        )
    if not tensors:
        raise RuntimeStateFrameError("layout.tensors must not be empty")
    return SimpleKvLayout(
        profile_digest=_require_digest(
            document["profile_digest"], "layout.profile_digest"
        ),
        n_stream=_require_positive_int(document["n_stream"], "layout.n_stream"),
        n_ctx=_require_positive_int(document["n_ctx"], "layout.n_ctx"),
        n_seq_max=_require_positive_int(
            document["n_seq_max"], "layout.n_seq_max"
        ),
        n_layer=_require_positive_int(document["n_layer"], "layout.n_layer"),
        v_trans=_require_bool(document["v_trans"], "layout.v_trans"),
        tensors=tuple(tensors),
        n_pos_per_embd=_require_positive_int(
            document["n_pos_per_embd"], "layout.n_pos_per_embd"
        ),
    )


def _parse_pair_array(
    value: object,
    field: str,
    first_name: str,
    second_name: str,
) -> tuple[tuple[int, int], ...]:
    result: list[tuple[int, int]] = []
    seen: set[int] = set()
    for index, raw in enumerate(_require_array(value, field)):
        item = _require_exact_fields(
            raw, frozenset({first_name, second_name}), f"{field}[{index}]"
        )
        first = _require_nonnegative_int(item[first_name], f"{field}.{first_name}")
        second = _require_nonnegative_int(
            item[second_name], f"{field}.{second_name}"
        )
        if first in seen:
            raise RuntimeStateFrameError(f"{field} contains a duplicate key")
        seen.add(first)
        result.append((first, second))
    return tuple(result)


def runtime_state_snapshot_from_frame(document: object) -> RuntimeStateSnapshot:
    top = _require_exact_fields(document, _FRAME_FIELDS, "frame")
    if top["frame_state"] != "complete":
        raise RuntimeStateFrameError("frame_state must be complete")
    if top["schema_version"] != RUNTIME_STATE_FRAME_SCHEMA_VERSION:
        raise RuntimeStateFrameError(
            f"schema_version must be {RUNTIME_STATE_FRAME_SCHEMA_VERSION}"
        )

    build_id = _require_text(
        top["mapped_libllama_elf_build_id"], "mapped_libllama_elf_build_id"
    )
    if _BUILD_ID.fullmatch(build_id) is None:
        raise RuntimeStateFrameError(
            "mapped_libllama_elf_build_id must be 40 lowercase hex characters"
        )
    backing_events: list[KvBackingEvent] = []
    for index, raw in enumerate(_require_array(top["backing_events"], "backing_events")):
        item = _require_exact_fields(
            raw,
            frozenset(
                {
                    "event_id",
                    "tgid",
                    "tid",
                    "monotonic_ns",
                    "base_address",
                    "size_bytes",
                }
            ),
            f"backing_events[{index}]",
        )
        backing_events.append(
            KvBackingEvent(
                event_id=_require_nonnegative_int(item["event_id"], "event_id"),
                tgid=_require_positive_int(item["tgid"], "tgid"),
                tid=_require_positive_int(item["tid"], "tid"),
                monotonic_ns=_require_nonnegative_int(
                    item["monotonic_ns"], "monotonic_ns"
                ),
                base_address=_require_nonnegative_int(
                    item["base_address"], "base_address"
                ),
                size_bytes=_require_positive_int(item["size_bytes"], "size_bytes"),
            )
        )
    token_cells: list[TokenCellEvent] = []
    for index, raw in enumerate(_require_array(top["token_cells"], "token_cells")):
        item = _require_exact_fields(
            raw,
            frozenset(
                {
                    "apply_call_id",
                    "tgid",
                    "tid",
                    "monotonic_ns",
                    "seq_id",
                    "stream_index",
                    "token_position",
                    "cell_index",
                }
            ),
            f"token_cells[{index}]",
        )
        token_cells.append(
            TokenCellEvent(
                apply_call_id=_require_nonnegative_int(
                    item["apply_call_id"], "apply_call_id"
                ),
                tgid=_require_positive_int(item["tgid"], "tgid"),
                tid=_require_positive_int(item["tid"], "tid"),
                monotonic_ns=_require_nonnegative_int(
                    item["monotonic_ns"], "monotonic_ns"
                ),
                seq_id=_require_nonnegative_int(item["seq_id"], "seq_id"),
                stream_index=_require_nonnegative_int(
                    item["stream_index"], "stream_index"
                ),
                token_position=_require_nonnegative_int(
                    item["token_position"], "token_position"
                ),
                cell_index=_require_nonnegative_int(
                    item["cell_index"], "cell_index"
                ),
            )
        )
    mutation_counters: list[tuple[str, int]] = []
    seen_mutations: set[str] = set()
    for index, raw in enumerate(
        _require_array(top["mutation_counters"], "mutation_counters")
    ):
        item = _require_exact_fields(
            raw, frozenset({"mutation", "count"}), f"mutation_counters[{index}]"
        )
        name = _require_text(item["mutation"], "mutation")
        if name in seen_mutations:
            raise RuntimeStateFrameError(
                "mutation_counters contains a duplicate key"
            )
        seen_mutations.add(name)
        mutation_counters.append(
            (name, _require_nonnegative_int(item["count"], "mutation count"))
        )

    snapshot = RuntimeStateSnapshot(
        frame_id=_require_text(top["frame_id"], "frame_id"),
        frame_sequence=_require_nonnegative_int(
            top["frame_sequence"], "frame_sequence"
        ),
        runtime_instance_id=_require_text(
            top["runtime_instance_id"], "runtime_instance_id"
        ),
        runtime_pid=_require_positive_int(top["runtime_pid"], "runtime_pid"),
        runtime_start_time_ticks=_require_positive_int(
            top["runtime_start_time_ticks"], "runtime_start_time_ticks"
        ),
        runtime_binary_sha256=_require_digest(
            top["runtime_binary_sha256"], "runtime_binary_sha256"
        ),
        mapped_libllama_sha256=_require_digest(
            top["mapped_libllama_sha256"], "mapped_libllama_sha256"
        ),
        mapped_libllama_elf_build_id=build_id,
        profile_digest=_require_digest(top["profile_digest"], "profile_digest"),
        memory_family=_require_text(top["memory_family"], "memory_family"),
        layout=_parse_layout(top["layout"]),
        observed_n_stream=_require_positive_int(
            top["observed_n_stream"], "observed_n_stream"
        ),
        seq_to_stream=_parse_pair_array(
            top["seq_to_stream"], "seq_to_stream", "seq_id", "stream_index"
        ),
        backing_events=tuple(backing_events),
        token_cells=tuple(token_cells),
        decode_counts=_parse_pair_array(
            top["decode_counts"], "decode_counts", "seq_id", "count"
        ),
        mapping_mutation_count=_require_nonnegative_int(
            top["mapping_mutation_count"], "mapping_mutation_count"
        ),
        mutation_counters=tuple(mutation_counters),
        observer_loss_count=_require_nonnegative_int(
            top["observer_loss_count"], "observer_loss_count"
        ),
        parse_failure_count=_require_nonnegative_int(
            top["parse_failure_count"], "parse_failure_count"
        ),
        unmatched_call_count=_require_nonnegative_int(
            top["unmatched_call_count"], "unmatched_call_count"
        ),
        unsupported_mutation_count=_require_nonnegative_int(
            top["unsupported_mutation_count"], "unsupported_mutation_count"
        ),
        window_start_monotonic_ns=_require_nonnegative_int(
            top["window_start_monotonic_ns"], "window_start_monotonic_ns"
        ),
        window_end_monotonic_ns=_require_nonnegative_int(
            top["window_end_monotonic_ns"], "window_end_monotonic_ns"
        ),
        cutoff_monotonic_ns=_require_nonnegative_int(
            top["cutoff_monotonic_ns"], "cutoff_monotonic_ns"
        ),
    )

    supplied_digest = _require_digest(top["frame_digest"], "frame_digest")
    unsigned = dict(top)
    del unsigned["frame_digest"]
    expected_digest = _frame_digest(unsigned)
    if supplied_digest != expected_digest:
        raise RuntimeStateFrameError("frame digest does not match canonical content")
    return snapshot


__all__ = [
    "KvBackingEvent",
    "RUNTIME_STATE_FRAME_SCHEMA_VERSION",
    "RuntimeStateFrameError",
    "RuntimeStateObservationSource",
    "RuntimeStateSnapshot",
    "TokenCellEvent",
    "runtime_state_snapshot_from_frame",
    "runtime_state_snapshot_to_frame",
]
