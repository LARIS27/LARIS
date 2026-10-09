from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Final, Protocol

from laris_onpremise.kv_state.acquisition.acquirer import (
    AcquisitionObserverIntegrityError,
    ObservedKvBackingRange,
    RuntimeStateAcquisitionView,
    RuntimeStateTargetMarker,
)
from laris_onpremise.kv_state.acquisition.contracts import RuntimeStateCaptureRequest
from laris_onpremise.kv_state.acquisition.layout import SelectedCell

from .runtime_state_contract import (
    RuntimeStateFrameError,
    RuntimeStateSnapshot,
    TokenCellEvent,
    runtime_state_snapshot_from_frame,
    runtime_state_snapshot_to_frame,
)
from .runtime_state_profile import RuntimeStateObserverProfile
from .runtime_state_service import LogLinkedRuntimeCut


_MAX_FRAME_BYTES: Final[int] = 64 * 1024 * 1024


def _canonical_digest(domain: bytes, value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha256(domain)
    digest.update(encoded)
    return "sha256:" + digest.hexdigest()


def _latest_cell_owners(
    snapshot: RuntimeStateSnapshot,
) -> dict[tuple[int, int], TokenCellEvent]:
    owners: dict[tuple[int, int], TokenCellEvent] = {}
    for event in sorted(
        snapshot.token_cells,
        key=lambda item: (
            item.apply_call_id,
            item.monotonic_ns,
            item.seq_id,
            item.stream_index,
            item.token_position,
            item.cell_index,
        ),
    ):
        key = (event.stream_index, event.cell_index)
        previous = owners.get(key)
        if previous is not None and previous.apply_call_id == event.apply_call_id:
            raise RuntimeStateObservationError(
                "one target cell has ambiguous owners in the same apply generation"
            )
        owners[key] = event
    return owners


def _target_cell_map_digest(events: tuple[TokenCellEvent, ...]) -> str:
    document = [
        [
            event.apply_call_id,
            event.seq_id,
            event.stream_index,
            event.token_position,
            event.cell_index,
        ]
        for event in sorted(events, key=lambda item: item.token_position)
    ]
    return _canonical_digest(b"LARIS-TARGET-CELL-MAP-V1\x00", document)


def _layout_topology_digest(snapshot: RuntimeStateSnapshot) -> str:
    layout = snapshot.layout
    document = {
        "profile_digest": layout.profile_digest,
        "n_stream": layout.n_stream,
        "n_ctx": layout.n_ctx,
        "n_seq_max": layout.n_seq_max,
        "n_layer": layout.n_layer,
        "n_pos_per_embd": layout.n_pos_per_embd,
        "v_trans": layout.v_trans,
        "tensors": [
            [
                tensor.kind,
                tensor.layer,
                tensor.stream_index,
                tensor.tensor_address,
                tensor.row_size,
                tensor.cell_capacity,
                tensor.ggml_type_id,
                tensor.element_size,
                tensor.embedding_width,
            ]
            for tensor in sorted(
                layout.tensors,
                key=lambda item: (item.kind, item.layer, item.stream_index),
            )
        ],
    }
    return _canonical_digest(b"LARIS-RUNTIME-LAYOUT-TOPOLOGY-V1\x00", document)


def _backing_identity_digest(
    ranges: tuple[ObservedKvBackingRange, ...],
) -> str:
    document = [
        [
            item.backing_ref,
            item.address_start,
            item.address_end,
            item.provenance_digest,
        ]
        for item in ranges
    ]
    return _canonical_digest(b"LARIS-KV-BACKING-IDENTITY-V1\x00", document)


class RuntimeStateObservationError(AcquisitionObserverIntegrityError):
    """Raised when observer metadata cannot prove a safe target selection."""


class ObservationFrameIncompleteError(RuntimeStateObservationError):
    """Raised when no atomically published complete observation frame exists."""


@dataclass(frozen=True)
class _LoadedFrame:
    snapshot: RuntimeStateSnapshot
    frame_digest: str
    path: Path


class _RuntimeStateSnapshotClient(Protocol):
    def snapshot(
        self,
        *,
        not_before_ns: int,
        not_after_ns: int,
        observed_seq_id: int,
    ) -> RuntimeStateSnapshot: ...

    def log_linked_cut(
        self,
        *,
        not_before_ns: int,
        not_after_ns: int,
        observed_seq_id: int,
    ) -> LogLinkedRuntimeCut: ...


class RuntimeStateObservationRepository:
    """Read-only repository for native observer metadata frames.

    It never reads process-memory backing bytes. It converts a complete, sealed
    metadata frame into the exact address-selection view consumed by the
    selective acquirer.
    """

    def __init__(
        self, *, frame_dir: Path, profile: RuntimeStateObserverProfile
    ) -> None:
        if not isinstance(frame_dir, Path):
            raise RuntimeStateObservationError(
                "frame_dir must be a pathlib.Path"
            )
        if not isinstance(profile, RuntimeStateObserverProfile):
            raise RuntimeStateObservationError(
                "profile must be a RuntimeStateObserverProfile"
            )
        self.frame_dir = frame_dir
        self.profile = profile

    def snapshot(
        self,
        *,
        not_before_ns: int,
        not_after_ns: int,
        observed_seq_id: int,
    ) -> RuntimeStateSnapshot:
        return self._load_for_interval(
            not_before_ns,
            not_after_ns,
            observed_seq_id,
        ).snapshot

    def resolve(
        self, request: RuntimeStateCaptureRequest
    ) -> RuntimeStateAcquisitionView:
        if not isinstance(request, RuntimeStateCaptureRequest):
            raise RuntimeStateObservationError(
                "request must be a RuntimeStateCaptureRequest"
            )
        loaded = self._load_for_interval(
            request.interval_start_monotonic_ns,
            request.interval_end_monotonic_ns,
            request.observed_seq_id,
        )
        snapshot = loaded.snapshot
        self._validate_snapshot_binding(snapshot, request)
        cells = self._select_target_cells(snapshot, request)
        backing_ranges = self._backing_ranges(snapshot)
        marker = self._marker(
            snapshot,
            request.observed_seq_id,
            cells,
            backing_ranges,
        )
        return RuntimeStateAcquisitionView(
            runtime_instance_id=snapshot.runtime_instance_id,
            runtime_pid=snapshot.runtime_pid,
            runtime_start_time_ticks=snapshot.runtime_start_time_ticks,
            observed_seq_id=request.observed_seq_id,
            snapshot_id=snapshot.frame_id,
            snapshot_cut_monotonic_ns=snapshot.cutoff_monotonic_ns,
            target_marker=marker,
            layout=snapshot.layout,
            cells=cells,
            kv_backing_ranges=backing_ranges,
            acquisition_evidence_ref=self._evidence_ref(loaded),
            observer_loss_count=snapshot.observer_loss_count,
            parse_failure_count=snapshot.parse_failure_count,
            unmatched_call_count=snapshot.unmatched_call_count,
            unsupported_mutation_count=snapshot.unsupported_mutation_count,
        )

    def validated_snapshot(
        self, request: RuntimeStateCaptureRequest
    ) -> RuntimeStateSnapshot:
        """Return one profile-bound snapshot without selecting Proxy cells."""

        if not isinstance(request, RuntimeStateCaptureRequest):
            raise RuntimeStateObservationError(
                "request must be a RuntimeStateCaptureRequest"
            )
        loaded = self._load_for_interval(
            request.interval_start_monotonic_ns,
            request.interval_end_monotonic_ns,
            request.observed_seq_id,
        )
        self._validate_snapshot_binding(loaded.snapshot, request)
        return loaded.snapshot

    def backing_ranges(
        self, snapshot: RuntimeStateSnapshot
    ) -> tuple[ObservedKvBackingRange, ...]:
        """Expose validated backing metadata to additive acquisition modes."""

        if not isinstance(snapshot, RuntimeStateSnapshot):
            raise RuntimeStateObservationError(
                "snapshot must be a RuntimeStateSnapshot"
            )
        return self._backing_ranges(snapshot)

    def marker_for_cells(
        self,
        snapshot: RuntimeStateSnapshot,
        observed_seq_id: int,
        cells: tuple[SelectedCell, ...],
        backing_ranges: tuple[ObservedKvBackingRange, ...],
    ) -> RuntimeStateTargetMarker:
        """Build the same stability marker used by Proxy acquisition."""

        return self._marker(
            snapshot,
            observed_seq_id,
            cells,
            backing_ranges,
        )

    def target_marker(
        self,
        request: RuntimeStateCaptureRequest,
        view: RuntimeStateAcquisitionView,
    ) -> RuntimeStateTargetMarker:
        if not isinstance(view, RuntimeStateAcquisitionView):
            raise RuntimeStateObservationError(
                "view must be a RuntimeStateAcquisitionView"
            )
        loaded = self._load_for_interval(
            request.interval_start_monotonic_ns,
            request.interval_end_monotonic_ns,
            request.observed_seq_id,
        )
        snapshot = loaded.snapshot
        self._validate_snapshot_binding(snapshot, request)
        cells = self._select_target_cells(snapshot, request)
        backing_ranges = self._backing_ranges(snapshot)
        return self._marker(
            snapshot,
            request.observed_seq_id,
            cells,
            backing_ranges,
        )

    def _load_for_interval(
        self,
        not_before_ns: int,
        not_after_ns: int,
        observed_seq_id: int,
    ) -> _LoadedFrame:
        _require_nonnegative_int(not_before_ns, "not_before_ns")
        _require_nonnegative_int(not_after_ns, "not_after_ns")
        _require_nonnegative_int(observed_seq_id, "observed_seq_id")
        if not_after_ns < not_before_ns:
            raise RuntimeStateObservationError(
                "not_after_ns must not precede not_before_ns"
            )
        if self.frame_dir.is_symlink():
            raise RuntimeStateObservationError("frame directory must not be a symlink")
        if not self.frame_dir.is_dir():
            raise ObservationFrameIncompleteError(
                "no complete runtime-state observation frame is available"
            )

        candidates = sorted(self.frame_dir.glob("*.frame.json"))
        loaded: list[_LoadedFrame] = []
        for path in candidates:
            item = self._read_complete_frame(path)
            snapshot = item.snapshot
            if (
                snapshot.window_start_monotonic_ns == not_before_ns
                and snapshot.window_end_monotonic_ns == not_after_ns
            ):
                loaded.append(item)
        if not loaded:
            has_partial = any(
                path.is_file()
                for path in self.frame_dir.iterdir()
                if path.name.endswith((".tmp", ".partial", ".writing"))
            )
            qualifier = "complete " if has_partial else ""
            raise ObservationFrameIncompleteError(
                f"no {qualifier}runtime-state observation frame matches the requested interval"
            )
        loaded.sort(
            key=lambda item: (
                item.snapshot.frame_sequence,
                item.snapshot.cutoff_monotonic_ns,
                item.snapshot.frame_id,
            )
        )
        latest = loaded[-1]
        same_sequence = [
            item
            for item in loaded
            if item.snapshot.frame_sequence == latest.snapshot.frame_sequence
        ]
        if len({item.frame_digest for item in same_sequence}) != 1:
            raise RuntimeStateObservationError(
                "conflicting complete frames share a frame_sequence"
            )
        return latest

    def _evidence_ref(self, loaded: _LoadedFrame) -> str:
        return (
            f"runtime-state-frame:{loaded.snapshot.frame_id}:"
            f"{loaded.frame_digest}"
        )

    def _read_complete_frame(self, path: Path) -> _LoadedFrame:
        if path.is_symlink():
            raise RuntimeStateObservationError(
                "runtime-state observation frame must not be a symlink"
            )
        try:
            before = path.stat()
            if before.st_size <= 0 or before.st_size > _MAX_FRAME_BYTES:
                raise RuntimeStateObservationError(
                    "runtime-state observation frame size is invalid"
                )
            raw = path.read_bytes()
            after = path.stat()
        except OSError as error:
            raise RuntimeStateObservationError(
                "runtime-state observation frame could not be read"
            ) from error
        if (
            before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or getattr(before, "st_ino", None) != getattr(after, "st_ino", None)
            or len(raw) != after.st_size
        ):
            raise ObservationFrameIncompleteError(
                "runtime-state observation frame changed while being read"
            )

        def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
            result: dict[str, object] = {}
            for key, value in pairs:
                if key in result:
                    raise RuntimeStateFrameError(f"duplicate JSON key: {key}")
                result[key] = value
            return result

        try:
            document = json.loads(
                raw.decode("utf-8"), object_pairs_hook=reject_duplicates
            )
            snapshot = runtime_state_snapshot_from_frame(document)
        except (UnicodeError, json.JSONDecodeError, RuntimeStateFrameError) as error:
            raise RuntimeStateObservationError(
                f"invalid complete runtime-state observation frame: {path.name}"
            ) from error
        if not isinstance(document, dict):
            raise RuntimeStateObservationError(
                "runtime-state observation frame must be an object"
            )
        return _LoadedFrame(
            snapshot=snapshot,
            frame_digest=str(document["frame_digest"]),
            path=path,
        )

    def _validate_snapshot_binding(
        self,
        snapshot: RuntimeStateSnapshot,
        request: RuntimeStateCaptureRequest,
    ) -> None:
        if (
            snapshot.window_start_monotonic_ns
            != request.interval_start_monotonic_ns
            or snapshot.window_end_monotonic_ns
            < request.interval_end_monotonic_ns
        ):
            raise RuntimeStateObservationError(
                "snapshot interval does not cover the capture request"
            )
        expected = {
            "runtime_instance_id": request.runtime_instance_id,
            "runtime_pid": request.runtime_pid,
            "runtime_start_time_ticks": request.runtime_start_time_ticks,
            "profile_digest": self.profile.profile_digest,
            "runtime_binary_sha256": self.profile.runtime_binary_sha256,
            "mapped_libllama_sha256": self.profile.mapped_libllama_sha256,
            "mapped_libllama_elf_build_id": (
                self.profile.mapped_libllama_elf_build_id
            ),
            "memory_family": self.profile.memory_family,
            "observed_n_stream": self.profile.n_stream,
        }
        for field, expected_value in expected.items():
            if getattr(snapshot, field) != expected_value:
                raise RuntimeStateObservationError(
                    f"snapshot {field} does not match the qualified capture binding"
                )
        layout_expected = {
            "profile_digest": self.profile.profile_digest,
            "n_stream": self.profile.n_stream,
            "n_seq_max": self.profile.n_seq_max,
        }
        for field, expected_value in layout_expected.items():
            if getattr(snapshot.layout, field) != expected_value:
                raise RuntimeStateObservationError(
                    f"layout {field} does not match the qualified profile"
                )
        if self.profile.n_layer:
            if snapshot.layout.n_layer != self.profile.n_layer:
                raise RuntimeStateObservationError(
                    "layout n_layer does not match the qualified profile"
                )
        elif not 0 < snapshot.layout.n_layer <= self.profile.max_n_layer:
            raise RuntimeStateObservationError(
                "observed n_layer is outside the qualified profile limit"
            )
        if self.profile.n_pos_per_embd:
            if snapshot.layout.n_pos_per_embd != self.profile.n_pos_per_embd:
                raise RuntimeStateObservationError(
                    "layout n_pos_per_embd does not match the qualified profile"
                )
        elif not (
            0
            < snapshot.layout.n_pos_per_embd
            <= self.profile.max_n_pos_per_embd
        ):
            raise RuntimeStateObservationError(
                "observed n_pos_per_embd is outside the qualified profile limit"
            )
        if (
            self.profile.v_trans is not None
            and snapshot.layout.v_trans != self.profile.v_trans
        ):
            raise RuntimeStateObservationError(
                "layout v_trans does not match the qualified profile"
            )
        if request.observed_seq_id >= snapshot.layout.n_seq_max:
            raise RuntimeStateObservationError(
                "observed_seq_id is outside the qualified n_seq_max"
            )
        self._validate_health(snapshot)
        self._validate_process_events(snapshot)
        self._validate_layout_topology(snapshot)

    def _validate_health(self, snapshot: RuntimeStateSnapshot) -> None:
        for field in (
            "observer_loss_count",
            "parse_failure_count",
            "unmatched_call_count",
            "unsupported_mutation_count",
        ):
            if getattr(snapshot, field) != 0:
                raise RuntimeStateObservationError(f"{field} must be zero")
        counters = dict(snapshot.mutation_counters)
        if set(counters) != set(self.profile.forbidden_mutations):
            raise RuntimeStateObservationError(
                "mutation_counters do not match the qualified profile"
            )
        for name in self.profile.forbidden_mutations:
            if counters[name] != 0:
                raise RuntimeStateObservationError(
                    f"forbidden runtime mutation observed: {name}"
                )

    def _validate_process_events(self, snapshot: RuntimeStateSnapshot) -> None:
        mappings = dict(snapshot.seq_to_stream)
        for seq_id, stream_index in snapshot.seq_to_stream:
            if seq_id >= snapshot.layout.n_seq_max:
                raise RuntimeStateObservationError(
                    "seq_to_stream seq_id is outside n_seq_max"
                )
            if stream_index >= snapshot.observed_n_stream:
                raise RuntimeStateObservationError(
                    "seq_to_stream stream_index is outside observed_n_stream"
                )
        for event in snapshot.backing_events:
            if event.tgid != snapshot.runtime_pid:
                raise RuntimeStateObservationError(
                    "KV backing event tgid does not match runtime_pid"
                )
        for event in snapshot.token_cells:
            if event.tgid != snapshot.runtime_pid:
                raise RuntimeStateObservationError(
                    "token-cell event tgid does not match runtime_pid"
                )
            if event.seq_id >= snapshot.layout.n_seq_max:
                raise RuntimeStateObservationError(
                    "token-cell seq_id is outside n_seq_max"
                )
            if event.stream_index >= snapshot.observed_n_stream:
                raise RuntimeStateObservationError(
                    "token-cell stream_index is outside observed_n_stream"
                )
            if not (
                snapshot.window_start_monotonic_ns
                <= event.monotonic_ns
                <= snapshot.window_end_monotonic_ns
            ):
                raise RuntimeStateObservationError(
                    "token-cell event is outside the observation interval"
                )
            if event.seq_id not in mappings:
                raise RuntimeStateObservationError(
                    "token-cell sequence has no seq_to_stream mapping"
                )
            if mappings[event.seq_id] != event.stream_index:
                raise RuntimeStateObservationError(
                    "observed sequence moved between streams"
                )

    def _validate_layout_topology(self, snapshot: RuntimeStateSnapshot) -> None:
        seen: set[tuple[str, int, int]] = set()
        for tensor in snapshot.layout.tensors:
            if tensor.stream_index >= snapshot.layout.n_stream:
                raise RuntimeStateObservationError(
                    "tensor stream_index is outside n_stream"
                )
            key = (tensor.kind, tensor.layer, tensor.stream_index)
            if key in seen:
                raise RuntimeStateObservationError(
                    "duplicate tensor projection in runtime layout"
                )
            seen.add(key)
            expected_type = (
                self.profile.key_ggml_type_id
                if tensor.kind == "k"
                else self.profile.value_ggml_type_id
            )
            if tensor.ggml_type_id != expected_type:
                raise RuntimeStateObservationError(
                    f"{tensor.kind.upper()} tensor ggml_type_id does not match the profile"
                )
        streams = {stream for _, _, stream in seen}
        if streams != set(range(self.profile.n_stream)):
            raise RuntimeStateObservationError(
                "runtime layout does not project all qualified streams"
            )
        for stream in range(self.profile.n_stream):
            k_layers = {layer for kind, layer, item_stream in seen if kind == "k" and item_stream == stream}
            v_layers = {layer for kind, layer, item_stream in seen if kind == "v" and item_stream == stream}
            if (
                k_layers != set(range(snapshot.layout.n_layer))
                or v_layers != set(range(snapshot.layout.n_layer))
            ):
                raise RuntimeStateObservationError(
                    "runtime layout has incomplete K/V layer topology"
                )

    def _select_target_cells(
        self,
        snapshot: RuntimeStateSnapshot,
        request: RuntimeStateCaptureRequest,
    ) -> tuple[SelectedCell, ...]:
        mappings = dict(snapshot.seq_to_stream)
        if request.observed_seq_id not in mappings:
            raise RuntimeStateObservationError(
                "target sequence has no observed seq_to_stream mapping"
            )
        target_stream = mappings[request.observed_seq_id]
        if target_stream >= snapshot.observed_n_stream:
            raise RuntimeStateObservationError(
                "target seq_to_stream mapping is outside observed_n_stream"
            )
        events = [
            event
            for event in snapshot.token_cells
            if event.seq_id == request.observed_seq_id
        ]
        if not events:
            raise RuntimeStateObservationError(
                "target sequence has no complete token-cell mapping"
            )
        for event in events:
            if not (
                request.interval_start_monotonic_ns
                <= event.monotonic_ns
                <= request.interval_end_monotonic_ns
            ):
                raise RuntimeStateObservationError(
                    "target token-cell event is outside the Lease interval"
                )
            if event.stream_index != target_stream:
                raise RuntimeStateObservationError(
                    "target sequence moved between streams or contradicts seq_to_stream"
                )
        by_position: dict[int, TokenCellEvent] = {}
        for event in sorted(
            events,
            key=lambda item: (
                item.apply_call_id,
                item.monotonic_ns,
                item.cell_index,
            ),
        ):
            previous = by_position.get(event.token_position)
            if (
                previous is not None
                and previous.apply_call_id == event.apply_call_id
            ):
                raise RuntimeStateObservationError(
                    "duplicate target sequence position was observed"
                )
            by_position[event.token_position] = event
        cells_seen: set[int] = set()
        for event in by_position.values():
            if event.cell_index in cells_seen:
                raise RuntimeStateObservationError(
                    "one physical KV cell was assigned to multiple positions"
                )
            cells_seen.add(event.cell_index)
        maximum_position = max(by_position)
        expected_positions = set(range(maximum_position + 1))
        if set(by_position) != expected_positions:
            raise RuntimeStateObservationError(
                "missing token position in full-sequence Lease rewrite"
            )
        selected = tuple(
            sorted(
                (
                    SelectedCell(
                        position=event.token_position,
                        cell_index=event.cell_index,
                        seq_id=event.seq_id,
                        stream_index=event.stream_index,
                    )
                    for event in by_position.values()
                ),
                key=lambda cell: cell.cell_index,
            )
        )
        return selected

    def _backing_ranges(
        self, snapshot: RuntimeStateSnapshot
    ) -> tuple[ObservedKvBackingRange, ...]:
        ordered = sorted(snapshot.backing_events, key=lambda item: item.base_address)
        result: list[ObservedKvBackingRange] = []
        previous_end = -1
        event_ids: set[int] = set()
        for event in ordered:
            if event.event_id in event_ids:
                raise RuntimeStateObservationError(
                    "duplicate KV backing event_id was observed"
                )
            event_ids.add(event.event_id)
            address_end = event.base_address + event.size_bytes
            if event.base_address < previous_end:
                raise RuntimeStateObservationError(
                    "overlapping KV backing ranges are ambiguous"
                )
            previous_end = address_end
            provenance = {
                "event_id": event.event_id,
                "tgid": event.tgid,
                "tid": event.tid,
                "monotonic_ns": event.monotonic_ns,
                "base_address": event.base_address,
                "size_bytes": event.size_bytes,
            }
            encoded = json.dumps(
                provenance,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            result.append(
                ObservedKvBackingRange(
                    backing_ref=f"kv-backing:{event.event_id}",
                    address_start=event.base_address,
                    address_end=address_end,
                    provenance_digest=(
                        "sha256:" + hashlib.sha256(encoded).hexdigest()
                    ),
                )
            )
        return tuple(result)

    @staticmethod
    def _marker(
        snapshot: RuntimeStateSnapshot,
        observed_seq_id: int,
        cells: tuple[SelectedCell, ...],
        backing_ranges: tuple[ObservedKvBackingRange, ...],
    ) -> RuntimeStateTargetMarker:
        decode_counts = dict(snapshot.decode_counts)
        if observed_seq_id not in decode_counts:
            raise RuntimeStateObservationError(
                "target sequence has no complete decode counter"
            )
        if not cells:
            raise RuntimeStateObservationError("target marker has no selected cells")
        stream_indices = {cell.stream_index for cell in cells}
        if len(stream_indices) != 1:
            raise RuntimeStateObservationError(
                "target marker spans more than one stream"
            )
        latest_owners = _latest_cell_owners(snapshot)
        target_events: list[TokenCellEvent] = []
        for cell in sorted(cells, key=lambda item: item.position):
            owner = latest_owners.get((cell.stream_index, cell.cell_index))
            if owner is None:
                raise RuntimeStateObservationError(
                    "target cell owner is missing from the observation"
                )
            if (
                owner.seq_id != observed_seq_id
                or owner.token_position != cell.position
            ):
                raise RuntimeStateObservationError(
                    "target cell owner changed after the terminal cut"
                )
            target_events.append(owner)
        apply_watermark = max(event.apply_call_id for event in target_events)
        return RuntimeStateTargetMarker(
            observed_seq_id=observed_seq_id,
            stream_index=next(iter(stream_indices)),
            target_decode_count=decode_counts[observed_seq_id],
            target_apply_watermark=apply_watermark,
            target_ownership_generation=apply_watermark,
            target_cell_map_digest=_target_cell_map_digest(
                tuple(target_events)
            ),
            layout_profile_digest=snapshot.layout.profile_digest,
            layout_topology_digest=_layout_topology_digest(snapshot),
            kv_backing_provenance_digest=_backing_identity_digest(
                backing_ranges
            ),
            observer_loss_count=snapshot.observer_loss_count,
            parse_failure_count=snapshot.parse_failure_count,
            unmatched_call_count=snapshot.unmatched_call_count,
            unsupported_mutation_count=snapshot.unsupported_mutation_count,
        )


class RuntimeStateClientObservationSource(RuntimeStateObservationRepository):
    """Adapt the long-lived local snapshot socket to the acquisition source.

    The client is queried for every resolve/stability cut. No frame-directory
    lookup is used as production truth, and every response is validated by the
    same qualified profile and full-sequence rules as sealed frame evidence.
    """

    def __init__(
        self,
        *,
        client: _RuntimeStateSnapshotClient,
        profile: RuntimeStateObserverProfile,
    ) -> None:
        if not callable(getattr(client, "snapshot", None)):
            raise RuntimeStateObservationError(
                "client must implement snapshot"
            )
        if not isinstance(profile, RuntimeStateObserverProfile):
            raise RuntimeStateObservationError(
                "profile must be a RuntimeStateObserverProfile"
            )
        self.client = client
        self.profile = profile

    def _load_for_interval(
        self,
        not_before_ns: int,
        not_after_ns: int,
        observed_seq_id: int,
    ) -> _LoadedFrame:
        _require_nonnegative_int(not_before_ns, "not_before_ns")
        _require_nonnegative_int(not_after_ns, "not_after_ns")
        _require_nonnegative_int(observed_seq_id, "observed_seq_id")
        if not_after_ns < not_before_ns:
            raise RuntimeStateObservationError(
                "not_after_ns must not precede not_before_ns"
            )
        try:
            snapshot = self.client.snapshot(
                not_before_ns=not_before_ns,
                not_after_ns=not_after_ns,
                observed_seq_id=observed_seq_id,
            )
            document = runtime_state_snapshot_to_frame(snapshot)
            frame_digest = document["frame_digest"]
        except Exception as error:
            if isinstance(error, RuntimeStateObservationError):
                raise
            raise RuntimeStateObservationError(
                "runtime-state snapshot service did not return a trusted frame"
            ) from error
        if not isinstance(frame_digest, str):
            raise RuntimeStateObservationError(
                "runtime-state snapshot frame digest is invalid"
            )
        return _LoadedFrame(
            snapshot=snapshot,
            frame_digest=frame_digest,
            path=Path("<runtime-state-socket>"),
        )

    def log_linked_cut(
        self, request: RuntimeStateCaptureRequest
    ) -> LogLinkedRuntimeCut:
        if not isinstance(request, RuntimeStateCaptureRequest):
            raise RuntimeStateObservationError(
                "request must be a RuntimeStateCaptureRequest"
            )
        operation = getattr(self.client, "log_linked_cut", None)
        if not callable(operation):
            raise RuntimeStateObservationError(
                "runtime-state client does not implement Log-linked cuts"
            )
        try:
            cut = operation(
                not_before_ns=request.interval_start_monotonic_ns,
                not_after_ns=request.interval_end_monotonic_ns,
                observed_seq_id=request.observed_seq_id,
            )
        except Exception as error:
            if isinstance(error, RuntimeStateObservationError):
                raise
            raise RuntimeStateObservationError(
                "runtime-state service did not return a trusted Log-linked cut"
            ) from error
        if not isinstance(cut, LogLinkedRuntimeCut):
            raise RuntimeStateObservationError(
                "runtime-state Log-linked cut is invalid"
            )
        self._validate_snapshot_binding(cut.snapshot, request)
        return cut

    def _evidence_ref(self, loaded: _LoadedFrame) -> str:
        return (
            f"runtime-state-socket-snapshot:{loaded.snapshot.frame_id}:"
            f"{loaded.frame_digest}"
        )


def _require_nonnegative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeStateObservationError(
            f"{field} must be a non-negative integer"
        )
    return value


__all__ = [
    "ObservationFrameIncompleteError",
    "RuntimeStateObservationError",
    "RuntimeStateObservationRepository",
    "RuntimeStateClientObservationSource",
]
