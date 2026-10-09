from __future__ import annotations

import argparse
import base64
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import stat
import struct
import subprocess
import sys
import tempfile
import threading
import time
from typing import BinaryIO, Callable, Final, Sequence

from laris_onpremise.kv_state.acquisition.layout import SimpleKvLayout, TensorRowLayout
from laris_onpremise.lab_agent.model_registration import (
    ModelRegistrationError,
    RegisteredModelHandle,
    load_model_registration_receipt,
    open_registered_model,
)

from .runtime_state_contract import (
    KvBackingEvent,
    RuntimeStateSnapshot,
    TokenCellEvent,
    runtime_state_snapshot_to_frame,
)
from .runtime_state_ranges import (
    NativeTokenCellRange,
    TokenCellRangeError,
    contains_token_cell,
    logical_token_cell_count,
    select_token_cells,
)
from .runtime_state_profile import (
    RuntimeStateObserverProfile,
    RuntimeStateProfileError,
    load_runtime_state_qualification_probe_profile,
    load_runtime_state_observer_profile,
    validate_runtime_state_capture_mode,
)
from .runtime_state_native_units import (
    CompletedNativeUnitPage,
    NativeCapturedUnitDescriptor,
    NativeUnitSpool,
    NativeUnitSpoolError,
    native_unit_schedule_payload_summary,
)
from .runtime_state_bulk import (
    BULK_PAYLOAD_BYTES,
    BULK_PREFIX_BYTES,
    NativeBulkDecodeError,
    decode_native_bulk_payload,
    iter_native_bulk_fragment_views,
)
from laris_onpremise.runtime.elf import (
    ElfInspectionError,
    ProcessInspectionBackend,
    ProcessInspectionError,
    SystemProcessInspectionBackend,
    inspect_elf,
    trusted_file_key,
)


def _emit_e6_phase(phase: str, elapsed_ns: int, byte_count: int) -> None:
    """Emit only numeric diagnostics in the isolated E6 candidate process."""

    if os.environ.get("LARIS_E6_PHASE_DIAGNOSTIC") != "1":
        return
    row = {"phase": phase, "elapsed_ns": elapsed_ns, "byte_count": byte_count}
    sys.stderr.write(
        "LARIS_E6_PHASE=" + json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
    )
    sys.stderr.flush()


_E6_CAPTURE_PHASES = (
    "observer_unit_read",
    "observer_fragment_persist",
    "observer_seal_persist",
)


def _publish_e6_capture_metrics(
    path: Path,
    document: dict[str, object],
) -> None:
    """Exclusively publish one canonical numeric document after clean drain."""

    data = (
        json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    descriptor: int | None = None
    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        temporary_path = Path(temporary_name)
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.close(descriptor)
        descriptor = None
        os.link(temporary_path, path)
        if os.name == "posix":
            directory_descriptor = os.open(
                path.parent,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
    except OSError as error:
        raise RuntimeStateServiceError(
            "E6 capture metrics could not be published"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


NATIVE_WIRE_MAGIC: Final[int] = 0x5352534C
NATIVE_WIRE_VERSION: Final[int] = 5
_PREVIOUS_NATIVE_WIRE_VERSION: Final[int] = 4
_LEGACY_NATIVE_WIRE_VERSION: Final[int] = 3
FRAME_RUNTIME_TOPOLOGY: Final[int] = 1
FRAME_TENSOR_LAYOUT: Final[int] = 2
FRAME_KV_BACKING: Final[int] = 3
FRAME_TOKEN_CELL: Final[int] = 4
FRAME_APPLY_COMPLETE: Final[int] = 5
FRAME_WATERMARK_COUNTERS: Final[int] = 6
FRAME_TERMINAL_COUNTERS: Final[int] = 7
FRAME_MUTATION: Final[int] = 8
FRAME_STREAM_VERSION: Final[int] = 9
FRAME_STATE_UNIT_FRAGMENT: Final[int] = 10
FRAME_STATE_UNIT_SEAL: Final[int] = 11
FRAME_TOKEN_CELL_RANGE: Final[int] = 12
FRAME_STATE_UNIT_FRAGMENT_BATCH: Final[int] = 13
FRAME_STATE_UNIT_SCHEDULE_BATCH: Final[int] = 14
FRAME_STATE_UNIT_BULK: Final[int] = 15

NATIVE_CHANNEL_METADATA: Final[int] = 0
NATIVE_CHANNEL_STATE_UNIT: Final[int] = 1
_MAX_STATE_UNIT_FRAGMENT_BYTES: Final[int] = 4096
_NATIVE_OBSERVER_PIPE_BUFFER_BYTES: Final[int] = 1024 * 1024
_NATIVE_UNIT_FRAGMENT_BATCH_SIZE: Final[int] = 4096
_MAX_NATIVE_UNIT_READ_BYTES: Final[int] = 1024 * 1024
_NATIVE_UNIT_ID: Final[re.Pattern[str]] = re.compile(r"unit_[0-9a-f]{64}")

_HEADER = struct.Struct("<IHHIIQ")
_TOPOLOGY = struct.Struct("<QIIQIIIIII")
_TENSOR = struct.Struct("<QIIQIIIIQQQQQ")
_BACKING = struct.Struct("<QIIQQQ")
_TOKEN = struct.Struct("<QIIQIIQQ")
_TOKEN_CELL_RANGE = struct.Struct("<QIIQIIQQII")
_APPLY_COMPLETE = struct.Struct("<QIIQ")
_MUTATION = struct.Struct("<QIIQII")
_COUNTERS = struct.Struct("<17Q")
_STREAM_VERSION_V3 = struct.Struct("<QQQQ4Q4I")
_STREAM_VERSION_V4 = struct.Struct("<QQQQI4x8Q8I")
_STATE_UNIT_FRAGMENT = struct.Struct("<QIIIIQQIIQQQ")
_STATE_UNIT_FRAGMENT_BATCH = struct.Struct("<II")
_STATE_UNIT_SCHEDULE_BATCH = struct.Struct("<IIQIIIIIIQQIIIIIIIIQ")
_MAX_STATE_UNIT_FRAGMENT_BATCH_RECORD_BYTES: Final[int] = (
    _STATE_UNIT_FRAGMENT.size + _MAX_STATE_UNIT_FRAGMENT_BYTES
)
_STATE_UNIT_SEAL = struct.Struct("<QIIIIQQQQQQQ")
_PAYLOAD_SIZES: Final[dict[int, int]] = {
    FRAME_RUNTIME_TOPOLOGY: _TOPOLOGY.size,
    FRAME_TENSOR_LAYOUT: _TENSOR.size,
    FRAME_KV_BACKING: _BACKING.size,
    FRAME_TOKEN_CELL: _TOKEN.size,
    FRAME_TOKEN_CELL_RANGE: _TOKEN_CELL_RANGE.size,
    FRAME_APPLY_COMPLETE: _APPLY_COMPLETE.size,
    FRAME_MUTATION: _MUTATION.size,
    FRAME_STREAM_VERSION: _STREAM_VERSION_V4.size,
    FRAME_WATERMARK_COUNTERS: _COUNTERS.size,
    FRAME_TERMINAL_COUNTERS: _COUNTERS.size,
    FRAME_STATE_UNIT_SEAL: _STATE_UNIT_SEAL.size,
}
_MAX_REQUEST_BYTES: Final[int] = 512
_MAX_RESPONSE_BYTES: Final[int] = 64 * 1024 * 1024
_MAX_TERMINAL_SEAL_BYTES: Final[int] = 64 * 1024
_MAX_COMPLETED_APPLY_PAGE: Final[int] = 256
_NATIVE_STARTUP_FAILURE_DIAGNOSTICS: Final[frozenset[bytes]] = frozenset(
    {
        b"runtime-state observer failed to open BPF object",
        b"runtime-state observer failed to size BPF maps",
        b"runtime-state observer failed to load BPF object",
        b"runtime-state observer failed to configure maps",
        b"runtime-state observer failed to attach qualified probes",
        b"runtime-state observer failed to attach mutation probes",
        b"runtime-state observer failed to create ring consumer",
    }
)
_LIBBPF_STARTUP_DIAGNOSTIC = re.compile(
    rb"libbpf: [\x20-\x7e]{1,1000}\Z"
)
_TENSOR_READ_DIAGNOSTIC = re.compile(
    rb"DIAGNOSTIC runtime_state candidate_missing=(0|[1-9][0-9]{0,19}) "
    rb"shape_invalid=(0|[1-9][0-9]{0,19}) "
    rb"data_read_failed=(0|[1-9][0-9]{0,19}) "
    rb"type_read_failed=(0|[1-9][0-9]{0,19}) "
    rb"data_null=(0|[1-9][0-9]{0,19})\Z"
)
_APPLY_REJECTION_DIAGNOSTIC = re.compile(
    rb"APPLY_DIAGNOSTIC runtime_state state_conflict=(0|[1-9][0-9]{0,19}) "
    rb"header_read_failed=(0|[1-9][0-9]{0,19}) "
    rb"slot_shape_invalid=(0|[1-9][0-9]{0,19}) "
    rb"batch_shape_invalid=(0|[1-9][0-9]{0,19}) "
    rb"stream_count_zero=(0|[1-9][0-9]{0,19}) "
    rb"stream_count_exceeds=(0|[1-9][0-9]{0,19}) "
    rb"index_vector_extent_invalid=(0|[1-9][0-9]{0,19}) "
    rb"token_count_zero=(0|[1-9][0-9]{0,19}) "
    rb"token_count_exceeds=(0|[1-9][0-9]{0,19}) "
    rb"token_count_exceeds_observed=(0|[1-9][0-9]{0,19}) "
    rb"token_count_exceeds_streams=(0|[1-9][0-9]{0,19}) "
    rb"token_stream_remainder=(0|[1-9][0-9]{0,19}) "
    rb"token_read_failed=(0|[1-9][0-9]{0,19}) "
    rb"token_shape_invalid=(0|[1-9][0-9]{0,19}) "
    rb"return_unmatched=(0|[1-9][0-9]{0,19}) "
    rb"state_delete_failed=(0|[1-9][0-9]{0,19}) "
    rb"process_state_conflict=(0|[1-9][0-9]{0,19}) "
    rb"process_return_unmatched=(0|[1-9][0-9]{0,19}) "
    rb"process_apply_missing=(0|[1-9][0-9]{0,19}) "
    rb"process_graph_compute_failed=(0|[1-9][0-9]{0,19})\Z"
)
_NATIVE_DRAIN_DIAGNOSTIC = re.compile(
    rb"DRAIN_FAILED runtime_state submitted=(0|[1-9][0-9]{0,19}) "
    rb"received=(0|[1-9][0-9]{0,19}) active=(0|[1-9][0-9]{0,19})\Z"
)
_NATIVE_DRAIN_STALL_DIAGNOSTIC = re.compile(
    rb"DRAIN_STALLED runtime_state "
    rb"submitted=(0|[1-9][0-9]{0,19}) "
    rb"received=(0|[1-9][0-9]{0,19}) "
    rb"active=(0|[1-9][0-9]{0,19}) "
    rb"inflight=(0|[1-9][0-9]{0,19}) "
    rb"pending=(0|[1-9][0-9]{0,9}) stopped=(0|1) "
    rb"plans=(0|[1-9][0-9]{0,19}) "
    rb"ready=(0|[1-9][0-9]{0,19}) "
    rb"reserve=(0|[1-9][0-9]{0,19}) "
    rb"unmatched=(0|[1-9][0-9]{0,19})\Z"
)
_NATIVE_GATHER_FAILURE_DIAGNOSTIC = re.compile(
    rb"GATHER_FAILED runtime_state "
    rb"stage=(plan|ready|stop|memory_read|frame_emit|seal_emit|resume) "
    rb"error=(0|[1-9][0-9]{0,9}) pending=(0|[1-9][0-9]{0,9})\Z"
)
_BULK_AGGREGATE_NATIVE_FIELDS: Final[tuple[str, ...]] = (
    "native_read_plan_count",
    "native_read_syscall_count",
    "native_copied_bytes",
    "native_bulk_frame_count",
    "native_bulk_payload_bytes",
    "native_logical_fragment_count",
    "native_logical_bytes",
    "native_encode_write_ns",
    "capture_boundary_hold_ns",
    "capture_boundary_total_ns",
)
_NATIVE_BULK_METRICS_DIAGNOSTIC = re.compile(
    rb"BULK_METRICS runtime_state "
    + rb" ".join(
        field.encode("ascii") + rb"=(0|[1-9][0-9]{0,19})"
        for field in _BULK_AGGREGATE_NATIVE_FIELDS
    )
    + rb"\Z"
)
_FORBIDDEN_LAUNCH_OPTIONS: Final[frozenset[str]] = frozenset(
    {
        "--cache-idle-slots",
        "--cache-prompt",
        "--cache-reuse",
        "--context-shift",
        "--draft",
        "--kv-unified",
        "--mmproj",
        "--speculative",
    }
)
_FORBIDDEN_LAUNCH_ALIASES: Final[frozenset[str]] = frozenset(
    {
        "-b",
        "-c",
        "-ctk",
        "-ctv",
        "-fa",
        "-ngl",
        "-np",
        "-ub",
        "--gpu-layers",
        "--n-parallel",
    }
)


class NativeRuntimeStateWireError(ValueError):
    """Raised when the native runtime-state stream is incomplete or malformed."""


class RuntimeStateServiceError(RuntimeError):
    """Raised when an observation cannot be sealed as a coherent snapshot."""


@dataclass(frozen=True)
class NativeStreamVersionSnapshot:
    snapshot_id: int
    mutation_generation: int
    backing_generation: int
    layout_generation: int
    stream_apply_call_ids: tuple[int, ...]
    stream_states: tuple[str, ...]
    active_stream_count: int = 4

    def __post_init__(self) -> None:
        if _require_uint(self.snapshot_id, "snapshot_id") == 0:
            raise NativeRuntimeStateWireError("snapshot_id must be positive")
        for field in (
            "mutation_generation",
            "backing_generation",
            "layout_generation",
        ):
            _require_uint(getattr(self, field), field)
        if self.active_stream_count not in {4, 8}:
            raise NativeRuntimeStateWireError(
                "active_stream_count must be four or eight"
            )
        if (
            not isinstance(self.stream_apply_call_ids, tuple)
            or len(self.stream_apply_call_ids)
                not in {self.active_stream_count, 8}
        ):
            raise NativeRuntimeStateWireError(
                "stream_apply_call_ids do not match the active stream count"
            )
        if any(
            _require_uint(item, "stream_apply_call_id") < 0
            for item in self.stream_apply_call_ids
        ):
            raise NativeRuntimeStateWireError("stream apply id is invalid")
        if (
            not isinstance(self.stream_states, tuple)
            or len(self.stream_states)
                not in {self.active_stream_count, 8}
            or len(self.stream_states) != len(self.stream_apply_call_ids)
            or any(
                item not in {"EMPTY", "BUSY", "READY"}
                for item in self.stream_states
            )
        ):
            raise NativeRuntimeStateWireError("stream states are invalid")
        normalized_apply_ids = self.stream_apply_call_ids + (
            (0,) * (8 - len(self.stream_apply_call_ids))
        )
        normalized_states = self.stream_states + (
            ("EMPTY",) * (8 - len(self.stream_states))
        )
        object.__setattr__(
            self, "stream_apply_call_ids", normalized_apply_ids
        )
        object.__setattr__(self, "stream_states", normalized_states)
        for apply_id, status in zip(
            normalized_apply_ids,
            normalized_states,
            strict=True,
        ):
            if (status == "EMPTY") != (apply_id == 0):
                raise NativeRuntimeStateWireError(
                    "empty stream version and apply id disagree"
                )
        if any(normalized_apply_ids[self.active_stream_count :]) or any(
            state != "EMPTY"
            for state in normalized_states[self.active_stream_count :]
        ):
            raise NativeRuntimeStateWireError(
                "inactive stream version entries must be empty"
            )


@dataclass(frozen=True)
class LogLinkedRuntimeCut:
    snapshot: RuntimeStateSnapshot
    stream_version: NativeStreamVersionSnapshot

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, RuntimeStateSnapshot):
            raise RuntimeStateServiceError(
                "Log-linked cut snapshot is invalid"
            )
        if not isinstance(
            self.stream_version, NativeStreamVersionSnapshot
        ):
            raise RuntimeStateServiceError(
                "Log-linked cut stream version is invalid"
            )


class NativeObservationCounterError(RuntimeStateServiceError):
    """Carry only fixed names for nonzero native safety counters."""

    def __init__(
        self,
        counter_codes: tuple[str, ...],
        descriptions: tuple[str, ...],
    ) -> None:
        self.counter_codes = counter_codes
        super().__init__(
            "native observation contains loss or anomaly counters: "
            + ", ".join(descriptions)
        )


_NATIVE_COUNTER_FAILURES: Final[
    tuple[tuple[str, str, str], ...]
] = (
    (
        "ringbuf_reserve_fail_count",
        "ring_buffer_reserve_failure",
        "ring buffer reserve failure",
    ),
    (
        "user_memory_read_failure_count",
        "user_memory_read_failure",
        "user-memory read failure",
    ),
    ("unmatched_call_count", "unmatched_probe_call", "unmatched probe call"),
    (
        "unsupported_mutation_count",
        "unsupported_mutation",
        "unsupported mutation",
    ),
    ("mapping_mutation_count", "mapping_mutation", "mapping mutation"),
    ("active_emitter_count", "active_emitter", "active emitter"),
)


_SNAPSHOT_FAILURE_REASONS: Final[dict[str, str]] = {
    "native observer did not publish a drained watermark": (
        "watermark_not_published"
    ),
    "native observer drain delivery mismatch": (
        "native_drain_delivery_mismatch"
    ),
    "native observer drain has an active emitter": (
        "native_drain_active_emitter"
    ),
    "native observer drain did not reach stable parity": (
        "native_drain_unstable_parity"
    ),
    "native mutation probes are not ready to arm": "mutation_probe_not_ready",
    "native observer did not arm mutation probes": "mutation_probe_arm_timeout",
    "native observer control channel failed": "native_control_channel_failed",
    "native observer diagnostic line is invalid": "native_diagnostic_invalid",
    "native observer failed to attach mutation probes": (
        "mutation_probe_attach_failed"
    ),
    "native observation contains loss or anomaly counters": (
        "native_loss_or_anomaly_counters"
    ),
    "native terminal event counter mismatch": "native_event_counter_mismatch",
    "native delivery counter mismatch": "native_delivery_counter_mismatch",
    "native terminal has an inflight call": "native_inflight_call",
    "native structural evidence occurs after the first watermark cutoff": (
        "native_structural_evidence_after_cutoff"
    ),
    "native structural evidence follows the first watermark": (
        "native_structural_evidence_after_first_watermark"
    ),
    "native watermark cutoff regressed or was repeated": (
        "native_watermark_cutoff_regressed"
    ),
    "native watermark counters regressed": "native_watermark_counters_regressed",
    "native frame sequence is not contiguous": "native_frame_sequence_gap",
    "native event_id is duplicated": "native_event_id_duplicated",
    "native apply completion is duplicated": "native_apply_completion_duplicated",
    "native mutation code or reserved field is invalid": (
        "native_mutation_frame_invalid"
    ),
    "incremental native stream does not extend prior evidence": (
        "native_stream_prefix_changed"
    ),
    "runtime topology event must be complete and unique": (
        "runtime_topology_not_unique"
    ),
    "duplicate runtime tensor layout event": "runtime_tensor_layout_duplicated",
    "runtime tensor layout is not complete": "runtime_tensor_layout_incomplete",
    "KV backing provenance is missing": "kv_backing_provenance_missing",
    "KV backing ranges overlap": "kv_backing_ranges_overlap",
    "runtime tensor is outside KV backing provenance": (
        "runtime_tensor_outside_kv_backing"
    ),
    "native event belongs to another runtime process": (
        "native_event_runtime_identity_mismatch"
    ),
    "native predecode slot reset must be unique": "predecode_reset_not_unique",
    "forbidden operational runtime mutation observed": (
        "forbidden_operational_mutation"
    ),
    "runtime mutation occurred after decode began": "mutation_after_decode_began",
    "sequence mapping mutation counter mismatch": (
        "sequence_mapping_counter_mismatch"
    ),
    "native apply completion has no matching token cells": (
        "apply_completion_without_token_cells"
    ),
    "native apply call topology is inconsistent": (
        "apply_call_topology_inconsistent"
    ),
    "snapshot interval extends beyond the native cutoff": (
        "interval_beyond_native_cutoff"
    ),
    "sequence stream mapping changed inside the snapshot": (
        "sequence_stream_mapping_changed"
    ),
    "target sequence is not quiescent: apply completion is missing": (
        "apply_completion_missing"
    ),
    "target sequence is not quiescent after the requested cut": (
        "sequence_activity_after_cut"
    ),
}


def _snapshot_failure_reason(error: BaseException) -> str:
    """Return only a fixed diagnostic code from a bounded exception chain."""

    reason = "unclassified"
    current: BaseException | None = error
    visited: set[int] = set()
    for _ in range(8):
        if current is None or id(current) in visited:
            break
        visited.add(id(current))
        mapped = _SNAPSHOT_FAILURE_REASONS.get(str(current))
        if mapped is not None:
            reason = mapped
        if isinstance(current, NativeObservationCounterError):
            reason = "native_counter_" + "_and_".join(
                current.counter_codes
            )
        current = current.__cause__ or current.__context__
    return reason


def _report_runtime_state_service_failure(error: BaseException) -> None:
    """Report the public failure plus one value-free root-cause code."""

    print(f"runtime-state service failed: {error}", file=sys.stderr)
    print(
        "runtime-state startup diagnostic: reason="
        + _snapshot_failure_reason(error),
        file=sys.stderr,
        flush=True,
    )


@dataclass(frozen=True)
class VerifiedProbeTargetIdentity:
    probe_id: str
    owner_role: str
    path_display: str
    device_major: int
    device_minor: int
    inode: int
    sha256: str
    elf_build_id: str
    symbol: str
    resolved_file_offset: int
    attachment_path: str


@dataclass(frozen=True)
class VerifiedRuntimeIdentity:
    pid: int
    start_time_ticks: int
    runtime_binary_sha256: str
    qualified_model_sha256: str
    mapped_libllama_sha256: str
    mapped_libllama_elf_build_id: str
    probe_targets: tuple[VerifiedProbeTargetIdentity, ...]
    required_launch_arguments: tuple[str, ...]


def _same_runtime_security_identity(
    attached: VerifiedRuntimeIdentity,
    live: VerifiedRuntimeIdentity,
) -> bool:
    if attached is live:
        return True

    def probe_identity(
        target: VerifiedProbeTargetIdentity,
    ) -> tuple[object, ...]:
        return (
            target.probe_id,
            target.owner_role,
            target.device_major,
            target.device_minor,
            target.inode,
            target.sha256,
            target.elf_build_id,
            target.symbol,
            target.resolved_file_offset,
        )

    return (
        attached.pid,
        attached.start_time_ticks,
        attached.runtime_binary_sha256,
        attached.qualified_model_sha256,
        attached.mapped_libllama_sha256,
        attached.mapped_libllama_elf_build_id,
        tuple(probe_identity(target) for target in attached.probe_targets),
        attached.required_launch_arguments,
    ) == (
        live.pid,
        live.start_time_ticks,
        live.runtime_binary_sha256,
        live.qualified_model_sha256,
        live.mapped_libllama_sha256,
        live.mapped_libllama_elf_build_id,
        tuple(probe_identity(target) for target in live.probe_targets),
        live.required_launch_arguments,
    )


def _probe_identity_document(
    identity: VerifiedProbeTargetIdentity,
) -> bytes:
    return _canonical_json(
        {
            "elf_build_id": identity.elf_build_id,
            "inode": identity.inode,
            "owner_role": identity.owner_role,
            "probe_id": identity.probe_id,
            "resolved_file_offset": identity.resolved_file_offset,
            "sha256": identity.sha256,
            "symbol": identity.symbol,
        }
    )


def _probe_identity_diagnostic(
    identity: VerifiedProbeTargetIdentity,
) -> bytes:
    diagnostic = b"ATTACHED_PROBE runtime_state " + _probe_identity_document(identity)
    if len(diagnostic) >= 1024 or b"\n" in diagnostic or b"\r" in diagnostic:
        raise RuntimeStateServiceError(
            "verified probe identity diagnostic exceeds the protocol limit"
        )
    return diagnostic


def _validate_verified_runtime_binding(
    profile: RuntimeStateObserverProfile,
    verified: VerifiedRuntimeIdentity,
) -> None:
    expected = (
        ("runtime binary", verified.runtime_binary_sha256, profile.runtime_binary_sha256),
        ("libllama digest", verified.mapped_libllama_sha256, profile.mapped_libllama_sha256),
        (
            "libllama build ID",
            verified.mapped_libllama_elf_build_id,
            profile.mapped_libllama_elf_build_id,
        ),
        (
            "launch arguments",
            verified.required_launch_arguments,
            profile.required_launch_arguments,
        ),
    )
    if not profile.is_model_generic:
        expected += (
            (
                "qualified model",
                verified.qualified_model_sha256,
                profile.qualified_model_sha256,
            ),
        )
    elif re.fullmatch(r"sha256:[0-9a-f]{64}", verified.qualified_model_sha256) is None:
        raise RuntimeStateServiceError(
            "verified runtime registered model digest is invalid"
        )
    for field, observed, required in expected:
        if type(observed) is not type(required) or observed != required:
            raise RuntimeStateServiceError(
                f"verified runtime {field} contradicts the profile"
            )
    if (
        isinstance(verified.pid, bool)
        or not isinstance(verified.pid, int)
        or verified.pid <= 0
        or isinstance(verified.start_time_ticks, bool)
        or not isinstance(verified.start_time_ticks, int)
        or verified.start_time_ticks <= 0
    ):
        raise RuntimeStateServiceError(
            "verified runtime process identity is invalid"
        )
    if not profile.is_linux_live_attachable:
        if verified.probe_targets:
            raise RuntimeStateServiceError(
                "pending profile cannot carry verified probe targets"
            )
        return
    required_targets = {item.probe_id: item for item in profile.probe_targets}
    observed_targets = {item.probe_id: item for item in verified.probe_targets}
    if len(observed_targets) != len(verified.probe_targets) or set(observed_targets) != set(required_targets):
        raise RuntimeStateServiceError(
            "verified runtime probe targets contradict the profile"
        )
    for probe_id, target in required_targets.items():
        observed = observed_targets[probe_id]
        live_attachment = re.fullmatch(
            rf"/proc/{verified.pid}/map_files/[0-9a-f]+-[0-9a-f]+",
            observed.attachment_path,
        ) is not None
        if (
            observed.owner_role != target.owner_role
            or (
                observed.path_display.removesuffix(" (deleted)") != str(target.path)
                and not live_attachment
            )
            or observed.inode <= 0
            or observed.sha256 != target.sha256
            or observed.elf_build_id != target.elf_build_id
            or observed.symbol != target.symbol
            or observed.resolved_file_offset != target.resolved_file_offset
        ):
            raise RuntimeStateServiceError(
                f"verified runtime probe {probe_id} contradicts the profile"
            )


def _read_proc_cmdline(pid: int, proc_root: Path = Path("/proc")) -> tuple[str, ...]:
    try:
        raw = (proc_root / str(pid) / "cmdline").read_bytes()
        values = tuple(
            item.decode("utf-8", errors="strict")
            for item in raw.split(b"\x00")
            if item
        )
    except (OSError, UnicodeError) as error:
        raise RuntimeStateServiceError("runtime command line could not be read") from error
    if not values:
        raise RuntimeStateServiceError("runtime command line is empty")
    return values


def _read_live_model_digest(pid: int, model_path: object) -> str:
    relative = str(model_path).lstrip("/")
    path = Path("/proc") / str(pid) / "root" / relative
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise RuntimeStateServiceError("qualified live model could not be opened") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeStateServiceError("qualified live model is not a regular file")
        digest = hashlib.sha256()
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
        return "sha256:" + digest.hexdigest()
    finally:
        os.close(descriptor)


def _verify_live_registered_model(
    pid: int, registered_model: RegisteredModelHandle
) -> None:
    relative = str(registered_model.path).lstrip("/")
    path = Path("/proc") / str(pid) / "root" / relative
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise RuntimeStateServiceError(
            "registered live model could not be opened"
        ) from error
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or int(metadata.st_dev) != registered_model.device
            or int(metadata.st_ino) != registered_model.inode
            or int(metadata.st_size) != registered_model.size_bytes
            or int(metadata.st_mtime_ns) != registered_model.mtime_ns
        ):
            raise RuntimeStateServiceError(
                "live model identity differs from its registration"
            )
    finally:
        os.close(descriptor)


def _validate_live_launch_arguments(
    cmdline: tuple[str, ...],
    profile: RuntimeStateObserverProfile,
    *,
    selected_model_path: object | None = None,
) -> None:
    required_options: dict[str, str | None] = {}
    required = profile.required_launch_arguments
    index = 0
    while index < len(required):
        option = required[index]
        if not option.startswith("--") or "=" in option:
            raise RuntimeStateServiceError(
                "qualified launch policy is not canonical split-form argv"
            )
        value: str | None = None
        if index + 1 < len(required) and not required[index + 1].startswith("-"):
            value = required[index + 1]
            index += 1
        if option in required_options:
            raise RuntimeStateServiceError(
                f"qualified launch policy repeats argument {option}"
            )
        required_options[option] = value
        index += 1

    seen: set[str] = set()
    index = 0
    while index < len(cmdline):
        argument = cmdline[index]
        option = argument.split("=", 1)[0]
        if option in _FORBIDDEN_LAUNCH_ALIASES:
            raise RuntimeStateServiceError(
                f"runtime launch argument alias {option} is not permitted"
            )
        if option in _FORBIDDEN_LAUNCH_OPTIONS:
            raise RuntimeStateServiceError(
                f"runtime launch argument {option} violates the qualified policy"
            )
        if option not in required_options:
            index += 1
            continue
        if argument != option:
            raise RuntimeStateServiceError(
                f"runtime launch argument {option} is missing, repeated, or overridden; "
                f"observed noncanonical form {argument!r}"
            )
        if option in seen:
            raise RuntimeStateServiceError(
                f"runtime launch argument {option} is missing, repeated, or overridden; "
                "observed_count is greater than one"
            )
        seen.add(option)
        expected_value = required_options[option]
        if expected_value is not None:
            observed_value = (
                None if index + 1 >= len(cmdline) else cmdline[index + 1]
            )
            if observed_value != expected_value:
                raise RuntimeStateServiceError(
                    f"runtime launch argument {option} is missing, repeated, or overridden; "
                    f"expected={expected_value!r}, observed={observed_value!r}"
                )
            index += 1
        index += 1

    missing = tuple(option for option in required_options if option not in seen)
    if missing:
        raise RuntimeStateServiceError(
            f"runtime launch argument {missing[0]} is missing, repeated, or overridden; "
            "observed_count=0"
        )

    model_bindings: list[str] = []
    index = 0
    while index < len(cmdline):
        argument = cmdline[index]
        if argument in {"-m", "--model"}:
            if index + 1 >= len(cmdline):
                raise RuntimeStateServiceError(
                    "runtime model argument has no value"
                )
            model_bindings.append(cmdline[index + 1])
            index += 2
            continue
        if argument.startswith("-m=") or argument.startswith("--model="):
            raise RuntimeStateServiceError(
                "runtime model argument equals-form is not permitted"
            )
        index += 1
    expected_model_path = (
        selected_model_path
        if selected_model_path is not None
        else profile.qualified_model_path
    )
    if expected_model_path is None or model_bindings != [str(expected_model_path)]:
        raise RuntimeStateServiceError(
            "runtime model argument is missing, repeated, or overridden"
        )


def _profile_file_identity(path: object) -> tuple[os.stat_result, str, int]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise RuntimeStateServiceError(
                "profile attachment input is not a unique regular file"
            )
        digest = hashlib.sha256()
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
        os.lseek(descriptor, 0, os.SEEK_SET)
        return metadata, "sha256:" + digest.hexdigest(), descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _prepare_runtime_attachment_identity(
    *,
    profile: RuntimeStateObserverProfile,
    runtime_pid: int,
    expected_start_time_ticks: int,
    backend: ProcessInspectionBackend | None = None,
    registered_model: RegisteredModelHandle | None = None,
    runtime_binary_path: Path | None = None,
    probe_path_resolver: Callable[[Path], Path] | None = None,
) -> VerifiedRuntimeIdentity:
    """Pin fixed probe files while the eventual runtime PID is still gated."""

    if not isinstance(profile, RuntimeStateObserverProfile):
        raise RuntimeStateServiceError("profile must be RuntimeStateObserverProfile")
    if not profile.is_linux_live_attachable:
        raise RuntimeStateServiceError(
            "runtime-state profile is linux_live_pending: "
            + profile.qualification_reason
        )
    if (
        isinstance(runtime_pid, bool)
        or not isinstance(runtime_pid, int)
        or runtime_pid <= 0
        or isinstance(expected_start_time_ticks, bool)
        or not isinstance(expected_start_time_ticks, int)
        or expected_start_time_ticks <= 0
    ):
        raise RuntimeStateServiceError("runtime process identity is invalid")
    active_backend = backend or SystemProcessInspectionBackend()
    try:
        if active_backend.read_start_time_ticks(runtime_pid) != expected_start_time_ticks:
            raise RuntimeStateServiceError(
                "pre-exec gate start_time_ticks does not match the target process"
            )
        selected_runtime_binary = runtime_binary_path or profile.runtime_binary_path
        runtime_metadata, runtime_digest, runtime_fd = _profile_file_identity(
            selected_runtime_binary
        )
        os.close(runtime_fd)
        if runtime_digest != profile.runtime_binary_sha256:
            raise RuntimeStateServiceError(
                "fixed runtime executable digest does not match the profile"
            )
        if profile.is_model_generic:
            if not isinstance(registered_model, RegisteredModelHandle):
                raise RuntimeStateServiceError(
                    "dynamic observer profile requires a registered model"
                )
            model_digest = registered_model.digest
        else:
            model_metadata, model_digest, model_fd = _profile_file_identity(
                profile.qualified_model_path
            )
            os.close(model_fd)
            if model_digest != profile.qualified_model_sha256:
                raise RuntimeStateServiceError(
                    "fixed qualified model digest does not match the profile"
                )
            del model_metadata
        del runtime_metadata

        verified: list[VerifiedProbeTargetIdentity] = []
        for target in profile.probe_targets:
            attachment_path = (
                probe_path_resolver(target.path)
                if probe_path_resolver is not None
                else target.path
            )
            metadata, digest, descriptor = _profile_file_identity(attachment_path)
            try:
                symbol = inspect_elf(descriptor, target.symbol)
            finally:
                os.close(descriptor)
            if (
                digest != target.sha256
                or symbol.elf_build_id != target.elf_build_id
                or symbol.name != target.symbol
                or symbol.resolved_file_offset != target.resolved_file_offset
            ):
                raise RuntimeStateServiceError(
                    f"probe {target.probe_id} fixed ELF identity or symbol offset mismatch"
                )
            if hasattr(os, "major") and hasattr(os, "minor"):
                device_major = os.major(metadata.st_dev)
                device_minor = os.minor(metadata.st_dev)
            else:
                device_major = 0
                device_minor = int(metadata.st_dev)
            verified.append(
                VerifiedProbeTargetIdentity(
                    probe_id=target.probe_id,
                    owner_role=target.owner_role,
                    path_display=str(target.path),
                    device_major=device_major,
                    device_minor=device_minor,
                    inode=int(metadata.st_ino),
                    sha256=digest,
                    elf_build_id=symbol.elf_build_id,
                    symbol=symbol.name,
                    resolved_file_offset=symbol.resolved_file_offset,
                    attachment_path=str(attachment_path),
                )
            )
        if active_backend.read_start_time_ticks(runtime_pid) != expected_start_time_ticks:
            raise RuntimeStateServiceError(
                "pre-exec gate identity changed during attachment preparation"
            )
    except (ElfInspectionError, ProcessInspectionError, OSError) as error:
        raise RuntimeStateServiceError(
            "runtime attachment identity preparation failed"
        ) from error

    attachment = VerifiedRuntimeIdentity(
        pid=runtime_pid,
        start_time_ticks=expected_start_time_ticks,
        runtime_binary_sha256=runtime_digest,
        qualified_model_sha256=model_digest,
        mapped_libllama_sha256=profile.mapped_libllama_sha256,
        mapped_libllama_elf_build_id=profile.mapped_libllama_elf_build_id,
        probe_targets=tuple(sorted(verified, key=lambda item: item.probe_id)),
        required_launch_arguments=profile.required_launch_arguments,
    )
    _validate_verified_runtime_binding(profile, attachment)
    return attachment


def verify_live_runtime_identity(
    *,
    profile: RuntimeStateObserverProfile,
    runtime_pid: int,
    expected_start_time_ticks: int,
    backend: ProcessInspectionBackend | None = None,
    cmdline_reader: Callable[[int], tuple[str, ...]] | None = None,
    model_digest_reader: Callable[[int, object], str] | None = None,
    registered_model: RegisteredModelHandle | None = None,
) -> VerifiedRuntimeIdentity:
    """Pin every qualified probe to its live mapped ELF, failing closed."""
    if not isinstance(profile, RuntimeStateObserverProfile):
        raise RuntimeStateServiceError("profile must be RuntimeStateObserverProfile")
    if not profile.is_linux_live_attachable:
        raise RuntimeStateServiceError(
            "runtime-state profile is linux_live_pending: "
            + profile.qualification_reason
        )
    if (
        isinstance(runtime_pid, bool)
        or not isinstance(runtime_pid, int)
        or runtime_pid <= 0
        or isinstance(expected_start_time_ticks, bool)
        or not isinstance(expected_start_time_ticks, int)
        or expected_start_time_ticks <= 0
    ):
        raise RuntimeStateServiceError("runtime process identity is invalid")
    active_backend = backend or SystemProcessInspectionBackend()
    try:
        before = active_backend.read_start_time_ticks(runtime_pid)
        if before != expected_start_time_ticks:
            raise RuntimeStateServiceError(
                "runtime start_time_ticks does not match the live process"
            )
        executable = active_backend.open_executable(runtime_pid)
        try:
            executable_identity = executable.identity
        finally:
            os.close(executable.fd)
        if executable_identity.digest != profile.runtime_binary_sha256:
            raise RuntimeStateServiceError(
                "runtime executable digest does not match the profile"
            )
        cmdline = (cmdline_reader or _read_proc_cmdline)(runtime_pid)
        if profile.is_model_generic:
            if not isinstance(registered_model, RegisteredModelHandle):
                raise RuntimeStateServiceError(
                    "dynamic observer profile requires a registered model"
                )
            selected_model_path = registered_model.path
            _verify_live_registered_model(runtime_pid, registered_model)
            model_digest = registered_model.digest
        else:
            selected_model_path = profile.qualified_model_path
            model_digest = (model_digest_reader or _read_live_model_digest)(
                runtime_pid, profile.qualified_model_path
            )
            if model_digest != profile.qualified_model_sha256:
                raise RuntimeStateServiceError(
                    "live model digest does not match the qualified profile"
                )
        if str(selected_model_path) not in cmdline:
            raise RuntimeStateServiceError(
                "runtime command line does not select the registered model"
            )
        _validate_live_launch_arguments(
            cmdline, profile, selected_model_path=selected_model_path
        )
        maps = active_backend.read_maps(runtime_pid)
        executable_mappings: dict[
            tuple[int, int, int], tuple[object, object]
        ] = {}
        for mapping in maps:
            if mapping.inode <= 0 or mapping.permissions[2] != "x":
                continue
            key = mapping.identity_key
            current = executable_mappings.get(key)
            if current is None or mapping.start < current[0].start:
                executable_mappings[key] = (mapping, None)
        for key, (mapping, _pinned) in tuple(executable_mappings.items()):
            pinned = active_backend.open_mapping(runtime_pid, mapping)
            if trusted_file_key(pinned.identity)[:3] != key:
                os.close(pinned.fd)
                raise RuntimeStateServiceError(
                    "probe ELF mapping inode changed while opening"
                )
            executable_mappings[key] = (mapping, pinned)
        verified: list[VerifiedProbeTargetIdentity] = []
        try:
            for target in profile.probe_targets:
                candidates = tuple(
                    (mapping, pinned)
                    for mapping, pinned in executable_mappings.values()
                    if pinned.identity.digest == target.sha256
                )
                if len(candidates) != 1:
                    raise RuntimeStateServiceError(
                        f"probe {target.probe_id} does not have one approved live ELF mapping"
                    )
                mapping, pinned = candidates[0]
                symbol = inspect_elf(pinned.fd, target.symbol)
                identity = pinned.identity
                if (
                    symbol.elf_build_id != target.elf_build_id
                    or symbol.name != target.symbol
                    or symbol.resolved_file_offset != target.resolved_file_offset
                ):
                    raise RuntimeStateServiceError(
                        f"probe {target.probe_id} live ELF identity or symbol offset mismatch"
                    )
                verified.append(
                    VerifiedProbeTargetIdentity(
                        probe_id=target.probe_id,
                        owner_role=target.owner_role,
                        path_display=identity.path_display,
                        device_major=identity.device_major,
                        device_minor=identity.device_minor,
                        inode=identity.inode,
                        sha256=identity.digest,
                        elf_build_id=symbol.elf_build_id,
                        symbol=symbol.name,
                        resolved_file_offset=symbol.resolved_file_offset,
                        attachment_path=(
                            f"/proc/{runtime_pid}/map_files/"
                            f"{mapping.start:x}-{mapping.end:x}"
                        ),
                    )
                )
        finally:
            for _mapping, pinned in executable_mappings.values():
                if pinned is not None:
                    os.close(pinned.fd)
        confirmed_maps = active_backend.read_maps(runtime_pid)
        confirmed_keys = {
            entry.identity_key
            for entry in confirmed_maps
            if entry.inode > 0 and entry.permissions[2] == "x"
        }
        if any(
            (item.device_major, item.device_minor, item.inode) not in confirmed_keys
            for item in verified
        ):
            raise RuntimeStateServiceError(
                "probe ELF mapping changed during live verification"
            )
        after = active_backend.read_start_time_ticks(runtime_pid)
        if after != before:
            raise RuntimeStateServiceError(
                "runtime process identity changed during live verification"
            )
    except (ElfInspectionError, ProcessInspectionError, OSError) as error:
        raise RuntimeStateServiceError(
            "runtime live ELF verification failed"
        ) from error
    libllama = {
        (item.sha256, item.elf_build_id)
        for item in verified
        if item.owner_role == "libllama"
    }
    if len(libllama) != 1:
        raise RuntimeStateServiceError(
            "qualified probes do not resolve to one verified libllama"
        )
    libllama_sha256, libllama_build_id = next(iter(libllama))
    return VerifiedRuntimeIdentity(
        pid=runtime_pid,
        start_time_ticks=before,
        runtime_binary_sha256=executable_identity.digest,
        qualified_model_sha256=model_digest,
        mapped_libllama_sha256=libllama_sha256,
        mapped_libllama_elf_build_id=libllama_build_id,
        probe_targets=tuple(sorted(verified, key=lambda item: item.probe_id)),
        required_launch_arguments=profile.required_launch_arguments,
    )


def _wait_for_live_runtime_identity(
    *,
    profile: RuntimeStateObserverProfile,
    runtime_pid: int,
    expected_start_time_ticks: int,
    timeout_seconds: float,
    verifier: Callable[..., VerifiedRuntimeIdentity] | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    registered_model: RegisteredModelHandle | None = None,
) -> VerifiedRuntimeIdentity:
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or timeout_seconds <= 0
    ):
        raise RuntimeStateServiceError("live runtime identity timeout must be positive")
    active_verifier = verifier or verify_live_runtime_identity
    deadline = monotonic() + float(timeout_seconds)
    last_error: RuntimeStateServiceError | None = None
    while True:
        try:
            return active_verifier(
                profile=profile,
                runtime_pid=runtime_pid,
                expected_start_time_ticks=expected_start_time_ticks,
                registered_model=registered_model,
            )
        except RuntimeStateServiceError as error:
            last_error = error
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise RuntimeStateServiceError(
                "runtime did not reach its verified post-exec identity; "
                f"last verification error: {last_error}"
            ) from last_error
        sleep(min(0.01, remaining))


def _require_uint(value: object, field: str, bits: int = 64) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value >= 1 << bits
    ):
        raise NativeRuntimeStateWireError(
            f"{field} must be an unsigned {bits}-bit integer"
        )
    return value


@dataclass(frozen=True)
class RuntimeTopologyEvent:
    event_id: int
    tgid: int
    tid: int
    monotonic_ns: int
    n_ctx: int
    n_stream: int
    n_seq_max: int
    n_layer: int
    v_trans: bool
    n_pos_per_embd: int

    def __post_init__(self) -> None:
        if _require_uint(self.event_id, "event_id") == 0:
            raise NativeRuntimeStateWireError("event_id must be positive")
        for field in ("tgid", "tid", "n_ctx", "n_stream", "n_seq_max", "n_layer", "n_pos_per_embd"):
            value = _require_uint(getattr(self, field), field, 32)
            if value == 0:
                raise NativeRuntimeStateWireError(f"{field} must be positive")
        _require_uint(self.monotonic_ns, "monotonic_ns")
        if not isinstance(self.v_trans, bool):
            raise NativeRuntimeStateWireError("v_trans must be a boolean")


@dataclass(frozen=True)
class TensorLayoutEvent:
    event_id: int
    tgid: int
    tid: int
    monotonic_ns: int
    kind: str
    layer: int
    stream_index: int
    tensor_address: int
    row_size: int
    cell_capacity: int
    ggml_type_id: int
    element_size: int
    embedding_width: int

    def __post_init__(self) -> None:
        if _require_uint(self.event_id, "event_id") == 0:
            raise NativeRuntimeStateWireError("event_id must be positive")
        for field in ("tgid", "tid"):
            if _require_uint(getattr(self, field), field, 32) == 0:
                raise NativeRuntimeStateWireError(f"{field} must be positive")
        _require_uint(self.monotonic_ns, "monotonic_ns")
        if self.kind not in {"k", "v"}:
            raise NativeRuntimeStateWireError("kind must be k or v")
        for field in ("layer", "stream_index", "ggml_type_id"):
            _require_uint(getattr(self, field), field, 32)
        for field in (
            "tensor_address",
            "row_size",
            "cell_capacity",
            "element_size",
            "embedding_width",
        ):
            if _require_uint(getattr(self, field), field) == 0:
                raise NativeRuntimeStateWireError(f"{field} must be positive")


@dataclass(frozen=True)
class NativeApplyCompleteEvent:
    apply_call_id: int
    tgid: int
    tid: int
    monotonic_ns: int

    def __post_init__(self) -> None:
        if _require_uint(self.apply_call_id, "apply_call_id") == 0:
            raise NativeRuntimeStateWireError("apply_call_id must be positive")
        for field in ("tgid", "tid"):
            if _require_uint(getattr(self, field), field, 32) == 0:
                raise NativeRuntimeStateWireError(f"{field} must be positive")
        _require_uint(self.monotonic_ns, "monotonic_ns")


@dataclass(frozen=True)
class ObservedApplyTokenCell:
    seq_id: int
    stream_index: int
    token_position: int
    cell_index: int
    ownership_generation: int


@dataclass(frozen=True)
class ObservedApplyBoundary:
    cursor: int
    runtime_instance_id: str
    runtime_pid: int
    runtime_start_time_ticks: int
    apply_call_id: int
    monotonic_ns: int
    sequence_ids: tuple[int, ...]
    token_cells: tuple[ObservedApplyTokenCell, ...]
    mutation_kinds: tuple[str, ...]
    observer_loss_count: int
    parse_failure_count: int
    unmatched_call_count: int
    unsupported_mutation_count: int
    watermark_monotonic_ns: int


@dataclass(frozen=True)
class CompletedApplyPage:
    next_cursor: int
    truncated: bool
    boundaries: tuple[ObservedApplyBoundary, ...]


_MUTATION_KIND_BY_CODE: Final[dict[int, str]] = {
    1: "cache_shift",
    2: "context_shift",
    3: "kv_defragmentation",
    4: "multimodal_position",
    5: "sequence_copy",
    6: "sequence_remove",
    7: "sequence_stream_remap",
    8: "speculative_context",
}
_MUTATION_CODE_BY_KIND: Final[dict[str, int]] = {
    value: key for key, value in _MUTATION_KIND_BY_CODE.items()
}


@dataclass(frozen=True)
class NativeMutationEvent:
    event_id: int
    tgid: int
    tid: int
    monotonic_ns: int
    mutation_kind: str

    def __post_init__(self) -> None:
        if _require_uint(self.event_id, "event_id") == 0:
            raise NativeRuntimeStateWireError("event_id must be positive")
        for field in ("tgid", "tid"):
            if _require_uint(getattr(self, field), field, 32) == 0:
                raise NativeRuntimeStateWireError(f"{field} must be positive")
        _require_uint(self.monotonic_ns, "monotonic_ns")
        if self.mutation_kind not in _MUTATION_CODE_BY_KIND:
            raise NativeRuntimeStateWireError("mutation_kind is invalid")


@dataclass(frozen=True)
class NativeMutationPolicySummary:
    predecode_reset_event_count: int
    operational_mutation_event_count: int

    def __post_init__(self) -> None:
        for field in self.__dataclass_fields__:
            _require_uint(getattr(self, field), field)


@dataclass(frozen=True)
class NativeObservationCounters:
    cutoff_monotonic_ns: int
    topology_event_count: int
    tensor_layout_event_count: int
    kv_backing_event_count: int
    token_cell_event_count: int
    apply_complete_event_count: int
    kernel_event_submit_count: int
    native_event_receive_count: int
    native_wire_frame_count: int
    ringbuf_reserve_fail_count: int
    user_memory_read_failure_count: int
    unmatched_call_count: int
    unsupported_mutation_count: int
    mapping_mutation_count: int
    final_inflight_count: int
    mutation_event_count: int = 0
    active_emitter_count: int = 0

    def __post_init__(self) -> None:
        for field in self.__dataclass_fields__:
            _require_uint(getattr(self, field), field)


@dataclass(frozen=True)
class NativeWatermarkCounters(NativeObservationCounters):
    pass


@dataclass(frozen=True)
class _NativeTensorReadDiagnostics:
    candidate_missing: int
    shape_invalid: int
    data_read_failed: int
    type_read_failed: int
    data_null: int

    def __post_init__(self) -> None:
        for field in self.__dataclass_fields__:
            _require_uint(getattr(self, field), field)


@dataclass(frozen=True)
class _NativeApplyRejectionDiagnostics:
    state_conflict: int
    header_read_failed: int
    slot_shape_invalid: int
    batch_shape_invalid: int
    stream_count_zero: int
    stream_count_exceeds: int
    index_vector_extent_invalid: int
    token_count_zero: int
    token_count_exceeds: int
    token_count_exceeds_observed: int
    token_count_exceeds_streams: int
    token_stream_remainder: int
    token_read_failed: int
    token_shape_invalid: int
    return_unmatched: int
    state_delete_failed: int
    process_state_conflict: int
    process_return_unmatched: int
    process_apply_missing: int
    process_graph_compute_failed: int

    def __post_init__(self) -> None:
        for field in self.__dataclass_fields__:
            _require_uint(getattr(self, field), field)


@dataclass(frozen=True)
class NativeTerminalCounters(NativeObservationCounters):
    pass


@dataclass(frozen=True)
class RuntimeStateTopologySummary:
    n_ctx: int
    n_layer: int
    n_pos_per_embd: int
    n_seq_max: int
    n_stream: int
    profile_digest: str
    v_trans: bool


@dataclass(frozen=True)
class RuntimeStateTerminalSeal:
    runtime_instance_id: str
    runtime_pid: int
    runtime_start_time_ticks: int
    runtime_binary_sha256: str
    qualified_model_sha256: str
    mapped_libllama_sha256: str
    mapped_libllama_elf_build_id: str
    profile_digest: str
    memory_family: str
    native_stream_artifact: str
    native_stream_sha256: str
    topology: RuntimeStateTopologySummary
    topology_digest: str
    watermark_frame_count: int
    native_unit_frame_count: int
    terminal_counters: NativeTerminalCounters
    seal_digest: str


@dataclass(frozen=True)
class NativeStateUnitFragment:
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

    def __post_init__(self) -> None:
        if _require_uint(self.apply_call_id, "apply_call_id") == 0:
            raise NativeRuntimeStateWireError(
                "apply_call_id must be positive"
            )
        for field in ("fragment_index", "seq_id", "stream_index", "layer"):
            _require_uint(getattr(self, field), field, 32)
        if self.stream_index >= 8:
            raise NativeRuntimeStateWireError("stream_index is invalid")
        for field in (
            "token_position",
            "cell_index",
            "span_cell_count",
            "tensor_byte_offset",
            "writer_monotonic_ns",
        ):
            _require_uint(getattr(self, field), field)
        if self.span_cell_count == 0:
            raise NativeRuntimeStateWireError(
                "span_cell_count must be positive"
            )
        if self.tensor_kind not in {"k", "v"}:
            raise NativeRuntimeStateWireError("tensor_kind must be k or v")
        if not isinstance(self.payload, bytes) or not self.payload:
            raise NativeRuntimeStateWireError(
                "state-unit fragment payload must be non-empty bytes"
            )


@dataclass(frozen=True)
class NativeStateUnitSeal:
    apply_call_id: int
    expected_fragment_count: int
    expected_token_cell_count: int
    stream_mask: int
    writer_state: str
    layout_generation: int
    backing_generation: int
    mutation_generation: int
    copy_byte_count: int
    reserve_failure_baseline: int
    reserve_failure_final: int
    writer_return_monotonic_ns: int

    def __post_init__(self) -> None:
        if _require_uint(self.apply_call_id, "apply_call_id") == 0:
            raise NativeRuntimeStateWireError(
                "apply_call_id must be positive"
            )
        for field in (
            "expected_fragment_count",
            "expected_token_cell_count",
            "stream_mask",
        ):
            value = _require_uint(getattr(self, field), field, 32)
            if value == 0:
                raise NativeRuntimeStateWireError(f"{field} must be positive")
        if self.stream_mask & ~0xFF:
            raise NativeRuntimeStateWireError("stream_mask is invalid")
        if self.writer_state != "COMPLETE":
            raise NativeRuntimeStateWireError("writer_state is invalid")
        for field in (
            "layout_generation",
            "backing_generation",
            "mutation_generation",
            "copy_byte_count",
            "reserve_failure_baseline",
            "reserve_failure_final",
            "writer_return_monotonic_ns",
        ):
            _require_uint(getattr(self, field), field)
        if self.copy_byte_count == 0:
            raise NativeRuntimeStateWireError(
                "copy_byte_count must be positive"
            )
        if self.reserve_failure_final != self.reserve_failure_baseline:
            raise NativeRuntimeStateWireError(
                "successful state-unit seal contains a reserve failure"
            )


@dataclass(frozen=True)
class ParsedNativeRuntimeStateStream:
    topology_events: tuple[RuntimeTopologyEvent, ...]
    tensor_layout_events: tuple[TensorLayoutEvent, ...]
    backing_events: tuple[KvBackingEvent, ...]
    token_cells: tuple[TokenCellEvent, ...]
    token_cell_ranges: tuple[NativeTokenCellRange, ...]
    apply_completions: tuple[NativeApplyCompleteEvent, ...]
    mutations: tuple[NativeMutationEvent, ...]
    stream_versions: tuple[NativeStreamVersionSnapshot, ...]
    state_unit_fragments: tuple[NativeStateUnitFragment, ...]
    state_unit_seals: tuple[NativeStateUnitSeal, ...]
    watermarks: tuple[NativeWatermarkCounters, ...]
    terminal: NativeTerminalCounters | None


def validate_native_runtime_state_mutations(
    parsed: ParsedNativeRuntimeStateStream,
    *,
    require_predecode_reset: bool = False,
    bound_decode_token_cells: tuple[TokenCellEvent, ...] | None = None,
) -> NativeMutationPolicySummary:
    """Validate normal request-start resets without hiding raw events."""

    if not isinstance(parsed, ParsedNativeRuntimeStateStream):
        raise RuntimeStateServiceError("native mutation stream is invalid")
    if not isinstance(require_predecode_reset, bool):
        raise RuntimeStateServiceError("predecode reset requirement is invalid")
    if bound_decode_token_cells is not None:
        if (
            not require_predecode_reset
            or not isinstance(bound_decode_token_cells, tuple)
            or not bound_decode_token_cells
            or any(
                not isinstance(item, TokenCellEvent)
                for item in bound_decode_token_cells
            )
            or any(
                not contains_token_cell(
                    parsed.token_cells,
                    parsed.token_cell_ranges,
                    item,
                )
                for item in bound_decode_token_cells
            )
        ):
            raise RuntimeStateServiceError(
                "bound predecode token-cell evidence is invalid"
            )
    mutations = parsed.mutations
    if not mutations:
        if require_predecode_reset:
            raise RuntimeStateServiceError(
                "native predecode slot reset is missing"
            )
        return NativeMutationPolicySummary(0, 0)
    if any(item.mutation_kind != "sequence_remove" for item in mutations):
        raise RuntimeStateServiceError(
            "forbidden operational runtime mutation observed"
        )
    if not require_predecode_reset:
        return NativeMutationPolicySummary(
            predecode_reset_event_count=len(mutations),
            operational_mutation_event_count=0,
        )
    if len(mutations) != 1:
        raise RuntimeStateServiceError(
            "native predecode slot reset must be unique"
        )
    reset = mutations[0]
    decode_evidence: tuple[TokenCellEvent | NativeTokenCellRange, ...] = (
        tuple(parsed.token_cells) + tuple(parsed.token_cell_ranges)
    )
    if parsed.watermarks:
        request_window_start_ns = parsed.watermarks[0].cutoff_monotonic_ns
        if reset.monotonic_ns > request_window_start_ns:
            decode_evidence = tuple(
                event
                for event in decode_evidence
                if event.monotonic_ns > request_window_start_ns
            )
        elif len(parsed.watermarks) > 1:
            raise RuntimeStateServiceError(
                "native predecode slot reset precedes the request observation window"
            )
    if bound_decode_token_cells is not None:
        decode_evidence = bound_decode_token_cells
    if decode_evidence and reset.monotonic_ns >= min(
        event.monotonic_ns for event in decode_evidence
    ):
        raise RuntimeStateServiceError(
            "runtime mutation occurred after decode began"
        )
    return NativeMutationPolicySummary(
        predecode_reset_event_count=1,
        operational_mutation_event_count=0,
    )


def _encode_frame(
    frame_type: int,
    payload: bytes,
    sequence: int,
    *,
    channel_id: int = NATIVE_CHANNEL_METADATA,
) -> bytes:
    _require_uint(sequence, "frame_sequence")
    if sequence == 0:
        raise NativeRuntimeStateWireError("frame_sequence must be positive")
    if channel_id not in {
        NATIVE_CHANNEL_METADATA,
        NATIVE_CHANNEL_STATE_UNIT,
    }:
        raise NativeRuntimeStateWireError("native frame channel is invalid")
    expected_size = _PAYLOAD_SIZES.get(frame_type)
    if expected_size is None or len(payload) != expected_size:
        raise NativeRuntimeStateWireError("frame payload size does not match its type")
    return _HEADER.pack(
        NATIVE_WIRE_MAGIC,
        NATIVE_WIRE_VERSION,
        frame_type,
        len(payload),
        channel_id,
        sequence,
    ) + payload


def _payload_size_for_version(
    version: int,
    frame_type: int,
) -> int | None:
    if version not in {
        _LEGACY_NATIVE_WIRE_VERSION,
        _PREVIOUS_NATIVE_WIRE_VERSION,
        NATIVE_WIRE_VERSION,
    }:
        return None
    if (
        version == _LEGACY_NATIVE_WIRE_VERSION
        and frame_type == FRAME_TOKEN_CELL_RANGE
    ):
        return None
    if frame_type == FRAME_STATE_UNIT_BULK:
        return None
    if frame_type == FRAME_STREAM_VERSION:
        return (
            _STREAM_VERSION_V3.size
            if version == _LEGACY_NATIVE_WIRE_VERSION
            else _STREAM_VERSION_V4.size
        )
    return _PAYLOAD_SIZES.get(frame_type)


def _encode_state_unit_fragment_payload(
    fragment: NativeStateUnitFragment,
) -> bytes:
    if not isinstance(fragment, NativeStateUnitFragment):
        raise NativeRuntimeStateWireError(
            "fragment must be NativeStateUnitFragment"
        )
    if len(fragment.payload) > _MAX_STATE_UNIT_FRAGMENT_BYTES:
        raise NativeRuntimeStateWireError(
            "state-unit fragment payload exceeds 4,096 bytes"
        )
    metadata = _STATE_UNIT_FRAGMENT.pack(
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
    return metadata + fragment.payload


def encode_state_unit_fragment_frame(
    fragment: NativeStateUnitFragment,
    sequence: int,
) -> bytes:
    _require_uint(sequence, "frame_sequence")
    if sequence == 0:
        raise NativeRuntimeStateWireError("frame_sequence must be positive")
    payload = _encode_state_unit_fragment_payload(fragment)
    return _HEADER.pack(
        NATIVE_WIRE_MAGIC,
        NATIVE_WIRE_VERSION,
        FRAME_STATE_UNIT_FRAGMENT,
        len(payload),
        NATIVE_CHANNEL_STATE_UNIT,
        sequence,
    ) + payload


def encode_state_unit_fragment_batch_frame(
    fragments: Sequence[NativeStateUnitFragment],
    sequence: int,
) -> bytes:
    _require_uint(sequence, "frame_sequence")
    if sequence == 0:
        raise NativeRuntimeStateWireError("frame_sequence must be positive")
    records = tuple(
        _encode_state_unit_fragment_payload(fragment)
        for fragment in fragments
    )
    if not records:
        raise NativeRuntimeStateWireError(
            "state-unit fragment batch must contain a record"
        )
    record_bytes = sum(len(record) for record in records)
    if record_bytes > _MAX_STATE_UNIT_FRAGMENT_BATCH_RECORD_BYTES:
        raise NativeRuntimeStateWireError(
            "state-unit fragment batch exceeds its record-byte bound"
        )
    payload = _STATE_UNIT_FRAGMENT_BATCH.pack(
        len(records), record_bytes
    ) + b"".join(records)
    return _HEADER.pack(
        NATIVE_WIRE_MAGIC,
        NATIVE_WIRE_VERSION,
        FRAME_STATE_UNIT_FRAGMENT_BATCH,
        len(payload),
        NATIVE_CHANNEL_STATE_UNIT,
        sequence,
    ) + payload


def encode_state_unit_seal_frame(
    seal: NativeStateUnitSeal,
    sequence: int,
) -> bytes:
    if not isinstance(seal, NativeStateUnitSeal):
        raise NativeRuntimeStateWireError("seal must be NativeStateUnitSeal")
    return _encode_frame(
        FRAME_STATE_UNIT_SEAL,
        _STATE_UNIT_SEAL.pack(
            seal.apply_call_id,
            seal.expected_fragment_count,
            seal.expected_token_cell_count,
            seal.stream_mask,
            1,
            seal.layout_generation,
            seal.backing_generation,
            seal.mutation_generation,
            seal.copy_byte_count,
            seal.reserve_failure_baseline,
            seal.reserve_failure_final,
            seal.writer_return_monotonic_ns,
        ),
        sequence,
        channel_id=NATIVE_CHANNEL_STATE_UNIT,
    )


def _decode_state_unit_fragment_payload(
    payload: bytes,
) -> NativeStateUnitFragment:
    if len(payload) <= _STATE_UNIT_FRAGMENT.size:
        raise NativeRuntimeStateWireError(
            "state-unit fragment payload is truncated"
        )
    values = _STATE_UNIT_FRAGMENT.unpack(
        payload[: _STATE_UNIT_FRAGMENT.size]
    )
    if values[4] not in {1, 2} or values[11] == 0:
        raise NativeRuntimeStateWireError(
            "state-unit fragment metadata is invalid"
        )
    fragment_payload = payload[_STATE_UNIT_FRAGMENT.size :]
    if values[8] != len(fragment_payload):
        raise NativeRuntimeStateWireError(
            "state-unit fragment payload length is invalid"
        )
    return NativeStateUnitFragment(
        apply_call_id=values[0],
        fragment_index=values[1],
        seq_id=values[2],
        stream_index=values[3],
        tensor_kind="k" if values[4] == 1 else "v",
        token_position=values[5],
        cell_index=values[6],
        layer=values[7],
        span_cell_count=values[11],
        tensor_byte_offset=values[9],
        writer_monotonic_ns=values[10],
        payload=fragment_payload,
    )


def _decode_state_unit_fragment_batch_payload(
    payload: bytes,
) -> tuple[bytes, ...]:
    if len(payload) < _STATE_UNIT_FRAGMENT_BATCH.size:
        raise NativeRuntimeStateWireError(
            "state-unit fragment batch prefix is truncated"
        )
    record_count, record_bytes = _STATE_UNIT_FRAGMENT_BATCH.unpack_from(
        payload
    )
    records_payload = payload[_STATE_UNIT_FRAGMENT_BATCH.size :]
    if (
        record_count == 0
        or record_bytes != len(records_payload)
        or record_bytes > _MAX_STATE_UNIT_FRAGMENT_BATCH_RECORD_BYTES
    ):
        raise NativeRuntimeStateWireError(
            "state-unit fragment batch bounds are invalid"
        )
    records: list[bytes] = []
    offset = 0
    for _index in range(record_count):
        if len(records_payload) - offset <= _STATE_UNIT_FRAGMENT.size:
            raise NativeRuntimeStateWireError(
                "state-unit fragment batch record is truncated"
            )
        values = _STATE_UNIT_FRAGMENT.unpack_from(records_payload, offset)
        record_size = _STATE_UNIT_FRAGMENT.size + values[8]
        record_end = offset + record_size
        if (
            values[8] == 0
            or values[8] > _MAX_STATE_UNIT_FRAGMENT_BYTES
            or record_end > len(records_payload)
        ):
            raise NativeRuntimeStateWireError(
                "state-unit fragment batch record length is invalid"
            )
        record = records_payload[offset:record_end]
        _decode_state_unit_fragment_payload(record)
        records.append(record)
        offset = record_end
    if offset != len(records_payload):
        raise NativeRuntimeStateWireError(
            "state-unit fragment batch has trailing records or bytes"
        )
    return tuple(records)


def _decode_state_unit_schedule_batch_payload(
    payload: bytes,
) -> tuple[bytes, ...]:
    if len(payload) <= _STATE_UNIT_SCHEDULE_BATCH.size:
        raise NativeRuntimeStateWireError(
            "state-unit schedule batch is truncated"
        )
    (
        record_count,
        payload_size,
        apply_call_id,
        first_fragment_index,
        first_flat_index,
        seq_id,
        stream_index,
        tensor_kind,
        fragments_per_tensor,
        first_token_position,
        first_cell_index,
        chunks_per_dimension,
        token_count,
        row_size,
        cell_capacity,
        element_size,
        embedding_width,
        transposed_value,
        reserved,
        writer_monotonic_ns,
    ) = _STATE_UNIT_SCHEDULE_BATCH.unpack_from(payload)
    packed_payload = payload[_STATE_UNIT_SCHEDULE_BATCH.size :]
    if (
        record_count == 0
        or record_count > 32
        or payload_size == 0
        or payload_size > _MAX_STATE_UNIT_FRAGMENT_BYTES
        or payload_size != len(packed_payload)
        or apply_call_id == 0
        or stream_index >= 8
        or tensor_kind != 2
        or fragments_per_tensor == 0
        or chunks_per_dimension == 0
        or token_count == 0
        or row_size == 0
        or cell_capacity == 0
        or element_size == 0
        or embedding_width == 0
        or transposed_value != 1
        or reserved != 0
        or row_size != element_size * embedding_width
        or fragments_per_tensor
        != chunks_per_dimension * embedding_width
        or first_cell_index >= cell_capacity
        or token_count > cell_capacity - first_cell_index
        or first_fragment_index > (1 << 32) - record_count
        or first_flat_index > (1 << 32) - record_count
        or first_token_position > (1 << 64) - token_count
    ):
        raise NativeRuntimeStateWireError(
            "state-unit schedule batch metadata is invalid"
        )
    units_per_fragment = _MAX_STATE_UNIT_FRAGMENT_BYTES // element_size
    if units_per_fragment == 0:
        raise NativeRuntimeStateWireError(
            "state-unit schedule batch element size is invalid"
        )
    records: list[bytes] = []
    payload_offset = 0
    for record_index in range(record_count):
        flat_index = first_flat_index + record_index
        layer = flat_index // fragments_per_tensor
        fragment_in_tensor = flat_index % fragments_per_tensor
        dimension = fragment_in_tensor // chunks_per_dimension
        chunk_index = fragment_in_tensor % chunks_per_dimension
        if dimension >= embedding_width:
            raise NativeRuntimeStateWireError(
                "state-unit schedule batch dimension is invalid"
            )
        cell_offset = chunk_index * units_per_fragment
        if cell_offset >= token_count:
            raise NativeRuntimeStateWireError(
                "state-unit schedule batch cell offset is invalid"
            )
        span_cell_count = min(
            token_count - cell_offset,
            units_per_fragment,
        )
        fragment_payload_size = span_cell_count * element_size
        payload_end = payload_offset + fragment_payload_size
        if (
            fragment_payload_size == 0
            or fragment_payload_size > 128
            or payload_end > len(packed_payload)
        ):
            raise NativeRuntimeStateWireError(
                "state-unit schedule batch record size is invalid"
            )
        cell_index = first_cell_index + cell_offset
        tensor_byte_offset = (
            dimension * element_size * cell_capacity
            + cell_index * element_size
        )
        records.append(
            _STATE_UNIT_FRAGMENT.pack(
                apply_call_id,
                first_fragment_index + record_index,
                seq_id,
                stream_index,
                tensor_kind,
                first_token_position + cell_offset,
                cell_index,
                layer,
                fragment_payload_size,
                tensor_byte_offset,
                writer_monotonic_ns,
                span_cell_count,
            )
            + packed_payload[payload_offset:payload_end]
        )
        payload_offset = payload_end
    if payload_offset != len(packed_payload):
        raise NativeRuntimeStateWireError(
            "state-unit schedule batch has trailing payload bytes"
        )
    return tuple(records)


def _decode_state_unit_seal_payload(payload: bytes) -> NativeStateUnitSeal:
    if len(payload) != _STATE_UNIT_SEAL.size:
        raise NativeRuntimeStateWireError(
            "state-unit seal payload size is invalid"
        )
    values = _STATE_UNIT_SEAL.unpack(payload)
    if values[4] != 1:
        raise NativeRuntimeStateWireError(
            "state-unit seal writer state is invalid"
        )
    return NativeStateUnitSeal(
        apply_call_id=values[0],
        expected_fragment_count=values[1],
        expected_token_cell_count=values[2],
        stream_mask=values[3],
        writer_state="COMPLETE",
        layout_generation=values[5],
        backing_generation=values[6],
        mutation_generation=values[7],
        copy_byte_count=values[8],
        reserve_failure_baseline=values[9],
        reserve_failure_final=values[10],
        writer_return_monotonic_ns=values[11],
    )


def encode_topology_frame(event: RuntimeTopologyEvent, sequence: int) -> bytes:
    if not isinstance(event, RuntimeTopologyEvent):
        raise NativeRuntimeStateWireError("event must be RuntimeTopologyEvent")
    return _encode_frame(
        FRAME_RUNTIME_TOPOLOGY,
        _TOPOLOGY.pack(
            event.event_id,
            event.tgid,
            event.tid,
            event.monotonic_ns,
            event.n_ctx,
            event.n_stream,
            event.n_seq_max,
            event.n_layer,
            int(event.v_trans),
            event.n_pos_per_embd,
        ),
        sequence,
    )


def encode_tensor_layout_frame(event: TensorLayoutEvent, sequence: int) -> bytes:
    if not isinstance(event, TensorLayoutEvent):
        raise NativeRuntimeStateWireError("event must be TensorLayoutEvent")
    return _encode_frame(
        FRAME_TENSOR_LAYOUT,
        _TENSOR.pack(
            event.event_id,
            event.tgid,
            event.tid,
            event.monotonic_ns,
            1 if event.kind == "k" else 2,
            event.layer,
            event.stream_index,
            event.ggml_type_id,
            event.tensor_address,
            event.row_size,
            event.cell_capacity,
            event.element_size,
            event.embedding_width,
        ),
        sequence,
    )


def encode_kv_backing_frame(event: KvBackingEvent, sequence: int) -> bytes:
    if not isinstance(event, KvBackingEvent):
        raise NativeRuntimeStateWireError("event must be KvBackingEvent")
    if _require_uint(event.event_id, "event_id") == 0:
        raise NativeRuntimeStateWireError("event_id must be positive")
    for field in ("tgid", "tid"):
        if _require_uint(getattr(event, field), field, 32) == 0:
            raise NativeRuntimeStateWireError(f"{field} must be positive")
    _require_uint(event.monotonic_ns, "monotonic_ns")
    for field in ("base_address", "size_bytes"):
        if _require_uint(getattr(event, field), field) == 0:
            raise NativeRuntimeStateWireError(f"{field} must be positive")
    return _encode_frame(
        FRAME_KV_BACKING,
        _BACKING.pack(
            event.event_id,
            event.tgid,
            event.tid,
            event.monotonic_ns,
            event.base_address,
            event.size_bytes,
        ),
        sequence,
    )


def encode_token_cell_frame(event: TokenCellEvent, sequence: int) -> bytes:
    if not isinstance(event, TokenCellEvent):
        raise NativeRuntimeStateWireError("event must be TokenCellEvent")
    if _require_uint(event.apply_call_id, "apply_call_id") == 0:
        raise NativeRuntimeStateWireError("apply_call_id must be positive")
    for field in ("tgid", "tid"):
        if _require_uint(getattr(event, field), field, 32) == 0:
            raise NativeRuntimeStateWireError(f"{field} must be positive")
    _require_uint(event.monotonic_ns, "monotonic_ns")
    for field in ("seq_id", "stream_index"):
        _require_uint(getattr(event, field), field, 32)
    for field in ("token_position", "cell_index"):
        _require_uint(getattr(event, field), field)
    return _encode_frame(
        FRAME_TOKEN_CELL,
        _TOKEN.pack(
            event.apply_call_id,
            event.tgid,
            event.tid,
            event.monotonic_ns,
            event.seq_id,
            event.stream_index,
            event.token_position,
            event.cell_index,
        ),
        sequence,
    )


def encode_token_cell_range_frame(
    item: NativeTokenCellRange,
    sequence: int,
) -> bytes:
    if not isinstance(item, NativeTokenCellRange):
        raise NativeRuntimeStateWireError(
            "item must be NativeTokenCellRange"
        )
    return _encode_frame(
        FRAME_TOKEN_CELL_RANGE,
        _TOKEN_CELL_RANGE.pack(
            item.apply_call_id,
            item.tgid,
            item.tid,
            item.monotonic_ns,
            item.seq_id,
            item.stream_index,
            item.first_token_position,
            item.first_cell_index,
            item.token_count,
            0,
        ),
        sequence,
    )


def encode_apply_complete_frame(
    event: NativeApplyCompleteEvent, sequence: int
) -> bytes:
    if not isinstance(event, NativeApplyCompleteEvent):
        raise NativeRuntimeStateWireError(
            "event must be NativeApplyCompleteEvent"
        )
    return _encode_frame(
        FRAME_APPLY_COMPLETE,
        _APPLY_COMPLETE.pack(
            event.apply_call_id,
            event.tgid,
            event.tid,
            event.monotonic_ns,
        ),
        sequence,
    )


def encode_mutation_frame(event: NativeMutationEvent, sequence: int) -> bytes:
    if not isinstance(event, NativeMutationEvent):
        raise NativeRuntimeStateWireError("event must be NativeMutationEvent")
    return _encode_frame(
        FRAME_MUTATION,
        _MUTATION.pack(
            event.event_id,
            event.tgid,
            event.tid,
            event.monotonic_ns,
            _MUTATION_CODE_BY_KIND[event.mutation_kind],
            0,
        ),
        sequence,
    )


def encode_stream_version_frame(
    snapshot: NativeStreamVersionSnapshot,
    sequence: int,
) -> bytes:
    if not isinstance(snapshot, NativeStreamVersionSnapshot):
        raise NativeRuntimeStateWireError(
            "snapshot must be a NativeStreamVersionSnapshot"
        )
    state_codes = {"EMPTY": 0, "BUSY": 1, "READY": 2}
    return _encode_frame(
        FRAME_STREAM_VERSION,
        _STREAM_VERSION_V4.pack(
            snapshot.snapshot_id,
            snapshot.mutation_generation,
            snapshot.backing_generation,
            snapshot.layout_generation,
            snapshot.active_stream_count,
            *snapshot.stream_apply_call_ids,
            *(state_codes[item] for item in snapshot.stream_states),
        ),
        sequence,
    )


def _encode_counters_frame(
    counters: NativeObservationCounters,
    sequence: int,
    frame_type: int,
) -> bytes:
    return _encode_frame(
        frame_type,
        _COUNTERS.pack(
            *(getattr(counters, field) for field in counters.__dataclass_fields__)
        ),
        sequence,
    )


def encode_watermark_frame(
    counters: NativeWatermarkCounters, sequence: int
) -> bytes:
    if not isinstance(counters, NativeWatermarkCounters):
        raise NativeRuntimeStateWireError(
            "counters must be NativeWatermarkCounters"
        )
    return _encode_counters_frame(
        counters, sequence, FRAME_WATERMARK_COUNTERS
    )


def encode_terminal_frame(counters: NativeTerminalCounters, sequence: int) -> bytes:
    if not isinstance(counters, NativeTerminalCounters):
        raise NativeRuntimeStateWireError("counters must be NativeTerminalCounters")
    return _encode_counters_frame(
        counters, sequence, FRAME_TERMINAL_COUNTERS
    )


def parse_native_runtime_state_stream(
    data: bytes, *, allow_nonterminal_watermark: bool = False
) -> ParsedNativeRuntimeStateStream:
    if not isinstance(data, bytes) or not data:
        raise NativeRuntimeStateWireError("native stream must be non-empty bytes")
    offset = 0
    expected_sequences = {
        NATIVE_CHANNEL_METADATA: 1,
        NATIVE_CHANNEL_STATE_UNIT: 1,
    }
    topology: list[RuntimeTopologyEvent] = []
    tensors: list[TensorLayoutEvent] = []
    backing: list[KvBackingEvent] = []
    tokens: list[TokenCellEvent] = []
    token_ranges: list[NativeTokenCellRange] = []
    completions: list[NativeApplyCompleteEvent] = []
    mutations: list[NativeMutationEvent] = []
    stream_versions: list[NativeStreamVersionSnapshot] = []
    state_unit_fragments: list[NativeStateUnitFragment] = []
    state_unit_seals: list[NativeStateUnitSeal] = []
    fragments_by_apply: dict[int, dict[int, NativeStateUnitFragment]] = {}
    sealed_apply_ids: set[int] = set()
    watermarks: list[NativeWatermarkCounters] = []
    terminal: NativeTerminalCounters | None = None
    event_ids: set[int] = set()
    completion_ids: set[int] = set()
    prior_counters: NativeObservationCounters | None = None
    previous_metadata_frame_type: int | None = None
    stream_wire_version: int | None = None

    def append_state_unit_fragment(fragment_payload: bytes) -> None:
        item = _decode_state_unit_fragment_payload(fragment_payload)
        if item.apply_call_id in sealed_apply_ids:
            raise NativeRuntimeStateWireError(
                "state-unit fragment follows its seal"
            )
        apply_fragments = fragments_by_apply.setdefault(
            item.apply_call_id, {}
        )
        if item.fragment_index in apply_fragments:
            raise NativeRuntimeStateWireError(
                "state-unit fragment index is duplicated"
            )
        apply_fragments[item.fragment_index] = item
        state_unit_fragments.append(item)

    while offset < len(data):
        if len(data) - offset < _HEADER.size:
            raise NativeRuntimeStateWireError("native frame header is truncated")
        magic, version, frame_type, payload_size, channel_id, sequence = _HEADER.unpack_from(data, offset)
        if magic != NATIVE_WIRE_MAGIC:
            raise NativeRuntimeStateWireError("native frame magic does not match")
        if version not in {
            _LEGACY_NATIVE_WIRE_VERSION,
            _PREVIOUS_NATIVE_WIRE_VERSION,
            NATIVE_WIRE_VERSION,
        }:
            raise NativeRuntimeStateWireError("native frame version does not match")
        if stream_wire_version is None:
            stream_wire_version = version
        elif version != stream_wire_version:
            raise NativeRuntimeStateWireError(
                "native frame version changes within the stream"
            )
        if channel_id not in expected_sequences:
            raise NativeRuntimeStateWireError("native frame channel is invalid")
        state_unit_frame = frame_type in {
            FRAME_STATE_UNIT_FRAGMENT,
            FRAME_STATE_UNIT_FRAGMENT_BATCH,
            FRAME_STATE_UNIT_SCHEDULE_BATCH,
            FRAME_STATE_UNIT_SEAL,
            FRAME_STATE_UNIT_BULK,
        }
        expected_channel = (
            NATIVE_CHANNEL_STATE_UNIT
            if state_unit_frame
            else NATIVE_CHANNEL_METADATA
        )
        if channel_id != expected_channel:
            raise NativeRuntimeStateWireError(
                "native frame type does not belong to its channel"
            )
        if sequence != expected_sequences[channel_id]:
            raise NativeRuntimeStateWireError("native frame sequence is not contiguous")
        expected_size = _payload_size_for_version(version, frame_type)
        valid_fragment_size = (
            frame_type == FRAME_STATE_UNIT_FRAGMENT
            and _STATE_UNIT_FRAGMENT.size < payload_size
            <= _STATE_UNIT_FRAGMENT.size + _MAX_STATE_UNIT_FRAGMENT_BYTES
        )
        valid_fragment_batch_size = (
            frame_type == FRAME_STATE_UNIT_FRAGMENT_BATCH
            and _STATE_UNIT_FRAGMENT_BATCH.size + _STATE_UNIT_FRAGMENT.size
                < payload_size
            <= _STATE_UNIT_FRAGMENT_BATCH.size
                + _MAX_STATE_UNIT_FRAGMENT_BATCH_RECORD_BYTES
        )
        valid_schedule_batch_size = (
            frame_type == FRAME_STATE_UNIT_SCHEDULE_BATCH
            and _STATE_UNIT_SCHEDULE_BATCH.size < payload_size
            <= _STATE_UNIT_SCHEDULE_BATCH.size
                + _MAX_STATE_UNIT_FRAGMENT_BYTES
        )
        valid_bulk_size = (
            version == NATIVE_WIRE_VERSION
            and frame_type == FRAME_STATE_UNIT_BULK
            and BULK_PREFIX_BYTES < payload_size
            <= BULK_PREFIX_BYTES + BULK_PAYLOAD_BYTES
        )
        if (
            not valid_fragment_size
            and not valid_fragment_batch_size
            and not valid_schedule_batch_size
            and not valid_bulk_size
            and (
            expected_size is None or payload_size != expected_size
            )
        ):
            raise NativeRuntimeStateWireError("native frame type or payload length is invalid")
        frame_end = offset + _HEADER.size + payload_size
        if frame_end > len(data):
            raise NativeRuntimeStateWireError("native frame payload is truncated")
        if terminal is not None:
            raise NativeRuntimeStateWireError("native event follows terminal counters")
        payload = data[offset + _HEADER.size : frame_end]
        if frame_type == FRAME_RUNTIME_TOPOLOGY:
            if watermarks:
                raise NativeRuntimeStateWireError(
                    "native structural evidence follows the first watermark"
                )
            values = _TOPOLOGY.unpack(payload)
            if values[8] not in {0, 1}:
                raise NativeRuntimeStateWireError(
                    "topology v_trans code is invalid"
                )
            item = RuntimeTopologyEvent(*values[:8], bool(values[8]), values[9])
            topology.append(item)
            event_id = item.event_id
        elif frame_type == FRAME_TENSOR_LAYOUT:
            if watermarks:
                raise NativeRuntimeStateWireError(
                    "native structural evidence follows the first watermark"
                )
            values = _TENSOR.unpack(payload)
            kind_value = values[4]
            if kind_value not in {1, 2}:
                raise NativeRuntimeStateWireError("tensor kind code is invalid")
            item = TensorLayoutEvent(
                event_id=values[0],
                tgid=values[1],
                tid=values[2],
                monotonic_ns=values[3],
                kind="k" if kind_value == 1 else "v",
                layer=values[5],
                stream_index=values[6],
                ggml_type_id=values[7],
                tensor_address=values[8],
                row_size=values[9],
                cell_capacity=values[10],
                element_size=values[11],
                embedding_width=values[12],
            )
            tensors.append(item)
            event_id = item.event_id
        elif frame_type == FRAME_KV_BACKING:
            if watermarks:
                raise NativeRuntimeStateWireError(
                    "native structural evidence follows the first watermark"
                )
            item = KvBackingEvent(*_BACKING.unpack(payload))
            encode_kv_backing_frame(item, sequence)
            backing.append(item)
            event_id = item.event_id
        elif frame_type == FRAME_TOKEN_CELL:
            item = TokenCellEvent(*_TOKEN.unpack(payload))
            encode_token_cell_frame(item, sequence)
            tokens.append(item)
            event_id = None
        elif frame_type == FRAME_TOKEN_CELL_RANGE:
            values = _TOKEN_CELL_RANGE.unpack(payload)
            if values[9] != 0:
                raise NativeRuntimeStateWireError(
                    "native token-cell range reserved field is invalid"
                )
            try:
                item = NativeTokenCellRange(*values[:9])
            except TokenCellRangeError as error:
                raise NativeRuntimeStateWireError(str(error)) from error
            token_ranges.append(item)
            event_id = None
        elif frame_type == FRAME_APPLY_COMPLETE:
            item = NativeApplyCompleteEvent(*_APPLY_COMPLETE.unpack(payload))
            if item.apply_call_id in completion_ids:
                raise NativeRuntimeStateWireError(
                    "native apply completion is duplicated"
                )
            completion_ids.add(item.apply_call_id)
            completions.append(item)
            event_id = None
        elif frame_type == FRAME_MUTATION:
            values = _MUTATION.unpack(payload)
            if values[5] != 0 or values[4] not in _MUTATION_KIND_BY_CODE:
                raise NativeRuntimeStateWireError(
                    "native mutation code or reserved field is invalid"
                )
            item = NativeMutationEvent(
                event_id=values[0],
                tgid=values[1],
                tid=values[2],
                monotonic_ns=values[3],
                mutation_kind=_MUTATION_KIND_BY_CODE[values[4]],
            )
            mutations.append(item)
            event_id = item.event_id
        elif frame_type == FRAME_STREAM_VERSION:
            if previous_metadata_frame_type == FRAME_STREAM_VERSION:
                raise NativeRuntimeStateWireError(
                    "native stream version watermark is duplicated"
                )
            if version == _LEGACY_NATIVE_WIRE_VERSION:
                values = _STREAM_VERSION_V3.unpack(payload)
                active_stream_count = 4
                apply_call_ids = tuple(values[4:8]) + (0,) * 4
                state_codes = tuple(values[8:12]) + (0,) * 4
            else:
                if payload[36:40] != b"\x00" * 4:
                    raise NativeRuntimeStateWireError(
                        "native stream version reserved field is invalid"
                    )
                values = _STREAM_VERSION_V4.unpack(payload)
                active_stream_count = values[4]
                apply_call_ids = tuple(values[5:13])
                state_codes = tuple(values[13:21])
            state_names = {0: "EMPTY", 1: "BUSY", 2: "READY"}
            if any(value not in state_names for value in state_codes):
                raise NativeRuntimeStateWireError(
                    "native stream version state code is invalid"
                )
            item = NativeStreamVersionSnapshot(
                snapshot_id=values[0],
                mutation_generation=values[1],
                backing_generation=values[2],
                layout_generation=values[3],
                stream_apply_call_ids=apply_call_ids,
                stream_states=tuple(
                    state_names[value] for value in state_codes
                ),
                active_stream_count=active_stream_count,
            )
            if stream_versions and item.snapshot_id <= stream_versions[-1].snapshot_id:
                raise NativeRuntimeStateWireError(
                    "native stream version snapshot id regressed"
                )
            stream_versions.append(item)
            event_id = None
        elif frame_type == FRAME_STATE_UNIT_FRAGMENT:
            append_state_unit_fragment(payload)
            event_id = None
        elif frame_type == FRAME_STATE_UNIT_FRAGMENT_BATCH:
            for record in _decode_state_unit_fragment_batch_payload(payload):
                append_state_unit_fragment(record)
            event_id = None
        elif frame_type == FRAME_STATE_UNIT_SCHEDULE_BATCH:
            for record in _decode_state_unit_schedule_batch_payload(payload):
                append_state_unit_fragment(record)
            event_id = None
        elif frame_type == FRAME_STATE_UNIT_BULK:
            try:
                bulk = decode_native_bulk_payload(payload)
                bulk_views = iter_native_bulk_fragment_views(bulk)
                for view in bulk_views:
                    append_state_unit_fragment(
                        _STATE_UNIT_FRAGMENT.pack(
                            view.apply_call_id,
                            view.fragment_index,
                            view.seq_id,
                            view.stream_index,
                            1 if view.tensor_kind == "k" else 2,
                            view.token_position,
                            view.cell_index,
                            view.layer,
                            view.payload_size,
                            view.tensor_byte_offset,
                            view.writer_monotonic_ns,
                            view.span_cell_count,
                        )
                        + bytes(view.payload)
                    )
            except NativeBulkDecodeError as error:
                raise NativeRuntimeStateWireError(
                    "state-unit bulk payload is invalid"
                ) from error
            event_id = None
        elif frame_type == FRAME_STATE_UNIT_SEAL:
            item = _decode_state_unit_seal_payload(payload)
            if item.apply_call_id in sealed_apply_ids:
                raise NativeRuntimeStateWireError(
                    "state-unit seal is duplicated"
                )
            apply_fragments = fragments_by_apply.get(
                item.apply_call_id, {}
            )
            expected_indexes = set(range(item.expected_fragment_count))
            if set(apply_fragments) != expected_indexes:
                raise NativeRuntimeStateWireError(
                    "state-unit fragment set is incomplete"
                )
            ordered_fragments = tuple(
                apply_fragments[index]
                for index in range(item.expected_fragment_count)
            )
            if sum(len(fragment.payload) for fragment in ordered_fragments) != (
                item.copy_byte_count
            ):
                raise NativeRuntimeStateWireError(
                    "state-unit fragment byte count disagrees with its seal"
                )
            stream_mask = 0
            for fragment in ordered_fragments:
                stream_mask |= 1 << fragment.stream_index
                if (
                    fragment.writer_monotonic_ns
                    > item.writer_return_monotonic_ns
                ):
                    raise NativeRuntimeStateWireError(
                        "state-unit fragment follows writer return"
                    )
            if stream_mask != item.stream_mask:
                raise NativeRuntimeStateWireError(
                    "state-unit fragment stream mask disagrees with its seal"
                )
            sealed_apply_ids.add(item.apply_call_id)
            state_unit_seals.append(item)
            event_id = None
        elif frame_type == FRAME_WATERMARK_COUNTERS:
            watermark = NativeWatermarkCounters(*_COUNTERS.unpack(payload))
            if stream_versions and (
                previous_metadata_frame_type != FRAME_STREAM_VERSION
                or len(stream_versions) != len(watermarks) + 1
            ):
                raise NativeRuntimeStateWireError(
                    "native watermark has no adjacent stream version"
                )
            if stream_versions:
                current_version = stream_versions[-1]
                if (
                    current_version.mutation_generation
                    != watermark.mutation_event_count
                    or current_version.backing_generation
                    != watermark.kv_backing_event_count
                    or current_version.layout_generation
                    != watermark.tensor_layout_event_count
                ):
                    raise NativeRuntimeStateWireError(
                        "native watermark generations disagree"
                    )
            if watermarks and (
                watermark.cutoff_monotonic_ns
                <= watermarks[-1].cutoff_monotonic_ns
            ):
                raise NativeRuntimeStateWireError(
                    "native watermark cutoff regressed or was repeated"
                )
            if prior_counters is not None and any(
                getattr(watermark, field) < getattr(prior_counters, field)
                for field in watermark.__dataclass_fields__
                if field not in {
                    "cutoff_monotonic_ns",
                    "final_inflight_count",
                }
            ):
                raise NativeRuntimeStateWireError(
                    "native watermark counters regressed"
                )
            watermarks.append(watermark)
            prior_counters = watermark
            event_id = None
        else:
            terminal = NativeTerminalCounters(*_COUNTERS.unpack(payload))
            if stream_versions and (
                previous_metadata_frame_type != FRAME_STREAM_VERSION
                or len(stream_versions) != len(watermarks) + 1
            ):
                raise NativeRuntimeStateWireError(
                    "native terminal has no adjacent stream version"
                )
            if stream_versions:
                current_version = stream_versions[-1]
                if (
                    current_version.mutation_generation
                    != terminal.mutation_event_count
                    or current_version.backing_generation
                    != terminal.kv_backing_event_count
                    or current_version.layout_generation
                    != terminal.tensor_layout_event_count
                ):
                    raise NativeRuntimeStateWireError(
                        "native terminal generations disagree"
                    )
            if watermarks and (
                terminal.cutoff_monotonic_ns
                < watermarks[-1].cutoff_monotonic_ns
            ):
                raise NativeRuntimeStateWireError(
                    "native terminal cutoff regressed"
                )
            if prior_counters is not None and any(
                getattr(terminal, field) < getattr(prior_counters, field)
                for field in terminal.__dataclass_fields__
                if field not in {
                    "cutoff_monotonic_ns",
                    "final_inflight_count",
                }
            ):
                raise NativeRuntimeStateWireError(
                    "native terminal counters regressed"
                )
            event_id = None
            if frame_end != len(data):
                raise NativeRuntimeStateWireError("terminal counters must be the last frame")
        if event_id is not None:
            if event_id in event_ids:
                raise NativeRuntimeStateWireError("native event_id is duplicated")
            event_ids.add(event_id)
        offset = frame_end
        expected_sequences[channel_id] += 1
        if channel_id == NATIVE_CHANNEL_METADATA:
            previous_metadata_frame_type = frame_type
    if terminal is None:
        if not allow_nonterminal_watermark or not watermarks:
            raise NativeRuntimeStateWireError(
                "native stream has no terminal counters"
            )
        if frame_type != FRAME_WATERMARK_COUNTERS:
            raise NativeRuntimeStateWireError(
                "nonterminal native stream must end at a watermark"
            )
    if set(fragments_by_apply) != sealed_apply_ids:
        raise NativeRuntimeStateWireError(
            "state-unit fragment set has no seal"
        )
    if tokens and token_ranges:
        raise NativeRuntimeStateWireError(
            "mixed legacy and compact token events"
        )
    active_stream_counts = {
        item.active_stream_count for item in stream_versions
    }
    active_stream_counts.update(
        item.n_stream for item in topology if item.n_stream in {4, 8}
    )
    if len(active_stream_counts) > 1:
        raise NativeRuntimeStateWireError(
            "native active stream topology disagrees"
        )
    if state_unit_seals:
        if not active_stream_counts:
            raise NativeRuntimeStateWireError(
                "state-unit active stream topology is unavailable"
            )
        active_stream_count = next(iter(active_stream_counts))
        active_stream_mask = (1 << active_stream_count) - 1
        if any(
            item.stream_mask & ~active_stream_mask
            for item in state_unit_seals
        ) or any(
            item.stream_index >= active_stream_count
            for item in state_unit_fragments
        ):
            raise NativeRuntimeStateWireError(
                "state-unit stream mask exceeds the active topology"
            )
    return ParsedNativeRuntimeStateStream(
        topology_events=tuple(topology),
        tensor_layout_events=tuple(tensors),
        backing_events=tuple(backing),
        token_cells=tuple(tokens),
        token_cell_ranges=tuple(token_ranges),
        apply_completions=tuple(completions),
        mutations=tuple(mutations),
        stream_versions=tuple(stream_versions),
        state_unit_fragments=tuple(state_unit_fragments),
        state_unit_seals=tuple(state_unit_seals),
        watermarks=tuple(watermarks),
        terminal=terminal,
    )


def encode_snapshot_request(
    not_before_ns: int,
    not_after_ns: int,
    observed_seq_id: int,
) -> bytes:
    _validate_interval(not_before_ns, not_after_ns)
    _validate_observed_seq_id(observed_seq_id)
    document = {
        "not_before_ns": not_before_ns,
        "not_after_ns": not_after_ns,
        "observed_seq_id": observed_seq_id,
    }
    return b"SNAPSHOT " + _canonical_json(document) + b"\n"


def encode_log_linked_cut_request(
    not_before_ns: int,
    not_after_ns: int,
    observed_seq_id: int,
) -> bytes:
    encoded = encode_snapshot_request(
        not_before_ns,
        not_after_ns,
        observed_seq_id,
    )
    return b"LOG_LINKED_CUT " + encoded.removeprefix(b"SNAPSHOT ")


def parse_snapshot_request(payload: bytes) -> tuple[int, int, int]:
    if not isinstance(payload, bytes) or len(payload) > _MAX_REQUEST_BYTES:
        raise RuntimeStateServiceError("snapshot request exceeds the size limit")
    if not payload.startswith(b"SNAPSHOT ") or not payload.endswith(b"\n"):
        raise RuntimeStateServiceError("snapshot request framing is invalid")
    encoded = payload[len(b"SNAPSHOT ") : -1]

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeStateServiceError("snapshot request contains a duplicate field")
            result[key] = value
        return result

    try:
        document = json.loads(encoded, object_pairs_hook=reject_duplicates)
    except RuntimeStateServiceError:
        raise
    except (UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeStateServiceError("snapshot request JSON is invalid") from error
    if not isinstance(document, dict) or set(document) != {
        "not_before_ns",
        "not_after_ns",
        "observed_seq_id",
    }:
        raise RuntimeStateServiceError("snapshot request fields are invalid")
    before = document["not_before_ns"]
    after = document["not_after_ns"]
    observed_seq_id = document["observed_seq_id"]
    _validate_interval(before, after)
    _validate_observed_seq_id(observed_seq_id)
    if encoded != _canonical_json(document):
        raise RuntimeStateServiceError("snapshot request must use canonical JSON")
    return before, after, observed_seq_id


def parse_log_linked_cut_request(
    payload: bytes,
) -> tuple[int, int, int]:
    prefix = b"LOG_LINKED_CUT "
    if not isinstance(payload, bytes) or not payload.startswith(prefix):
        raise RuntimeStateServiceError(
            "Log-linked cut request framing is invalid"
        )
    return parse_snapshot_request(
        b"SNAPSHOT " + payload.removeprefix(prefix)
    )


def native_stream_version_to_document(
    snapshot: NativeStreamVersionSnapshot,
) -> dict[str, object]:
    if not isinstance(snapshot, NativeStreamVersionSnapshot):
        raise RuntimeStateServiceError(
            "native stream version snapshot is invalid"
        )
    return {
        "schema_version": "1.0",
        "snapshot_id": snapshot.snapshot_id,
        "mutation_generation": snapshot.mutation_generation,
        "backing_generation": snapshot.backing_generation,
        "layout_generation": snapshot.layout_generation,
        "streams": [
            {
                "stream_index": stream_index,
                "apply_call_id": apply_call_id,
                "state": stream_state,
            }
            for stream_index, (apply_call_id, stream_state) in enumerate(
                zip(
                    snapshot.stream_apply_call_ids[
                        : snapshot.active_stream_count
                    ],
                    snapshot.stream_states[: snapshot.active_stream_count],
                    strict=True,
                )
            )
        ],
    }


def native_stream_version_from_document(
    value: object,
) -> NativeStreamVersionSnapshot:
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "snapshot_id",
        "mutation_generation",
        "backing_generation",
        "layout_generation",
        "streams",
    }:
        raise NativeRuntimeStateWireError(
            "native stream version document fields are invalid"
        )
    if value["schema_version"] != "1.0":
        raise NativeRuntimeStateWireError(
            "native stream version schema is invalid"
        )
    streams = value["streams"]
    if not isinstance(streams, list) or len(streams) not in {4, 8}:
        raise NativeRuntimeStateWireError(
            "native stream version stream list is invalid"
        )
    apply_ids: list[int] = []
    states: list[str] = []
    for expected_index, item in enumerate(streams):
        if not isinstance(item, dict) or set(item) != {
            "stream_index",
            "apply_call_id",
            "state",
        }:
            raise NativeRuntimeStateWireError(
                "native stream version entry fields are invalid"
            )
        if item["stream_index"] != expected_index:
            raise NativeRuntimeStateWireError(
                "native stream version indexes are not canonical"
            )
        apply_ids.append(
            _require_uint(item["apply_call_id"], "apply_call_id")
        )
        stream_state = item["state"]
        if stream_state not in {"EMPTY", "BUSY", "READY"}:
            raise NativeRuntimeStateWireError(
                "native stream version state is invalid"
            )
        states.append(stream_state)
    return NativeStreamVersionSnapshot(
        snapshot_id=_require_uint(value["snapshot_id"], "snapshot_id"),
        mutation_generation=_require_uint(
            value["mutation_generation"], "mutation_generation"
        ),
        backing_generation=_require_uint(
            value["backing_generation"], "backing_generation"
        ),
        layout_generation=_require_uint(
            value["layout_generation"], "layout_generation"
        ),
        stream_apply_call_ids=tuple(apply_ids),
        stream_states=tuple(states),
        active_stream_count=len(streams),
    )


def encode_completed_applies_request(after_cursor: int) -> bytes:
    _validate_completed_apply_cursor(after_cursor)
    return b"COMPLETED_APPLIES " + _canonical_json(
        {"after_cursor": after_cursor}
    ) + b"\n"


def parse_completed_applies_request(payload: bytes) -> int:
    if not isinstance(payload, bytes) or len(payload) > _MAX_REQUEST_BYTES:
        raise RuntimeStateServiceError(
            "completed-apply request exceeds the size limit"
        )
    prefix = b"COMPLETED_APPLIES "
    if not payload.startswith(prefix) or not payload.endswith(b"\n"):
        raise RuntimeStateServiceError("completed-apply request framing is invalid")
    encoded = payload[len(prefix) : -1]

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeStateServiceError(
                    "completed-apply request contains a duplicate field"
                )
            result[key] = value
        return result

    try:
        document = json.loads(encoded, object_pairs_hook=reject_duplicates)
    except RuntimeStateServiceError:
        raise
    except (UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeStateServiceError(
            "completed-apply request JSON is invalid"
        ) from error
    if not isinstance(document, dict) or set(document) != {"after_cursor"}:
        raise RuntimeStateServiceError("completed-apply request fields are invalid")
    cursor = document["after_cursor"]
    _validate_completed_apply_cursor(cursor)
    if encoded != _canonical_json(document):
        raise RuntimeStateServiceError(
            "completed-apply request must use canonical JSON"
        )
    return cursor


def encode_completed_native_units_request(after_cursor: int) -> bytes:
    _validate_completed_apply_cursor(after_cursor)
    return b"COMPLETED_NATIVE_UNITS " + _canonical_json(
        {"after_cursor": after_cursor}
    ) + b"\n"


def parse_completed_native_units_request(payload: bytes) -> int:
    document = _parse_exact_canonical_request(
        payload,
        prefix=b"COMPLETED_NATIVE_UNITS ",
        fields={"after_cursor"},
        subject="completed native-unit",
    )
    cursor = document["after_cursor"]
    _validate_completed_apply_cursor(cursor)
    return cursor


def encode_read_native_unit_request(
    *, unit_id: str, offset: int, size: int
) -> bytes:
    _validate_native_unit_read(unit_id=unit_id, offset=offset, size=size)
    return b"READ_NATIVE_UNIT " + _canonical_json(
        {"offset": offset, "size": size, "unit_id": unit_id}
    ) + b"\n"


def parse_read_native_unit_request(payload: bytes) -> tuple[str, int, int]:
    document = _parse_exact_canonical_request(
        payload,
        prefix=b"READ_NATIVE_UNIT ",
        fields={"offset", "size", "unit_id"},
        subject="read native-unit",
    )
    unit_id = document["unit_id"]
    offset = document["offset"]
    size = document["size"]
    _validate_native_unit_read(unit_id=unit_id, offset=offset, size=size)
    return unit_id, offset, size


def _parse_exact_canonical_request(
    payload: bytes,
    *,
    prefix: bytes,
    fields: set[str],
    subject: str,
) -> dict[str, object]:
    if not isinstance(payload, bytes) or len(payload) > _MAX_REQUEST_BYTES:
        raise RuntimeStateServiceError(f"{subject} request exceeds the size limit")
    if not payload.startswith(prefix) or not payload.endswith(b"\n"):
        raise RuntimeStateServiceError(f"{subject} request framing is invalid")
    encoded = payload[len(prefix) : -1]

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeStateServiceError(
                    f"{subject} request contains a duplicate field"
                )
            result[key] = value
        return result

    try:
        document = json.loads(encoded, object_pairs_hook=reject_duplicates)
    except RuntimeStateServiceError:
        raise
    except (UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeStateServiceError(
            f"{subject} request JSON is invalid"
        ) from error
    if not isinstance(document, dict) or set(document) != fields:
        raise RuntimeStateServiceError(f"{subject} request fields are invalid")
    if encoded != _canonical_json(document):
        raise RuntimeStateServiceError(f"{subject} request must use canonical JSON")
    return document


def _validate_native_unit_read(
    *, unit_id: object, offset: object, size: object
) -> None:
    if not isinstance(unit_id, str) or not _NATIVE_UNIT_ID.fullmatch(unit_id):
        raise RuntimeStateServiceError("native-unit id is invalid")
    if (
        isinstance(offset, bool)
        or not isinstance(offset, int)
        or offset < 0
        or offset >= 1 << 64
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size <= 0
        or size > _MAX_NATIVE_UNIT_READ_BYTES
    ):
        raise RuntimeStateServiceError("native-unit byte range is invalid")


def _validate_completed_apply_cursor(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeStateServiceError(
            "completed-apply cursor must be a non-negative integer"
        )


def completed_apply_page_to_document(
    page: CompletedApplyPage,
) -> dict[str, object]:
    return {
        "boundaries": [
            {
                "apply_call_id": boundary.apply_call_id,
                "cursor": boundary.cursor,
                "monotonic_ns": boundary.monotonic_ns,
                "mutation_kinds": list(boundary.mutation_kinds),
                "observer_loss_count": boundary.observer_loss_count,
                "parse_failure_count": boundary.parse_failure_count,
                "runtime_instance_id": boundary.runtime_instance_id,
                "runtime_pid": boundary.runtime_pid,
                "runtime_start_time_ticks": boundary.runtime_start_time_ticks,
                "sequence_ids": list(boundary.sequence_ids),
                "token_cells": [
                    {
                        "cell_index": cell.cell_index,
                        "ownership_generation": cell.ownership_generation,
                        "seq_id": cell.seq_id,
                        "stream_index": cell.stream_index,
                        "token_position": cell.token_position,
                    }
                    for cell in boundary.token_cells
                ],
                "unmatched_call_count": boundary.unmatched_call_count,
                "unsupported_mutation_count": (
                    boundary.unsupported_mutation_count
                ),
                "watermark_monotonic_ns": boundary.watermark_monotonic_ns,
            }
            for boundary in page.boundaries
        ],
        "next_cursor": page.next_cursor,
        "truncated": page.truncated,
    }


def completed_native_unit_page_to_document(
    page: CompletedNativeUnitPage,
) -> dict[str, object]:
    return {
        "next_cursor": page.next_cursor,
        "truncated": page.truncated,
        "units": [
            {
                "apply_call_id": descriptor.apply_call_id,
                "cursor": descriptor.cursor,
                "fragment_count": descriptor.fragment_count,
                "payload_sha256": descriptor.payload_sha256,
                "payload_size": descriptor.payload_size,
                "runtime_instance_id": descriptor.runtime_instance_id,
                "seq_id": descriptor.seq_id,
                "stream_index": descriptor.stream_index,
                "token_cells": [list(item) for item in descriptor.token_cells],
                "unit_id": descriptor.unit_id,
                "writer_monotonic_ns": descriptor.writer_monotonic_ns,
            }
            for descriptor in page.units
        ],
    }


def _validate_interval(not_before_ns: object, not_after_ns: object) -> None:
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (not_before_ns, not_after_ns)):
        raise RuntimeStateServiceError("snapshot interval must contain non-negative integers")
    if not_after_ns < not_before_ns:
        raise RuntimeStateServiceError("not_after_ns must not precede not_before_ns")


def _validate_observed_seq_id(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeStateServiceError(
            "observed_seq_id must be a non-negative integer"
        )


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
        raise RuntimeStateServiceError("value is not canonical JSON data") from error


def _validate_observation_counters(
    parsed: ParsedNativeRuntimeStateStream,
    counters: NativeObservationCounters,
    *,
    require_zero_inflight: bool,
    external_native_unit_frame_count: int = 0,
    external_native_unit_delivery_event_count: int | None = 0,
) -> None:
    if (
        not isinstance(external_native_unit_frame_count, int)
        or isinstance(external_native_unit_frame_count, bool)
        or external_native_unit_frame_count < 0
    ):
        raise RuntimeStateServiceError(
            "external native-unit frame count is invalid"
        )
    if (
        external_native_unit_delivery_event_count is not None
        and (
            not isinstance(external_native_unit_delivery_event_count, int)
            or isinstance(external_native_unit_delivery_event_count, bool)
            or external_native_unit_delivery_event_count < 0
        )
    ):
        raise RuntimeStateServiceError(
            "external native-unit delivery event count is invalid"
        )
    counts = (
        (counters.topology_event_count, len(parsed.topology_events)),
        (counters.tensor_layout_event_count, len(parsed.tensor_layout_events)),
        (counters.kv_backing_event_count, len(parsed.backing_events)),
        (
            counters.apply_complete_event_count,
            len(parsed.apply_completions),
        ),
        (counters.mutation_event_count, len(parsed.mutations)),
    )
    if any(observed != actual for observed, actual in counts):
        raise RuntimeStateServiceError("native terminal event counter mismatch")
    try:
        logical_cells = logical_token_cell_count(
            parsed.token_cells,
            parsed.token_cell_ranges,
        )
    except TokenCellRangeError as error:
        raise RuntimeStateServiceError(str(error)) from error
    if counters.token_cell_event_count != logical_cells:
        raise RuntimeStateServiceError("native terminal event counter mismatch")
    common_total = (
        sum(actual for _, actual in counts)
        + len(parsed.token_cells)
        + len(parsed.token_cell_ranges)
        + len(parsed.state_unit_fragments)
        + len(parsed.state_unit_seals)
    )
    wire_total = common_total + external_native_unit_frame_count
    delivery_mismatch = (
        counters.kernel_event_submit_count
        != counters.native_event_receive_count
    )
    if external_native_unit_delivery_event_count is not None:
        delivery_total = (
            common_total + external_native_unit_delivery_event_count
        )
        delivery_mismatch = delivery_mismatch or any(
            count != delivery_total
            for count in (
                counters.kernel_event_submit_count,
                counters.native_event_receive_count,
            )
        )
    if delivery_mismatch or counters.native_wire_frame_count != wire_total:
        raise RuntimeStateServiceError("native delivery counter mismatch")
    if require_zero_inflight and counters.final_inflight_count != 0:
        raise RuntimeStateServiceError("native terminal has an inflight call")
    nonzero = tuple(
        (code, description)
        for field, code, description in _NATIVE_COUNTER_FAILURES
        if getattr(counters, field) != 0
    )
    if nonzero:
        raise NativeObservationCounterError(
            tuple(code for code, _description in nonzero),
            tuple(description for _code, description in nonzero),
        )


def _build_observed_layout(
    profile: RuntimeStateObserverProfile,
    parsed: ParsedNativeRuntimeStateStream,
) -> SimpleKvLayout:
    if len(parsed.topology_events) != 1:
        raise RuntimeStateServiceError(
            "runtime topology event must be complete and unique"
        )
    topology = parsed.topology_events[0]
    expected = {
        "n_stream": profile.n_stream,
        "n_seq_max": profile.n_seq_max,
    }
    if not profile.has_dynamic_topology:
        expected.update(
            {
                "n_layer": profile.n_layer,
                "n_pos_per_embd": profile.n_pos_per_embd,
                "v_trans": profile.v_trans,
            }
        )
    for field, required in expected.items():
        if getattr(topology, field) != required:
            raise RuntimeStateServiceError(
                f"runtime topology {field} does not match the profile"
            )
    if profile.has_dynamic_topology:
        if topology.n_layer <= 0 or topology.n_layer > profile.max_n_layer:
            raise RuntimeStateServiceError(
                "runtime topology n_layer exceeds the observer capability"
            )
        if (
            topology.n_pos_per_embd <= 0
            or topology.n_pos_per_embd > profile.max_n_pos_per_embd
        ):
            raise RuntimeStateServiceError(
                "runtime topology n_pos_per_embd exceeds the observer capability"
            )
        if type(topology.v_trans) is not bool:
            raise RuntimeStateServiceError(
                "runtime topology v_trans is not observable"
            )
    if topology.n_ctx % topology.n_stream:
        raise RuntimeStateServiceError(
            "runtime n_ctx does not divide across streams"
        )
    expected_keys = {
        (kind, layer, stream_index)
        for kind in ("k", "v")
        for stream_index in range(topology.n_stream)
        for layer in range(topology.n_layer)
    }
    tensors: dict[tuple[str, int, int], TensorLayoutEvent] = {}
    for event in parsed.tensor_layout_events:
        key = (event.kind, event.layer, event.stream_index)
        if key in tensors:
            raise RuntimeStateServiceError(
                "duplicate runtime tensor layout event"
            )
        tensors[key] = event
    if set(tensors) != expected_keys:
        raise RuntimeStateServiceError("runtime tensor layout is not complete")
    capacity = topology.n_ctx // topology.n_stream
    rows: list[TensorRowLayout] = []
    storage_intervals: list[tuple[int, int]] = []
    for key in sorted(tensors, key=lambda item: (item[2], item[1], item[0])):
        event = tensors[key]
        if event.cell_capacity != capacity:
            raise RuntimeStateServiceError(
                "runtime tensor capacity contradicts topology"
            )
        expected_type = (
            profile.key_ggml_type_id
            if event.kind == "k"
            else profile.value_ggml_type_id
        )
        if event.ggml_type_id != expected_type:
            raise RuntimeStateServiceError(
                "runtime tensor ggml type contradicts profile"
            )
        if event.row_size != event.element_size * event.embedding_width:
            raise RuntimeStateServiceError(
                "runtime tensor row shape is inconsistent"
            )
        storage_end = event.tensor_address + event.row_size * event.cell_capacity
        if storage_end > 1 << 64:
            raise RuntimeStateServiceError(
                "runtime tensor storage exceeds uint64 address space"
            )
        storage_intervals.append((event.tensor_address, storage_end))
        rows.append(
            TensorRowLayout(
                kind=event.kind,
                layer=event.layer,
                stream_index=event.stream_index,
                tensor_address=event.tensor_address,
                row_size=event.row_size,
                cell_capacity=event.cell_capacity,
                ggml_type_id=event.ggml_type_id,
                element_size=event.element_size,
                embedding_width=event.embedding_width,
            )
        )
    storage_intervals.sort()
    for previous, current in zip(storage_intervals, storage_intervals[1:]):
        if current[0] < previous[1]:
            raise RuntimeStateServiceError(
                "runtime tensor storage intervals overlap"
            )
    return SimpleKvLayout(
        profile_digest=profile.profile_digest,
        n_stream=topology.n_stream,
        n_ctx=topology.n_ctx,
        n_seq_max=topology.n_seq_max,
        n_layer=topology.n_layer,
        v_trans=topology.v_trans,
        tensors=tuple(rows),
        n_pos_per_embd=topology.n_pos_per_embd,
    )


def _validate_observation_events(
    *,
    runtime_pid: int,
    parsed: ParsedNativeRuntimeStateStream,
    layout: SimpleKvLayout,
    counters: NativeObservationCounters,
    terminal: bool,
) -> None:
    all_events = (
        tuple(parsed.topology_events)
        + tuple(parsed.tensor_layout_events)
        + tuple(parsed.backing_events)
        + tuple(parsed.token_cells)
        + tuple(parsed.token_cell_ranges)
        + tuple(parsed.apply_completions)
        + tuple(parsed.mutations)
    )
    structural_events = (
        tuple(parsed.topology_events)
        + tuple(parsed.tensor_layout_events)
        + tuple(parsed.backing_events)
    )
    structural_cutoff = (
        parsed.watermarks[0].cutoff_monotonic_ns
        if parsed.watermarks
        else counters.cutoff_monotonic_ns
    )
    if any(event.monotonic_ns > structural_cutoff for event in structural_events):
        raise RuntimeStateServiceError(
            "native structural evidence occurs after the first watermark cutoff"
        )
    if not parsed.backing_events:
        raise RuntimeStateServiceError("KV backing provenance is missing")
    for event in all_events:
        if event.tgid != runtime_pid:
            raise RuntimeStateServiceError(
                "native event belongs to another runtime process"
            )
        if terminal and event.monotonic_ns > counters.cutoff_monotonic_ns:
            raise RuntimeStateServiceError(
                "native event occurs after the terminal cutoff"
            )
    validate_native_runtime_state_mutations(parsed)

    intervals: list[tuple[int, int]] = []
    for event in parsed.backing_events:
        end = event.base_address + event.size_bytes
        if end > 1 << 64:
            raise RuntimeStateServiceError(
                "KV backing range exceeds uint64 address space"
            )
        intervals.append((event.base_address, end))
    intervals.sort()
    for previous, current in zip(intervals, intervals[1:]):
        if current[0] < previous[1]:
            raise RuntimeStateServiceError("KV backing ranges overlap")
    for tensor in layout.tensors:
        end = tensor.tensor_address + tensor.row_size * tensor.cell_capacity
        containing = sum(
            start <= tensor.tensor_address and end <= limit
            for start, limit in intervals
        )
        if containing != 1:
            raise RuntimeStateServiceError(
                "runtime tensor is outside KV backing provenance"
            )

    capacity = layout.n_ctx // layout.n_stream
    mappings: dict[int, int] = {}
    observed_mutations = 0
    tokens_by_call: dict[
        int, list[TokenCellEvent | NativeTokenCellRange]
    ] = {}
    for event in parsed.token_cells:
        if event.seq_id >= layout.n_seq_max or event.stream_index >= layout.n_stream:
            raise RuntimeStateServiceError(
                "token mapping is outside runtime topology"
            )
        if event.token_position >= layout.n_ctx or event.cell_index >= capacity:
            raise RuntimeStateServiceError(
                "token position or cell is outside runtime topology"
            )
        prior = mappings.setdefault(event.seq_id, event.stream_index)
        if prior != event.stream_index:
            observed_mutations += 1
        tokens_by_call.setdefault(event.apply_call_id, []).append(event)
    for item in parsed.token_cell_ranges:
        if item.seq_id >= layout.n_seq_max or item.stream_index >= layout.n_stream:
            raise RuntimeStateServiceError(
                "token mapping is outside runtime topology"
            )
        if (
            item.first_token_position >= layout.n_ctx
            or item.last_token_position >= layout.n_ctx
            or item.first_cell_index >= capacity
            or item.last_cell_index >= capacity
        ):
            raise RuntimeStateServiceError(
                "token position or cell is outside runtime topology"
            )
        prior = mappings.setdefault(item.seq_id, item.stream_index)
        if prior != item.stream_index:
            observed_mutations += 1
        tokens_by_call.setdefault(item.apply_call_id, []).append(item)
    if observed_mutations != counters.mapping_mutation_count:
        raise RuntimeStateServiceError(
            "sequence mapping mutation counter mismatch"
        )

    completions_by_call = {
        event.apply_call_id: event for event in parsed.apply_completions
    }
    for call_id, completion in completions_by_call.items():
        call_tokens = tokens_by_call.get(call_id)
        if not call_tokens:
            raise RuntimeStateServiceError(
                "native apply completion has no matching token cells"
            )
        if any(
            token.tgid != completion.tgid
            or token.tid != completion.tid
            or token.monotonic_ns > completion.monotonic_ns
            for token in call_tokens
        ):
            raise RuntimeStateServiceError(
                "native apply call topology is inconsistent"
            )
    if terminal and set(tokens_by_call) != set(completions_by_call):
        raise RuntimeStateServiceError(
            "native terminal apply call topology is incomplete"
        )


def _selected_token_cells(
    parsed: ParsedNativeRuntimeStateStream,
    *,
    seq_id: int | None = None,
    apply_call_id: int | None = None,
    not_before_ns: int | None = None,
    not_after_ns: int | None = None,
) -> tuple[TokenCellEvent, ...]:
    try:
        return select_token_cells(
            parsed.token_cells,
            parsed.token_cell_ranges,
            seq_id=seq_id,
            apply_call_id=apply_call_id,
            not_before_ns=not_before_ns,
            not_after_ns=not_after_ns,
        )
    except TokenCellRangeError as error:
        raise RuntimeStateServiceError(str(error)) from error


def _validate_native_observation(
    *,
    profile: RuntimeStateObserverProfile,
    runtime_pid: int,
    parsed: ParsedNativeRuntimeStateStream,
    counters: NativeObservationCounters,
    terminal: bool,
    external_native_unit_frame_count: int = 0,
    external_native_unit_delivery_event_count: int | None = 0,
) -> SimpleKvLayout:
    """Apply the one production semantic boundary to live and sealed streams."""
    _validate_observation_counters(
        parsed,
        counters,
        require_zero_inflight=terminal,
        external_native_unit_frame_count=external_native_unit_frame_count,
        external_native_unit_delivery_event_count=(
            external_native_unit_delivery_event_count
        ),
    )
    layout = _build_observed_layout(profile, parsed)
    _validate_observation_events(
        runtime_pid=runtime_pid,
        parsed=parsed,
        layout=layout,
        counters=counters,
        terminal=terminal,
    )
    return layout


_TERMINAL_SEAL_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "artifact",
        "schema_version",
        "runtime_instance_id",
        "runtime_pid",
        "runtime_start_time_ticks",
        "runtime_binary_sha256",
        "qualified_model_sha256",
        "mapped_libllama_sha256",
        "mapped_libllama_elf_build_id",
        "profile_digest",
        "memory_family",
        "native_stream_artifact",
        "native_stream_sha256",
        "topology",
        "topology_digest",
        "watermark_frame_count",
        "native_unit_frame_count",
        "terminal_counters",
        "seal_digest",
    }
)
_TOPOLOGY_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "n_ctx",
        "n_layer",
        "n_pos_per_embd",
        "n_seq_max",
        "n_stream",
        "profile_digest",
        "v_trans",
    }
)
_TERMINAL_COUNTER_FIELDS: Final[frozenset[str]] = frozenset(
    NativeTerminalCounters.__dataclass_fields__
)


def _seal_uint(
    value: object,
    field: str,
    *,
    positive: bool = False,
    bits: int = 64,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value >= 1 << bits
        or (positive and value == 0)
    ):
        qualifier = "positive " if positive else ""
        raise RuntimeStateServiceError(
            f"terminal seal {field} must be a {qualifier}unsigned {bits}-bit integer"
        )
    return value


def _seal_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeStateServiceError(
            f"terminal seal {field} must be a non-empty string"
        )
    return value


def _seal_digest(value: object, field: str) -> str:
    text = _seal_text(value, field)
    if (
        len(text) != 71
        or not text.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in text[7:])
    ):
        raise RuntimeStateServiceError(
            f"terminal seal {field} must be lowercase sha256:<64 hex>"
        )
    return text


def _exact_object(
    value: object, fields: frozenset[str], subject: str
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise RuntimeStateServiceError(f"terminal seal {subject} must be an object")
    actual = set(value)
    if actual != fields:
        raise RuntimeStateServiceError(
            f"terminal seal {subject} fields differ: "
            f"missing={sorted(fields - actual)}, "
            f"unexpected={sorted(actual - fields)}"
        )
    return value


def _parse_terminal_seal_json(path: Path) -> dict[str, object]:
    if not isinstance(path, Path):
        raise RuntimeStateServiceError("terminal seal path must be a pathlib.Path")
    if path.is_symlink() or not path.is_file():
        raise RuntimeStateServiceError("terminal seal must be a regular non-symlink file")
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise RuntimeStateServiceError("terminal seal could not be read") from error
    if (
        not raw
        or len(raw) > _MAX_TERMINAL_SEAL_BYTES
        or not raw.endswith(b"\n")
        or raw.count(b"\n") != 1
    ):
        raise RuntimeStateServiceError("terminal seal framing is invalid")

    def reject_duplicates(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeStateServiceError(
                    f"terminal seal contains duplicate field: {key}"
                )
            result[key] = value
        return result

    try:
        document = json.loads(
            raw[:-1], object_pairs_hook=reject_duplicates
        )
    except RuntimeStateServiceError:
        raise
    except (UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeStateServiceError("terminal seal JSON is invalid") from error
    top = _exact_object(document, _TERMINAL_SEAL_FIELDS, "top-level")
    if raw != _canonical_json(top) + b"\n":
        raise RuntimeStateServiceError("terminal seal is not canonical JSON")
    return top


def load_and_verify_runtime_state_terminal_seal(
    path: Path,
    *,
    profile: RuntimeStateObserverProfile,
    runtime_instance_id: str,
    verified_runtime: VerifiedRuntimeIdentity,
) -> RuntimeStateTerminalSeal:
    if not isinstance(profile, RuntimeStateObserverProfile):
        raise RuntimeStateServiceError(
            "terminal seal profile must be RuntimeStateObserverProfile"
        )
    expected_runtime_instance_id = _seal_text(
        runtime_instance_id, "expected runtime_instance_id"
    )
    if not isinstance(verified_runtime, VerifiedRuntimeIdentity):
        raise RuntimeStateServiceError(
            "terminal seal requires verified live runtime identity"
        )
    _validate_verified_runtime_binding(profile, verified_runtime)
    expected_runtime_pid = _seal_uint(
        verified_runtime.pid, "expected runtime_pid", positive=True, bits=32
    )
    expected_start_ticks = _seal_uint(
        verified_runtime.start_time_ticks,
        "expected runtime_start_time_ticks",
        positive=True,
    )
    top = _parse_terminal_seal_json(path)
    if top["artifact"] != "runtime_state_terminal_seal":
        raise RuntimeStateServiceError("terminal seal artifact is invalid")
    if top["schema_version"] != "1.0":
        raise RuntimeStateServiceError("terminal seal schema_version is invalid")

    supplied_seal_digest = _seal_digest(top["seal_digest"], "seal_digest")
    unsigned = dict(top)
    del unsigned["seal_digest"]
    expected_seal_digest = "sha256:" + hashlib.sha256(
        _canonical_json(unsigned)
    ).hexdigest()
    if supplied_seal_digest != expected_seal_digest:
        raise RuntimeStateServiceError("terminal seal seal_digest mismatch")

    identities: tuple[tuple[str, object, object], ...] = (
        ("runtime_instance_id", top["runtime_instance_id"], expected_runtime_instance_id),
        ("runtime_pid", top["runtime_pid"], expected_runtime_pid),
        (
            "runtime_start_time_ticks",
            top["runtime_start_time_ticks"],
            expected_start_ticks,
        ),
        (
            "runtime_binary_sha256",
            top["runtime_binary_sha256"],
            verified_runtime.runtime_binary_sha256,
        ),
        (
            "qualified_model_sha256",
            top["qualified_model_sha256"],
            verified_runtime.qualified_model_sha256,
        ),
        (
            "mapped_libllama_sha256",
            top["mapped_libllama_sha256"],
            verified_runtime.mapped_libllama_sha256,
        ),
        (
            "mapped_libllama_elf_build_id",
            top["mapped_libllama_elf_build_id"],
            verified_runtime.mapped_libllama_elf_build_id,
        ),
        ("profile_digest", top["profile_digest"], profile.profile_digest),
        ("memory_family", top["memory_family"], profile.memory_family),
    )
    for field, observed, expected in identities:
        if type(observed) is not type(expected) or observed != expected:
            raise RuntimeStateServiceError(
                f"terminal seal {field} does not match the qualified runtime"
            )
    native_stream_sha256 = _seal_digest(
        top["native_stream_sha256"], "native_stream_sha256"
    )
    native_stream_artifact = _seal_text(
        top["native_stream_artifact"], "native_stream_artifact"
    )
    if native_stream_artifact != "runtime_state_native_stream.bin":
        raise RuntimeStateServiceError(
            "terminal seal native_stream_artifact is invalid"
        )

    topology_value = _exact_object(
        top["topology"], _TOPOLOGY_FIELDS, "topology"
    )
    topology = RuntimeStateTopologySummary(
        n_ctx=_seal_uint(
            topology_value["n_ctx"], "topology.n_ctx", positive=True, bits=32
        ),
        n_layer=_seal_uint(
            topology_value["n_layer"], "topology.n_layer", positive=True, bits=32
        ),
        n_pos_per_embd=_seal_uint(
            topology_value["n_pos_per_embd"],
            "topology.n_pos_per_embd",
            positive=True,
            bits=32,
        ),
        n_seq_max=_seal_uint(
            topology_value["n_seq_max"],
            "topology.n_seq_max",
            positive=True,
            bits=32,
        ),
        n_stream=_seal_uint(
            topology_value["n_stream"],
            "topology.n_stream",
            positive=True,
            bits=32,
        ),
        profile_digest=_seal_digest(
            topology_value["profile_digest"], "topology.profile_digest"
        ),
        v_trans=topology_value["v_trans"],
    )
    if not isinstance(topology.v_trans, bool):
        raise RuntimeStateServiceError(
            "terminal seal topology.v_trans must be a boolean"
        )
    qualified_topology = (
        ("n_seq_max", topology.n_seq_max, profile.n_seq_max),
        ("n_stream", topology.n_stream, profile.n_stream),
        ("profile_digest", topology.profile_digest, profile.profile_digest),
    )
    if not profile.has_dynamic_topology:
        qualified_topology += (
            ("n_layer", topology.n_layer, profile.n_layer),
            ("n_pos_per_embd", topology.n_pos_per_embd, profile.n_pos_per_embd),
            ("v_trans", topology.v_trans, profile.v_trans),
        )
    for field, observed, expected in qualified_topology:
        if type(observed) is not type(expected) or observed != expected:
            raise RuntimeStateServiceError(
                f"terminal seal topology.{field} does not match the profile"
            )
    if profile.has_dynamic_topology:
        if topology.n_layer > profile.max_n_layer:
            raise RuntimeStateServiceError(
                "terminal seal topology.n_layer exceeds the observer capability"
            )
        if topology.n_pos_per_embd > profile.max_n_pos_per_embd:
            raise RuntimeStateServiceError(
                "terminal seal topology.n_pos_per_embd exceeds the observer capability"
            )
    if topology.n_ctx % topology.n_stream != 0:
        raise RuntimeStateServiceError(
            "terminal seal topology.n_ctx is not divisible by n_stream"
        )
    topology_digest = _seal_digest(top["topology_digest"], "topology_digest")
    expected_topology_digest = "sha256:" + hashlib.sha256(
        _canonical_json(topology_value)
    ).hexdigest()
    if topology_digest != expected_topology_digest:
        raise RuntimeStateServiceError("terminal seal topology_digest mismatch")

    counter_value = _exact_object(
        top["terminal_counters"],
        _TERMINAL_COUNTER_FIELDS,
        "terminal_counters",
    )
    counters = NativeTerminalCounters(
        **{
            field: _seal_uint(counter_value[field], f"terminal_counters.{field}")
            for field in NativeTerminalCounters.__dataclass_fields__
        }
    )
    watermark_frame_count = _seal_uint(
        top["watermark_frame_count"], "watermark_frame_count"
    )
    native_unit_frame_count = _seal_uint(
        top["native_unit_frame_count"], "native_unit_frame_count"
    )
    native_stream_path = path.parent / native_stream_artifact
    if native_stream_path.is_symlink() or not native_stream_path.is_file():
        raise RuntimeStateServiceError(
            "terminal seal native stream must be a regular non-symlink file"
        )
    try:
        native_stream = native_stream_path.read_bytes()
    except OSError as error:
        raise RuntimeStateServiceError(
            "terminal seal native stream could not be read"
        ) from error
    if not native_stream or len(native_stream) > _MAX_RESPONSE_BYTES:
        raise RuntimeStateServiceError(
            "terminal seal native stream framing is invalid"
        )
    observed_stream_digest = "sha256:" + hashlib.sha256(native_stream).hexdigest()
    if observed_stream_digest != native_stream_sha256:
        raise RuntimeStateServiceError(
            "terminal seal native stream digest mismatch"
        )
    try:
        parsed_stream = parse_native_runtime_state_stream(native_stream)
    except NativeRuntimeStateWireError as error:
        raise RuntimeStateServiceError(
            "terminal seal native stream is invalid"
        ) from error
    _validate_observation_counters(
        parsed_stream,
        counters,
        require_zero_inflight=True,
        external_native_unit_frame_count=native_unit_frame_count,
        external_native_unit_delivery_event_count=None,
    )
    if parsed_stream.terminal != counters:
        raise RuntimeStateServiceError(
            "terminal seal native stream counters mismatch"
        )
    if len(parsed_stream.watermarks) != watermark_frame_count:
        raise RuntimeStateServiceError(
            "terminal seal native stream watermark count mismatch"
        )
    validated_layout = _validate_native_observation(
        profile=profile,
        runtime_pid=expected_runtime_pid,
        parsed=parsed_stream,
        counters=counters,
        terminal=True,
        external_native_unit_frame_count=native_unit_frame_count,
        external_native_unit_delivery_event_count=None,
    )
    live_topology = parsed_stream.topology_events[0]
    if (
        live_topology.tgid != expected_runtime_pid
        or validated_layout.n_ctx != topology.n_ctx
        or validated_layout.n_layer != topology.n_layer
        or validated_layout.n_pos_per_embd != topology.n_pos_per_embd
        or validated_layout.n_seq_max != topology.n_seq_max
        or validated_layout.n_stream != topology.n_stream
        or validated_layout.v_trans != topology.v_trans
    ):
        raise RuntimeStateServiceError(
            "terminal seal native stream topology mismatch"
        )
    return RuntimeStateTerminalSeal(
        runtime_instance_id=expected_runtime_instance_id,
        runtime_pid=expected_runtime_pid,
        runtime_start_time_ticks=expected_start_ticks,
        runtime_binary_sha256=verified_runtime.runtime_binary_sha256,
        qualified_model_sha256=verified_runtime.qualified_model_sha256,
        mapped_libllama_sha256=verified_runtime.mapped_libllama_sha256,
        mapped_libllama_elf_build_id=verified_runtime.mapped_libllama_elf_build_id,
        profile_digest=profile.profile_digest,
        memory_family=profile.memory_family,
        native_stream_artifact=native_stream_artifact,
        native_stream_sha256=native_stream_sha256,
        topology=topology,
        topology_digest=topology_digest,
        watermark_frame_count=watermark_frame_count,
        native_unit_frame_count=native_unit_frame_count,
        terminal_counters=counters,
        seal_digest=supplied_seal_digest,
    )


class RuntimeStateSnapshotService:
    def __init__(
        self,
        *,
        profile: RuntimeStateObserverProfile,
        runtime_instance_id: str,
        verified_runtime: VerifiedRuntimeIdentity,
        frame_dir: Path,
        capture_mode: str | None = None,
    ) -> None:
        if not isinstance(profile, RuntimeStateObserverProfile):
            raise RuntimeStateServiceError("profile must be RuntimeStateObserverProfile")
        if not isinstance(runtime_instance_id, str) or not runtime_instance_id:
            raise RuntimeStateServiceError("runtime_instance_id must be non-empty")
        if not isinstance(verified_runtime, VerifiedRuntimeIdentity):
            raise RuntimeStateServiceError(
                "service requires verified live runtime identity"
            )
        _validate_verified_runtime_binding(profile, verified_runtime)
        if not isinstance(frame_dir, Path):
            raise RuntimeStateServiceError("frame_dir must be a pathlib.Path")
        self.profile = profile
        self.runtime_instance_id = runtime_instance_id
        self.verified_runtime = verified_runtime
        self.runtime_pid = verified_runtime.pid
        self.runtime_start_time_ticks = verified_runtime.start_time_ticks
        self.frame_dir = frame_dir
        self._capture_mode = capture_mode
        try:
            self.native_unit_spool = NativeUnitSpool(
                frame_dir=frame_dir,
                runtime_instance_id=runtime_instance_id,
                active_stream_count=profile.n_stream,
            )
        except NativeUnitSpoolError as error:
            raise RuntimeStateServiceError(
                "native-unit spool initialization failed"
            ) from error
        self._parsed: ParsedNativeRuntimeStateStream | None = None
        self._layout: SimpleKvLayout | None = None
        self._native_stream = b""
        self._sealed = False
        self._sequence = 0
        self._native_unit_wire_frame_count = 0
        self._native_unit_delivery_event_count = 0
        self._native_bulk_frame_count = 0
        self._native_bulk_payload_bytes = 0
        self._native_bulk_logical_fragment_count = 0
        self._lock = threading.Lock()

    def ingest_native_stream(self, data: bytes) -> None:
        parsed = parse_native_runtime_state_stream(
            data, allow_nonterminal_watermark=True
        )
        with self._lock:
            if self._sealed:
                raise RuntimeStateServiceError("native observation is already sealed")
            if self._native_stream and not data.startswith(self._native_stream):
                raise RuntimeStateServiceError(
                    "incremental native stream does not extend prior evidence"
                )
            counters = parsed.terminal or parsed.watermarks[-1]
            layout = _validate_native_observation(
                profile=self.profile,
                runtime_pid=self.runtime_pid,
                parsed=parsed,
                counters=counters,
                terminal=parsed.terminal is not None,
                external_native_unit_frame_count=(
                    self._native_unit_wire_frame_count
                    if not parsed.state_unit_fragments
                    and not parsed.state_unit_seals
                    else 0
                ),
                external_native_unit_delivery_event_count=(
                    self._native_unit_delivery_event_count
                    if not parsed.state_unit_fragments
                    and not parsed.state_unit_seals
                    else 0
                ),
            )
            if (
                parsed.terminal is not None
                and self.native_unit_spool.active_apply_count != 0
            ):
                raise RuntimeStateServiceError(
                    "native observation ended with an unsealed state unit"
                )
            if parsed.terminal is not None:
                self._publish_terminal_seal(
                    data,
                    parsed.terminal,
                    parsed,
                    layout,
                )
            self._native_stream = data
            self._parsed = parsed
            self._layout = layout
            self._sealed = parsed.terminal is not None

    def ingest_native_stdout(self, stream: BinaryIO) -> None:
        if not hasattr(stream, "read"):
            raise RuntimeStateServiceError("native stdout must be readable")
        data = stream.read(_MAX_RESPONSE_BYTES + 1)
        if not isinstance(data, bytes) or len(data) > _MAX_RESPONSE_BYTES:
            raise RuntimeStateServiceError("native stdout exceeds the size limit")
        self.ingest_native_stream(data)

    def snapshot(
        self,
        *,
        not_before_ns: int,
        not_after_ns: int,
        observed_seq_id: int,
    ) -> RuntimeStateSnapshot:
        _validate_interval(not_before_ns, not_after_ns)
        _validate_observed_seq_id(observed_seq_id)
        with self._lock:
            if self._parsed is None or self._layout is None:
                raise RuntimeStateServiceError("native observation is not sealed")
            if observed_seq_id >= self._layout.n_seq_max:
                raise RuntimeStateServiceError(
                    "observed_seq_id is outside runtime topology"
                )
            parsed = self._parsed
            counters = parsed.terminal or parsed.watermarks[-1]
            if counters.cutoff_monotonic_ns < not_after_ns:
                raise RuntimeStateServiceError("snapshot interval extends beyond the native cutoff")
            effective_not_after_ns = counters.cutoff_monotonic_ns
            selected = _selected_token_cells(
                parsed,
                seq_id=observed_seq_id,
                not_before_ns=not_before_ns,
                not_after_ns=effective_not_after_ns,
            )
            seq_to_stream: dict[int, int] = {}
            decode_calls: dict[int, set[int]] = {}
            for item in selected:
                prior = seq_to_stream.setdefault(item.seq_id, item.stream_index)
                if prior != item.stream_index:
                    raise RuntimeStateServiceError("sequence stream mapping changed inside the snapshot")
                decode_calls.setdefault(item.seq_id, set()).add(item.apply_call_id)
            completed_calls = {
                item.apply_call_id
                for item in parsed.apply_completions
                if item.monotonic_ns <= counters.cutoff_monotonic_ns
            }
            selected_calls = {item.apply_call_id for item in selected}
            if not selected_calls.issubset(completed_calls):
                raise RuntimeStateServiceError(
                    "target sequence is not quiescent: apply completion is missing"
                )
            mutation_policy = validate_native_runtime_state_mutations(parsed)
            if mutation_policy.operational_mutation_event_count != 0:
                raise RuntimeStateServiceError(
                    "forbidden operational runtime mutation observed"
                )
            self._sequence += 1
            frame_sequence = self._sequence
            digest = hashlib.sha256(
                f"{self.runtime_instance_id}:{observed_seq_id}:"
                f"{not_before_ns}:{effective_not_after_ns}:"
                f"{counters.cutoff_monotonic_ns}:{frame_sequence}".encode("utf-8")
            ).hexdigest()
            snapshot = RuntimeStateSnapshot(
                frame_id=f"rsf_{digest[:32]}",
                frame_sequence=frame_sequence,
                runtime_instance_id=self.runtime_instance_id,
                runtime_pid=self.runtime_pid,
                runtime_start_time_ticks=self.runtime_start_time_ticks,
                runtime_binary_sha256=self.verified_runtime.runtime_binary_sha256,
                mapped_libllama_sha256=self.verified_runtime.mapped_libllama_sha256,
                mapped_libllama_elf_build_id=self.verified_runtime.mapped_libllama_elf_build_id,
                profile_digest=self.profile.profile_digest,
                memory_family=self.profile.memory_family,
                layout=self._layout,
                observed_n_stream=self._layout.n_stream,
                seq_to_stream=tuple(sorted(seq_to_stream.items())),
                backing_events=parsed.backing_events,
                token_cells=selected,
                decode_counts=tuple(
                    sorted((seq_id, len(calls)) for seq_id, calls in decode_calls.items())
                ),
                mapping_mutation_count=counters.mapping_mutation_count,
                mutation_counters=tuple(
                    (name, 0)
                    for name in self.profile.forbidden_mutations
                ),
                observer_loss_count=counters.ringbuf_reserve_fail_count,
                parse_failure_count=counters.user_memory_read_failure_count,
                unmatched_call_count=(
                    counters.unmatched_call_count
                ),
                unsupported_mutation_count=counters.unsupported_mutation_count,
                window_start_monotonic_ns=not_before_ns,
                window_end_monotonic_ns=effective_not_after_ns,
                cutoff_monotonic_ns=counters.cutoff_monotonic_ns,
            )
            self._publish(snapshot)
            return snapshot

    def log_linked_cut(
        self,
        *,
        not_before_ns: int,
        not_after_ns: int,
        observed_seq_id: int,
    ) -> LogLinkedRuntimeCut:
        """Return metadata and a writer-synchronous stream version cut."""

        snapshot = self.snapshot(
            not_before_ns=not_before_ns,
            not_after_ns=not_after_ns,
            observed_seq_id=observed_seq_id,
        )
        with self._lock:
            parsed = self._parsed
            if parsed is None or not parsed.watermarks:
                raise RuntimeStateServiceError(
                    "Log-linked cut has no drained native watermark"
                )
            watermark_indexes = tuple(
                index
                for index, watermark in enumerate(parsed.watermarks)
                if watermark.cutoff_monotonic_ns
                == snapshot.cutoff_monotonic_ns
            )
            if len(watermark_indexes) != 1:
                raise RuntimeStateServiceError(
                    "Log-linked cut watermark is not unique"
                )
            watermark_index = watermark_indexes[0]
            if watermark_index >= len(parsed.stream_versions):
                raise RuntimeStateServiceError(
                    "Log-linked cut has no stream-version watermark"
                )
            version = parsed.stream_versions[watermark_index]
            watermark = parsed.watermarks[watermark_index]
        if (
            version.mutation_generation != watermark.mutation_event_count
            or version.backing_generation
            != watermark.kv_backing_event_count
            or version.layout_generation
            != watermark.tensor_layout_event_count
        ):
            raise RuntimeStateServiceError(
                "Log-linked cut generations disagree with native counters"
            )
        stream_by_sequence = dict(snapshot.seq_to_stream)
        stream_index = stream_by_sequence.get(observed_seq_id)
        if (
            stream_index is None
            or stream_index >= version.active_stream_count
        ):
            raise RuntimeStateServiceError(
                "Log-linked cut target stream is unavailable"
            )
        if version.stream_states[stream_index] != "READY":
            raise RuntimeStateServiceError(
                "Log-linked cut target stream has an active writer"
            )
        target_apply_ids = tuple(
            event.apply_call_id
            for event in snapshot.token_cells
            if event.seq_id == observed_seq_id
            and event.stream_index == stream_index
        )
        if not target_apply_ids or version.stream_apply_call_ids[
            stream_index
        ] != max(target_apply_ids):
            raise RuntimeStateServiceError(
                "Log-linked cut stream version differs from cell ownership"
            )
        return LogLinkedRuntimeCut(
            snapshot=snapshot,
            stream_version=version,
        )

    def completed_applies(self, *, after_cursor: int) -> CompletedApplyPage:
        _validate_completed_apply_cursor(after_cursor)
        with self._lock:
            if self._parsed is None:
                raise RuntimeStateServiceError(
                    "native observation has no drained watermark"
                )
            parsed = self._parsed
            counters = parsed.terminal or parsed.watermarks[-1]
            completions = parsed.apply_completions
            if after_cursor > len(completions):
                raise RuntimeStateServiceError(
                    "completed-apply cursor is beyond the current watermark"
                )
            selected_completions = completions[
                after_cursor : after_cursor + _MAX_COMPLETED_APPLY_PAGE
            ]
            boundaries: list[ObservedApplyBoundary] = []
            previous_completion_ns = (
                completions[after_cursor - 1].monotonic_ns
                if after_cursor > 0
                else -1
            )
            for ordinal, completion in enumerate(
                selected_completions, start=after_cursor + 1
            ):
                cells = tuple(
                    sorted(
                        _selected_token_cells(
                            parsed,
                            apply_call_id=completion.apply_call_id,
                        ),
                        key=lambda item: (
                            item.seq_id,
                            item.token_position,
                            item.stream_index,
                            item.cell_index,
                        ),
                    )
                )
                sequence_ids = tuple(sorted({item.seq_id for item in cells}))
                mutation_kinds = tuple(
                    sorted(
                        {
                            item.mutation_kind
                            for item in parsed.mutations
                            if previous_completion_ns
                            < item.monotonic_ns
                            <= completion.monotonic_ns
                        }
                    )
                )
                boundaries.append(
                    ObservedApplyBoundary(
                        cursor=ordinal,
                        runtime_instance_id=self.runtime_instance_id,
                        runtime_pid=self.runtime_pid,
                        runtime_start_time_ticks=self.runtime_start_time_ticks,
                        apply_call_id=completion.apply_call_id,
                        monotonic_ns=completion.monotonic_ns,
                        sequence_ids=sequence_ids,
                        token_cells=tuple(
                            ObservedApplyTokenCell(
                                seq_id=item.seq_id,
                                stream_index=item.stream_index,
                                token_position=item.token_position,
                                cell_index=item.cell_index,
                                ownership_generation=item.apply_call_id,
                            )
                            for item in cells
                        ),
                        mutation_kinds=mutation_kinds,
                        observer_loss_count=counters.ringbuf_reserve_fail_count,
                        parse_failure_count=(
                            counters.user_memory_read_failure_count
                        ),
                        unmatched_call_count=counters.unmatched_call_count,
                        unsupported_mutation_count=(
                            counters.unsupported_mutation_count
                        ),
                        watermark_monotonic_ns=counters.cutoff_monotonic_ns,
                    )
                )
                previous_completion_ns = completion.monotonic_ns
            truncated = any(
                (
                    counters.ringbuf_reserve_fail_count,
                    counters.user_memory_read_failure_count,
                    counters.unmatched_call_count,
                    counters.unsupported_mutation_count,
                )
            )
            return CompletedApplyPage(
                next_cursor=after_cursor + len(boundaries),
                truncated=truncated,
                boundaries=tuple(boundaries),
            )

    def consume_native_unit_frame(
        self,
        *,
        frame_type: int,
        payload: bytes,
    ) -> None:
        try:
            if frame_type == FRAME_STATE_UNIT_FRAGMENT:
                self.native_unit_spool.consume_fragment_payload(payload)
                self._native_unit_wire_frame_count += 1
                if self._capture_mode != "gather-writer-boundary":
                    self._native_unit_delivery_event_count += 1
                return
            if frame_type == FRAME_STATE_UNIT_SEAL:
                seal = _decode_state_unit_seal_payload(payload)
                self.native_unit_spool.consume_seal(seal)
                self._native_unit_wire_frame_count += 1
                if self._capture_mode == "gather-writer-boundary":
                    layer_count = (
                        self._layout.n_layer
                        if self._layout is not None
                        else self.profile.n_layer
                    )
                    if layer_count <= 0:
                        raise RuntimeStateServiceError(
                            "gather state-unit topology is unavailable"
                        )
                    self._native_unit_delivery_event_count += (
                        2 * layer_count * seal.stream_mask.bit_count() + 1
                    )
                else:
                    self._native_unit_delivery_event_count += 1
                return
        except (NativeRuntimeStateWireError, NativeUnitSpoolError) as error:
            raise RuntimeStateServiceError(
                "native state-unit frame is invalid"
            ) from error
        raise RuntimeStateServiceError("native state-unit frame type is invalid")

    def consume_native_unit_fragment_payloads(
        self,
        payloads: tuple[bytes, ...],
        *,
        wire_frame_count: int | None = None,
    ) -> None:
        if wire_frame_count is None:
            wire_frame_count = len(payloads)
        if (
            not isinstance(wire_frame_count, int)
            or isinstance(wire_frame_count, bool)
            or wire_frame_count <= 0
            or wire_frame_count > len(payloads)
        ):
            raise RuntimeStateServiceError(
                "native state-unit wire frame count is invalid"
            )
        try:
            self.native_unit_spool.consume_fragment_payloads(payloads)
        except NativeUnitSpoolError as error:
            raise RuntimeStateServiceError(
                "native state-unit frame is invalid"
            ) from error
        self._native_unit_wire_frame_count += wire_frame_count
        if self._capture_mode != "gather-writer-boundary":
            self._native_unit_delivery_event_count += wire_frame_count

    def consume_native_unit_schedule_payloads(
        self,
        payloads: tuple[bytes, ...],
        *,
        wire_frame_count: int | None = None,
    ) -> tuple[int, int]:
        if wire_frame_count is None:
            wire_frame_count = len(payloads)
        if (
            not isinstance(wire_frame_count, int)
            or isinstance(wire_frame_count, bool)
            or wire_frame_count <= 0
            or wire_frame_count > len(payloads)
        ):
            raise RuntimeStateServiceError(
                "native state-unit wire frame count is invalid"
            )
        try:
            logical_fragments, copied_bytes = (
                self.native_unit_spool.consume_schedule_payloads(payloads)
            )
        except NativeUnitSpoolError as error:
            raise RuntimeStateServiceError(
                "native state-unit frame is invalid"
            ) from error
        self._native_unit_wire_frame_count += wire_frame_count
        if self._capture_mode != "gather-writer-boundary":
            self._native_unit_delivery_event_count += wire_frame_count
        return logical_fragments, copied_bytes

    def consume_native_unit_bulk_payloads(
        self,
        payloads: tuple[bytes, ...],
        *,
        wire_frame_count: int | None = None,
    ) -> tuple[int, int]:
        if wire_frame_count is None:
            wire_frame_count = len(payloads)
        if (
            not isinstance(payloads, tuple)
            or not payloads
            or not isinstance(wire_frame_count, int)
            or isinstance(wire_frame_count, bool)
            or wire_frame_count != len(payloads)
        ):
            raise RuntimeStateServiceError(
                "native state-unit bulk frame count is invalid"
            )
        try:
            logical_fragments, copied_bytes = (
                self.native_unit_spool.consume_bulk_payloads(payloads)
            )
        except NativeUnitSpoolError as error:
            raise RuntimeStateServiceError(
                "native state-unit bulk frame is invalid"
            ) from error
        self._native_unit_wire_frame_count += wire_frame_count
        self._native_bulk_frame_count += wire_frame_count
        self._native_bulk_payload_bytes += copied_bytes
        self._native_bulk_logical_fragment_count += logical_fragments
        if self._capture_mode != "gather-writer-boundary":
            self._native_unit_delivery_event_count += wire_frame_count
        return logical_fragments, copied_bytes

    @property
    def native_bulk_frame_count(self) -> int:
        return self._native_bulk_frame_count

    @property
    def native_bulk_payload_bytes(self) -> int:
        return self._native_bulk_payload_bytes

    @property
    def native_bulk_logical_fragment_count(self) -> int:
        return self._native_bulk_logical_fragment_count

    def completed_native_units(
        self,
        *,
        after_cursor: int,
    ) -> CompletedNativeUnitPage:
        try:
            return self.native_unit_spool.completed(
                after_cursor=after_cursor
            )
        except NativeUnitSpoolError as error:
            raise RuntimeStateServiceError(
                "native-unit page is unavailable"
            ) from error

    def read_native_unit(
        self,
        *,
        unit_id: str,
        offset: int,
        size: int,
    ) -> tuple[NativeCapturedUnitDescriptor, bytes]:
        try:
            descriptor = self.native_unit_spool.descriptor(unit_id)
            payload = self.native_unit_spool.read(
                unit_id=unit_id,
                offset=offset,
                size=size,
            )
        except NativeUnitSpoolError as error:
            raise RuntimeStateServiceError(
                "native-unit byte range is unavailable"
            ) from error
        return descriptor, payload

    def handle_request(self, payload: bytes) -> bytes:
        if payload.startswith(b"LOG_LINKED_CUT "):
            before, after, observed_seq_id = (
                parse_log_linked_cut_request(payload)
            )
            cut = self.log_linked_cut(
                not_before_ns=before,
                not_after_ns=after,
                observed_seq_id=observed_seq_id,
            )
            return _canonical_json(
                {
                    "snapshot": runtime_state_snapshot_to_frame(
                        cut.snapshot
                    ),
                    "stream_version": native_stream_version_to_document(
                        cut.stream_version
                    ),
                }
            ) + b"\n"
        if payload.startswith(b"COMPLETED_APPLIES "):
            return _canonical_json(
                completed_apply_page_to_document(
                    self.completed_applies(
                        after_cursor=parse_completed_applies_request(payload)
                    )
                )
            ) + b"\n"
        if payload.startswith(b"COMPLETED_NATIVE_UNITS "):
            return _canonical_json(
                completed_native_unit_page_to_document(
                    self.completed_native_units(
                        after_cursor=parse_completed_native_units_request(
                            payload
                        )
                    )
                )
            ) + b"\n"
        if payload.startswith(b"READ_NATIVE_UNIT "):
            unit_id, offset, size = parse_read_native_unit_request(payload)
            descriptor, unit_payload = self.read_native_unit(
                unit_id=unit_id,
                offset=offset,
                size=size,
            )
            return _canonical_json(
                {
                    "byte_count": len(unit_payload),
                    "data_base64": base64.b64encode(unit_payload).decode(
                        "ascii"
                    ),
                    "offset": offset,
                    "payload_sha256": descriptor.payload_sha256,
                    "schema_version": "1.0",
                    "total_size": descriptor.payload_size,
                    "unit_id": descriptor.unit_id,
                }
            ) + b"\n"
        before, after, observed_seq_id = parse_snapshot_request(payload)
        return _canonical_json(runtime_state_snapshot_to_frame(
            self.snapshot(
                not_before_ns=before,
                not_after_ns=after,
                observed_seq_id=observed_seq_id,
            )
        )) + b"\n"

    def topology_marker_line(self) -> str:
        with self._lock:
            if self._layout is None:
                raise RuntimeStateServiceError(
                    "no drained native topology is available"
                )
            encoded = _canonical_json(
                self._topology_document(self._layout)
            ).decode("ascii")
        return f"TOPOLOGY runtime-state-service {encoded}"

    @property
    def is_sealed(self) -> bool:
        with self._lock:
            return self._sealed

    def expected_tensor_layout_count(self) -> int:
        with self._lock:
            if self._layout is None:
                raise RuntimeStateServiceError(
                    "no observed runtime topology is available"
                )
            return 2 * self._layout.n_stream * self._layout.n_layer

    def _validate_counters(
        self,
        parsed: ParsedNativeRuntimeStateStream,
        counters: NativeObservationCounters,
        *,
        require_zero_inflight: bool,
    ) -> None:
        _validate_observation_counters(
            parsed,
            counters,
            require_zero_inflight=require_zero_inflight,
        )

    def _build_layout(self, parsed: ParsedNativeRuntimeStateStream) -> SimpleKvLayout:
        return _build_observed_layout(self.profile, parsed)

    def _validate_runtime_events(
        self,
        parsed: ParsedNativeRuntimeStateStream,
        layout: SimpleKvLayout,
        counters: NativeObservationCounters,
        *,
        terminal: bool,
    ) -> None:
        _validate_observation_events(
            runtime_pid=self.runtime_pid,
            parsed=parsed,
            layout=layout,
            counters=counters,
            terminal=terminal,
        )

    def _publish(self, snapshot: RuntimeStateSnapshot) -> None:
        self._publish_immutable_json(
            f"{snapshot.frame_id}.frame.json",
            runtime_state_snapshot_to_frame(snapshot),
            error_message="complete frame publication failed",
        )

    def _publish_terminal_seal(
        self,
        native_stream: bytes,
        counters: NativeTerminalCounters,
        parsed: ParsedNativeRuntimeStateStream,
        layout: SimpleKvLayout,
    ) -> None:
        topology = self._topology_document(layout)
        topology_digest = "sha256:" + hashlib.sha256(
            _canonical_json(topology)
        ).hexdigest()
        self._publish_immutable_bytes(
            "runtime_state_native_stream.bin",
            native_stream,
            error_message="native stream publication failed",
        )
        document: dict[str, object] = {
            "artifact": "runtime_state_terminal_seal",
            "schema_version": "1.0",
            "runtime_instance_id": self.runtime_instance_id,
            "runtime_pid": self.runtime_pid,
            "runtime_start_time_ticks": self.runtime_start_time_ticks,
            "runtime_binary_sha256": self.verified_runtime.runtime_binary_sha256,
            "qualified_model_sha256": self.verified_runtime.qualified_model_sha256,
            "mapped_libllama_sha256": self.verified_runtime.mapped_libllama_sha256,
            "mapped_libllama_elf_build_id": (
                self.verified_runtime.mapped_libllama_elf_build_id
            ),
            "profile_digest": self.profile.profile_digest,
            "memory_family": self.profile.memory_family,
            "native_stream_artifact": "runtime_state_native_stream.bin",
            "native_stream_sha256": "sha256:"
            + hashlib.sha256(native_stream).hexdigest(),
            "topology": topology,
            "topology_digest": topology_digest,
            "watermark_frame_count": len(parsed.watermarks),
            "native_unit_frame_count": (
                self._native_unit_wire_frame_count
            ),
            "terminal_counters": {
                field: getattr(counters, field)
                for field in counters.__dataclass_fields__
            },
        }
        document["seal_digest"] = "sha256:" + hashlib.sha256(
            _canonical_json(document)
        ).hexdigest()
        self._publish_immutable_json(
            "runtime_state_terminal_seal.json",
            document,
            error_message="terminal seal publication failed",
        )

    def _topology_document(
        self, layout: SimpleKvLayout
    ) -> dict[str, object]:
        return {
            "n_ctx": layout.n_ctx,
            "n_layer": layout.n_layer,
            "n_pos_per_embd": layout.n_pos_per_embd,
            "n_seq_max": layout.n_seq_max,
            "n_stream": layout.n_stream,
            "profile_digest": self.profile.profile_digest,
            "v_trans": layout.v_trans,
        }

    def _publish_immutable_json(
        self,
        filename: str,
        document: object,
        *,
        error_message: str,
    ) -> None:
        if self.frame_dir.is_symlink():
            raise RuntimeStateServiceError("frame directory must not be a symlink")
        try:
            self.frame_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        except OSError as error:
            raise RuntimeStateServiceError("frame directory cannot be created") from error
        path = self.frame_dir / filename
        encoded = _canonical_json(document) + b"\n"
        self._publish_immutable_bytes(
            filename, encoded, error_message=error_message
        )

    def _publish_immutable_bytes(
        self,
        filename: str,
        encoded: bytes,
        *,
        error_message: str,
    ) -> None:
        if not isinstance(encoded, bytes) or not encoded:
            raise RuntimeStateServiceError(error_message)
        if Path(filename).name != filename:
            raise RuntimeStateServiceError(error_message)
        if self.frame_dir.is_symlink():
            raise RuntimeStateServiceError("frame directory must not be a symlink")
        try:
            self.frame_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        except OSError as error:
            raise RuntimeStateServiceError("frame directory cannot be created") from error
        path = self.frame_dir / filename
        descriptor: int | None = None
        temporary_path: Path | None = None
        try:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=".runtime-state-writing-",
                suffix=".tmp",
                dir=self.frame_dir,
            )
            temporary_path = Path(temporary_name)
            with os.fdopen(descriptor, "wb", closefd=True) as stream:
                descriptor = None
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary_path, path)
            if os.name == "posix":
                directory_descriptor = os.open(
                    self.frame_dir,
                    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
                )
                try:
                    os.fsync(directory_descriptor)
                finally:
                    os.close(directory_descriptor)
        except OSError as error:
            raise RuntimeStateServiceError(error_message) from error
        finally:
            if descriptor is not None:
                os.close(descriptor)
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)


class NativeRuntimeStateProcess:
    def __init__(
        self,
        *,
        command: Sequence[str],
        service: RuntimeStateSnapshotService,
        ready_timeout_seconds: float = 10.0,
        control_timeout_seconds: float = 10.0,
        capture_metrics_path: Path | None = None,
        capture_metrics_mode: str | None = None,
    ) -> None:
        if (
            not isinstance(command, (tuple, list))
            or not command
            or any(not isinstance(item, str) or not item for item in command)
        ):
            raise RuntimeStateServiceError(
                "native observer command must contain non-empty strings"
            )
        if not isinstance(service, RuntimeStateSnapshotService):
            raise RuntimeStateServiceError(
                "service must be RuntimeStateSnapshotService"
            )
        for value, field in (
            (ready_timeout_seconds, "ready_timeout_seconds"),
            (control_timeout_seconds, "control_timeout_seconds"),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or value <= 0
            ):
                raise RuntimeStateServiceError(f"{field} must be positive")
        if (capture_metrics_path is None) != (capture_metrics_mode is None):
            raise RuntimeStateServiceError(
                "E6 capture metrics path and mode must be configured together"
            )
        if capture_metrics_path is not None:
            if (
                not isinstance(capture_metrics_path, Path)
                or not capture_metrics_path.is_absolute()
                or capture_metrics_path.exists()
                or capture_metrics_path.is_symlink()
                or capture_metrics_mode not in {
                    "metadata",
                    "writer-boundary",
                    "gather-writer-boundary",
                    "compact-metadata",
                }
            ):
                raise RuntimeStateServiceError(
                    "E6 capture metrics configuration is invalid"
                )
        self.command = tuple(command)
        self.service = service
        self.ready_timeout_seconds = float(ready_timeout_seconds)
        self.control_timeout_seconds = float(control_timeout_seconds)
        self.capture_metrics_path = capture_metrics_path
        self.capture_metrics_mode = capture_metrics_mode
        self._capture_metrics_lock = threading.Lock()
        self._capture_metrics = {
            phase: {"count": 0, "elapsed_ns": 0, "byte_count": 0}
            for phase in _E6_CAPTURE_PHASES
        }
        self._process: subprocess.Popen[bytes] | None = None
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._condition = threading.Condition()
        self._control_lock = threading.Lock()
        self._expected_probe_diagnostics = tuple(
            _probe_identity_diagnostic(identity)
            for identity in sorted(
                service.verified_runtime.probe_targets,
                key=lambda item: item.probe_id,
            )
        )
        self._probe_diagnostic_index = 0
        self._attached = False
        self._ready = False
        self._armed = False
        self._watermark_generation = 0
        self._snapshot_request_generation = 0
        self._snapshot_generation = 0
        self._diagnostic_generation = 0
        self._pre_ready_counter_generation = 0
        self._pre_ready_counters: NativeWatermarkCounters | None = None
        self._tensor_read_diagnostics: _NativeTensorReadDiagnostics | None = None
        self._apply_rejection_diagnostic_reported = False
        self._native_bulk_metrics: dict[str, int] | None = None
        self._pre_ready_mutation_kinds: tuple[str, ...] = ()
        self._terminal_received = False
        self._failure: BaseException | None = None

    @property
    def is_running(self) -> bool:
        process = self._process
        return process is not None and process.poll() is None

    @property
    def is_attached(self) -> bool:
        with self._condition:
            return self._attached

    @property
    def is_ready(self) -> bool:
        with self._condition:
            return self._ready

    @property
    def is_armed(self) -> bool:
        with self._condition:
            return self._armed

    def start(self) -> None:
        if self._process is not None:
            raise RuntimeStateServiceError("native observer process already started")
        try:
            process = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=_NATIVE_OBSERVER_PIPE_BUFFER_BYTES,
            )
        except OSError as error:
            raise RuntimeStateServiceError(
                "native observer process could not be started"
            ) from error
        self._process = process
        self._stdout_thread = threading.Thread(
            target=self._consume_stdout,
            name="laris-runtime-state-stdout",
            daemon=True,
        )
        self._stderr_thread = threading.Thread(
            target=self._consume_stderr,
            name="laris-runtime-state-stderr",
            daemon=True,
        )
        self._stdout_thread.start()
        self._stderr_thread.start()
        deadline = time.monotonic() + self.ready_timeout_seconds
        with self._condition:
            while not self._attached and self._failure is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)
            if self._failure is not None:
                self._abort_process()
                raise RuntimeStateServiceError(
                    "native observer failed before READY: "
                    + str(self._failure)
                ) from self._failure
            if not self._attached:
                self._abort_process()
                raise RuntimeStateServiceError(
                    "native observer did not report ATTACHED"
                )

    def wait_ready(self) -> None:
        message = "native observer did not report topology READY"
        try:
            self._wait_for(
                lambda: self._ready,
                message,
                timeout_seconds=self.ready_timeout_seconds,
            )
            return
        except RuntimeStateServiceError:
            with self._condition:
                if self._failure is not None:
                    raise
        snapshot = self._request_pre_ready_snapshot()
        if snapshot is None:
            raise RuntimeStateServiceError(message)
        counters, diagnostics, mutation_kinds = snapshot
        raise RuntimeStateServiceError(
            message
            + ": "
            + _format_pre_ready_counters(counters)
            + " "
            + _format_tensor_read_diagnostics(diagnostics)
            + " "
            + _format_pre_ready_mutation_kinds(mutation_kinds)
        )

    def _request_pre_ready_snapshot(
        self,
    ) -> tuple[
        NativeWatermarkCounters,
        _NativeTensorReadDiagnostics,
        tuple[str, ...],
    ] | None:
        with self._control_lock:
            with self._condition:
                if self._ready:
                    return None
                self._snapshot_request_generation += 1
                requested_generation = self._snapshot_request_generation
            try:
                self._write_control(b"SNAPSHOT\n")
            except RuntimeStateServiceError:
                return None
            deadline = time.monotonic() + self.control_timeout_seconds
            with self._condition:
                while (
                    self._snapshot_generation < requested_generation
                    or self._diagnostic_generation < requested_generation
                ) and self._failure is None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return None
                    self._condition.wait(remaining)
                if self._failure is not None:
                    raise RuntimeStateServiceError(
                        "native observer did not report topology READY"
                    ) from self._failure
                if (
                    self._pre_ready_counters is None
                    or self._tensor_read_diagnostics is None
                ):
                    return None
                return (
                    self._pre_ready_counters,
                    self._tensor_read_diagnostics,
                    self._pre_ready_mutation_kinds,
                )

    def request_watermark(self) -> None:
        diagnostic_enabled = os.environ.get("LARIS_E6_PHASE_DIAGNOSTIC") == "1"
        started_ns = time.monotonic_ns() if diagnostic_enabled else 0
        with self._control_lock:
            if diagnostic_enabled:
                _emit_e6_phase(
                    "watermark_control_lock",
                    time.monotonic_ns() - started_ns,
                    0,
                )
            with self._condition:
                baseline = self._watermark_generation
                self._snapshot_request_generation += 1
                requested_generation = self._snapshot_request_generation
            started_ns = time.monotonic_ns() if diagnostic_enabled else 0
            self._write_control(b"SNAPSHOT\n")
            self._wait_for(
                lambda: (
                    self._watermark_generation > baseline
                    and self._snapshot_generation >= requested_generation
                    and self._diagnostic_generation >= requested_generation
                ),
                "native observer did not publish a drained watermark",
            )
            if diagnostic_enabled:
                _emit_e6_phase(
                    "watermark_completion_wait",
                    time.monotonic_ns() - started_ns,
                    0,
                )

    def arm_mutations(self) -> None:
        with self._control_lock:
            with self._condition:
                if not self._ready or self._armed:
                    raise RuntimeStateServiceError(
                        "native mutation probes are not ready to arm"
                    )
            self._write_control(b"ARM\n")
            self._wait_for(
                lambda: self._armed,
                "native observer did not arm mutation probes",
            )

    def stop(self) -> None:
        with self._control_lock:
            if not self.is_armed:
                raise RuntimeStateServiceError(
                    "native mutation probes were not armed"
                )
            if self._terminal_received:
                return
            self._write_control(b"STOP\n")
            self._wait_for(
                lambda: self._terminal_received,
                "native observer did not publish terminal counters",
            )
        process = self._process
        if process is None:
            raise RuntimeStateServiceError("native observer was not started")
        try:
            return_code = process.wait(timeout=self.control_timeout_seconds)
        except subprocess.TimeoutExpired as error:
            self._abort_process()
            raise RuntimeStateServiceError(
                "native observer did not stop after terminal counters"
            ) from error
        for thread in (self._stdout_thread, self._stderr_thread):
            if thread is not None:
                thread.join(timeout=self.control_timeout_seconds)
                if thread.is_alive():
                    raise RuntimeStateServiceError(
                        "native observer consumer thread did not stop"
                    )
        with self._condition:
            failure = self._failure
        if failure is not None:
            raise RuntimeStateServiceError(
                "native observer emitted an invalid diagnostic or evidence stream"
            ) from failure
        if return_code != 0:
            raise RuntimeStateServiceError(
                "native observer exited with a failure status"
            )
        if self.capture_metrics_path is not None:
            _publish_e6_capture_metrics(
                self.capture_metrics_path,
                self.capture_metrics_snapshot(),
            )

    def _record_capture_metric(
        self,
        phase: str,
        elapsed_ns: int,
        byte_count: int,
    ) -> None:
        if phase not in self._capture_metrics:
            raise RuntimeStateServiceError("E6 capture metric phase is unknown")
        if (
            type(elapsed_ns) is not int
            or elapsed_ns < 0
            or type(byte_count) is not int
            or byte_count < 0
        ):
            raise RuntimeStateServiceError("E6 capture metric value is invalid")
        with self._capture_metrics_lock:
            total = self._capture_metrics[phase]
            total["count"] += 1
            total["elapsed_ns"] += elapsed_ns
            total["byte_count"] += byte_count

    def capture_metrics_snapshot(self) -> dict[str, object]:
        with self._capture_metrics_lock:
            phase_totals = {
                phase: dict(self._capture_metrics[phase])
                for phase in _E6_CAPTURE_PHASES
            }
        kv_capture_work_ns = sum(
            int(total["elapsed_ns"]) for total in phase_totals.values()
        )
        native_metrics = (
            {field: 0 for field in _BULK_AGGREGATE_NATIVE_FIELDS}
            if self._native_bulk_metrics is None
            else dict(self._native_bulk_metrics)
        )
        if self.capture_metrics_mode == "gather-writer-boundary":
            if self._native_bulk_metrics is None:
                raise RuntimeStateServiceError(
                    "native bulk aggregate metric is missing"
                )
            if (
                native_metrics["native_bulk_frame_count"]
                    != self.service.native_bulk_frame_count
                or native_metrics["native_bulk_payload_bytes"]
                    != self.service.native_bulk_payload_bytes
                or native_metrics["native_logical_fragment_count"]
                    != self.service.native_bulk_logical_fragment_count
                or native_metrics["native_copied_bytes"]
                    != self.service.native_bulk_payload_bytes
                or native_metrics["native_logical_bytes"]
                    != self.service.native_bulk_payload_bytes
                or native_metrics["native_read_plan_count"] <= 0
                or native_metrics["native_read_syscall_count"]
                    < native_metrics["native_read_plan_count"]
                or native_metrics["native_bulk_frame_count"] <= 0
                or native_metrics["native_bulk_frame_count"]
                    > native_metrics["native_logical_fragment_count"]
                or native_metrics["capture_boundary_hold_ns"]
                    > native_metrics["capture_boundary_total_ns"]
            ):
                raise RuntimeStateServiceError(
                    "native bulk aggregate metric is inconsistent"
                )
        bulk_aggregates = {
            **native_metrics,
            "python_decode_persist_ns": kv_capture_work_ns,
        }
        return {
            "schema_version": "1.0",
            "metric_kind": "E6_LOG_LINKED_KV_CAPTURE_WORK",
            "capture_mode": self.capture_metrics_mode,
            "kv_capture_work_ns": kv_capture_work_ns,
            "kv_capture_bytes": int(
                phase_totals["observer_unit_read"]["byte_count"]
            ),
            "phase_totals": phase_totals,
            "bulk_aggregates": bulk_aggregates,
        }

    def _write_control(self, payload: bytes) -> None:
        process = self._process
        if process is None or process.stdin is None or process.poll() is not None:
            raise RuntimeStateServiceError("native observer is not running")
        try:
            process.stdin.write(payload)
            process.stdin.flush()
        except OSError as error:
            raise RuntimeStateServiceError(
                "native observer control channel failed"
            ) from error

    def _wait_for(
        self,
        predicate: Callable[[], bool],
        message: str,
        *,
        timeout_seconds: float | None = None,
    ) -> None:
        timeout = (
            self.control_timeout_seconds
            if timeout_seconds is None
            else timeout_seconds
        )
        deadline = time.monotonic() + timeout
        with self._condition:
            while not predicate() and self._failure is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)
            if self._failure is not None:
                raise RuntimeStateServiceError(message) from self._failure
            if not predicate():
                raise RuntimeStateServiceError(message)

    def _consume_stdout(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            self._record_failure(
                RuntimeStateServiceError("native observer stdout is unavailable")
            )
            return
        accumulated = bytearray()
        fragment_batch: list[bytes] = []
        fragment_batch_wire_frame_count = 0
        schedule_batch: list[bytes] = []
        schedule_batch_wire_frame_count = 0
        schedule_batch_copied_bytes = 0
        bulk_batch: list[bytes] = []
        bulk_batch_wire_frame_count = 0
        bulk_batch_logical_fragments = 0
        bulk_batch_copied_bytes = 0
        unit_read_ns = unit_read_bytes = 0
        fragment_persist_ns = fragment_persist_bytes = 0
        seal_persist_ns = seal_persist_bytes = 0
        diagnostic_enabled = os.environ.get("LARIS_E6_PHASE_DIAGNOSTIC") == "1"
        aggregate_enabled = (
            self.capture_metrics_path is not None
            and self.capture_metrics_mode
            in {"writer-boundary", "gather-writer-boundary"}
        )
        timing_enabled = diagnostic_enabled or aggregate_enabled
        expected_sequences = {
            NATIVE_CHANNEL_METADATA: 1,
            NATIVE_CHANNEL_STATE_UNIT: 1,
        }

        def flush_fragment_batch() -> None:
            nonlocal fragment_batch_wire_frame_count
            nonlocal fragment_persist_ns, fragment_persist_bytes
            if not fragment_batch:
                return
            payloads = tuple(fragment_batch)
            fragment_batch.clear()
            started_ns = time.monotonic_ns() if timing_enabled else 0
            self.service.consume_native_unit_fragment_payloads(
                payloads,
                wire_frame_count=fragment_batch_wire_frame_count,
            )
            fragment_batch_wire_frame_count = 0
            if timing_enabled:
                fragment_persist_ns += time.monotonic_ns() - started_ns
                fragment_persist_bytes += sum(
                    len(payload) - _STATE_UNIT_FRAGMENT.size
                    for payload in payloads
                )

        def flush_schedule_batch() -> None:
            nonlocal schedule_batch_wire_frame_count
            nonlocal schedule_batch_copied_bytes
            nonlocal fragment_persist_ns, fragment_persist_bytes
            if not schedule_batch:
                return
            payloads = tuple(schedule_batch)
            schedule_batch.clear()
            expected_copied_bytes = schedule_batch_copied_bytes
            schedule_batch_copied_bytes = 0
            started_ns = time.monotonic_ns() if timing_enabled else 0
            _logical_fragments, copied_bytes = (
                self.service.consume_native_unit_schedule_payloads(
                    payloads,
                    wire_frame_count=schedule_batch_wire_frame_count,
                )
            )
            schedule_batch_wire_frame_count = 0
            if copied_bytes != expected_copied_bytes:
                raise NativeRuntimeStateWireError(
                    "native schedule copied-byte total changed during persistence"
                )
            if timing_enabled:
                fragment_persist_ns += time.monotonic_ns() - started_ns
                fragment_persist_bytes += copied_bytes

        def flush_bulk_batch() -> None:
            nonlocal bulk_batch_wire_frame_count
            nonlocal bulk_batch_logical_fragments
            nonlocal bulk_batch_copied_bytes
            nonlocal fragment_persist_ns, fragment_persist_bytes
            if not bulk_batch:
                return
            payloads = tuple(bulk_batch)
            bulk_batch.clear()
            expected_logical_fragments = bulk_batch_logical_fragments
            expected_copied_bytes = bulk_batch_copied_bytes
            bulk_batch_logical_fragments = 0
            bulk_batch_copied_bytes = 0
            started_ns = time.monotonic_ns() if timing_enabled else 0
            logical_fragments, copied_bytes = (
                self.service.consume_native_unit_bulk_payloads(
                    payloads,
                    wire_frame_count=bulk_batch_wire_frame_count,
                )
            )
            bulk_batch_wire_frame_count = 0
            if (
                logical_fragments != expected_logical_fragments
                or copied_bytes != expected_copied_bytes
            ):
                raise NativeRuntimeStateWireError(
                    "native bulk totals changed during persistence"
                )
            if timing_enabled:
                fragment_persist_ns += time.monotonic_ns() - started_ns
                fragment_persist_bytes += copied_bytes

        try:
            while True:
                header = _read_exact(process.stdout, _HEADER.size)
                if header is None:
                    flush_fragment_batch()
                    flush_schedule_batch()
                    flush_bulk_batch()
                    break
                (
                    magic,
                    version,
                    frame_type,
                    payload_size,
                    channel_id,
                    sequence,
                ) = _HEADER.unpack(header)
                expected = _PAYLOAD_SIZES.get(frame_type)
                state_unit_frame = frame_type in {
                    FRAME_STATE_UNIT_FRAGMENT,
                    FRAME_STATE_UNIT_FRAGMENT_BATCH,
                    FRAME_STATE_UNIT_SCHEDULE_BATCH,
                    FRAME_STATE_UNIT_SEAL,
                    FRAME_STATE_UNIT_BULK,
                }
                expected_channel = (
                    NATIVE_CHANNEL_STATE_UNIT
                    if state_unit_frame
                    else NATIVE_CHANNEL_METADATA
                )
                valid_fragment_size = (
                    frame_type == FRAME_STATE_UNIT_FRAGMENT
                    and _STATE_UNIT_FRAGMENT.size < payload_size
                    <= _STATE_UNIT_FRAGMENT.size
                        + _MAX_STATE_UNIT_FRAGMENT_BYTES
                )
                valid_fragment_batch_size = (
                    frame_type == FRAME_STATE_UNIT_FRAGMENT_BATCH
                    and _STATE_UNIT_FRAGMENT_BATCH.size
                        + _STATE_UNIT_FRAGMENT.size
                        < payload_size
                    <= _STATE_UNIT_FRAGMENT_BATCH.size
                        + _MAX_STATE_UNIT_FRAGMENT_BATCH_RECORD_BYTES
                )
                valid_schedule_batch_size = (
                    frame_type == FRAME_STATE_UNIT_SCHEDULE_BATCH
                    and _STATE_UNIT_SCHEDULE_BATCH.size < payload_size
                    <= _STATE_UNIT_SCHEDULE_BATCH.size
                        + _MAX_STATE_UNIT_FRAGMENT_BYTES
                )
                valid_bulk_size = (
                    frame_type == FRAME_STATE_UNIT_BULK
                    and BULK_PREFIX_BYTES < payload_size
                    <= BULK_PREFIX_BYTES + BULK_PAYLOAD_BYTES
                )
                if (
                    magic != NATIVE_WIRE_MAGIC
                    or version != NATIVE_WIRE_VERSION
                    or channel_id != expected_channel
                    or sequence != expected_sequences.get(channel_id)
                    or (
                        not valid_fragment_size
                        and not valid_fragment_batch_size
                        and not valid_schedule_batch_size
                        and not valid_bulk_size
                        and (expected is None or payload_size != expected)
                    )
                ):
                    raise NativeRuntimeStateWireError(
                        "native observer emitted an invalid frame header"
                    )
                expected_sequences[channel_id] += 1
                payload = _read_exact(process.stdout, payload_size)
                if payload is None:
                    raise NativeRuntimeStateWireError(
                        "native observer frame payload is truncated"
                    )
                if state_unit_frame:
                    started_ns = time.monotonic_ns() if timing_enabled else 0
                    fragment_payloads: tuple[bytes, ...] = ()
                    copied_bytes = 0
                    if frame_type == FRAME_STATE_UNIT_FRAGMENT:
                        fragment_payloads = (payload,)
                    elif frame_type == FRAME_STATE_UNIT_FRAGMENT_BATCH:
                        fragment_payloads = (
                            _decode_state_unit_fragment_batch_payload(payload)
                        )
                    elif frame_type == FRAME_STATE_UNIT_SCHEDULE_BATCH:
                        _logical_fragments, copied_bytes = (
                            native_unit_schedule_payload_summary(payload)
                        )
                    elif frame_type == FRAME_STATE_UNIT_BULK:
                        try:
                            bulk = decode_native_bulk_payload(payload)
                        except NativeBulkDecodeError as error:
                            raise NativeRuntimeStateWireError(
                                "native observer emitted an invalid bulk frame"
                            ) from error
                        _logical_fragments = bulk.frame_fragment_count
                        copied_bytes = bulk.payload_size
                    if timing_enabled:
                        unit_read_ns += time.monotonic_ns() - started_ns
                        unit_read_bytes += (
                            sum(
                                len(item) - _STATE_UNIT_FRAGMENT.size
                                for item in fragment_payloads
                            )
                            if fragment_payloads
                            else copied_bytes
                        )
                    if fragment_payloads:
                        flush_schedule_batch()
                        flush_bulk_batch()
                        fragment_batch.extend(fragment_payloads)
                        fragment_batch_wire_frame_count += 1
                        if (
                            len(fragment_batch)
                            >= _NATIVE_UNIT_FRAGMENT_BATCH_SIZE
                        ):
                            flush_fragment_batch()
                        continue
                    if frame_type == FRAME_STATE_UNIT_SCHEDULE_BATCH:
                        flush_fragment_batch()
                        flush_bulk_batch()
                        schedule_batch.append(payload)
                        schedule_batch_wire_frame_count += 1
                        schedule_batch_copied_bytes += copied_bytes
                        if (
                            len(schedule_batch)
                            >= _NATIVE_UNIT_FRAGMENT_BATCH_SIZE
                        ):
                            flush_schedule_batch()
                        continue
                    if frame_type == FRAME_STATE_UNIT_BULK:
                        flush_fragment_batch()
                        flush_schedule_batch()
                        bulk_batch.append(payload)
                        bulk_batch_wire_frame_count += 1
                        bulk_batch_logical_fragments += _logical_fragments
                        bulk_batch_copied_bytes += copied_bytes
                        if len(bulk_batch) >= _NATIVE_UNIT_FRAGMENT_BATCH_SIZE:
                            flush_bulk_batch()
                        continue
                    flush_fragment_batch()
                    flush_schedule_batch()
                    flush_bulk_batch()
                    started_ns = time.monotonic_ns() if timing_enabled else 0
                    self.service.consume_native_unit_frame(
                        frame_type=frame_type,
                        payload=payload,
                    )
                    if timing_enabled:
                        elapsed_ns = time.monotonic_ns() - started_ns
                        seal_persist_ns += elapsed_ns
                        seal_persist_bytes += len(payload)
                    continue
                flush_fragment_batch()
                flush_schedule_batch()
                flush_bulk_batch()
                accumulated.extend(header)
                accumulated.extend(payload)
                if len(accumulated) > _MAX_RESPONSE_BYTES:
                    raise NativeRuntimeStateWireError(
                        "native observation exceeds the size limit"
                    )
                if frame_type in {
                    FRAME_WATERMARK_COUNTERS,
                    FRAME_TERMINAL_COUNTERS,
                }:
                    if timing_enabled and (
                        frame_type == FRAME_WATERMARK_COUNTERS
                        or unit_read_bytes
                        or fragment_persist_bytes
                        or seal_persist_bytes
                    ):
                        rows = (
                            ("observer_unit_read", unit_read_ns, unit_read_bytes),
                            (
                                "observer_fragment_persist",
                                fragment_persist_ns,
                                fragment_persist_bytes,
                            ),
                            (
                                "observer_seal_persist",
                                seal_persist_ns,
                                seal_persist_bytes,
                            ),
                        )
                        if aggregate_enabled:
                            for phase, elapsed_ns, byte_count in rows:
                                self._record_capture_metric(
                                    phase, elapsed_ns, byte_count
                                )
                        if diagnostic_enabled:
                            for phase, elapsed_ns, byte_count in rows:
                                _emit_e6_phase(phase, elapsed_ns, byte_count)
                        unit_read_ns = unit_read_bytes = 0
                        fragment_persist_ns = fragment_persist_bytes = 0
                        seal_persist_ns = seal_persist_bytes = 0
                    if frame_type == FRAME_WATERMARK_COUNTERS:
                        started_ns = time.monotonic_ns() if diagnostic_enabled else 0
                        parsed = parse_native_runtime_state_stream(
                            bytes(accumulated),
                            allow_nonterminal_watermark=True,
                        )
                        if diagnostic_enabled:
                            _emit_e6_phase(
                                "observer_prefix_parse",
                                time.monotonic_ns() - started_ns,
                                len(accumulated),
                            )
                        counters = parsed.watermarks[-1]
                        with self._condition:
                            if (
                                self._snapshot_generation
                                >= self._snapshot_request_generation
                            ):
                                raise RuntimeStateServiceError(
                                    "native observer published an unsolicited watermark"
                                )
                            self._snapshot_generation += 1
                            if not self._ready:
                                self._pre_ready_counters = counters
                                self._pre_ready_mutation_kinds = tuple(
                                    sorted(
                                        {
                                            item.mutation_kind
                                            for item in parsed.mutations
                                        }
                                    )
                                )
                                self._pre_ready_counter_generation += 1
                                self._condition.notify_all()
                                continue
                    started_ns = time.monotonic_ns() if diagnostic_enabled else 0
                    self.service.ingest_native_stream(bytes(accumulated))
                    if diagnostic_enabled:
                        _emit_e6_phase(
                            "observer_ingest",
                            time.monotonic_ns() - started_ns,
                            len(accumulated),
                        )
                    with self._condition:
                        if frame_type == FRAME_WATERMARK_COUNTERS:
                            self._watermark_generation += 1
                        else:
                            self._terminal_received = True
                        self._condition.notify_all()
        except BaseException as error:
            self._record_failure(error)

    def _consume_stderr(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            self._record_failure(
                RuntimeStateServiceError("native observer stderr is unavailable")
            )
            return
        try:
            while True:
                line = process.stderr.readline(1025)
                if not line:
                    break
                if len(line) > 1024 or not line.endswith(b"\n"):
                    raise RuntimeStateServiceError(
                        "native observer diagnostic line is invalid"
                    )
                diagnostic = line[:-1]
                if diagnostic.startswith(b"ATTACHED_PROBE "):
                    if (
                        self._attached
                        or self._ready
                        or self._probe_diagnostic_index
                        >= len(self._expected_probe_diagnostics)
                        or diagnostic
                        != self._expected_probe_diagnostics[
                            self._probe_diagnostic_index
                        ]
                    ):
                        raise RuntimeStateServiceError(
                            "native observer probe identity diagnostic is invalid"
                        )
                    self._probe_diagnostic_index += 1
                elif diagnostic.startswith(b"ATTACHED "):
                    attached = re.fullmatch(
                        rb"ATTACHED runtime_state probes=([1-9][0-9]{0,2}) "
                        rb"identities=([1-9][0-9]{0,2})",
                        diagnostic,
                    )
                    expected_identities = len(self._expected_probe_diagnostics)
                    if (
                        attached is None
                        or int(attached.group(1)) < expected_identities
                        or int(attached.group(1)) > 64
                        or int(attached.group(2)) != expected_identities
                        or self._attached
                        or self._ready
                        or self._probe_diagnostic_index
                        != expected_identities
                    ):
                        raise RuntimeStateServiceError(
                            "native observer ATTACHED diagnostic is invalid"
                        )
                    with self._condition:
                        self._attached = True
                        self._condition.notify_all()
                elif diagnostic.startswith(b"ARMED "):
                    if (
                        diagnostic != b"ARMED runtime_state mutations=2"
                        or not self._attached
                        or not self._ready
                        or self._armed
                    ):
                        raise RuntimeStateServiceError(
                            "native observer ARMED diagnostic is invalid"
                        )
                    with self._condition:
                        self._armed = True
                        self._condition.notify_all()
                elif diagnostic.startswith(b"READY "):
                    if self.service.profile.has_dynamic_topology:
                        tensor_pattern = rb"([1-9][0-9]*)"
                    else:
                        expected_tensors = (
                            2
                            * self.service.profile.n_stream
                            * self.service.profile.n_layer
                        )
                        tensor_pattern = str(expected_tensors).encode("ascii")
                    match = re.fullmatch(
                        b"READY runtime_state topology=1 tensors="
                        + tensor_pattern
                        + rb" backing=([1-9][0-9]*)",
                        diagnostic,
                    )
                    if (
                        match is None
                        or self._ready
                    ):
                        raise RuntimeStateServiceError(
                            "native observer READY diagnostic is invalid"
                        )
                    with self._condition:
                        if not self._attached:
                            raise RuntimeStateServiceError(
                                "native observer reported READY before ATTACHED"
                            )
                        self._ready = True
                        self._condition.notify_all()
                elif diagnostic.startswith(b"DIAGNOSTIC runtime_state "):
                    match = _TENSOR_READ_DIAGNOSTIC.fullmatch(diagnostic)
                    if match is None:
                        raise RuntimeStateServiceError(
                            "native observer snapshot diagnostic is invalid"
                        )
                    values = tuple(int(item) for item in match.groups())
                    parsed_diagnostic = _NativeTensorReadDiagnostics(*values)
                    with self._condition:
                        if (
                            self._diagnostic_generation
                            >= self._snapshot_request_generation
                        ):
                            raise RuntimeStateServiceError(
                                "native observer snapshot diagnostic is invalid"
                            )
                        self._tensor_read_diagnostics = parsed_diagnostic
                        self._diagnostic_generation += 1
                        self._condition.notify_all()
                elif diagnostic.startswith(b"APPLY_DIAGNOSTIC runtime_state "):
                    match = _APPLY_REJECTION_DIAGNOSTIC.fullmatch(diagnostic)
                    if match is None:
                        raise RuntimeStateServiceError(
                            "native observer apply diagnostic is invalid"
                        )
                    parsed_apply = _NativeApplyRejectionDiagnostics(
                        *(int(item) for item in match.groups())
                    )
                    if (
                        any(
                            getattr(parsed_apply, field)
                            for field in parsed_apply.__dataclass_fields__
                        )
                        and not self._apply_rejection_diagnostic_reported
                    ):
                        print(
                            "runtime-state apply rejection diagnostic: "
                            + _format_apply_rejection_diagnostics(parsed_apply),
                            file=sys.stderr,
                            flush=True,
                        )
                        self._apply_rejection_diagnostic_reported = True
                elif diagnostic.startswith(b"GATHER_FAILED runtime_state "):
                    match = _NATIVE_GATHER_FAILURE_DIAGNOSTIC.fullmatch(
                        diagnostic
                    )
                    if match is None:
                        raise RuntimeStateServiceError(
                            "native observer gather diagnostic is invalid"
                        )
                    stage, error_code, pending_count = match.groups()
                    print(
                        "runtime-state gather diagnostic: "
                        f"stage={stage.decode('ascii')} "
                        f"error={int(error_code)} "
                        f"pending={int(pending_count)}",
                        file=sys.stderr,
                        flush=True,
                    )
                elif diagnostic.startswith(b"BULK_METRICS runtime_state "):
                    match = _NATIVE_BULK_METRICS_DIAGNOSTIC.fullmatch(
                        diagnostic
                    )
                    if (
                        match is None
                        or self._native_bulk_metrics is not None
                        or not self._attached
                        or not self._ready
                        or not self._armed
                    ):
                        raise RuntimeStateServiceError(
                            "native observer bulk metric is invalid"
                        )
                    self._native_bulk_metrics = dict(
                        zip(
                            _BULK_AGGREGATE_NATIVE_FIELDS,
                            (int(item) for item in match.groups()),
                            strict=True,
                        )
                    )
                elif diagnostic.startswith(b"DRAIN_STALLED runtime_state "):
                    match = _NATIVE_DRAIN_STALL_DIAGNOSTIC.fullmatch(
                        diagnostic
                    )
                    if match is None:
                        raise RuntimeStateServiceError(
                            "native observer drain stall diagnostic is invalid"
                        )
                    (
                        submitted,
                        received,
                        active,
                        inflight,
                        pending,
                        stopped,
                        plans,
                        ready,
                        reserve,
                        unmatched,
                    ) = (int(item) for item in match.groups())
                    print(
                        "runtime-state drain stall diagnostic: "
                        f"submitted={submitted} received={received} "
                        f"active={active} inflight={inflight} "
                        f"pending={pending} stopped={stopped} "
                        f"plans={plans} ready={ready} "
                        f"reserve={reserve} unmatched={unmatched}",
                        file=sys.stderr,
                        flush=True,
                    )
                elif diagnostic.startswith(b"DRAIN_FAILED runtime_state "):
                    match = _NATIVE_DRAIN_DIAGNOSTIC.fullmatch(diagnostic)
                    if match is None:
                        raise RuntimeStateServiceError(
                            "native observer drain diagnostic is invalid"
                        )
                    submitted, received, active = (
                        int(item) for item in match.groups()
                    )
                    print(
                        "runtime-state drain diagnostic: "
                        f"submitted={submitted} received={received} "
                        f"active={active}",
                        file=sys.stderr,
                        flush=True,
                    )
                    if submitted != received:
                        raise RuntimeStateServiceError(
                            "native observer drain delivery mismatch"
                        )
                    if active != 0:
                        raise RuntimeStateServiceError(
                            "native observer drain has an active emitter"
                        )
                    raise RuntimeStateServiceError(
                        "native observer drain did not reach stable parity"
                    )
                elif diagnostic in _NATIVE_STARTUP_FAILURE_DIAGNOSTICS:
                    raise RuntimeStateServiceError(
                        diagnostic.decode("ascii")
                    )
                elif diagnostic == (
                    b"runtime-state observer failed to attach mutation probes"
                ):
                    raise RuntimeStateServiceError(
                        "native observer failed to attach mutation probes"
                    )
                elif (
                    not self._attached
                    and not self._ready
                    and _LIBBPF_STARTUP_DIAGNOSTIC.fullmatch(diagnostic)
                    is not None
                ):
                    print(
                        diagnostic.decode("ascii"),
                        file=sys.stderr,
                        flush=True,
                    )
                else:
                    raise RuntimeStateServiceError(
                        "native observer diagnostic is not allowlisted: "
                        "diagnostic_hex=" + diagnostic.hex()
                    )
        except BaseException as error:
            self._record_failure(error)

    def _record_failure(self, error: BaseException) -> None:
        with self._condition:
            if self._failure is None:
                self._failure = error
            self._condition.notify_all()

    def _abort_process(self) -> None:
        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)

    def abort(self) -> None:
        self._abort_process()


def _format_pre_ready_counters(counters: NativeWatermarkCounters) -> str:
    return (
        f"topology={counters.topology_event_count} "
        f"tensors={counters.tensor_layout_event_count} "
        f"backing={counters.kv_backing_event_count} "
        f"tokens={counters.token_cell_event_count} "
        f"applies={counters.apply_complete_event_count} "
        f"submitted={counters.kernel_event_submit_count} "
        f"received={counters.native_event_receive_count} "
        f"frames={counters.native_wire_frame_count} "
        f"reserve_failures={counters.ringbuf_reserve_fail_count} "
        f"read_failures={counters.user_memory_read_failure_count} "
        f"unmatched={counters.unmatched_call_count} "
        f"unsupported={counters.unsupported_mutation_count} "
        f"mapping_mutations={counters.mapping_mutation_count} "
        f"inflight={counters.final_inflight_count} "
        f"mutations={counters.mutation_event_count} "
        f"emitters={counters.active_emitter_count}"
    )


def _format_tensor_read_diagnostics(
    diagnostics: _NativeTensorReadDiagnostics,
) -> str:
    return (
        f"candidate_missing={diagnostics.candidate_missing} "
        f"shape_invalid={diagnostics.shape_invalid} "
        f"data_read_failed={diagnostics.data_read_failed} "
        f"type_read_failed={diagnostics.type_read_failed} "
        f"data_null={diagnostics.data_null}"
    )


def _format_apply_rejection_diagnostics(
    diagnostics: _NativeApplyRejectionDiagnostics,
) -> str:
    return (
        f"state_conflict={diagnostics.state_conflict} "
        f"header_read_failed={diagnostics.header_read_failed} "
        f"slot_shape_invalid={diagnostics.slot_shape_invalid} "
        f"batch_shape_invalid={diagnostics.batch_shape_invalid} "
        f"stream_count_zero={diagnostics.stream_count_zero} "
        f"stream_count_exceeds={diagnostics.stream_count_exceeds} "
        "index_vector_extent_invalid="
        f"{diagnostics.index_vector_extent_invalid} "
        f"token_count_zero={diagnostics.token_count_zero} "
        f"token_count_exceeds={diagnostics.token_count_exceeds} "
        "token_count_exceeds_observed="
        f"{diagnostics.token_count_exceeds_observed} "
        "token_count_exceeds_streams="
        f"{diagnostics.token_count_exceeds_streams} "
        f"token_stream_remainder={diagnostics.token_stream_remainder} "
        f"token_read_failed={diagnostics.token_read_failed} "
        f"token_shape_invalid={diagnostics.token_shape_invalid} "
        f"return_unmatched={diagnostics.return_unmatched} "
        f"state_delete_failed={diagnostics.state_delete_failed} "
        f"process_state_conflict={diagnostics.process_state_conflict} "
        f"process_return_unmatched={diagnostics.process_return_unmatched} "
        f"process_apply_missing={diagnostics.process_apply_missing} "
        "process_graph_compute_failed="
        f"{diagnostics.process_graph_compute_failed}"
    )


def _format_pre_ready_mutation_kinds(kinds: tuple[str, ...]) -> str:
    canonical = tuple(sorted(set(kinds)))
    return "mutation_kinds=" + (",".join(canonical) if canonical else "none")


def _read_exact(stream: BinaryIO, size: int) -> bytes | None:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = stream.read(size - len(chunks))
        if not chunk:
            if not chunks:
                return None
            raise NativeRuntimeStateWireError(
                "native observer stream ended inside a frame"
            )
        chunks.extend(chunk)
    return bytes(chunks)


PeerAuthorizer = Callable[[socket.socket], bool]


def _same_user_peer(peer: socket.socket) -> bool:
    if hasattr(socket, "SO_PEERCRED") and hasattr(os, "getuid"):
        try:
            credentials = peer.getsockopt(
                socket.SOL_SOCKET,
                socket.SO_PEERCRED,
                struct.calcsize("3i"),
            )
            _pid, uid, _gid = struct.unpack("3i", credentials)
            return uid == os.getuid()
        except OSError:
            return False
    return os.name == "nt"


class RuntimeStateUnixSnapshotServer:
    def __init__(
        self,
        *,
        service: RuntimeStateSnapshotService,
        socket_path: Path,
        peer_authorizer: PeerAuthorizer | None = None,
        arm_requester: Callable[[], None] | None = None,
        watermark_requester: Callable[[], None] | None = None,
        drain_requester: Callable[[], None] | None = None,
        stop_requester: Callable[[], None] | None = None,
    ) -> None:
        if not isinstance(service, RuntimeStateSnapshotService):
            raise RuntimeStateServiceError("service must be RuntimeStateSnapshotService")
        if not isinstance(socket_path, Path) or not socket_path.is_absolute():
            raise RuntimeStateServiceError("socket_path must be an absolute pathlib.Path")
        self.service = service
        self.socket_path = socket_path
        self.peer_authorizer = peer_authorizer or _same_user_peer
        self.arm_requester = arm_requester
        self.watermark_requester = watermark_requester
        self.drain_requester = drain_requester
        self.stop_requester = stop_requester
        self._listener: socket.socket | None = None
        self._stop_requested = threading.Event()

    def __enter__(self) -> RuntimeStateUnixSnapshotServer:
        self.socket_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.socket_path.exists() or self.socket_path.is_symlink():
            raise RuntimeStateServiceError("snapshot socket path already exists")
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            listener.bind(str(self.socket_path))
            os.chmod(self.socket_path, 0o600)
            listener.listen(self.service.profile.n_stream)
        except BaseException:
            listener.close()
            self.socket_path.unlink(missing_ok=True)
            raise
        self._listener = listener
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        if self._listener is not None:
            self._listener.close()
            self._listener = None
        self.socket_path.unlink(missing_ok=True)

    def serve_once(self) -> bool:
        if self._listener is None:
            raise RuntimeStateServiceError("snapshot server is not bound")
        peer, _ = self._listener.accept()
        return self._serve_peer(peer)

    def _serve_peer(self, peer: socket.socket) -> bool:
        with peer:
            peer.settimeout(5.0)
            if not self.peer_authorizer(peer):
                peer.sendall(_error_response("PEER_NOT_AUTHORIZED", "peer is not authorized"))
                return True
            payload = bytearray()
            while b"\n" not in payload and len(payload) <= _MAX_REQUEST_BYTES:
                chunk = peer.recv(_MAX_REQUEST_BYTES + 1 - len(payload))
                if not chunk:
                    break
                payload.extend(chunk)
            try:
                request = bytes(payload)
                if request == b"STOP {}\n":
                    if self.stop_requester is None:
                        raise RuntimeStateServiceError(
                            "service stop control is unavailable"
                        )
                    self.stop_requester()
                    self._stop_requested.set()
                    response = _canonical_json({"state": "sealed"}) + b"\n"
                elif request == b"ARM {}\n":
                    if self.arm_requester is None:
                        raise RuntimeStateServiceError(
                            "service arm control is unavailable"
                        )
                    self.arm_requester()
                    response = _canonical_json({"state": "armed"}) + b"\n"
                elif request == b"DRAIN {}\n":
                    if self.drain_requester is None:
                        raise RuntimeStateServiceError(
                            "service drain control is unavailable"
                        )
                    self.drain_requester()
                    response = _canonical_json({"state": "drained"}) + b"\n"
                else:
                    if self.watermark_requester is not None:
                        diagnostic_enabled = os.environ.get("LARIS_E6_PHASE_DIAGNOSTIC") == "1"
                        started_ns = time.monotonic_ns() if diagnostic_enabled else 0
                        self.watermark_requester()
                        if diagnostic_enabled:
                            _emit_e6_phase(
                                "snapshot_service_watermark",
                                time.monotonic_ns() - started_ns,
                                0,
                            )
                    else:
                        diagnostic_enabled = False
                    started_ns = time.monotonic_ns() if diagnostic_enabled else 0
                    response = self.service.handle_request(request)
                    if diagnostic_enabled:
                        _emit_e6_phase(
                            "snapshot_service_handle",
                            time.monotonic_ns() - started_ns,
                            len(response),
                        )
            except RuntimeStateServiceError as error:
                print(
                    "runtime-state snapshot unavailable: reason="
                    + _snapshot_failure_reason(error),
                    file=sys.stderr,
                    flush=True,
                )
                response = _error_response(
                    "SNAPSHOT_UNAVAILABLE",
                    "runtime-state snapshot is unavailable",
                )
            peer.sendall(response)
        return not self._stop_requested.is_set()

    def serve_forever(self) -> None:
        listener = self._listener
        if listener is None:
            raise RuntimeStateServiceError("snapshot server is not bound")
        workers: set[Future[bool]] = set()
        listener.settimeout(0.1)
        try:
            with ThreadPoolExecutor(
                max_workers=self.service.profile.n_stream,
                thread_name_prefix="laris-runtime-state-snapshot",
            ) as executor:
                while not self._stop_requested.is_set():
                    if len(workers) >= self.service.profile.n_stream:
                        completed, workers = wait(
                            workers,
                            timeout=0.1,
                            return_when=FIRST_COMPLETED,
                        )
                        for future in completed:
                            future.result()
                        continue
                    try:
                        peer, _ = listener.accept()
                    except socket.timeout:
                        continue
                    workers.add(executor.submit(self._serve_peer, peer))
                for future in workers:
                    future.result()
        finally:
            listener.settimeout(None)


def _error_response(code: str, message: str) -> bytes:
    return _canonical_json({"code": code, "message": message, "state": "error"}) + b"\n"


def _positive_cli_int(value: str) -> int:
    try:
        parsed = int(value, 0)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be an integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_cli_int(value: str) -> int:
    try:
        parsed = int(value, 0)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be an integer") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _service_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m laris_onpremise.observer.runtime_state_service",
        description=(
            "Run the identity-neutral LARIS runtime-state observer service."
        ),
    )
    parser.add_argument("--socket-path", type=Path, required=True)
    parser.add_argument("--frame-dir", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--static-qualification-receipt", type=Path)
    parser.add_argument("--qualification-runtime-root", type=Path)
    parser.add_argument("--model-registration-receipt", type=Path)
    parser.add_argument("--runtime-instance-id", required=True)
    parser.add_argument("--runtime-pid", type=_positive_cli_int, required=True)
    parser.add_argument(
        "--runtime-start-time-ticks", type=_positive_cli_int, required=True
    )
    parser.add_argument("--native-observer", type=Path, required=True)
    parser.add_argument("--runtime-binary", type=Path, required=True)
    parser.add_argument(
        "--attachment-mode",
        choices=("preexec", "live"),
        default="preexec",
    )
    parser.add_argument(
        "--capture-mode",
        choices=(
            "metadata",
            "writer-boundary",
            "gather-writer-boundary",
            "compact-metadata",
        ),
        default="metadata",
    )
    parser.add_argument(
        "--session-mode",
        choices=("snapshot", "log-linked"),
        default="snapshot",
    )
    parser.add_argument("--bpf-object", type=Path, required=True)
    parser.add_argument(
        "--constructor-offset", type=_nonnegative_cli_int, required=True
    )
    parser.add_argument(
        "--n-pos-per-embd-offset", type=_nonnegative_cli_int, required=True
    )
    parser.add_argument("--view-offset", type=_nonnegative_cli_int, required=True)
    parser.add_argument(
        "--buffer-base-offset", type=_nonnegative_cli_int, required=True
    )
    parser.add_argument(
        "--buffer-size-offset", type=_nonnegative_cli_int, required=True
    )
    parser.add_argument("--apply-offset", type=_nonnegative_cli_int, required=True)
    parser.add_argument(
        "--tensor-data-offset", type=_nonnegative_cli_int, required=True
    )
    parser.add_argument(
        "--tensor-type-offset", type=_nonnegative_cli_int, required=True
    )
    parser.add_argument(
        "--ready-timeout-seconds", type=float, default=10.0
    )
    parser.add_argument(
        "--control-timeout-seconds", type=float, default=10.0
    )
    parser.add_argument("--e6-capture-metrics-path", type=Path)
    return parser


def _require_absolute_paths(
    parser: argparse.ArgumentParser, values: Sequence[Path]
) -> None:
    if any(not value.is_absolute() for value in values):
        parser.error("all filesystem paths must be absolute")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _service_argument_parser()
    args = parser.parse_args(argv)
    _require_absolute_paths(
        parser,
        (
            args.socket_path,
            args.frame_dir,
            args.profile,
            args.native_observer,
            args.runtime_binary,
            args.bpf_object,
            *(
                (args.static_qualification_receipt,)
                if args.static_qualification_receipt is not None
                else ()
            ),
            *(
                (args.qualification_runtime_root,)
                if args.qualification_runtime_root is not None
                else ()
            ),
            *(
                (args.model_registration_receipt,)
                if args.model_registration_receipt is not None
                else ()
            ),
            *(
                (args.e6_capture_metrics_path,)
                if args.e6_capture_metrics_path is not None
                else ()
            ),
        ),
    )
    controller: NativeRuntimeStateProcess | None = None
    registered_model: RegisteredModelHandle | None = None
    try:
        profile = (
            load_runtime_state_observer_profile(args.profile)
            if args.static_qualification_receipt is None
            else load_runtime_state_qualification_probe_profile(
                args.profile, args.static_qualification_receipt
            )
        )
        if not (
            profile.is_linux_live_qualified
            or getattr(profile, "qualification_state", None)
            == "linux_live_capture"
        ):
            raise RuntimeStateServiceError(
                "runtime-state profile is linux_live_pending: "
                + profile.qualification_reason
            )
        validate_runtime_state_capture_mode(profile, args.capture_mode)
        if getattr(profile, "schema_version", None) in {"1.4", "1.5"}:
            if args.model_registration_receipt is None:
                raise RuntimeStateServiceError(
                    "dynamic observer profile requires a model registration receipt"
                )
            registration = load_model_registration_receipt(
                args.model_registration_receipt
            )
            registered_model = open_registered_model(
                path=registration.canonical_path,
                receipt=registration,
            )
        targets = {target.probe_id: target for target in profile.probe_targets}
        expected_cli_offsets = {
            "constructor_offset": targets["kv_constructor"].resolved_file_offset,
            "n_pos_per_embd_offset": targets["n_pos_per_embd"].resolved_file_offset,
            "view_offset": targets["view_2d"].resolved_file_offset,
            "buffer_base_offset": targets["buffer_base"].resolved_file_offset,
            "buffer_size_offset": targets["buffer_size"].resolved_file_offset,
            "apply_offset": targets["apply_ubatch"].resolved_file_offset,
            "tensor_data_offset": profile.offsets.tensor_data,
            "tensor_type_offset": profile.offsets.tensor_type,
        }
        for field, expected in expected_cli_offsets.items():
            if getattr(args, field) != expected:
                raise RuntimeStateServiceError(
                    f"CLI {field} does not match the qualified profile manifest"
                )
        qualification_runtime_root: Path | None = None
        probe_path_resolver: Callable[[Path], Path] | None = None
        if args.qualification_runtime_root is not None:
            if args.static_qualification_receipt is None:
                raise RuntimeStateServiceError(
                    "qualification runtime root requires a static qualification receipt"
                )
            if args.attachment_mode != "preexec":
                raise RuntimeStateServiceError(
                    "qualification runtime root is valid only for preexec attachment"
                )
            root_metadata = args.qualification_runtime_root.lstat()
            if not stat.S_ISDIR(root_metadata.st_mode):
                raise RuntimeStateServiceError(
                    "qualification runtime root is not a directory"
                )
            qualification_runtime_root = args.qualification_runtime_root.resolve(
                strict=True
            )
            if qualification_runtime_root != args.qualification_runtime_root:
                raise RuntimeStateServiceError(
                    "qualification runtime root must be a normalized real path"
                )

            logical_runtime_root = Path("/opt/laris/llama-runtime")

            def resolve_qualification_probe_path(logical_path: Path) -> Path:
                try:
                    relative_path = logical_path.relative_to(logical_runtime_root)
                except ValueError as error:
                    raise RuntimeStateServiceError(
                        "qualified probe path is outside the runtime contract root"
                    ) from error
                if not relative_path.parts or any(
                    part in ("", ".", "..") for part in relative_path.parts
                ):
                    raise RuntimeStateServiceError(
                        "qualified probe path is not a canonical runtime path"
                    )
                return qualification_runtime_root.joinpath(*relative_path.parts)

            probe_path_resolver = resolve_qualification_probe_path
            if args.runtime_binary != resolve_qualification_probe_path(
                profile.runtime_binary_path
            ):
                raise RuntimeStateServiceError(
                    "CLI runtime_binary does not match the qualification runtime root"
                )
        elif str(args.runtime_binary) != str(profile.runtime_binary_path):
            raise RuntimeStateServiceError(
                "CLI runtime_binary does not match the qualified profile manifest"
            )
        attachment_identity = (
            _wait_for_live_runtime_identity(
                profile=profile,
                runtime_pid=args.runtime_pid,
                expected_start_time_ticks=args.runtime_start_time_ticks,
                timeout_seconds=args.ready_timeout_seconds,
                registered_model=registered_model,
            )
            if args.attachment_mode == "live"
            else _prepare_runtime_attachment_identity(
                profile=profile,
                runtime_pid=args.runtime_pid,
                expected_start_time_ticks=args.runtime_start_time_ticks,
                registered_model=registered_model,
                runtime_binary_path=(
                    args.runtime_binary
                    if qualification_runtime_root is not None
                    else None
                ),
                probe_path_resolver=probe_path_resolver,
            )
        )
        attached_targets = {
            target.probe_id: target
            for target in attachment_identity.probe_targets
        }
        command = (
            str(args.native_observer),
            "--target-tgid",
            str(args.runtime_pid),
            "--bpf-object",
            str(args.bpf_object),
            "--expected-n-stream",
            str(profile.n_stream),
            "--expected-n-seq-max",
            str(profile.n_seq_max),
            "--expected-n-layer",
            str(profile.n_layer),
            "--constructor-offset",
            str(targets["kv_constructor"].resolved_file_offset),
            "--constructor-binary",
            str(attached_targets["kv_constructor"].attachment_path),
            "--n-pos-per-embd-offset",
            str(targets["n_pos_per_embd"].resolved_file_offset),
            "--n-pos-per-embd-binary",
            str(attached_targets["n_pos_per_embd"].attachment_path),
            "--view-offset",
            str(targets["view_2d"].resolved_file_offset),
            "--view-binary",
            str(attached_targets["view_2d"].attachment_path),
            "--buffer-base-offset",
            str(targets["buffer_base"].resolved_file_offset),
            "--buffer-base-binary",
            str(attached_targets["buffer_base"].attachment_path),
            "--buffer-size-offset",
            str(targets["buffer_size"].resolved_file_offset),
            "--buffer-size-binary",
            str(attached_targets["buffer_size"].attachment_path),
            "--apply-offset",
            str(targets["apply_ubatch"].resolved_file_offset),
            "--apply-binary",
            str(attached_targets["apply_ubatch"].attachment_path),
            *(
                (
                    "--process-ubatch-offset",
                    str(targets["process_ubatch"].resolved_file_offset),
                    "--process-ubatch-binary",
                    str(
                        attached_targets[
                            "process_ubatch"
                        ].attachment_path
                    ),
                    "--capture-mode",
                    args.capture_mode,
                )
                if args.capture_mode
                in {"writer-boundary", "gather-writer-boundary"}
                else ()
            ),
            *(
                ("--capture-mode", "compact-metadata")
                if args.capture_mode == "compact-metadata"
                else ()
            ),
            "--tensor-data-offset",
            str(profile.offsets.tensor_data),
            "--tensor-type-offset",
            str(profile.offsets.tensor_type),
            *(
                item
                for identity in sorted(
                    attachment_identity.probe_targets,
                    key=lambda candidate: candidate.probe_id,
                )
                for item in (
                    "--probe-identity",
                    _probe_identity_document(identity).decode("ascii"),
                )
            ),
            *(
                item
                for target in profile.probe_targets
                if target.attachment == "entry"
                for item in (
                    "--mutation-probe",
                    f"{_MUTATION_CODE_BY_KIND[target.probe_id]}|"
                    f"{attached_targets[target.probe_id].attachment_path}|"
                    f"{target.resolved_file_offset}",
                )
            ),
        )
        service = RuntimeStateSnapshotService(
            profile=profile,
            runtime_instance_id=args.runtime_instance_id,
            verified_runtime=attachment_identity,
            frame_dir=args.frame_dir,
            capture_mode=args.capture_mode,
        )
        controller = NativeRuntimeStateProcess(
            command=command,
            service=service,
            ready_timeout_seconds=args.ready_timeout_seconds,
            control_timeout_seconds=args.control_timeout_seconds,
            capture_metrics_path=args.e6_capture_metrics_path,
            capture_metrics_mode=(
                args.capture_mode
                if args.e6_capture_metrics_path is not None
                else None
            ),
        )
        controller.start()
        print("ATTACHED runtime-state-service", flush=True)
        verified_runtime = _wait_for_live_runtime_identity(
            profile=profile,
            runtime_pid=args.runtime_pid,
            expected_start_time_ticks=args.runtime_start_time_ticks,
            timeout_seconds=args.ready_timeout_seconds,
            registered_model=registered_model,
        )
        if not _same_runtime_security_identity(
            attachment_identity,
            verified_runtime,
        ):
            raise RuntimeStateServiceError(
                "post-exec runtime identity differs from the attached probe identity"
            )
        controller.wait_ready()
        controller.request_watermark()
        with RuntimeStateUnixSnapshotServer(
            service=service,
            socket_path=args.socket_path,
            arm_requester=controller.arm_mutations,
            watermark_requester=(
                controller.request_watermark
                if args.session_mode == "snapshot"
                else None
            ),
            drain_requester=(
                controller.request_watermark
                if args.session_mode == "log-linked"
                else None
            ),
            stop_requester=controller.stop,
        ) as server:
            print(service.topology_marker_line(), flush=True)
            print("READY runtime-state-service", flush=True)
            server.serve_forever()
        if not service.is_sealed:
            raise RuntimeStateServiceError(
                "runtime-state observation did not reach terminal seal"
            )
        return 0
    except (
        ModelRegistrationError,
        OSError,
        RuntimeStateProfileError,
        RuntimeStateServiceError,
    ) as error:
        _report_runtime_state_service_failure(error)
        if controller is not None and controller.is_running:
            controller.abort()
        return 1
    finally:
        if registered_model is not None:
            os.close(registered_model.descriptor)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FRAME_KV_BACKING",
    "FRAME_APPLY_COMPLETE",
    "FRAME_MUTATION",
    "FRAME_STREAM_VERSION",
    "FRAME_STATE_UNIT_FRAGMENT",
    "FRAME_STATE_UNIT_FRAGMENT_BATCH",
    "FRAME_STATE_UNIT_SCHEDULE_BATCH",
    "FRAME_STATE_UNIT_BULK",
    "FRAME_STATE_UNIT_SEAL",
    "FRAME_RUNTIME_TOPOLOGY",
    "FRAME_TENSOR_LAYOUT",
    "FRAME_TERMINAL_COUNTERS",
    "FRAME_TOKEN_CELL",
    "FRAME_TOKEN_CELL_RANGE",
    "FRAME_WATERMARK_COUNTERS",
    "KvBackingEvent",
    "LogLinkedRuntimeCut",
    "NATIVE_WIRE_MAGIC",
    "NATIVE_WIRE_VERSION",
    "NATIVE_CHANNEL_METADATA",
    "NATIVE_CHANNEL_STATE_UNIT",
    "NativeRuntimeStateWireError",
    "NativeCapturedUnitDescriptor",
    "CompletedNativeUnitPage",
    "NativeApplyCompleteEvent",
    "NativeMutationEvent",
    "NativeMutationPolicySummary",
    "NativeTokenCellRange",
    "NativeStateUnitFragment",
    "NativeStateUnitSeal",
    "NativeStreamVersionSnapshot",
    "NativeObservationCounters",
    "NativeRuntimeStateProcess",
    "NativeTerminalCounters",
    "NativeWatermarkCounters",
    "CompletedApplyPage",
    "ObservedApplyBoundary",
    "ObservedApplyTokenCell",
    "ParsedNativeRuntimeStateStream",
    "RuntimeStateServiceError",
    "RuntimeStateSnapshotService",
    "RuntimeStateTerminalSeal",
    "RuntimeStateTopologySummary",
    "VerifiedProbeTargetIdentity",
    "VerifiedRuntimeIdentity",
    "RuntimeStateUnixSnapshotServer",
    "RuntimeTopologyEvent",
    "TensorLayoutEvent",
    "TokenCellEvent",
    "encode_kv_backing_frame",
    "encode_apply_complete_frame",
    "encode_mutation_frame",
    "encode_completed_applies_request",
    "encode_completed_native_units_request",
    "encode_read_native_unit_request",
    "encode_log_linked_cut_request",
    "encode_snapshot_request",
    "encode_stream_version_frame",
    "encode_state_unit_fragment_frame",
    "encode_state_unit_fragment_batch_frame",
    "encode_state_unit_seal_frame",
    "encode_tensor_layout_frame",
    "encode_token_cell_range_frame",
    "encode_terminal_frame",
    "encode_token_cell_frame",
    "encode_topology_frame",
    "encode_watermark_frame",
    "load_and_verify_runtime_state_terminal_seal",
    "main",
    "parse_native_runtime_state_stream",
    "parse_completed_applies_request",
    "parse_completed_native_units_request",
    "parse_read_native_unit_request",
    "parse_log_linked_cut_request",
    "parse_snapshot_request",
    "native_stream_version_from_document",
    "native_stream_version_to_document",
    "validate_native_runtime_state_mutations",
    "verify_live_runtime_identity",
    "runtime_state_snapshot_to_frame",
]
