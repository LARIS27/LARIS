from __future__ import annotations

import hashlib
import json
import re
import stat
from dataclasses import dataclass
from pathlib import Path

from .elf import (
    MappedElfIdentity,
    NativeDependency,
    ProcMapEntry,
    ProcessIdentity,
    ProcessInspectionBackend,
    SystemProcessInspectionBackend,
    inspect_single_mapped_libllama,
    read_process_identity,
    trusted_file_key,
)


_DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
_ATTACHED_PATTERN = re.compile(
    r"ATTACHED entry program_id=([1-9][0-9]*) "
    r"link_id=([1-9][0-9]*) tag=([0-9a-f]{16}) "
    r"return program_id=([1-9][0-9]*) "
    r"link_id=([1-9][0-9]*) tag=([0-9a-f]{16})"
)


class ProvenanceError(RuntimeError):
    """Raised when observer provenance is incomplete or inconsistent."""


@dataclass(frozen=True)
class LoadedBpfProgram:
    probe_role: str
    program_id: int
    link_id: int
    kernel_tag: str

    def to_mapping(self) -> dict[str, object]:
        return {
            "probe_role": self.probe_role,
            "program_id": self.program_id,
            "link_id": self.link_id,
            "kernel_tag": self.kernel_tag,
        }

    def tag_mapping(self) -> dict[str, str]:
        return {
            "probe_role": self.probe_role,
            "kernel_tag": self.kernel_tag,
        }


@dataclass(frozen=True)
class ThreadState:
    tid: int
    start_time_ticks: int
    state: str

    def to_mapping(self) -> dict[str, object]:
        return {
            "tid": self.tid,
            "start_time_ticks": self.start_time_ticks,
            "state": self.state,
        }


@dataclass(frozen=True)
class QuiescentMapping:
    ordinal: int
    length_bytes: int
    permissions: str
    file_offset: int
    device_major: int
    device_minor: int
    inode: int
    path_display: str

    def to_mapping(self) -> dict[str, object]:
        return {
            "ordinal": self.ordinal,
            "length_bytes": self.length_bytes,
            "permissions": self.permissions,
            "file_offset": self.file_offset,
            "device_major": self.device_major,
            "device_minor": self.device_minor,
            "inode": self.inode,
            "path_display": self.path_display,
        }


@dataclass(frozen=True)
class QuiescentSnapshot:
    thread_states: tuple[ThreadState, ...]
    mapping_snapshot: tuple[QuiescentMapping, ...]

    def to_mapping(self) -> dict[str, object]:
        return {
            "thread_states": [state.to_mapping() for state in self.thread_states],
            "mapping_snapshot": [
                mapping.to_mapping() for mapping in self.mapping_snapshot
            ],
        }


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_json_digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _require_digest(value: str, field: str) -> None:
    if not isinstance(value, str) or _DIGEST_PATTERN.fullmatch(value) is None:
        raise ProvenanceError(f"{field} must be a lowercase SHA-256 digest")


def _require_positive(value: int, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ProvenanceError(f"{field} must be a positive integer")


def _require_text(value: str, field: str, *, minimum: int = 1) -> None:
    if not isinstance(value, str) or len(value) < minimum:
        raise ProvenanceError(f"{field} must contain at least {minimum} characters")


def derive_runtime_instance_id(process: ProcessIdentity) -> str:
    return canonical_json_digest(
        {
            "host_boot_id": process.host_boot_id,
            "server_pid": process.pid,
            "server_start_time_ticks": process.start_time_ticks,
            "live_executable_digest": process.live_executable.digest,
        }
    )


def build_native_dependency_manifest(
    dependencies: tuple[NativeDependency, ...],
) -> dict[str, object]:
    required_roles = ("libbpf", "libelf", "zlib", "libc", "loader")
    by_role = {dependency.role: dependency for dependency in dependencies}
    if len(by_role) != len(dependencies) or tuple(by_role) != required_roles:
        raise ProvenanceError(
            "native dependencies must have fixed libbpf, libelf, zlib, libc, loader order"
        )
    body = {
        "schema_version": "1.0",
        "dependencies": [
            by_role[role].to_mapping() for role in required_roles
        ],
    }
    return {**body, "digest": canonical_json_digest(body)}


def build_source_bundle_manifest(
    source_root: Path | str,
    source_files: tuple[Path | str, ...],
) -> dict[str, object]:
    root = Path(source_root).resolve(strict=True)
    if not source_files:
        raise ProvenanceError("source bundle must contain at least one file")
    rows: list[dict[str, object]] = []
    observed_paths: set[str] = set()
    for supplied in source_files:
        path = Path(supplied)
        try:
            relative = path.resolve(strict=True).relative_to(root)
        except ValueError as error:
            raise ProvenanceError("source file is outside source root") from error
        result = path.lstat()
        if stat.S_ISLNK(result.st_mode) or not stat.S_ISREG(result.st_mode):
            raise ProvenanceError("source bundle entries must be regular non-symlink files")
        relative_text = relative.as_posix()
        if relative_text in observed_paths:
            raise ProvenanceError("source bundle contains a duplicate path")
        observed_paths.add(relative_text)
        rows.append(
            {
                "path": relative_text,
                "size_bytes": result.st_size,
                "digest": _sha256_path(path),
            }
        )
    rows.sort(key=lambda item: str(item["path"]))
    body = {"schema_version": "1.0", "files": rows}
    return {**body, "digest": canonical_json_digest(body)}


def build_abi_layout_manifest(
    probe_result: dict[str, object],
    *,
    source_commit: str,
    compiler: str,
    compiler_flags: tuple[str, ...],
    llama_header_digest: str,
    mapped_libllama_digest: str,
) -> dict[str, object]:
    required = {
        "batch_size_bytes",
        "n_tokens_offset",
        "n_seq_id_offset",
        "seq_id_offset",
    }
    if set(probe_result) != required:
        raise ProvenanceError("ABI probe result fields are not exact")
    values: dict[str, int] = {}
    for field in sorted(required):
        value = probe_result[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ProvenanceError(f"ABI probe {field} must be non-negative")
        values[field] = value
    if values["batch_size_bytes"] <= 0:
        raise ProvenanceError("ABI batch_size_bytes must be positive")
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ProvenanceError("source_commit must be a lowercase Git SHA-1")
    _require_text(compiler, "compiler")
    if not compiler_flags or any(
        not isinstance(flag, str) or not flag for flag in compiler_flags
    ):
        raise ProvenanceError("compiler_flags must be non-empty strings")
    _require_digest(llama_header_digest, "llama_header_digest")
    _require_digest(mapped_libllama_digest, "mapped_libllama_digest")
    return {
        "schema_version": "1.0",
        "source_commit": source_commit,
        "compiler": compiler,
        "compiler_flags": list(compiler_flags),
        "llama_header_digest": llama_header_digest,
        "mapped_libllama_digest": mapped_libllama_digest,
        "batch_size_bytes": values["batch_size_bytes"],
        "n_tokens_offset": values["n_tokens_offset"],
        "n_seq_id_offset": values["n_seq_id_offset"],
        "seq_id_offset": values["seq_id_offset"],
    }


def parse_attached_diagnostic(line: str) -> tuple[LoadedBpfProgram, ...]:
    if not isinstance(line, str):
        raise ProvenanceError("ATTACHED diagnostic must be text")
    match = _ATTACHED_PATTERN.fullmatch(line.strip())
    if match is None:
        raise ProvenanceError("malformed ATTACHED diagnostic")
    entry = LoadedBpfProgram(
        probe_role="entry",
        program_id=int(match.group(1), 10),
        link_id=int(match.group(2), 10),
        kernel_tag=match.group(3),
    )
    returned = LoadedBpfProgram(
        probe_role="return",
        program_id=int(match.group(4), 10),
        link_id=int(match.group(5), 10),
        kernel_tag=match.group(6),
    )
    if entry.program_id == returned.program_id:
        raise ProvenanceError("entry and return program IDs must be distinct")
    if entry.link_id == returned.link_id:
        raise ProvenanceError("entry and return link IDs must be distinct")
    if entry.kernel_tag == returned.kernel_tag:
        raise ProvenanceError("entry and return kernel tags must be distinct")
    return (entry, returned)


def derive_oracle_instance_id(
    *,
    host_boot_id: str,
    supervisor_pid: int,
    supervisor_start_time_ticks: int,
    supervisor_source_bundle_digest: str,
    supervisor_interpreter_digest: str,
    native_collector_pid: int,
    native_collector_start_time_ticks: int,
    native_collector_digest: str,
    native_dependency_manifest_digest: str,
    bpf_object_digest: str,
    loaded_bpf_program_tags: tuple[dict[str, str], ...],
) -> str:
    if not isinstance(host_boot_id, str) or not host_boot_id:
        raise ProvenanceError("host_boot_id must be non-empty")
    for value, field in (
        (supervisor_pid, "supervisor_pid"),
        (supervisor_start_time_ticks, "supervisor_start_time_ticks"),
        (native_collector_pid, "native_collector_pid"),
        (native_collector_start_time_ticks, "native_collector_start_time_ticks"),
    ):
        _require_positive(value, field)
    for value, field in (
        (supervisor_source_bundle_digest, "supervisor_source_bundle_digest"),
        (supervisor_interpreter_digest, "supervisor_interpreter_digest"),
        (native_collector_digest, "native_collector_digest"),
        (native_dependency_manifest_digest, "native_dependency_manifest_digest"),
        (bpf_object_digest, "bpf_object_digest"),
    ):
        _require_digest(value, field)
    if len(loaded_bpf_program_tags) != 2:
        raise ProvenanceError("loaded BPF tag projection must contain two roles")
    roles = tuple(item.get("probe_role") for item in loaded_bpf_program_tags)
    if roles != ("entry", "return"):
        raise ProvenanceError("loaded BPF tag roles must be entry, return")
    for item in loaded_bpf_program_tags:
        if set(item) != {"probe_role", "kernel_tag"}:
            raise ProvenanceError("loaded BPF tag fields are invalid")
        if re.fullmatch(r"[0-9a-f]{16}", item["kernel_tag"]) is None:
            raise ProvenanceError("loaded BPF kernel tag is invalid")
    if len({item["kernel_tag"] for item in loaded_bpf_program_tags}) != 2:
        raise ProvenanceError("loaded BPF kernel tags must be distinct")
    return canonical_json_digest(
        {
            "host_boot_id": host_boot_id,
            "supervisor_pid": supervisor_pid,
            "supervisor_start_time_ticks": supervisor_start_time_ticks,
            "supervisor_source_bundle_digest": supervisor_source_bundle_digest,
            "supervisor_interpreter_digest": supervisor_interpreter_digest,
            "native_collector_pid": native_collector_pid,
            "native_collector_start_time_ticks": (
                native_collector_start_time_ticks
            ),
            "native_collector_digest": native_collector_digest,
            "native_dependency_manifest_digest": (
                native_dependency_manifest_digest
            ),
            "bpf_object_digest": bpf_object_digest,
            "loaded_bpf_program_tags": list(loaded_bpf_program_tags),
        }
    )


def build_target_projection(
    process: ProcessIdentity,
    mapped_libllama: MappedElfIdentity,
) -> dict[str, object]:
    if (
        mapped_libllama.owner_pid != process.pid
        or mapped_libllama.owner_start_time_ticks != process.start_time_ticks
    ):
        raise ProvenanceError("mapped libllama owner lifetime does not match target")
    return {
        "host_boot_id": process.host_boot_id,
        "server_pid": process.pid,
        "server_start_time_ticks": process.start_time_ticks,
        "live_executable": process.live_executable.to_mapping(),
        "mapped_libllama": mapped_libllama.file_mapping(),
        "llama_decode_symbol": mapped_libllama.llama_decode_symbol.to_mapping(),
    }


def build_expected_target_manifest(
    *,
    run_id: str,
    run_nonce: str,
    runtime_instance_id: str,
    target_process: ProcessIdentity,
    mapped_libllama: MappedElfIdentity,
    model_digest: str,
    server_config_digest: str,
    probe_profile_digest: str,
    abi_layout_manifest_digest: str,
    bpf_object_digest: str,
    native_collector_digest: str,
    native_dependency_manifest_digest: str,
    supervisor_source_bundle_digest: str,
    supervisor_interpreter_digest: str,
) -> dict[str, object]:
    _require_text(run_id, "run_id")
    _require_text(run_nonce, "run_nonce", minimum=16)
    for value, field in (
        (runtime_instance_id, "runtime_instance_id"),
        (model_digest, "model_digest"),
        (server_config_digest, "server_config_digest"),
        (probe_profile_digest, "probe_profile_digest"),
        (abi_layout_manifest_digest, "abi_layout_manifest_digest"),
        (bpf_object_digest, "bpf_object_digest"),
        (native_collector_digest, "native_collector_digest"),
        (native_dependency_manifest_digest, "native_dependency_manifest_digest"),
        (supervisor_source_bundle_digest, "supervisor_source_bundle_digest"),
        (supervisor_interpreter_digest, "supervisor_interpreter_digest"),
    ):
        _require_digest(value, field)
    if runtime_instance_id != derive_runtime_instance_id(target_process):
        raise ProvenanceError("runtime_instance_id derivation mismatch")
    target = build_target_projection(target_process, mapped_libllama)
    return {
        "schema_version": "1.0",
        "run_id": run_id,
        "run_nonce": run_nonce,
        "host_boot_id": target_process.host_boot_id,
        "server_pid": target_process.pid,
        "server_start_time_ticks": target_process.start_time_ticks,
        "runtime_instance_id": runtime_instance_id,
        "live_executable": target["live_executable"],
        "mapped_libllama": target["mapped_libllama"],
        "llama_decode_symbol": target["llama_decode_symbol"],
        "model_digest": model_digest,
        "server_config_digest": server_config_digest,
        "probe_profile_digest": probe_profile_digest,
        "abi_layout_manifest_digest": abi_layout_manifest_digest,
        "bpf_object_digest": bpf_object_digest,
        "native_collector_digest": native_collector_digest,
        "native_dependency_manifest_digest": native_dependency_manifest_digest,
        "supervisor_source_bundle_digest": supervisor_source_bundle_digest,
        "supervisor_interpreter_digest": supervisor_interpreter_digest,
    }


def _mapped_trusted_key(
    identity: MappedElfIdentity,
) -> tuple[object, ...]:
    return (
        identity.device_major,
        identity.device_minor,
        identity.inode,
        identity.digest,
        identity.elf_build_id,
        identity.llama_decode_symbol.name,
        identity.llama_decode_symbol.symbol_value,
        identity.llama_decode_symbol.resolved_file_offset,
    )


def collect_quiescent_snapshot(
    pid: int,
    *,
    expected_process: ProcessIdentity,
    expected_libllama: MappedElfIdentity,
    backend: ProcessInspectionBackend | None = None,
) -> QuiescentSnapshot:
    active_backend = backend or SystemProcessInspectionBackend()
    current_process = read_process_identity(pid, backend=active_backend)
    if (
        current_process.host_boot_id != expected_process.host_boot_id
        or current_process.pid != expected_process.pid
        or current_process.start_time_ticks != expected_process.start_time_ticks
        or trusted_file_key(current_process.live_executable)
        != trusted_file_key(expected_process.live_executable)
    ):
        raise ProvenanceError("target process identity changed before quiescence")
    current_library = inspect_single_mapped_libllama(pid, backend=active_backend)
    if _mapped_trusted_key(current_library) != _mapped_trusted_key(
        expected_libllama
    ):
        raise ProvenanceError("mapped libllama identity changed before quiescence")

    thread_states: list[ThreadState] = []
    for tid in sorted(active_backend.list_tids(pid)):
        start_time_ticks, state = active_backend.read_thread_state(pid, tid)
        if state not in {"T", "t"}:
            raise ProvenanceError(f"target thread {tid} is not stopped")
        thread_states.append(
            ThreadState(
                tid=tid,
                start_time_ticks=start_time_ticks,
                state=state,
            )
        )
    if not any(
        item.tid == expected_process.pid
        and item.start_time_ticks == expected_process.start_time_ticks
        for item in thread_states
    ):
        raise ProvenanceError("quiescent snapshot lacks the target leader")

    source_mappings = tuple(
        mapping for mapping in active_backend.read_maps(pid) if mapping.inode > 0
    )
    if not source_mappings:
        raise ProvenanceError("quiescent file-backed mapping snapshot is empty")
    if not any(
        mapping.identity_key
        == (
            expected_libllama.device_major,
            expected_libllama.device_minor,
            expected_libllama.inode,
        )
        and mapping.permissions[2] == "x"
        for mapping in source_mappings
    ):
        raise ProvenanceError("quiescent snapshot lacks executable mapped libllama")
    mappings = tuple(
        QuiescentMapping(
            ordinal=index,
            length_bytes=mapping.length_bytes,
            permissions=mapping.permissions,
            file_offset=mapping.file_offset,
            device_major=mapping.device_major,
            device_minor=mapping.device_minor,
            inode=mapping.inode,
            path_display=mapping.path_display,
        )
        for index, mapping in enumerate(source_mappings)
    )
    if active_backend.read_start_time_ticks(pid) != expected_process.start_time_ticks:
        raise ProvenanceError("target process identity changed during quiescence")
    return QuiescentSnapshot(
        thread_states=tuple(thread_states),
        mapping_snapshot=mappings,
    )


def collect_retained_identity_snapshot(
    pid: int,
    *,
    expected_process: ProcessIdentity,
    expected_libllama: MappedElfIdentity,
    backend: ProcessInspectionBackend | None = None,
) -> tuple[ProcessIdentity, MappedElfIdentity]:
    """Revalidate a live external target without stopping any of its threads."""

    active_backend = backend or SystemProcessInspectionBackend()
    current_process = read_process_identity(pid, backend=active_backend)
    if (
        current_process.host_boot_id != expected_process.host_boot_id
        or current_process.pid != expected_process.pid
        or current_process.start_time_ticks != expected_process.start_time_ticks
        or trusted_file_key(current_process.live_executable)
        != trusted_file_key(expected_process.live_executable)
    ):
        raise ProvenanceError("target process identity changed before retention")
    current_library = inspect_single_mapped_libllama(pid, backend=active_backend)
    if _mapped_trusted_key(current_library) != _mapped_trusted_key(
        expected_libllama
    ):
        raise ProvenanceError("mapped libllama identity changed before retention")
    # The second start-time read brackets both executable and maps inspection.
    if active_backend.read_start_time_ticks(pid) != expected_process.start_time_ticks:
        raise ProvenanceError("target process identity changed during retained identity")
    return current_process, current_library


def build_observed_attach_manifest(
    *,
    run_id: str,
    run_nonce: str,
    runtime_instance_id: str,
    target_process: ProcessIdentity,
    mapped_libllama: MappedElfIdentity,
    attached_at_monotonic_ns: int,
    expected_target_manifest_digest: str,
    probe_profile_digest: str,
    abi_layout_manifest_digest: str,
    supervisor_process: ProcessIdentity,
    supervisor_source_bundle_digest: str,
    native_collector_process: ProcessIdentity,
    native_dependency_manifest_digest: str,
    bpf_object_digest: str,
    loaded_bpf_programs: tuple[LoadedBpfProgram, ...],
) -> dict[str, object]:
    _require_text(run_id, "run_id")
    _require_text(run_nonce, "run_nonce", minimum=16)
    if len(
        {
            target_process.pid,
            supervisor_process.pid,
            native_collector_process.pid,
        }
    ) != 3:
        raise ProvenanceError(
            "target, supervisor, and native collector must be three distinct processes"
        )
    if target_process.host_boot_id != supervisor_process.host_boot_id or (
        target_process.host_boot_id != native_collector_process.host_boot_id
    ):
        raise ProvenanceError("target and observer processes must share one boot ID")
    if len(loaded_bpf_programs) != 2 or tuple(
        item.probe_role for item in loaded_bpf_programs
    ) != ("entry", "return"):
        raise ProvenanceError("loaded BPF programs must be entry, return")
    for value, field in (
        (runtime_instance_id, "runtime_instance_id"),
        (expected_target_manifest_digest, "expected_target_manifest_digest"),
        (probe_profile_digest, "probe_profile_digest"),
        (abi_layout_manifest_digest, "abi_layout_manifest_digest"),
        (supervisor_source_bundle_digest, "supervisor_source_bundle_digest"),
        (native_dependency_manifest_digest, "native_dependency_manifest_digest"),
        (bpf_object_digest, "bpf_object_digest"),
    ):
        _require_digest(value, field)
    if runtime_instance_id != derive_runtime_instance_id(target_process):
        raise ProvenanceError("runtime_instance_id derivation mismatch")
    if (
        isinstance(attached_at_monotonic_ns, bool)
        or not isinstance(attached_at_monotonic_ns, int)
        or attached_at_monotonic_ns < 0
    ):
        raise ProvenanceError("attached_at_monotonic_ns must be non-negative")
    tags = tuple(item.tag_mapping() for item in loaded_bpf_programs)
    oracle_instance_id = derive_oracle_instance_id(
        host_boot_id=target_process.host_boot_id,
        supervisor_pid=supervisor_process.pid,
        supervisor_start_time_ticks=supervisor_process.start_time_ticks,
        supervisor_source_bundle_digest=supervisor_source_bundle_digest,
        supervisor_interpreter_digest=supervisor_process.live_executable.digest,
        native_collector_pid=native_collector_process.pid,
        native_collector_start_time_ticks=(
            native_collector_process.start_time_ticks
        ),
        native_collector_digest=native_collector_process.live_executable.digest,
        native_dependency_manifest_digest=native_dependency_manifest_digest,
        bpf_object_digest=bpf_object_digest,
        loaded_bpf_program_tags=tags,
    )
    target = build_target_projection(target_process, mapped_libllama)
    return {
        "schema_version": "1.0",
        "run_id": run_id,
        "run_nonce": run_nonce,
        "oracle_instance_id": oracle_instance_id,
        "runtime_instance_id": runtime_instance_id,
        "host_boot_id": target_process.host_boot_id,
        "server_pid": target_process.pid,
        "server_start_time_ticks": target_process.start_time_ticks,
        "attached_at_monotonic_ns": attached_at_monotonic_ns,
        "expected_target_manifest_digest": expected_target_manifest_digest,
        "probe_profile_digest": probe_profile_digest,
        "abi_layout_manifest_digest": abi_layout_manifest_digest,
        "live_executable": target["live_executable"],
        "mapped_libllama": target["mapped_libllama"],
        "llama_decode_symbol": target["llama_decode_symbol"],
        "supervisor_process": {
            "pid": supervisor_process.pid,
            "start_time_ticks": supervisor_process.start_time_ticks,
            "live_executable": supervisor_process.live_executable.to_mapping(),
            "source_bundle_digest": supervisor_source_bundle_digest,
        },
        "native_collector_process": {
            "pid": native_collector_process.pid,
            "start_time_ticks": native_collector_process.start_time_ticks,
            "live_executable": native_collector_process.live_executable.to_mapping(),
            "dependency_manifest_digest": native_dependency_manifest_digest,
        },
        "bpf_object_digest": bpf_object_digest,
        "loaded_bpf_programs": [
            item.to_mapping() for item in loaded_bpf_programs
        ],
        "loaded_bpf_program_tags_digest": canonical_json_digest(list(tags)),
        "attach_succeeded": True,
        "kernel_features": {
            "bpf_ring_buffer": True,
            "uprobe": True,
            "uretprobe": True,
        },
    }


def build_observed_quiescent_manifest(
    *,
    run_id: str,
    run_nonce: str,
    oracle_instance_id: str,
    runtime_instance_id: str,
    target_process: ProcessIdentity,
    mapped_libllama: MappedElfIdentity,
    quiescent_at_monotonic_ns: int,
    observed_attach_manifest_digest: str,
    snapshot: QuiescentSnapshot,
    identity_monitor_alert_count: int,
    final_inflight_count: int,
    anomaly_count: int,
) -> dict[str, object]:
    _require_text(run_id, "run_id")
    _require_text(run_nonce, "run_nonce", minimum=16)
    for value, field in (
        (oracle_instance_id, "oracle_instance_id"),
        (runtime_instance_id, "runtime_instance_id"),
        (observed_attach_manifest_digest, "observed_attach_manifest_digest"),
    ):
        _require_digest(value, field)
    if runtime_instance_id != derive_runtime_instance_id(target_process):
        raise ProvenanceError("runtime_instance_id derivation mismatch")
    if (
        isinstance(quiescent_at_monotonic_ns, bool)
        or not isinstance(quiescent_at_monotonic_ns, int)
        or quiescent_at_monotonic_ns < 0
    ):
        raise ProvenanceError("quiescent_at_monotonic_ns must be non-negative")
    for value, field in (
        (identity_monitor_alert_count, "identity_monitor_alert_count"),
        (final_inflight_count, "final_inflight_count"),
        (anomaly_count, "anomaly_count"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ProvenanceError(f"{field} must be non-negative")
    if not snapshot.thread_states or not snapshot.mapping_snapshot:
        raise ProvenanceError("quiescent snapshot must be non-empty")
    target = build_target_projection(target_process, mapped_libllama)
    return {
        "schema_version": "1.0",
        "run_id": run_id,
        "run_nonce": run_nonce,
        "oracle_instance_id": oracle_instance_id,
        "runtime_instance_id": runtime_instance_id,
        "host_boot_id": target_process.host_boot_id,
        "server_pid": target_process.pid,
        "server_start_time_ticks": target_process.start_time_ticks,
        "quiescent_at_monotonic_ns": quiescent_at_monotonic_ns,
        "observed_attach_manifest_digest": observed_attach_manifest_digest,
        "live_executable": target["live_executable"],
        "mapped_libllama": target["mapped_libllama"],
        "llama_decode_symbol": target["llama_decode_symbol"],
        "all_threads_stopped": True,
        "thread_states": [
            state.to_mapping() for state in snapshot.thread_states
        ],
        "mapping_snapshot": [
            mapping.to_mapping() for mapping in snapshot.mapping_snapshot
        ],
        "identity_monitor_alert_count": identity_monitor_alert_count,
        "final_inflight_count": final_inflight_count,
        "anomaly_count": anomaly_count,
    }
