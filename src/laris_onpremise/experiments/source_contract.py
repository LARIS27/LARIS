from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final


PINNED_LLAMA_CPP_COMMIT: Final[str] = (
    "daf6bc9f2d3516d7bda73d191c6a01388aff09c5"
)
_SOURCE_MARKERS: Final[dict[str, tuple[str, str]]] = {
    "request_slot_parse": (
        "tools/server/server-context.cpp",
        'task.id_slot = json_value(data, "id_slot", -1);',
    ),
    "requested_slot_lookup": (
        "tools/server/server-context.cpp",
        "get_slot_by_id(id_slot)",
    ),
    "response_slot_assignment": (
        "tools/server/server-context.cpp",
        "res->id_slot = slot.id;",
    ),
    "decode_sequence_assignment": (
        "tools/server/server-context.cpp",
        "{ slot.id },",
    ),
    "sequence_state_api": (
        "include/llama.h",
        "llama_state_seq_get_data_ext",
    ),
    "argv_exact_option_lookup": (
        "common/arg.cpp",
        "arg_to_options.find(arg)",
    ),
    "argv_split_value_consumption": (
        "common/arg.cpp",
        "std::string val = argv[++i];",
    ),
}
_SOURCE_PROVENANCE_NAME: Final[str] = "source-provenance.json"
_SOURCE_PROVENANCE_SCHEMA_VERSION: Final[str] = "1.0"
_MAX_SOURCE_PROVENANCE_MANIFEST_BYTES: Final[int] = 4 * 1024 * 1024
_MAX_SOURCE_PROVENANCE_FILES: Final[int] = 8192
_MAX_SOURCE_PROVENANCE_ENTRIES: Final[int] = 32768
_MAX_SOURCE_PROVENANCE_FILE_BYTES: Final[int] = 128 * 1024 * 1024
_MAX_SOURCE_PROVENANCE_TOTAL_BYTES: Final[int] = 1024 * 1024 * 1024
_MAX_SOURCE_PROVENANCE_PATH_BYTES: Final[int] = 1024
_SHA256_DIGEST_LENGTH: Final[int] = len("sha256:") + 64


class SourceContractError(RuntimeError):
    """Raised when source or live process provenance cannot be collected."""


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def digest_json(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _canonical_json_line(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise SourceContractError(
            "source provenance manifest is not canonical JSON"
        ) from error


def _validated_source_root(source_root: Path) -> Path:
    try:
        metadata = source_root.lstat()
    except OSError as error:
        raise SourceContractError("source provenance root is unavailable") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise SourceContractError(
            "source provenance root must be a non-symlink directory"
        )
    try:
        return source_root.resolve(strict=True)
    except OSError as error:
        raise SourceContractError("source provenance root is unavailable") from error


def _safe_source_relative_path(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise SourceContractError("source provenance path is invalid")
    try:
        encoded = value.encode("utf-8", errors="strict")
    except UnicodeError as error:
        raise SourceContractError("source provenance path is invalid") from error
    if (
        len(encoded) > _MAX_SOURCE_PROVENANCE_PATH_BYTES
        or "\\" in value
        or "\x00" in value
        or ":" in value
    ):
        raise SourceContractError("source provenance path is unsafe")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or relative.as_posix() != value
        or len(relative.parts) > 64
        or any(part in {"", ".", ".."} for part in relative.parts)
        or value == _SOURCE_PROVENANCE_NAME
    ):
        raise SourceContractError("source provenance path is unsafe")
    return value


@dataclass(frozen=True)
class _SourceFileObservation:
    size_bytes: int
    digest: str
    payload: bytes | None


def _read_source_file_once(
    source_root: Path,
    relative_name: str,
    *,
    maximum_bytes: int,
    capture_payload: bool,
    allow_manifest_name: bool = False,
) -> _SourceFileObservation:
    if allow_manifest_name and relative_name == _SOURCE_PROVENANCE_NAME:
        normalized = relative_name
    else:
        normalized = _safe_source_relative_path(relative_name)
    relative = PurePosixPath(normalized)
    current = source_root
    try:
        for part in relative.parts[:-1]:
            current = current / part
            metadata = current.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise SourceContractError(
                    "source provenance path parent is unsafe"
                )
        path = source_root.joinpath(*relative.parts)
        before = path.lstat()
    except SourceContractError:
        raise
    except OSError as error:
        raise SourceContractError(
            f"source provenance file is missing: {normalized}"
        ) from error
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise SourceContractError(
            f"source provenance file is not regular: {normalized}"
        )
    if before.st_size < 0 or before.st_size > maximum_bytes:
        raise SourceContractError(
            f"source provenance file exceeds its bound: {normalized}"
        )

    flags = os.O_RDONLY
    flags |= (
        getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_BINARY", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise SourceContractError(
            f"source provenance file cannot be opened: {normalized}"
        ) from error
    digest = hashlib.sha256()
    captured: list[bytes] | None = [] if capture_payload else None
    observed_size = 0
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
        ):
            raise SourceContractError(
                f"source provenance file changed during open: {normalized}"
            )
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, maximum_bytes + 1))
            if not chunk:
                break
            observed_size += len(chunk)
            if observed_size > maximum_bytes:
                raise SourceContractError(
                    f"source provenance file exceeds its bound: {normalized}"
                )
            digest.update(chunk)
            if captured is not None:
                captured.append(chunk)
        after = os.fstat(descriptor)
        if (
            (after.st_dev, after.st_ino, after.st_size)
            != (opened.st_dev, opened.st_ino, opened.st_size)
            or observed_size != after.st_size
            or getattr(after, "st_mtime_ns", None)
            != getattr(opened, "st_mtime_ns", None)
        ):
            raise SourceContractError(
                f"source provenance file changed while reading: {normalized}"
            )
    finally:
        os.close(descriptor)
    return _SourceFileObservation(
        size_bytes=observed_size,
        digest="sha256:" + digest.hexdigest(),
        payload=b"".join(captured) if captured is not None else None,
    )


def _inventory_source_files(source_root: Path) -> tuple[str, ...]:
    files: list[str] = []
    pending: list[tuple[Path, PurePosixPath | None]] = [(source_root, None)]
    entry_count = 0
    while pending:
        directory, relative_parent = pending.pop()
        try:
            entries = tuple(os.scandir(directory))
        except OSError as error:
            raise SourceContractError(
                "source provenance inventory cannot be read"
            ) from error
        for entry in entries:
            entry_count += 1
            if entry_count > _MAX_SOURCE_PROVENANCE_ENTRIES:
                raise SourceContractError(
                    "source provenance inventory exceeds its entry bound"
                )
            relative = (
                PurePosixPath(entry.name)
                if relative_parent is None
                else relative_parent / entry.name
            )
            relative_name = relative.as_posix()
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError as error:
                raise SourceContractError(
                    f"source provenance inventory is unstable: {relative_name}"
                ) from error
            if stat.S_ISLNK(metadata.st_mode):
                raise SourceContractError(
                    f"source provenance inventory contains a symlink: {relative_name}"
                )
            if stat.S_ISDIR(metadata.st_mode):
                pending.append((Path(entry.path), relative))
            elif stat.S_ISREG(metadata.st_mode):
                files.append(relative_name)
                if len(files) > _MAX_SOURCE_PROVENANCE_FILES + 1:
                    raise SourceContractError(
                        "source provenance inventory exceeds its file bound"
                    )
            else:
                raise SourceContractError(
                    f"source provenance inventory contains a special file: {relative_name}"
                )
    return tuple(sorted(files))


def _required_source_files() -> frozenset[str]:
    return frozenset(
        {relative_name for relative_name, _needle in _SOURCE_MARKERS.values()}
        | {"include/llama.h"}
    )


def _inspect_verified_source_mapping(
    payloads: dict[str, bytes],
) -> tuple[str, bool, tuple[str, ...]]:
    observations: dict[str, dict[str, object]] = {}
    missing: list[str] = []
    for marker_name, (relative_name, needle) in _SOURCE_MARKERS.items():
        try:
            source_text = payloads[relative_name].decode("utf-8", errors="strict")
        except (KeyError, UnicodeError):
            source_text = ""
        present = needle in source_text
        observations[marker_name] = {
            "file": relative_name,
            "needle_digest": _sha256_bytes(needle.encode("utf-8")),
            "present": present,
        }
        if not present:
            missing.append(marker_name)
    return digest_json(observations), not missing, tuple(missing)


def _decode_source_provenance(raw: bytes) -> dict[str, object]:
    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise SourceContractError(
                    "source provenance manifest contains duplicate fields"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=unique_object,
        )
    except SourceContractError:
        raise
    except (UnicodeError, json.JSONDecodeError) as error:
        raise SourceContractError(
            "source provenance manifest is invalid JSON"
        ) from error
    if not isinstance(value, dict):
        raise SourceContractError("source provenance manifest must be an object")
    if raw != _canonical_json_line(value):
        raise SourceContractError("source provenance manifest is not canonical")
    return value


@dataclass(frozen=True)
class _VerifiedSourceProvenance:
    observed_commit: str
    manifest_digest: str
    source_mapping_digest: str
    source_mapping_verified: bool
    missing_source_markers: tuple[str, ...]


def _verify_source_provenance(
    source_root: Path,
    *,
    expected_commit: str,
) -> _VerifiedSourceProvenance:
    root = _validated_source_root(source_root)
    manifest_observation = _read_source_file_once(
        root,
        _SOURCE_PROVENANCE_NAME,
        maximum_bytes=_MAX_SOURCE_PROVENANCE_MANIFEST_BYTES,
        capture_payload=True,
        allow_manifest_name=True,
    )
    assert manifest_observation.payload is not None
    document = _decode_source_provenance(manifest_observation.payload)
    if set(document) != {
        "schema_version",
        "commit",
        "files",
        "manifest_digest",
    }:
        raise SourceContractError("source provenance manifest schema is invalid")
    commit = document.get("commit")
    if (
        document.get("schema_version") != _SOURCE_PROVENANCE_SCHEMA_VERSION
        or not isinstance(commit, str)
        or commit != expected_commit
        or commit != PINNED_LLAMA_CPP_COMMIT
    ):
        raise SourceContractError("source provenance commit is not the pinned commit")
    manifest_digest = document.get("manifest_digest")
    if (
        not isinstance(manifest_digest, str)
        or len(manifest_digest) != _SHA256_DIGEST_LENGTH
        or not manifest_digest.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in manifest_digest[7:])
    ):
        raise SourceContractError("source provenance manifest digest is invalid")
    unsigned = dict(document)
    unsigned.pop("manifest_digest")
    if digest_json(unsigned) != manifest_digest:
        raise SourceContractError("source provenance manifest digest mismatch")

    raw_files = document.get("files")
    if (
        not isinstance(raw_files, list)
        or not raw_files
        or len(raw_files) > _MAX_SOURCE_PROVENANCE_FILES
    ):
        raise SourceContractError("source provenance file list is invalid")
    records: list[tuple[str, int, str]] = []
    for raw_record in raw_files:
        if not isinstance(raw_record, dict) or set(raw_record) != {
            "path",
            "size_bytes",
            "sha256",
        }:
            raise SourceContractError("source provenance file record schema is invalid")
        relative_name = _safe_source_relative_path(raw_record.get("path"))
        size_bytes = raw_record.get("size_bytes")
        digest = raw_record.get("sha256")
        if (
            isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes < 0
            or size_bytes > _MAX_SOURCE_PROVENANCE_FILE_BYTES
        ):
            raise SourceContractError(
                f"source provenance file size is invalid: {relative_name}"
            )
        if (
            not isinstance(digest, str)
            or len(digest) != _SHA256_DIGEST_LENGTH
            or not digest.startswith("sha256:")
            or any(character not in "0123456789abcdef" for character in digest[7:])
        ):
            raise SourceContractError(
                f"source provenance file digest is invalid: {relative_name}"
            )
        records.append((relative_name, size_bytes, digest))
    paths = [record[0] for record in records]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise SourceContractError(
            "source provenance file list must be sorted and unique"
        )
    required = _required_source_files()
    if not required.issubset(paths):
        raise SourceContractError("source provenance required files are missing")
    declared_total = sum(record[1] for record in records)
    if declared_total > _MAX_SOURCE_PROVENANCE_TOTAL_BYTES:
        raise SourceContractError("source provenance total size exceeds its bound")

    actual_files = set(_inventory_source_files(root))
    actual_files.discard(_SOURCE_PROVENANCE_NAME)
    listed_files = set(paths)
    if listed_files - actual_files:
        raise SourceContractError("source provenance has missing listed files")
    if actual_files - listed_files:
        raise SourceContractError("source provenance has extra files")

    payloads: dict[str, bytes] = {}
    observed_total = 0
    for relative_name, size_bytes, expected_digest in records:
        observation = _read_source_file_once(
            root,
            relative_name,
            maximum_bytes=_MAX_SOURCE_PROVENANCE_FILE_BYTES,
            capture_payload=relative_name in required,
        )
        observed_total += observation.size_bytes
        if observed_total > _MAX_SOURCE_PROVENANCE_TOTAL_BYTES:
            raise SourceContractError("source provenance total size exceeds its bound")
        if observation.size_bytes != size_bytes:
            raise SourceContractError(
                f"source provenance file size mismatch: {relative_name}"
            )
        if observation.digest != expected_digest:
            raise SourceContractError(
                f"source provenance file digest mismatch: {relative_name}"
            )
        if observation.payload is not None:
            payloads[relative_name] = observation.payload
    mapping_digest, mapping_verified, missing_markers = (
        _inspect_verified_source_mapping(payloads)
    )
    return _VerifiedSourceProvenance(
        observed_commit=commit,
        manifest_digest=manifest_digest,
        source_mapping_digest=mapping_digest,
        source_mapping_verified=mapping_verified,
        missing_source_markers=missing_markers,
    )


def build_source_provenance(
    source_root: Path,
    included_files: Iterable[str | Path],
    expected_commit: str,
) -> Path:
    """Build one canonical provenance receipt for a closed installed source."""

    if expected_commit != PINNED_LLAMA_CPP_COMMIT:
        raise SourceContractError("source provenance commit is not the pinned commit")
    root = _validated_source_root(source_root)
    manifest_path = root / _SOURCE_PROVENANCE_NAME
    if manifest_path.exists() or manifest_path.is_symlink():
        raise SourceContractError("source provenance manifest already exists")
    normalized_paths: list[str] = []
    for included in included_files:
        raw = included.as_posix() if isinstance(included, Path) else included
        normalized_paths.append(_safe_source_relative_path(raw))
    if (
        not normalized_paths
        or len(normalized_paths) > _MAX_SOURCE_PROVENANCE_FILES
        or len(normalized_paths) != len(set(normalized_paths))
    ):
        raise SourceContractError(
            "source provenance included file list must be unique and bounded"
        )
    paths = sorted(normalized_paths)
    required = _required_source_files()
    if not required.issubset(paths):
        raise SourceContractError("source provenance required files are missing")
    actual_files = set(_inventory_source_files(root))
    if set(paths) - actual_files:
        raise SourceContractError("source provenance has missing listed files")
    if actual_files - set(paths):
        raise SourceContractError("source provenance has extra files")

    records: list[dict[str, object]] = []
    total_bytes = 0
    for relative_name in paths:
        observation = _read_source_file_once(
            root,
            relative_name,
            maximum_bytes=_MAX_SOURCE_PROVENANCE_FILE_BYTES,
            capture_payload=False,
        )
        total_bytes += observation.size_bytes
        if total_bytes > _MAX_SOURCE_PROVENANCE_TOTAL_BYTES:
            raise SourceContractError("source provenance total size exceeds its bound")
        records.append(
            {
                "path": relative_name,
                "size_bytes": observation.size_bytes,
                "sha256": observation.digest,
            }
        )
    unsigned: dict[str, object] = {
        "schema_version": _SOURCE_PROVENANCE_SCHEMA_VERSION,
        "commit": expected_commit,
        "files": records,
    }
    document = dict(unsigned)
    document["manifest_digest"] = digest_json(unsigned)
    payload = _canonical_json_line(document)
    if len(payload) > _MAX_SOURCE_PROVENANCE_MANIFEST_BYTES:
        raise SourceContractError("source provenance manifest exceeds its bound")

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= (
        getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_BINARY", 0)
    )
    try:
        descriptor = os.open(manifest_path, flags, 0o600)
        try:
            offset = 0
            while offset < len(payload):
                written = os.write(descriptor, payload[offset:])
                if written <= 0:
                    raise SourceContractError(
                        "source provenance manifest write made no progress"
                    )
                offset += written
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except SourceContractError:
        raise
    except OSError as error:
        raise SourceContractError(
            "source provenance manifest cannot be published"
        ) from error
    return manifest_path


def derive_runtime_instance_id(
    *,
    host_boot_id: str,
    server_pid: int,
    server_start_time_ticks: int,
    live_executable_digest: str,
) -> str:
    """Derive a stable identity for one live runtime process lifetime."""
    return digest_json(
        {
            "host_boot_id": host_boot_id,
            "server_pid": server_pid,
            "server_start_time_ticks": server_start_time_ticks,
            "live_executable_digest": live_executable_digest,
        }
    )


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
    """Derive the identity of one supervisor/native/BPF observer instance."""
    return digest_json(
        {
            "host_boot_id": host_boot_id,
            "supervisor_pid": supervisor_pid,
            "supervisor_start_time_ticks": supervisor_start_time_ticks,
            "supervisor_source_bundle_digest": (
                supervisor_source_bundle_digest
            ),
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


def select_mapped_libllama(paths: tuple[Path, ...]) -> Path:
    """Select the only mapped libllama object or fail closed."""
    matches = tuple(
        path
        for path in paths
        if path.name == "libllama.so" or path.name.startswith("libllama.so.")
    )
    if len(matches) != 1:
        raise ValueError("exactly one mapped libllama object is required")
    return matches[0]


@dataclass(frozen=True)
class SourceContractEvidence:
    schema_version: str
    run_id: str
    run_nonce: str
    profile_id: str
    config_digest: str
    expected_commit: str
    observed_commit: str
    source_clean: bool
    source_status_digest: str
    compiler_id: str
    compiler_version: str
    build_command: tuple[str, ...]
    build_command_digest: str
    binary_digest: str
    host_boot_id: str
    server_pid: int
    server_start_time_ticks: int
    runtime_instance_id: str
    live_executable_digest: str
    live_executable_matches_configured: bool
    link_mode: str
    libllama_digest: str | None
    dependency_list_digest: str
    model_digest: str
    runtime_argv_digest: str
    slot_count: int
    slot_save_enabled: bool
    cache_idle_slots: bool
    context_shift_enabled: bool
    listen_endpoint: str
    socket_owner_verified: bool
    source_mapping_digest: str
    source_mapping_verified: bool
    missing_source_markers: tuple[str, ...]
    evidence_method: str
    runtime_mode: str = "MANAGED_LEGACY"
    runtime_ownership: str = "laris"
    approved_runtime_profile_digest: str | None = None

    def to_mapping(self) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "run_nonce": self.run_nonce,
            "profile_id": self.profile_id,
            "config_digest": self.config_digest,
            "expected_commit": self.expected_commit,
            "observed_commit": self.observed_commit,
            "source_clean": self.source_clean,
            "source_status_digest": self.source_status_digest,
            "compiler_id": self.compiler_id,
            "compiler_version": self.compiler_version,
            "build_command": list(self.build_command),
            "build_command_digest": self.build_command_digest,
            "binary_digest": self.binary_digest,
            "host_boot_id": self.host_boot_id,
            "server_pid": self.server_pid,
            "server_start_time_ticks": self.server_start_time_ticks,
            "runtime_instance_id": self.runtime_instance_id,
            "live_executable_digest": self.live_executable_digest,
            "live_executable_matches_configured": (
                self.live_executable_matches_configured
            ),
            "link_mode": self.link_mode,
            "libllama_digest": self.libllama_digest,
            "dependency_list_digest": self.dependency_list_digest,
            "model_digest": self.model_digest,
            "runtime_argv_digest": self.runtime_argv_digest,
            "slot_count": self.slot_count,
            "slot_save_enabled": self.slot_save_enabled,
            "cache_idle_slots": self.cache_idle_slots,
            "context_shift_enabled": self.context_shift_enabled,
            "listen_endpoint": self.listen_endpoint,
            "socket_owner_verified": self.socket_owner_verified,
            "source_mapping_digest": self.source_mapping_digest,
            "source_mapping_verified": self.source_mapping_verified,
            "missing_source_markers": list(self.missing_source_markers),
            "evidence_method": self.evidence_method,
        }
        if self.schema_version == "1.1":
            value.update(
                {
                    "runtime_mode": self.runtime_mode,
                    "runtime_ownership": self.runtime_ownership,
                    "approved_runtime_profile_digest": (
                        self.approved_runtime_profile_digest
                    ),
                }
            )
        return value


def build_attached_source_contract(
    *,
    run_id: str,
    run_nonce: str,
    profile_id: str,
    approved_runtime_profile_digest: str,
    source_commit: str,
    compiler_id: str,
    compiler_version: str,
    executable_digest: str,
    host_boot_id: str,
    server_pid: int,
    server_start_time_ticks: int,
    runtime_instance_id: str,
    mapped_libllama_digest: str,
    dependency_list_digest: str,
    model_digest: str,
    runtime_argv_digest: str,
    slot_count: int,
    cache_idle_slots: bool,
    context_shift_enabled: bool,
    listen_endpoint: str,
) -> SourceContractEvidence:
    """Describe an inspected external runtime without claiming checkout custody."""

    return SourceContractEvidence(
        schema_version="1.1",
        run_id=run_id,
        run_nonce=run_nonce,
        profile_id=profile_id,
        config_digest=approved_runtime_profile_digest,
        expected_commit=source_commit,
        observed_commit=source_commit,
        source_clean=True,
        source_status_digest=approved_runtime_profile_digest,
        compiler_id=compiler_id,
        compiler_version=compiler_version,
        build_command=(),
        build_command_digest=approved_runtime_profile_digest,
        binary_digest=executable_digest,
        host_boot_id=host_boot_id,
        server_pid=server_pid,
        server_start_time_ticks=server_start_time_ticks,
        runtime_instance_id=runtime_instance_id,
        live_executable_digest=executable_digest,
        live_executable_matches_configured=True,
        link_mode="dynamic",
        libllama_digest=mapped_libllama_digest,
        dependency_list_digest=dependency_list_digest,
        model_digest=model_digest,
        runtime_argv_digest=runtime_argv_digest,
        slot_count=slot_count,
        slot_save_enabled=False,
        cache_idle_slots=cache_idle_slots,
        context_shift_enabled=context_shift_enabled,
        listen_endpoint=listen_endpoint,
        socket_owner_verified=True,
        source_mapping_digest=approved_runtime_profile_digest,
        source_mapping_verified=True,
        missing_source_markers=(),
        evidence_method="approved-runtime-profile+procfs+sha256",
        runtime_mode="ATTACHED",
        runtime_ownership="external",
        approved_runtime_profile_digest=approved_runtime_profile_digest,
    )


def inspect_source_mapping(source_dir: Path) -> tuple[str, bool, tuple[str, ...]]:
    observations: dict[str, dict[str, object]] = {}
    missing: list[str] = []
    for marker_name, (relative_name, needle) in _SOURCE_MARKERS.items():
        path = source_dir / Path(relative_name)
        present = path.is_file() and needle in path.read_text(
            encoding="utf-8", errors="strict"
        )
        observations[marker_name] = {
            "file": relative_name,
            "needle_digest": _sha256_bytes(needle.encode("utf-8")),
            "present": present,
        }
        if not present:
            missing.append(marker_name)
    return digest_json(observations), not missing, tuple(missing)


def _git(source_dir: Path, *args: str) -> str:
    trusted_source = str(source_dir.resolve(strict=True))
    result = subprocess.run(
        (
            "git",
            "-c",
            f"safe.directory={trusted_source}",
            "-C",
            trusted_source,
            *args,
        ),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise SourceContractError("git source provenance command failed")
    return result.stdout.strip()


def _read_proc_start_time_ticks(pid: int) -> int:
    raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    close_paren = raw.rfind(")")
    if close_paren < 0:
        raise SourceContractError("invalid proc stat record")
    fields_after_command = raw[close_paren + 2 :].split()
    try:
        return int(fields_after_command[19])
    except (IndexError, ValueError) as error:
        raise SourceContractError("proc stat is missing process start time") from error


def _read_proc_cmdline(pid: int) -> tuple[str, ...]:
    raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    return tuple(
        item.decode("utf-8", errors="surrogateescape")
        for item in raw.split(b"\0")
        if item
    )


def _mapped_files(pid: int) -> tuple[Path, ...]:
    paths: set[Path] = set()
    for line in Path(f"/proc/{pid}/maps").read_text(
        encoding="utf-8", errors="replace"
    ).splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) == 6 and parts[5].startswith("/"):
            candidate = Path(parts[5].removesuffix(" (deleted)"))
            if candidate.is_file():
                paths.add(candidate)
    return tuple(sorted(paths, key=lambda item: str(item)))


def _socket_owned_by_process(pid: int, endpoint: str) -> bool:
    try:
        host, raw_port = endpoint.rsplit(":", maxsplit=1)
        if host not in {"127.0.0.1", "0.0.0.0"}:
            return False
        port = int(raw_port)
        candidate_inodes: set[str] = set()
        for line in Path("/proc/net/tcp").read_text(encoding="ascii").splitlines()[1:]:
            fields = line.split()
            local_address, state, inode = fields[1], fields[3], fields[9]
            _, hex_port = local_address.split(":", maxsplit=1)
            if state == "0A" and int(hex_port, 16) == port:
                candidate_inodes.add(inode)
        if not candidate_inodes:
            return False
        for descriptor in Path(f"/proc/{pid}/fd").iterdir():
            try:
                target = os.readlink(descriptor)
            except OSError:
                continue
            if target.startswith("socket:[") and target[8:-1] in candidate_inodes:
                return True
        return False
    except (OSError, ValueError, IndexError):
        return False


def collect_source_contract(
    *,
    run_id: str,
    run_nonce: str,
    profile_id: str,
    config: object,
    source_dir: Path,
    server_binary: Path,
    libllama_path: Path | None,
    model_path: Path,
    server_pid: int,
    expected_commit: str,
    compiler_id: str,
    compiler_version: str,
    build_command: tuple[str, ...],
    slot_count: int,
    slot_save_enabled: bool,
    cache_idle_slots: bool,
    context_shift_enabled: bool,
    listen_endpoint: str,
) -> SourceContractEvidence:
    if os.name != "posix" or not Path("/proc").is_dir():
        raise SourceContractError("live process binding requires Linux procfs")
    source_dir = source_dir.resolve(strict=True)
    server_binary = server_binary.resolve(strict=True)
    model_path = model_path.resolve(strict=True)
    git_metadata = source_dir / ".git"
    if git_metadata.exists() or git_metadata.is_symlink():
        observed_commit = _git(source_dir, "rev-parse", "HEAD")
        source_status = _git(
            source_dir,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--ignore-submodules=none",
        )
        source_clean = not source_status
        source_status_digest = _sha256_bytes(source_status.encode("utf-8"))
        mapping_digest, mapping_verified, missing_markers = (
            inspect_source_mapping(source_dir)
        )
        evidence_method = "git+procfs+sha256+static-source-contract"
    else:
        provenance = _verify_source_provenance(
            source_dir,
            expected_commit=expected_commit,
        )
        observed_commit = provenance.observed_commit
        source_clean = True
        source_status_digest = provenance.manifest_digest
        mapping_digest = provenance.source_mapping_digest
        mapping_verified = provenance.source_mapping_verified
        missing_markers = provenance.missing_source_markers
        evidence_method = "manifest+procfs+sha256+static-source-contract"

    live_executable = Path(f"/proc/{server_pid}/exe").resolve(strict=True)
    binary_digest = sha256_file(server_binary)
    live_executable_digest = sha256_file(live_executable)
    host_boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(
        encoding="ascii"
    ).strip()
    start_ticks = _read_proc_start_time_ticks(server_pid)
    runtime_argv = _read_proc_cmdline(server_pid)
    mapped_files = _mapped_files(server_pid)
    dependency_manifest = [
        {"path": str(path), "digest": sha256_file(path)}
        for path in mapped_files
    ]

    link_mode = "dynamic"
    mapped_libllama = select_mapped_libllama(mapped_files)
    live_library_digest = sha256_file(mapped_libllama)
    if libllama_path is not None:
        configured_library_digest = sha256_file(
            libllama_path.resolve(strict=True)
        )
        libllama_digest = (
            live_library_digest
            if live_library_digest == configured_library_digest
            else None
        )
    else:
        libllama_digest = live_library_digest

    config_digest = digest_json(config)
    runtime_instance_id = derive_runtime_instance_id(
        host_boot_id=host_boot_id,
        server_pid=server_pid,
        server_start_time_ticks=start_ticks,
        live_executable_digest=live_executable_digest,
    )
    return SourceContractEvidence(
        schema_version="1.0",
        run_id=run_id,
        run_nonce=run_nonce,
        profile_id=profile_id,
        config_digest=config_digest,
        expected_commit=expected_commit,
        observed_commit=observed_commit,
        source_clean=source_clean,
        source_status_digest=source_status_digest,
        compiler_id=compiler_id,
        compiler_version=compiler_version,
        build_command=build_command,
        build_command_digest=digest_json(build_command),
        binary_digest=binary_digest,
        host_boot_id=host_boot_id,
        server_pid=server_pid,
        server_start_time_ticks=start_ticks,
        runtime_instance_id=runtime_instance_id,
        live_executable_digest=live_executable_digest,
        live_executable_matches_configured=(
            binary_digest == live_executable_digest
        ),
        link_mode=link_mode,
        libllama_digest=libllama_digest,
        dependency_list_digest=digest_json(dependency_manifest),
        model_digest=sha256_file(model_path),
        runtime_argv_digest=digest_json(runtime_argv),
        slot_count=slot_count,
        slot_save_enabled=slot_save_enabled,
        cache_idle_slots=cache_idle_slots,
        context_shift_enabled=context_shift_enabled,
        listen_endpoint=listen_endpoint,
        socket_owner_verified=_socket_owned_by_process(
            server_pid, listen_endpoint
        ),
        source_mapping_digest=mapping_digest,
        source_mapping_verified=mapping_verified,
        missing_source_markers=missing_markers,
        evidence_method=evidence_method,
    )
