from __future__ import annotations

import json
import re
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from .profile import (
    ObserverProfileError,
    RuntimeObserverProfile,
    derive_observer_artifact_profile_digest,
    derive_runtime_build_profile_digest,
)
from laris_onpremise.experiments.source_contract import (
    derive_oracle_instance_id,
    derive_runtime_instance_id,
    digest_json,
)
from .lifecycle import (
    LifecycleError,
    canonical_lifecycle_bytes,
    load_lifecycle_marker,
)


_SHA256_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"sha256:[0-9a-f]{64}\Z"
)
_GIT_COMMIT_PATTERN: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{40}\Z")
_BPF_TAG_PATTERN: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{16}\Z")
_ELF_BUILD_ID_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?:[0-9a-f]{2}){8,64}\Z"
)
_RAW_ADDRESS_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?i)(?:^|[^0-9a-f])0x[0-9a-f]{8,16}(?:$|[^0-9a-f])"
)
_RUN_NONCE_MIN_LENGTH: Final[int] = 16
_SEAL_KEYS: Final[frozenset[str]] = frozenset({"file", "digest"})
_BPF_TAG_KEYS: Final[frozenset[str]] = frozenset(
    {"probe_role", "kernel_tag"}
)
_FILE_IDENTITY_KEYS: Final[frozenset[str]] = frozenset(
    {
        "device_major",
        "device_minor",
        "inode",
        "path_display",
        "digest",
    }
)
_MAPPED_LIBRARY_IDENTITY_KEYS: Final[frozenset[str]] = frozenset(
    {*_FILE_IDENTITY_KEYS, "elf_build_id"}
)
_SYMBOL_KEYS: Final[frozenset[str]] = frozenset(
    {"name", "symbol_value", "resolved_file_offset"}
)
_SUPERVISOR_PROCESS_KEYS: Final[frozenset[str]] = frozenset(
    {"pid", "start_time_ticks", "live_executable", "source_bundle_digest"}
)
_NATIVE_PROCESS_KEYS: Final[frozenset[str]] = frozenset(
    {
        "pid",
        "start_time_ticks",
        "live_executable",
        "dependency_manifest_digest",
    }
)
_LOADED_BPF_PROGRAM_KEYS: Final[frozenset[str]] = frozenset(
    {"probe_role", "program_id", "link_id", "kernel_tag"}
)
_KERNEL_FEATURE_KEYS: Final[frozenset[str]] = frozenset(
    {"bpf_ring_buffer", "uprobe", "uretprobe"}
)
_THREAD_STATE_KEYS: Final[frozenset[str]] = frozenset(
    {"tid", "start_time_ticks", "state"}
)
_MAPPING_SNAPSHOT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "ordinal",
        "length_bytes",
        "permissions",
        "file_offset",
        "device_major",
        "device_minor",
        "inode",
        "path_display",
    }
)
_ARTIFACT_FILES: Final[dict[str, str]] = {
    "probe_profile": "x86_64_cpu_llama_decode.json",
    "abi_layout_manifest": "abi_layout_manifest.json",
    "expected_target_manifest": "expected_target_manifest.json",
    "observed_attach_manifest": "observed_attach_manifest.json",
    "observed_quiescent_manifest": "observed_quiescent_manifest.json",
    "target_terminated": "TARGET_TERMINATED.json",
}
_ABI_LAYOUT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "source_commit",
        "compiler",
        "compiler_flags",
        "llama_header_digest",
        "mapped_libllama_digest",
        "batch_size_bytes",
        "n_tokens_offset",
        "n_seq_id_offset",
        "seq_id_offset",
    }
)
_EXPECTED_TARGET_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "run_id",
        "run_nonce",
        "host_boot_id",
        "server_pid",
        "server_start_time_ticks",
        "runtime_instance_id",
        "live_executable",
        "mapped_libllama",
        "llama_decode_symbol",
        "model_digest",
        "server_config_digest",
        "probe_profile_digest",
        "abi_layout_manifest_digest",
        "bpf_object_digest",
        "native_collector_digest",
        "native_dependency_manifest_digest",
        "supervisor_source_bundle_digest",
        "supervisor_interpreter_digest",
    }
)
_OBSERVED_ATTACH_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "run_id",
        "run_nonce",
        "oracle_instance_id",
        "runtime_instance_id",
        "host_boot_id",
        "server_pid",
        "server_start_time_ticks",
        "attached_at_monotonic_ns",
        "expected_target_manifest_digest",
        "probe_profile_digest",
        "abi_layout_manifest_digest",
        "live_executable",
        "mapped_libllama",
        "llama_decode_symbol",
        "supervisor_process",
        "native_collector_process",
        "bpf_object_digest",
        "loaded_bpf_programs",
        "loaded_bpf_program_tags_digest",
        "attach_succeeded",
        "kernel_features",
    }
)
_OBSERVED_QUIESCENT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "run_id",
        "run_nonce",
        "oracle_instance_id",
        "runtime_instance_id",
        "host_boot_id",
        "server_pid",
        "server_start_time_ticks",
        "quiescent_at_monotonic_ns",
        "observed_attach_manifest_digest",
        "live_executable",
        "mapped_libllama",
        "llama_decode_symbol",
        "all_threads_stopped",
        "thread_states",
        "mapping_snapshot",
        "identity_monitor_alert_count",
        "final_inflight_count",
        "anomaly_count",
    }
)
_TARGET_TERMINATED_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "run_id",
        "run_nonce",
        "oracle_instance_id",
        "runtime_instance_id",
        "host_boot_id",
        "server_pid",
        "server_start_time_ticks",
        "pidfd_bound",
        "termination_signal",
        "wait_status",
        "observed_quiescent_manifest_digest",
    }
)
_TARGET_RETAINED_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "run_id",
        "run_nonce",
        "oracle_instance_id",
        "runtime_instance_id",
        "host_boot_id",
        "server_pid",
        "server_start_time_ticks",
        "runtime_mode",
        "runtime_ownership",
        "observer_detached_at_monotonic_ns",
        "identity_verified_at_monotonic_ns",
        "live_executable_digest",
        "mapped_libllama_digest",
        "observer_detached_digest",
    }
)
_BINDING_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "run_id",
        "run_nonce",
        "oracle_instance_id",
        "runtime_instance_id",
        "host_boot_id",
        "server_pid",
        "server_start_time_ticks",
        "supervisor_pid",
        "supervisor_start_time_ticks",
        "native_collector_pid",
        "native_collector_start_time_ticks",
        "configured_context_ref",
        "clock_id",
        *_ARTIFACT_FILES,
        "source_commit",
        "server_binary_digest",
        "mapped_libllama_digest",
        "model_digest",
        "server_config_digest",
        "build_command_digest",
        "bpf_object_digest",
        "native_collector_digest",
        "native_dependency_manifest_digest",
        "supervisor_source_bundle_digest",
        "supervisor_interpreter_digest",
        "loaded_bpf_program_tags",
        "loaded_bpf_program_tags_digest",
        "wire_protocol_version",
        "runtime_oracle_schema_version",
        "qualified_toolchain_profile",
        "runtime_build_profile_digest",
        "observer_artifact_profile_digest",
    }
)
_BINDING_V11_KEYS: Final[frozenset[str]] = frozenset(
    {
        *_BINDING_KEYS,
        "runtime_mode",
        "runtime_ownership",
        "observer_detached",
        "target_retained",
    }
)


class ObserverBindingError(ValueError):
    """Raised when observer evaluation binding metadata is malformed."""


@dataclass(frozen=True)
class LoadedBpfProgramTag:
    probe_role: str
    kernel_tag: str

    @classmethod
    def from_mapping(
        cls,
        value: object,
        *,
        field: str,
    ) -> LoadedBpfProgramTag:
        if not isinstance(value, dict):
            raise ObserverBindingError(f"{field} must be an object")
        _require_exact_fields(value, _BPF_TAG_KEYS, field)
        probe_role = _require_string(value["probe_role"], f"{field}.probe_role")
        kernel_tag = _require_string(value["kernel_tag"], f"{field}.kernel_tag")
        if _BPF_TAG_PATTERN.fullmatch(kernel_tag) is None:
            raise ObserverBindingError(
                f"{field}.kernel_tag must be 16 lowercase hexadecimal characters"
            )
        return cls(probe_role=probe_role, kernel_tag=kernel_tag)

    def to_mapping(self) -> dict[str, str]:
        return {
            "probe_role": self.probe_role,
            "kernel_tag": self.kernel_tag,
        }


def _decode_json(
    path: Path,
    *,
    subject: str = "observer binding",
) -> dict[str, object]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ObserverBindingError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
        )
    except ObserverBindingError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ObserverBindingError(f"invalid {subject} JSON") from error
    if not isinstance(value, dict):
        raise ObserverBindingError(f"{subject} must be an object")
    return value


def _decode_regular_artifact(path: Path, *, field: str) -> dict[str, object]:
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise ObserverBindingError(
            f"{field} artifact must be a regular file"
        ) from error
    if stat.S_ISLNK(mode):
        raise ObserverBindingError(f"{field} artifact must not be a symlink")
    if not stat.S_ISREG(mode):
        raise ObserverBindingError(
            f"{field} artifact must be a regular file"
        )
    return _decode_json(path, subject=f"{field} artifact")


def _require_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ObserverBindingError(f"{field} must be a non-empty string")
    return value


def _require_run_nonce(value: object) -> str:
    run_nonce = _require_string(value, "run_nonce")
    if len(run_nonce) < _RUN_NONCE_MIN_LENGTH:
        raise ObserverBindingError(
            "run_nonce must contain at least 16 characters"
        )
    return run_nonce


def _require_digest(value: object, field: str) -> str:
    digest = _require_string(value, field)
    if _SHA256_PATTERN.fullmatch(digest) is None:
        raise ObserverBindingError(
            f"{field} must be a lowercase SHA-256 digest"
        )
    return digest


def _require_positive_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ObserverBindingError(f"{field} must be a positive integer")
    return value


def _require_nonnegative_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ObserverBindingError(
            f"{field} must be a non-negative integer"
        )
    return value


def _require_boolean(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise ObserverBindingError(f"{field} must be a boolean")
    return value


def _require_string_list(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ObserverBindingError(f"{field} must be a list")
    return tuple(
        _require_string(item, f"{field}[{index}]")
        for index, item in enumerate(value)
    )


def _require_exact_fields(
    value: dict[str, object],
    expected: frozenset[str],
    subject: str,
) -> None:
    if set(value) != expected:
        raise ObserverBindingError(
            f"{subject} fields must be exactly {sorted(expected)}"
        )


def _require_schema_v1(value: object, subject: str) -> None:
    if _require_string(value, f"{subject}.schema_version") != "1.0":
        raise ObserverBindingError(
            f"{subject}.schema_version must be 1.0"
        )


def _reject_raw_address_material(value: object, subject: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            lowered = key.lower()
            if (
                "address" in lowered
                or "pointer" in lowered
                or lowered in {"raw_va", "virtual_address"}
            ):
                raise ObserverBindingError(
                    f"{subject} must not contain raw addresses or pointers"
                )
            _reject_raw_address_material(item, subject)
    elif isinstance(value, list):
        for item in value:
            _reject_raw_address_material(item, subject)
    elif isinstance(value, str) and _RAW_ADDRESS_PATTERN.search(value):
        raise ObserverBindingError(
            f"{subject} must not contain raw addresses or pointers"
        )


def _require_object(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ObserverBindingError(f"{field} must be an object")
    return value


def _parse_file_identity(
    value: object,
    *,
    field: str,
    mapped_library: bool = False,
) -> dict[str, object]:
    identity = _require_object(value, field)
    expected_keys = (
        _MAPPED_LIBRARY_IDENTITY_KEYS
        if mapped_library
        else _FILE_IDENTITY_KEYS
    )
    _require_exact_fields(identity, expected_keys, field)
    parsed: dict[str, object] = {
        "device_major": _require_nonnegative_integer(
            identity["device_major"], f"{field}.device_major"
        ),
        "device_minor": _require_nonnegative_integer(
            identity["device_minor"], f"{field}.device_minor"
        ),
        "inode": _require_positive_integer(
            identity["inode"], f"{field}.inode"
        ),
        "path_display": _require_string(
            identity["path_display"], f"{field}.path_display"
        ),
        "digest": _require_digest(identity["digest"], f"{field}.digest"),
    }
    if mapped_library:
        build_id = _require_string(
            identity["elf_build_id"], f"{field}.elf_build_id"
        )
        if _ELF_BUILD_ID_PATTERN.fullmatch(build_id) is None:
            raise ObserverBindingError(
                f"{field}.elf_build_id must be lowercase hexadecimal bytes"
            )
        parsed["elf_build_id"] = build_id
    return parsed


def _trusted_file_identity(
    value: dict[str, object],
) -> tuple[object, ...]:
    fields = ["device_major", "device_minor", "inode", "digest"]
    if "elf_build_id" in value:
        fields.append("elf_build_id")
    return tuple(value[field] for field in fields)


def _parse_symbol(value: object, *, field: str) -> tuple[str, int, int]:
    symbol = _require_object(value, field)
    _require_exact_fields(symbol, _SYMBOL_KEYS, field)
    name = _require_string(symbol["name"], f"{field}.name")
    if name != "llama_decode":
        raise ObserverBindingError(f"{field}.name must be llama_decode")
    return (
        name,
        _require_nonnegative_integer(
            symbol["symbol_value"], f"{field}.symbol_value"
        ),
        _require_nonnegative_integer(
            symbol["resolved_file_offset"],
            f"{field}.resolved_file_offset",
        ),
    )


def _parse_target_projection(
    value: dict[str, object],
    *,
    subject: str,
) -> dict[str, object]:
    live_executable = _parse_file_identity(
        value["live_executable"], field=f"{subject}.live_executable"
    )
    mapped_libllama = _parse_file_identity(
        value["mapped_libllama"],
        field=f"{subject}.mapped_libllama",
        mapped_library=True,
    )
    return {
        "host_boot_id": _require_string(
            value["host_boot_id"], f"{subject}.host_boot_id"
        ),
        "server_pid": _require_positive_integer(
            value["server_pid"], f"{subject}.server_pid"
        ),
        "server_start_time_ticks": _require_positive_integer(
            value["server_start_time_ticks"],
            f"{subject}.server_start_time_ticks",
        ),
        "runtime_instance_id": _require_digest(
            value["runtime_instance_id"], f"{subject}.runtime_instance_id"
        ),
        "live_executable": _trusted_file_identity(live_executable),
        "mapped_libllama": _trusted_file_identity(mapped_libllama),
        "mapped_libllama_mapping_key": (
            mapped_libllama["device_major"],
            mapped_libllama["device_minor"],
            mapped_libllama["inode"],
        ),
        "llama_decode_symbol": _parse_symbol(
            value["llama_decode_symbol"],
            field=f"{subject}.llama_decode_symbol",
        ),
    }


def _require_target_projection_matches(
    actual: dict[str, object],
    expected: dict[str, object],
    subject: str,
) -> None:
    for field in (
        "host_boot_id",
        "server_pid",
        "server_start_time_ticks",
        "runtime_instance_id",
        "live_executable",
        "mapped_libllama",
        "llama_decode_symbol",
    ):
        if actual[field] != expected[field]:
            raise ObserverBindingError(
                f"{subject}.{field} does not match the trusted target"
            )


def _require_matches(actual: object, expected: object, field: str) -> None:
    if actual != expected:
        raise ObserverBindingError(f"{field} does not match the binding")


@dataclass(frozen=True)
class ArtifactSeal:
    file: str
    digest: str

    @classmethod
    def from_mapping(
        cls,
        value: object,
        *,
        field: str,
        expected_file: str,
    ) -> ArtifactSeal:
        if not isinstance(value, dict):
            raise ObserverBindingError(f"{field} must be an object")
        actual_keys = set(value)
        if actual_keys != _SEAL_KEYS:
            raise ObserverBindingError(
                f"{field} seal fields must be exactly {sorted(_SEAL_KEYS)}"
            )
        artifact_file = _require_string(value["file"], f"{field}.file")
        if artifact_file != expected_file:
            raise ObserverBindingError(
                f"{field}.file must be {expected_file}"
            )
        return cls(
            file=artifact_file,
            digest=_require_digest(value["digest"], f"{field}.digest"),
        )

    def to_mapping(self) -> dict[str, str]:
        return {"file": self.file, "digest": self.digest}


def _validate_probe_profile(
    binding: RuntimeObserverEvaluationBinding,
    value: dict[str, object],
) -> RuntimeObserverProfile:
    try:
        profile = RuntimeObserverProfile.from_mapping(value)
    except ObserverProfileError as error:
        raise ObserverBindingError(
            f"probe_profile artifact is invalid: {error}"
        ) from error
    _require_matches(
        profile.llama_cpp_commit,
        binding.source_commit,
        "probe_profile.llama_cpp_commit",
    )
    _require_matches(
        profile.configured_context_ref,
        binding.configured_context_ref,
        "probe_profile.configured_context_ref",
    )
    return profile


def _validate_abi_layout(
    binding: RuntimeObserverEvaluationBinding,
    profile: RuntimeObserverProfile,
    value: dict[str, object],
) -> None:
    subject = "abi_layout_manifest"
    _require_exact_fields(value, _ABI_LAYOUT_KEYS, subject)
    _require_schema_v1(value["schema_version"], subject)
    source_commit = _require_string(
        value["source_commit"], f"{subject}.source_commit"
    )
    _require_string(value["compiler"], f"{subject}.compiler")
    _require_string_list(value["compiler_flags"], f"{subject}.compiler_flags")
    _require_digest(
        value["llama_header_digest"], f"{subject}.llama_header_digest"
    )
    mapped_libllama_digest = _require_digest(
        value["mapped_libllama_digest"],
        f"{subject}.mapped_libllama_digest",
    )
    layout = {
        field: _require_nonnegative_integer(
            value[field], f"{subject}.{field}"
        )
        for field in (
            "batch_size_bytes",
            "n_tokens_offset",
            "n_seq_id_offset",
            "seq_id_offset",
        )
    }
    _require_matches(
        source_commit,
        binding.source_commit,
        f"{subject}.source_commit",
    )
    _require_matches(
        mapped_libllama_digest,
        binding.mapped_libllama_digest,
        f"{subject}.mapped_libllama_digest",
    )
    for field, expected in (
        ("batch_size_bytes", profile.batch_size_bytes),
        ("n_tokens_offset", profile.n_tokens_offset),
        ("n_seq_id_offset", profile.n_seq_id_offset),
        ("seq_id_offset", profile.seq_id_offset),
    ):
        _require_matches(layout[field], expected, f"{subject}.{field}")


def _validate_expected_target(
    binding: RuntimeObserverEvaluationBinding,
    value: dict[str, object],
) -> dict[str, object]:
    subject = "expected_target_manifest"
    _require_exact_fields(value, _EXPECTED_TARGET_KEYS, subject)
    _require_schema_v1(value["schema_version"], subject)
    _reject_raw_address_material(value, subject)
    target = _parse_target_projection(value, subject=subject)
    actual = {
        "run_id": _require_string(value["run_id"], f"{subject}.run_id"),
        "run_nonce": _require_string(
            value["run_nonce"], f"{subject}.run_nonce"
        ),
        "model_digest": _require_digest(
            value["model_digest"], f"{subject}.model_digest"
        ),
        "server_config_digest": _require_digest(
            value["server_config_digest"],
            f"{subject}.server_config_digest",
        ),
        "probe_profile_digest": _require_digest(
            value["probe_profile_digest"],
            f"{subject}.probe_profile_digest",
        ),
        "abi_layout_manifest_digest": _require_digest(
            value["abi_layout_manifest_digest"],
            f"{subject}.abi_layout_manifest_digest",
        ),
        "bpf_object_digest": _require_digest(
            value["bpf_object_digest"], f"{subject}.bpf_object_digest"
        ),
        "native_collector_digest": _require_digest(
            value["native_collector_digest"],
            f"{subject}.native_collector_digest",
        ),
        "native_dependency_manifest_digest": _require_digest(
            value["native_dependency_manifest_digest"],
            f"{subject}.native_dependency_manifest_digest",
        ),
        "supervisor_source_bundle_digest": _require_digest(
            value["supervisor_source_bundle_digest"],
            f"{subject}.supervisor_source_bundle_digest",
        ),
        "supervisor_interpreter_digest": _require_digest(
            value["supervisor_interpreter_digest"],
            f"{subject}.supervisor_interpreter_digest",
        ),
    }
    expected = {
        "run_id": binding.run_id,
        "run_nonce": binding.run_nonce,
        "model_digest": binding.model_digest,
        "server_config_digest": binding.server_config_digest,
        "probe_profile_digest": binding.probe_profile.digest,
        "abi_layout_manifest_digest": binding.abi_layout_manifest.digest,
        "bpf_object_digest": binding.bpf_object_digest,
        "native_collector_digest": binding.native_collector_digest,
        "native_dependency_manifest_digest": (
            binding.native_dependency_manifest_digest
        ),
        "supervisor_source_bundle_digest": (
            binding.supervisor_source_bundle_digest
        ),
        "supervisor_interpreter_digest": binding.supervisor_interpreter_digest,
    }
    for field in actual:
        _require_matches(actual[field], expected[field], f"{subject}.{field}")
    for field, expected_value in (
        ("host_boot_id", binding.host_boot_id),
        ("server_pid", binding.server_pid),
        ("server_start_time_ticks", binding.server_start_time_ticks),
        ("runtime_instance_id", binding.runtime_instance_id),
    ):
        _require_matches(target[field], expected_value, f"{subject}.{field}")
    if target["live_executable"][3] != binding.server_binary_digest:
        raise ObserverBindingError(
            f"{subject}.live_executable.digest does not match the binding"
        )
    if target["mapped_libllama"][3] != binding.mapped_libllama_digest:
        raise ObserverBindingError(
            f"{subject}.mapped_libllama.digest does not match the binding"
        )
    return target


def _validate_observed_attach(
    binding: RuntimeObserverEvaluationBinding,
    value: dict[str, object],
    expected_target: dict[str, object],
) -> tuple[dict[str, object], int]:
    subject = "observed_attach_manifest"
    _require_exact_fields(value, _OBSERVED_ATTACH_KEYS, subject)
    _require_schema_v1(value["schema_version"], subject)
    _reject_raw_address_material(value, subject)
    target = _parse_target_projection(value, subject=subject)
    _require_target_projection_matches(target, expected_target, subject)
    attached_at = _require_nonnegative_integer(
        value["attached_at_monotonic_ns"],
        f"{subject}.attached_at_monotonic_ns",
    )
    actual = {
        "run_id": _require_string(value["run_id"], f"{subject}.run_id"),
        "run_nonce": _require_string(
            value["run_nonce"], f"{subject}.run_nonce"
        ),
        "oracle_instance_id": _require_digest(
            value["oracle_instance_id"],
            f"{subject}.oracle_instance_id",
        ),
        "runtime_instance_id": _require_digest(
            value["runtime_instance_id"],
            f"{subject}.runtime_instance_id",
        ),
        "host_boot_id": _require_string(
            value["host_boot_id"], f"{subject}.host_boot_id"
        ),
        "server_pid": _require_positive_integer(
            value["server_pid"], f"{subject}.server_pid"
        ),
        "server_start_time_ticks": _require_positive_integer(
            value["server_start_time_ticks"],
            f"{subject}.server_start_time_ticks",
        ),
        "expected_target_manifest_digest": _require_digest(
            value["expected_target_manifest_digest"],
            f"{subject}.expected_target_manifest_digest",
        ),
        "probe_profile_digest": _require_digest(
            value["probe_profile_digest"], f"{subject}.probe_profile_digest"
        ),
        "abi_layout_manifest_digest": _require_digest(
            value["abi_layout_manifest_digest"],
            f"{subject}.abi_layout_manifest_digest",
        ),
        "bpf_object_digest": _require_digest(
            value["bpf_object_digest"], f"{subject}.bpf_object_digest"
        ),
    }
    expected = {
        "run_id": binding.run_id,
        "run_nonce": binding.run_nonce,
        "oracle_instance_id": binding.oracle_instance_id,
        "runtime_instance_id": binding.runtime_instance_id,
        "host_boot_id": binding.host_boot_id,
        "server_pid": binding.server_pid,
        "server_start_time_ticks": binding.server_start_time_ticks,
        "expected_target_manifest_digest": (
            binding.expected_target_manifest.digest
        ),
        "probe_profile_digest": binding.probe_profile.digest,
        "abi_layout_manifest_digest": binding.abi_layout_manifest.digest,
        "bpf_object_digest": binding.bpf_object_digest,
    }
    for field in actual:
        _require_matches(actual[field], expected[field], f"{subject}.{field}")

    supervisor = _require_object(
        value["supervisor_process"], f"{subject}.supervisor_process"
    )
    _require_exact_fields(
        supervisor,
        _SUPERVISOR_PROCESS_KEYS,
        f"{subject}.supervisor_process",
    )
    supervisor_executable = _parse_file_identity(
        supervisor["live_executable"],
        field=f"{subject}.supervisor_process.live_executable",
    )
    supervisor_actual = (
        _require_positive_integer(
            supervisor["pid"], f"{subject}.supervisor_process.pid"
        ),
        _require_positive_integer(
            supervisor["start_time_ticks"],
            f"{subject}.supervisor_process.start_time_ticks",
        ),
        supervisor_executable["digest"],
        _require_digest(
            supervisor["source_bundle_digest"],
            f"{subject}.supervisor_process.source_bundle_digest",
        ),
    )
    supervisor_expected = (
        binding.supervisor_pid,
        binding.supervisor_start_time_ticks,
        binding.supervisor_interpreter_digest,
        binding.supervisor_source_bundle_digest,
    )
    _require_matches(
        supervisor_actual,
        supervisor_expected,
        f"{subject}.supervisor_process",
    )

    native = _require_object(
        value["native_collector_process"],
        f"{subject}.native_collector_process",
    )
    _require_exact_fields(
        native,
        _NATIVE_PROCESS_KEYS,
        f"{subject}.native_collector_process",
    )
    native_executable = _parse_file_identity(
        native["live_executable"],
        field=f"{subject}.native_collector_process.live_executable",
    )
    native_actual = (
        _require_positive_integer(
            native["pid"], f"{subject}.native_collector_process.pid"
        ),
        _require_positive_integer(
            native["start_time_ticks"],
            f"{subject}.native_collector_process.start_time_ticks",
        ),
        native_executable["digest"],
        _require_digest(
            native["dependency_manifest_digest"],
            f"{subject}.native_collector_process.dependency_manifest_digest",
        ),
    )
    native_expected = (
        binding.native_collector_pid,
        binding.native_collector_start_time_ticks,
        binding.native_collector_digest,
        binding.native_dependency_manifest_digest,
    )
    _require_matches(
        native_actual,
        native_expected,
        f"{subject}.native_collector_process",
    )

    programs_value = value["loaded_bpf_programs"]
    if not isinstance(programs_value, list):
        raise ObserverBindingError(f"{subject}.loaded_bpf_programs must be a list")
    programs: list[tuple[str, int, int, str]] = []
    for index, item in enumerate(programs_value):
        field = f"{subject}.loaded_bpf_programs[{index}]"
        program = _require_object(item, field)
        _require_exact_fields(program, _LOADED_BPF_PROGRAM_KEYS, field)
        role = _require_string(program["probe_role"], f"{field}.probe_role")
        tag = _require_string(program["kernel_tag"], f"{field}.kernel_tag")
        if _BPF_TAG_PATTERN.fullmatch(tag) is None:
            raise ObserverBindingError(
                f"{field}.kernel_tag must be 16 lowercase hexadecimal characters"
            )
        programs.append(
            (
                role,
                _require_positive_integer(
                    program["program_id"], f"{field}.program_id"
                ),
                _require_positive_integer(
                    program["link_id"], f"{field}.link_id"
                ),
                tag,
            )
        )
    if tuple(program[0] for program in programs) != ("entry", "return"):
        raise ObserverBindingError(
            f"{subject}.loaded_bpf_programs roles must be exactly entry, return"
        )
    if len({program[1] for program in programs}) != 2:
        raise ObserverBindingError(
            f"{subject}.loaded_bpf_programs program_id values must be unique"
        )
    if len({program[2] for program in programs}) != 2:
        raise ObserverBindingError(
            f"{subject}.loaded_bpf_programs link_id values must be unique"
        )
    if len({program[3] for program in programs}) != 2:
        raise ObserverBindingError(
            f"{subject}.loaded_bpf_programs kernel_tag values must be unique"
        )
    observed_tags = tuple(
        {"probe_role": role, "kernel_tag": tag}
        for role, _, _, tag in programs
    )
    expected_tags = tuple(
        tag.to_mapping() for tag in binding.loaded_bpf_program_tags
    )
    _require_matches(
        observed_tags,
        expected_tags,
        f"{subject}.loaded_bpf_programs",
    )
    tags_digest = _require_digest(
        value["loaded_bpf_program_tags_digest"],
        f"{subject}.loaded_bpf_program_tags_digest",
    )
    _require_matches(
        tags_digest,
        digest_json(list(observed_tags)),
        f"{subject}.loaded_bpf_program_tags_digest",
    )
    _require_matches(
        tags_digest,
        binding.loaded_bpf_program_tags_digest,
        f"{subject}.loaded_bpf_program_tags_digest",
    )
    if not _require_boolean(
        value["attach_succeeded"], f"{subject}.attach_succeeded"
    ):
        raise ObserverBindingError(f"{subject}.attach_succeeded must be true")
    features = _require_object(
        value["kernel_features"], f"{subject}.kernel_features"
    )
    _require_exact_fields(
        features, _KERNEL_FEATURE_KEYS, f"{subject}.kernel_features"
    )
    for feature in sorted(_KERNEL_FEATURE_KEYS):
        if not _require_boolean(
            features[feature], f"{subject}.kernel_features.{feature}"
        ):
            raise ObserverBindingError(
                f"{subject}.kernel_features.{feature} must be true"
            )
    return target, attached_at


def _validate_observed_quiescent(
    binding: RuntimeObserverEvaluationBinding,
    value: dict[str, object],
    attached_target: dict[str, object],
    attached_at_monotonic_ns: int,
) -> None:
    subject = "observed_quiescent_manifest"
    _require_exact_fields(value, _OBSERVED_QUIESCENT_KEYS, subject)
    _require_schema_v1(value["schema_version"], subject)
    _reject_raw_address_material(value, subject)
    target = _parse_target_projection(value, subject=subject)
    _require_target_projection_matches(target, attached_target, subject)
    actual = {
        "run_id": _require_string(value["run_id"], f"{subject}.run_id"),
        "run_nonce": _require_string(
            value["run_nonce"], f"{subject}.run_nonce"
        ),
        "oracle_instance_id": _require_digest(
            value["oracle_instance_id"], f"{subject}.oracle_instance_id"
        ),
        "runtime_instance_id": _require_digest(
            value["runtime_instance_id"],
            f"{subject}.runtime_instance_id",
        ),
        "observed_attach_manifest_digest": _require_digest(
            value["observed_attach_manifest_digest"],
            f"{subject}.observed_attach_manifest_digest",
        ),
    }
    expected = {
        "run_id": binding.run_id,
        "run_nonce": binding.run_nonce,
        "oracle_instance_id": binding.oracle_instance_id,
        "runtime_instance_id": binding.runtime_instance_id,
        "observed_attach_manifest_digest": (
            binding.observed_attach_manifest.digest
        ),
    }
    for field in actual:
        _require_matches(actual[field], expected[field], f"{subject}.{field}")
    quiescent_at = _require_nonnegative_integer(
        value["quiescent_at_monotonic_ns"],
        f"{subject}.quiescent_at_monotonic_ns",
    )
    if quiescent_at < attached_at_monotonic_ns:
        raise ObserverBindingError(
            f"{subject}.quiescent_at_monotonic_ns precedes attach"
        )
    if not _require_boolean(
        value["all_threads_stopped"], f"{subject}.all_threads_stopped"
    ):
        raise ObserverBindingError(f"{subject}.all_threads_stopped must be true")
    thread_values = value["thread_states"]
    if not isinstance(thread_values, list) or not thread_values:
        raise ObserverBindingError(f"{subject}.thread_states must be non-empty")
    threads: list[tuple[int, int, str]] = []
    for index, item in enumerate(thread_values):
        field = f"{subject}.thread_states[{index}]"
        thread = _require_object(item, field)
        _require_exact_fields(thread, _THREAD_STATE_KEYS, field)
        state = _require_string(thread["state"], f"{field}.state")
        if state not in {"T", "t"}:
            raise ObserverBindingError(f"{field}.state must be stopped")
        threads.append(
            (
                _require_positive_integer(thread["tid"], f"{field}.tid"),
                _require_positive_integer(
                    thread["start_time_ticks"], f"{field}.start_time_ticks"
                ),
                state,
            )
        )
    if tuple(item[0] for item in threads) != tuple(
        sorted(item[0] for item in threads)
    ) or len({item[0] for item in threads}) != len(threads):
        raise ObserverBindingError(
            f"{subject}.thread_states must have unique sorted tids"
        )
    if not any(
        tid == binding.server_pid
        and start == binding.server_start_time_ticks
        for tid, start, _ in threads
    ):
        raise ObserverBindingError(
            f"{subject}.thread_states must include the target leader"
        )

    mapping_values = value["mapping_snapshot"]
    if not isinstance(mapping_values, list) or not mapping_values:
        raise ObserverBindingError(
            f"{subject}.mapping_snapshot must be non-empty"
        )
    mappings: list[tuple[int, int, int, int, bool]] = []
    for index, item in enumerate(mapping_values):
        field = f"{subject}.mapping_snapshot[{index}]"
        mapping = _require_object(item, field)
        _require_exact_fields(mapping, _MAPPING_SNAPSHOT_KEYS, field)
        ordinal = _require_nonnegative_integer(
            mapping["ordinal"], f"{field}.ordinal"
        )
        _require_positive_integer(
            mapping["length_bytes"], f"{field}.length_bytes"
        )
        permissions = _require_string(
            mapping["permissions"], f"{field}.permissions"
        )
        if re.fullmatch(r"[r-][w-][x-][ps]", permissions) is None:
            raise ObserverBindingError(
                f"{field}.permissions must be a procfs permission token"
            )
        _require_nonnegative_integer(
            mapping["file_offset"], f"{field}.file_offset"
        )
        device_major = _require_nonnegative_integer(
            mapping["device_major"], f"{field}.device_major"
        )
        device_minor = _require_nonnegative_integer(
            mapping["device_minor"], f"{field}.device_minor"
        )
        inode = _require_positive_integer(
            mapping["inode"], f"{field}.inode"
        )
        _require_string(mapping["path_display"], f"{field}.path_display")
        mappings.append(
            (
                ordinal,
                device_major,
                device_minor,
                inode,
                permissions[2] == "x",
            )
        )
    if tuple(mapping[0] for mapping in mappings) != tuple(range(len(mappings))):
        raise ObserverBindingError(
            f"{subject}.mapping_snapshot ordinals must be continuous from zero"
        )
    executable_mapping_keys = {
        mapping[1:4] for mapping in mappings if mapping[4]
    }
    if (
        attached_target["mapped_libllama_mapping_key"]
        not in executable_mapping_keys
    ):
        raise ObserverBindingError(
            f"{subject}.mapping_snapshot must include executable mapped libllama"
        )
    for field in (
        "identity_monitor_alert_count",
        "final_inflight_count",
        "anomaly_count",
    ):
        count = _require_nonnegative_integer(
            value[field], f"{subject}.{field}"
        )
        if count != 0:
            raise ObserverBindingError(
                f"{subject}.{field} must be zero for healthy evidence"
            )


def _validate_target_terminated(
    binding: RuntimeObserverEvaluationBinding,
    value: dict[str, object],
) -> None:
    subject = "target_terminated"
    _require_exact_fields(value, _TARGET_TERMINATED_KEYS, subject)
    _require_schema_v1(value["schema_version"], subject)
    actual = {
        "run_id": _require_string(value["run_id"], f"{subject}.run_id"),
        "run_nonce": _require_string(
            value["run_nonce"], f"{subject}.run_nonce"
        ),
        "oracle_instance_id": _require_digest(
            value["oracle_instance_id"],
            f"{subject}.oracle_instance_id",
        ),
        "runtime_instance_id": _require_digest(
            value["runtime_instance_id"],
            f"{subject}.runtime_instance_id",
        ),
        "host_boot_id": _require_string(
            value["host_boot_id"], f"{subject}.host_boot_id"
        ),
        "server_pid": _require_positive_integer(
            value["server_pid"], f"{subject}.server_pid"
        ),
        "server_start_time_ticks": _require_positive_integer(
            value["server_start_time_ticks"],
            f"{subject}.server_start_time_ticks",
        ),
        "observed_quiescent_manifest_digest": _require_digest(
            value["observed_quiescent_manifest_digest"],
            f"{subject}.observed_quiescent_manifest_digest",
        ),
    }
    expected = {
        "run_id": binding.run_id,
        "run_nonce": binding.run_nonce,
        "oracle_instance_id": binding.oracle_instance_id,
        "runtime_instance_id": binding.runtime_instance_id,
        "host_boot_id": binding.host_boot_id,
        "server_pid": binding.server_pid,
        "server_start_time_ticks": binding.server_start_time_ticks,
        "observed_quiescent_manifest_digest": (
            binding.observed_quiescent_manifest.digest
        ),
    }
    for field in actual:
        _require_matches(actual[field], expected[field], f"{subject}.{field}")
    pidfd_bound = _require_boolean(
        value["pidfd_bound"], f"{subject}.pidfd_bound"
    )
    if not pidfd_bound:
        raise ObserverBindingError(f"{subject}.pidfd_bound must be true")
    termination_signal = _require_string(
        value["termination_signal"], f"{subject}.termination_signal"
    )
    if termination_signal != "SIGKILL":
        raise ObserverBindingError(
            f"{subject}.termination_signal must be SIGKILL"
        )
    wait_status = _require_nonnegative_integer(
        value["wait_status"], f"{subject}.wait_status"
    )
    if wait_status != 9:
        raise ObserverBindingError(f"{subject}.wait_status must be 9")


def _validate_observer_detached_retained(
    binding: RuntimeObserverEvaluationBinding,
    value: dict[str, object],
) -> dict[str, object]:
    try:
        marker = load_lifecycle_marker(canonical_lifecycle_bytes(value))
    except LifecycleError as error:
        raise ObserverBindingError("observer_detached artifact is invalid") from error
    for field, expected in (
        ("run_id", binding.run_id),
        ("run_nonce", binding.run_nonce),
        ("oracle_instance_id", binding.oracle_instance_id),
        ("runtime_instance_id", binding.runtime_instance_id),
        ("runtime_mode", "ATTACHED"),
        ("runtime_ownership", "external"),
    ):
        _require_matches(marker[field], expected, f"observer_detached.{field}")
    return marker


def _validate_target_retained(
    binding: RuntimeObserverEvaluationBinding,
    value: dict[str, object],
    detached: dict[str, object],
) -> None:
    subject = "target_retained"
    _require_exact_fields(value, _TARGET_RETAINED_KEYS, subject)
    _require_schema_v1(value["schema_version"], subject)
    expected = {
        "run_id": binding.run_id,
        "run_nonce": binding.run_nonce,
        "oracle_instance_id": binding.oracle_instance_id,
        "runtime_instance_id": binding.runtime_instance_id,
        "host_boot_id": binding.host_boot_id,
        "server_pid": binding.server_pid,
        "server_start_time_ticks": binding.server_start_time_ticks,
        "runtime_mode": "ATTACHED",
        "runtime_ownership": "external",
        "live_executable_digest": binding.server_binary_digest,
        "mapped_libllama_digest": binding.mapped_libllama_digest,
        "observer_detached_digest": (
            None
            if binding.observer_detached is None
            else binding.observer_detached.digest
        ),
        "observer_detached_at_monotonic_ns": detached[
            "detached_at_monotonic_ns"
        ],
    }
    for field, expected_value in expected.items():
        actual = value[field]
        if field.endswith("_digest"):
            actual = _require_digest(actual, f"{subject}.{field}")
        elif field in {
            "server_pid",
            "server_start_time_ticks",
        }:
            actual = _require_positive_integer(actual, f"{subject}.{field}")
        elif field == "observer_detached_at_monotonic_ns":
            actual = _require_nonnegative_integer(actual, f"{subject}.{field}")
        else:
            actual = _require_string(actual, f"{subject}.{field}")
        _require_matches(actual, expected_value, f"{subject}.{field}")
    verified_at = _require_nonnegative_integer(
        value["identity_verified_at_monotonic_ns"],
        f"{subject}.identity_verified_at_monotonic_ns",
    )
    if verified_at < int(detached["detached_at_monotonic_ns"]):
        raise ObserverBindingError(
            "target_retained identity verification precedes detach"
        )


def _validate_verified_artifacts(
    binding: RuntimeObserverEvaluationBinding,
    artifacts: dict[str, dict[str, object]],
) -> RuntimeObserverProfile:
    profile = _validate_probe_profile(binding, artifacts["probe_profile"])
    _validate_abi_layout(
        binding,
        profile,
        artifacts["abi_layout_manifest"],
    )
    expected_target = _validate_expected_target(
        binding, artifacts["expected_target_manifest"]
    )
    attached_target, attached_at = _validate_observed_attach(
        binding,
        artifacts["observed_attach_manifest"],
        expected_target,
    )
    if binding.runtime_mode == "ATTACHED":
        detached = _validate_observer_detached_retained(
            binding, artifacts["observer_detached"]
        )
        _validate_target_retained(binding, artifacts["target_retained"], detached)
    else:
        _validate_observed_quiescent(
            binding,
            artifacts["observed_quiescent_manifest"],
            attached_target,
            attached_at,
        )
        _validate_target_terminated(binding, artifacts["target_terminated"])
    return profile


@dataclass(frozen=True)
class RuntimeObserverEvaluationBinding:
    """Bind one evaluation run to frozen runtime and observer metadata.

    Metadata-only loading supports construction. Verified loading additionally
    reopens and authenticates every fixed-basename artifact before evaluation.
    """

    schema_version: str
    run_id: str
    run_nonce: str
    oracle_instance_id: str
    runtime_instance_id: str
    host_boot_id: str
    server_pid: int
    server_start_time_ticks: int
    supervisor_pid: int
    supervisor_start_time_ticks: int
    native_collector_pid: int
    native_collector_start_time_ticks: int
    configured_context_ref: str
    clock_id: str
    probe_profile: ArtifactSeal
    abi_layout_manifest: ArtifactSeal
    expected_target_manifest: ArtifactSeal
    observed_attach_manifest: ArtifactSeal
    observed_quiescent_manifest: ArtifactSeal | None
    target_terminated: ArtifactSeal | None
    source_commit: str
    server_binary_digest: str
    mapped_libllama_digest: str
    model_digest: str
    server_config_digest: str
    build_command_digest: str
    bpf_object_digest: str
    native_collector_digest: str
    native_dependency_manifest_digest: str
    supervisor_source_bundle_digest: str
    supervisor_interpreter_digest: str
    loaded_bpf_program_tags: tuple[LoadedBpfProgramTag, ...]
    loaded_bpf_program_tags_digest: str
    wire_protocol_version: str
    runtime_oracle_schema_version: str
    qualified_toolchain_profile: str
    runtime_build_profile_digest: str
    observer_artifact_profile_digest: str
    runtime_mode: str = "MANAGED_LEGACY"
    runtime_ownership: str = "laris"
    observer_detached: ArtifactSeal | None = None
    target_retained: ArtifactSeal | None = None
    _verified_probe_profile: RuntimeObserverProfile | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    @classmethod
    def load(cls, path: Path) -> RuntimeObserverEvaluationBinding:
        """Load and validate metadata without reopening referenced artifacts."""
        return cls.from_mapping(_decode_json(path))

    @classmethod
    def load_verified(cls, path: Path) -> RuntimeObserverEvaluationBinding:
        """Load metadata and verify all fixed-basename artifact seals."""
        binding = cls.load(path)
        artifacts: dict[str, dict[str, object]] = {}
        artifact_files = dict(_ARTIFACT_FILES)
        if binding.schema_version == "1.1":
            artifact_files.pop("target_terminated")
            if binding.runtime_mode == "ATTACHED":
                artifact_files.pop("observed_quiescent_manifest")
                artifact_files.update(
                    {
                        "observer_detached": "OBSERVER_DETACHED.json",
                        "target_retained": "TARGET_RETAINED.json",
                    }
                )
            else:
                artifact_files.update(
                    {"observer_detached": "OBSERVER_DETACHED.json"}
                )
                artifact_files["target_terminated"] = "TARGET_TERMINATED.json"
        for field, filename in artifact_files.items():
            seal = getattr(binding, field)
            if seal is None:
                raise ObserverBindingError(f"{field} artifact seal is unavailable")
            artifact = _decode_regular_artifact(
                path.parent / filename,
                field=field,
            )
            if digest_json(artifact) != seal.digest:
                raise ObserverBindingError(
                    f"{field} artifact digest mismatch"
                )
            artifacts[field] = artifact
        profile = _validate_verified_artifacts(binding, artifacts)
        object.__setattr__(binding, "_verified_probe_profile", profile)
        return binding

    @property
    def max_batch_tokens(self) -> int:
        """Return the limit authenticated by the sealed probe profile."""
        if self._verified_probe_profile is None:
            raise ObserverBindingError(
                "verified probe profile is required for event limits"
            )
        return self._verified_probe_profile.max_batch_tokens

    @property
    def allowed_seq_ids(self) -> tuple[int, ...]:
        """Return sequence IDs authenticated by the sealed probe profile."""
        if self._verified_probe_profile is None:
            raise ObserverBindingError(
                "verified probe profile is required for event limits"
            )
        return self._verified_probe_profile.allowed_seq_ids

    @classmethod
    def from_mapping(
        cls,
        value: dict[str, object],
    ) -> RuntimeObserverEvaluationBinding:
        if value.get("schema_version") == "1.1":
            return cls._from_mapping_v11(value)
        extra = set(value) - _BINDING_KEYS
        if extra:
            raise ObserverBindingError(
                f"unexpected binding fields: {sorted(extra)}"
            )
        missing = _BINDING_KEYS - set(value)
        if missing:
            raise ObserverBindingError(
                f"missing binding fields: {sorted(missing)}"
            )

        seals = {
            field: ArtifactSeal.from_mapping(
                value[field],
                field=field,
                expected_file=expected_file,
            )
            for field, expected_file in _ARTIFACT_FILES.items()
        }

        schema_version = _require_string(
            value["schema_version"], "schema_version"
        )
        if schema_version != "1.0":
            raise ObserverBindingError("schema_version must be 1.0")
        clock_id = _require_string(value["clock_id"], "clock_id")
        if clock_id != "CLOCK_MONOTONIC":
            raise ObserverBindingError("clock_id must be CLOCK_MONOTONIC")
        source_commit = _require_string(value["source_commit"], "source_commit")
        if _GIT_COMMIT_PATTERN.fullmatch(source_commit) is None:
            raise ObserverBindingError(
                "source_commit must be a lowercase 40-character Git commit"
            )
        host_boot_id = _require_string(value["host_boot_id"], "host_boot_id")
        server_pid = _require_positive_integer(
            value["server_pid"], "server_pid"
        )
        server_start_time_ticks = _require_positive_integer(
            value["server_start_time_ticks"], "server_start_time_ticks"
        )
        supervisor_pid = _require_positive_integer(
            value["supervisor_pid"], "supervisor_pid"
        )
        supervisor_start_time_ticks = _require_positive_integer(
            value["supervisor_start_time_ticks"],
            "supervisor_start_time_ticks",
        )
        native_collector_pid = _require_positive_integer(
            value["native_collector_pid"], "native_collector_pid"
        )
        native_collector_start_time_ticks = _require_positive_integer(
            value["native_collector_start_time_ticks"],
            "native_collector_start_time_ticks",
        )
        raw_tags = value["loaded_bpf_program_tags"]
        if not isinstance(raw_tags, list):
            raise ObserverBindingError(
                "loaded_bpf_program_tags must be a list"
            )
        loaded_bpf_program_tags = tuple(
            LoadedBpfProgramTag.from_mapping(
                item,
                field=f"loaded_bpf_program_tags[{index}]",
            )
            for index, item in enumerate(raw_tags)
        )
        roles = tuple(tag.probe_role for tag in loaded_bpf_program_tags)
        if roles != ("entry", "return"):
            raise ObserverBindingError(
                "loaded_bpf_program_tags roles must be exactly entry, return"
            )
        if len({tag.kernel_tag for tag in loaded_bpf_program_tags}) != 2:
            raise ObserverBindingError(
                "loaded_bpf_program_tags kernel tags must be unique"
            )

        digests = {
            field: _require_digest(value[field], field)
            for field in (
                "oracle_instance_id",
                "runtime_instance_id",
                "server_binary_digest",
                "mapped_libllama_digest",
                "model_digest",
                "server_config_digest",
                "build_command_digest",
                "bpf_object_digest",
                "native_collector_digest",
                "native_dependency_manifest_digest",
                "supervisor_source_bundle_digest",
                "supervisor_interpreter_digest",
                "loaded_bpf_program_tags_digest",
                "runtime_build_profile_digest",
                "observer_artifact_profile_digest",
            )
        }
        wire_protocol_version = _require_string(
            value["wire_protocol_version"], "wire_protocol_version"
        )
        if wire_protocol_version != "1.0":
            raise ObserverBindingError("wire_protocol_version must be 1.0")
        runtime_oracle_schema_version = _require_string(
            value["runtime_oracle_schema_version"],
            "runtime_oracle_schema_version",
        )
        if runtime_oracle_schema_version != "2.0":
            raise ObserverBindingError(
                "runtime_oracle_schema_version must be 2.0"
            )
        qualified_toolchain_profile = _require_string(
            value["qualified_toolchain_profile"],
            "qualified_toolchain_profile",
        )

        expected_runtime_instance_id = derive_runtime_instance_id(
            host_boot_id=host_boot_id,
            server_pid=server_pid,
            server_start_time_ticks=server_start_time_ticks,
            live_executable_digest=digests["server_binary_digest"],
        )
        if digests["runtime_instance_id"] != expected_runtime_instance_id:
            raise ObserverBindingError(
                "runtime_instance_id derivation mismatch"
            )

        canonical_tags = tuple(
            tag.to_mapping() for tag in loaded_bpf_program_tags
        )
        expected_tags_digest = digest_json(list(canonical_tags))
        if (
            digests["loaded_bpf_program_tags_digest"]
            != expected_tags_digest
        ):
            raise ObserverBindingError(
                "loaded_bpf_program_tags_digest mismatch"
            )
        expected_oracle_instance_id = derive_oracle_instance_id(
            host_boot_id=host_boot_id,
            supervisor_pid=supervisor_pid,
            supervisor_start_time_ticks=supervisor_start_time_ticks,
            supervisor_source_bundle_digest=digests[
                "supervisor_source_bundle_digest"
            ],
            supervisor_interpreter_digest=digests[
                "supervisor_interpreter_digest"
            ],
            native_collector_pid=native_collector_pid,
            native_collector_start_time_ticks=(
                native_collector_start_time_ticks
            ),
            native_collector_digest=digests["native_collector_digest"],
            native_dependency_manifest_digest=digests[
                "native_dependency_manifest_digest"
            ],
            bpf_object_digest=digests["bpf_object_digest"],
            loaded_bpf_program_tags=canonical_tags,
        )
        if digests["oracle_instance_id"] != expected_oracle_instance_id:
            raise ObserverBindingError(
                "oracle_instance_id derivation mismatch"
            )

        expected_runtime_digest = derive_runtime_build_profile_digest(
            source_commit=source_commit,
            server_binary_digest=digests["server_binary_digest"],
            mapped_libllama_digest=digests["mapped_libllama_digest"],
            model_digest=digests["model_digest"],
            server_config_digest=digests["server_config_digest"],
            build_command_digest=digests["build_command_digest"],
            probe_profile_digest=seals["probe_profile"].digest,
            abi_layout_manifest_digest=seals["abi_layout_manifest"].digest,
        )
        if digests["runtime_build_profile_digest"] != expected_runtime_digest:
            raise ObserverBindingError("runtime_build_profile_digest mismatch")

        expected_observer_digest = derive_observer_artifact_profile_digest(
            bpf_object_digest=digests["bpf_object_digest"],
            native_collector_digest=digests["native_collector_digest"],
            native_dependency_manifest_digest=digests[
                "native_dependency_manifest_digest"
            ],
            supervisor_source_bundle_digest=digests[
                "supervisor_source_bundle_digest"
            ],
            supervisor_interpreter_digest=digests[
                "supervisor_interpreter_digest"
            ],
            wire_protocol_version=wire_protocol_version,
            runtime_oracle_schema_version=runtime_oracle_schema_version,
            qualified_toolchain_profile=qualified_toolchain_profile,
        )
        if (
            digests["observer_artifact_profile_digest"]
            != expected_observer_digest
        ):
            raise ObserverBindingError(
                "observer_artifact_profile_digest mismatch"
            )

        return cls(
            schema_version=schema_version,
            run_id=_require_string(value["run_id"], "run_id"),
            run_nonce=_require_run_nonce(value["run_nonce"]),
            oracle_instance_id=digests["oracle_instance_id"],
            runtime_instance_id=digests["runtime_instance_id"],
            host_boot_id=host_boot_id,
            server_pid=server_pid,
            server_start_time_ticks=server_start_time_ticks,
            supervisor_pid=supervisor_pid,
            supervisor_start_time_ticks=supervisor_start_time_ticks,
            native_collector_pid=native_collector_pid,
            native_collector_start_time_ticks=(
                native_collector_start_time_ticks
            ),
            configured_context_ref=_require_string(
                value["configured_context_ref"], "configured_context_ref"
            ),
            clock_id=clock_id,
            probe_profile=seals["probe_profile"],
            abi_layout_manifest=seals["abi_layout_manifest"],
            expected_target_manifest=seals["expected_target_manifest"],
            observed_attach_manifest=seals["observed_attach_manifest"],
            observed_quiescent_manifest=seals["observed_quiescent_manifest"],
            target_terminated=seals["target_terminated"],
            source_commit=source_commit,
            server_binary_digest=digests["server_binary_digest"],
            mapped_libllama_digest=digests["mapped_libllama_digest"],
            model_digest=digests["model_digest"],
            server_config_digest=digests["server_config_digest"],
            build_command_digest=digests["build_command_digest"],
            bpf_object_digest=digests["bpf_object_digest"],
            native_collector_digest=digests["native_collector_digest"],
            native_dependency_manifest_digest=digests[
                "native_dependency_manifest_digest"
            ],
            supervisor_source_bundle_digest=digests[
                "supervisor_source_bundle_digest"
            ],
            supervisor_interpreter_digest=digests[
                "supervisor_interpreter_digest"
            ],
            loaded_bpf_program_tags=loaded_bpf_program_tags,
            loaded_bpf_program_tags_digest=digests[
                "loaded_bpf_program_tags_digest"
            ],
            wire_protocol_version=wire_protocol_version,
            runtime_oracle_schema_version=runtime_oracle_schema_version,
            qualified_toolchain_profile=qualified_toolchain_profile,
            runtime_build_profile_digest=digests[
                "runtime_build_profile_digest"
            ],
            observer_artifact_profile_digest=digests[
                "observer_artifact_profile_digest"
            ],
        )

    @classmethod
    def _from_mapping_v11(
        cls,
        value: dict[str, object],
    ) -> RuntimeObserverEvaluationBinding:
        extra = set(value) - _BINDING_V11_KEYS
        missing = _BINDING_V11_KEYS - set(value)
        if extra:
            raise ObserverBindingError(f"unexpected binding fields: {sorted(extra)}")
        if missing:
            raise ObserverBindingError(f"missing binding fields: {sorted(missing)}")

        runtime_mode = _require_string(value["runtime_mode"], "runtime_mode")
        runtime_ownership = _require_string(
            value["runtime_ownership"], "runtime_ownership"
        )
        expected_ownership = {
            "MANAGED_LEGACY": "laris",
            "ATTACHED": "external",
        }.get(runtime_mode)
        if expected_ownership != runtime_ownership:
            raise ObserverBindingError("runtime_mode and runtime_ownership mismatch")
        if value.get("runtime_oracle_schema_version") != "2.1":
            raise ObserverBindingError(
                "runtime_oracle_schema_version must be 2.1 for binding 1.1"
            )

        observer_detached = ArtifactSeal.from_mapping(
            value["observer_detached"],
            field="observer_detached",
            expected_file="OBSERVER_DETACHED.json",
        )
        raw_quiescent = value["observed_quiescent_manifest"]
        observed_quiescent = (
            None
            if raw_quiescent is None
            else ArtifactSeal.from_mapping(
                raw_quiescent,
                field="observed_quiescent_manifest",
                expected_file="observed_quiescent_manifest.json",
            )
        )
        raw_terminated = value["target_terminated"]
        target_terminated = (
            None
            if raw_terminated is None
            else ArtifactSeal.from_mapping(
                raw_terminated,
                field="target_terminated",
                expected_file="TARGET_TERMINATED.json",
            )
        )
        raw_retained = value["target_retained"]
        target_retained = (
            None
            if raw_retained is None
            else ArtifactSeal.from_mapping(
                raw_retained,
                field="target_retained",
                expected_file="TARGET_RETAINED.json",
            )
        )
        if (target_terminated is None) == (target_retained is None):
            raise ObserverBindingError("exactly one terminal target artifact is required")
        if runtime_mode == "ATTACHED":
            if target_retained is None or observed_quiescent is not None:
                raise ObserverBindingError(
                    "attached binding requires retained terminal evidence only"
                )
        elif target_terminated is None or observed_quiescent is None:
            raise ObserverBindingError(
                "managed binding requires quiescent and terminated evidence"
            )

        original_observer_digest = _require_digest(
            value["observer_artifact_profile_digest"],
            "observer_artifact_profile_digest",
        )
        expected_observer_digest = derive_observer_artifact_profile_digest(
            bpf_object_digest=_require_digest(
                value["bpf_object_digest"], "bpf_object_digest"
            ),
            native_collector_digest=_require_digest(
                value["native_collector_digest"], "native_collector_digest"
            ),
            native_dependency_manifest_digest=_require_digest(
                value["native_dependency_manifest_digest"],
                "native_dependency_manifest_digest",
            ),
            supervisor_source_bundle_digest=_require_digest(
                value["supervisor_source_bundle_digest"],
                "supervisor_source_bundle_digest",
            ),
            supervisor_interpreter_digest=_require_digest(
                value["supervisor_interpreter_digest"],
                "supervisor_interpreter_digest",
            ),
            wire_protocol_version=_require_string(
                value["wire_protocol_version"], "wire_protocol_version"
            ),
            runtime_oracle_schema_version="2.1",
            qualified_toolchain_profile=_require_string(
                value["qualified_toolchain_profile"],
                "qualified_toolchain_profile",
            ),
        )
        if original_observer_digest != expected_observer_digest:
            raise ObserverBindingError("observer_artifact_profile_digest mismatch")

        # Reuse all frozen v1 identity and derivation checks, substituting only
        # internal seals that are not part of the returned v1.1 contract.
        legacy = dict(value)
        for field in (
            "runtime_mode",
            "runtime_ownership",
            "observer_detached",
            "target_retained",
        ):
            legacy.pop(field)
        legacy["schema_version"] = "1.0"
        legacy["runtime_oracle_schema_version"] = "2.0"
        placeholder = {
            "file": "observed_quiescent_manifest.json",
            "digest": "sha256:" + "0" * 64,
        }
        if observed_quiescent is None:
            legacy["observed_quiescent_manifest"] = placeholder
        if target_terminated is None:
            legacy["target_terminated"] = {
                "file": "TARGET_TERMINATED.json",
                "digest": "sha256:" + "0" * 64,
            }
        legacy["observer_artifact_profile_digest"] = (
            derive_observer_artifact_profile_digest(
                bpf_object_digest=str(value["bpf_object_digest"]),
                native_collector_digest=str(value["native_collector_digest"]),
                native_dependency_manifest_digest=str(
                    value["native_dependency_manifest_digest"]
                ),
                supervisor_source_bundle_digest=str(
                    value["supervisor_source_bundle_digest"]
                ),
                supervisor_interpreter_digest=str(
                    value["supervisor_interpreter_digest"]
                ),
                wire_protocol_version=str(value["wire_protocol_version"]),
                runtime_oracle_schema_version="2.0",
                qualified_toolchain_profile=str(
                    value["qualified_toolchain_profile"]
                ),
            )
        )
        binding = cls.from_mapping(legacy)
        object.__setattr__(binding, "schema_version", "1.1")
        object.__setattr__(binding, "runtime_oracle_schema_version", "2.1")
        object.__setattr__(
            binding,
            "observer_artifact_profile_digest",
            original_observer_digest,
        )
        object.__setattr__(binding, "runtime_mode", runtime_mode)
        object.__setattr__(binding, "runtime_ownership", runtime_ownership)
        object.__setattr__(binding, "observer_detached", observer_detached)
        object.__setattr__(binding, "observed_quiescent_manifest", observed_quiescent)
        object.__setattr__(binding, "target_terminated", target_terminated)
        object.__setattr__(binding, "target_retained", target_retained)
        return binding

    @property
    def terminal_target_state(self) -> str:
        return "retained" if self.target_retained is not None else "terminated"

    @property
    def terminal_target_digest(self) -> str:
        seal = self.target_retained or self.target_terminated
        if seal is None:
            raise ObserverBindingError("terminal target artifact is unavailable")
        return seal.digest

    def to_mapping(self) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "run_nonce": self.run_nonce,
            "oracle_instance_id": self.oracle_instance_id,
            "runtime_instance_id": self.runtime_instance_id,
            "host_boot_id": self.host_boot_id,
            "server_pid": self.server_pid,
            "server_start_time_ticks": self.server_start_time_ticks,
            "supervisor_pid": self.supervisor_pid,
            "supervisor_start_time_ticks": self.supervisor_start_time_ticks,
            "native_collector_pid": self.native_collector_pid,
            "native_collector_start_time_ticks": (
                self.native_collector_start_time_ticks
            ),
            "configured_context_ref": self.configured_context_ref,
            "clock_id": self.clock_id,
            "probe_profile": self.probe_profile.to_mapping(),
            "abi_layout_manifest": self.abi_layout_manifest.to_mapping(),
            "expected_target_manifest": (
                self.expected_target_manifest.to_mapping()
            ),
            "observed_attach_manifest": (
                self.observed_attach_manifest.to_mapping()
            ),
            "observed_quiescent_manifest": (
                None
                if self.observed_quiescent_manifest is None
                else self.observed_quiescent_manifest.to_mapping()
            ),
            "source_commit": self.source_commit,
            "server_binary_digest": self.server_binary_digest,
            "mapped_libllama_digest": self.mapped_libllama_digest,
            "model_digest": self.model_digest,
            "server_config_digest": self.server_config_digest,
            "build_command_digest": self.build_command_digest,
            "bpf_object_digest": self.bpf_object_digest,
            "native_collector_digest": self.native_collector_digest,
            "native_dependency_manifest_digest": (
                self.native_dependency_manifest_digest
            ),
            "supervisor_source_bundle_digest": (
                self.supervisor_source_bundle_digest
            ),
            "supervisor_interpreter_digest": self.supervisor_interpreter_digest,
            "loaded_bpf_program_tags": [
                tag.to_mapping() for tag in self.loaded_bpf_program_tags
            ],
            "loaded_bpf_program_tags_digest": (
                self.loaded_bpf_program_tags_digest
            ),
            "wire_protocol_version": self.wire_protocol_version,
            "runtime_oracle_schema_version": (
                self.runtime_oracle_schema_version
            ),
            "qualified_toolchain_profile": self.qualified_toolchain_profile,
            "runtime_build_profile_digest": self.runtime_build_profile_digest,
            "observer_artifact_profile_digest": (
                self.observer_artifact_profile_digest
            ),
        }
        if self.schema_version == "1.0":
            if self.target_terminated is None:
                raise ObserverBindingError("legacy target_terminated is unavailable")
            value["target_terminated"] = self.target_terminated.to_mapping()
            return value
        if self.schema_version != "1.1":
            raise ObserverBindingError("unsupported binding schema_version")
        value.update(
            {
                "runtime_mode": self.runtime_mode,
                "runtime_ownership": self.runtime_ownership,
                "observer_detached": (
                    None
                    if self.observer_detached is None
                    else self.observer_detached.to_mapping()
                ),
                "target_terminated": (
                    None
                    if self.target_terminated is None
                    else self.target_terminated.to_mapping()
                ),
                "target_retained": (
                    None
                    if self.target_retained is None
                    else self.target_retained.to_mapping()
                ),
            }
        )
        return value
