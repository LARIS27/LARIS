from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat

from laris_onpremise.kv_state.observation.runtime_state_profile import (
    PINNED_LLAMA_CPP_COMMIT,
    QUALIFIED_MODEL_PROFILE_ID,
    QUALIFIED_MODEL_SHA256,
    RuntimeStateProfileError,
)

from .contracts import RehostRuntimeProfile


_MAXIMUM_PROFILE_BYTES = 1024 * 1024
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_BUILD_ID = re.compile(r"[0-9a-f]{40}\Z")
_PROFILE_IDS = {
    "1.3": "x86_64-cpu-simple-kv-qwen2.5-0.5b-q4",
    "1.4": "x86_64-cpu-simple-kv",
    "1.5": "x86_64-cpu-simple-kv-bounded",
    "1.6": "x86_64-cpu-simple-kv-rehost",
}


def _object(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise RuntimeStateProfileError(f"{field} must be an object")
    return value


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeStateProfileError(f"{field} must be non-empty text")
    return value


def _digest(value: object, field: str) -> str:
    result = _text(value, field)
    if _SHA256.fullmatch(result) is None:
        raise RuntimeStateProfileError(
            f"{field} must be lowercase sha256:<64 hex>"
        )
    return result


def _positive(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeStateProfileError(f"{field} must be a positive integer")
    return value


def _exact(value: object, expected: object, field: str) -> None:
    if type(value) is not type(expected) or value != expected:
        raise RuntimeStateProfileError(f"{field} must equal {expected!r}")


def _absolute_posix_path(value: object, field: str) -> str:
    result = _text(value, field)
    if (
        not result.startswith("/")
        or result == "/"
        or "\\" in result
        or "//" in result
        or any(part in {"", ".", ".."} for part in result.split("/")[1:])
    ):
        raise RuntimeStateProfileError(f"{field} must be a canonical absolute POSIX path")
    return result


def _read_profile(path: Path) -> dict[str, object]:
    if not isinstance(path, Path):
        raise RuntimeStateProfileError("profile path must be a pathlib.Path")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeStateProfileError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    descriptor = -1
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_BINARY", 0),
        )
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > _MAXIMUM_PROFILE_BYTES
        ):
            raise RuntimeStateProfileError("runtime profile identity is invalid")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 64 * 1024))
            if not chunk:
                raise RuntimeStateProfileError("runtime profile size is invalid")
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
        if len(payload) != before.st_size or len(payload) > _MAXIMUM_PROFILE_BYTES:
            raise RuntimeStateProfileError("runtime profile size is invalid")
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise RuntimeStateProfileError("runtime profile changed while read")
        document = json.loads(
            payload.decode("utf-8"), object_pairs_hook=reject_duplicates
        )
    except RuntimeStateProfileError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeStateProfileError("invalid rehost runtime profile JSON") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return _object(document, "runtime profile")


def _canonical_digest(document: dict[str, object]) -> str:
    payload = json.dumps(
        document,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def load_rehost_runtime_profile(path: Path) -> RehostRuntimeProfile:
    """Load only the qualified runtime facts required for payload rehosting.

    Capture-only ABI offsets and probe layouts intentionally remain under the
    stricter observer-profile loader. Rehosting validates only the runtime facts
    needed to restore a portable bundle, without weakening runtime, topology,
    model-library, or KV-type checks.
    """

    document = _read_profile(path)
    schema_version = _text(document.get("schema_version"), "schema_version")
    if schema_version not in _PROFILE_IDS:
        raise RuntimeStateProfileError("runtime profile schema is unsupported")
    _exact(document.get("profile_id"), _PROFILE_IDS[schema_version], "profile_id")
    if schema_version in {"1.4", "1.5", "1.6"}:
        _exact(
            document.get("adapter_kind"),
            "llama_cpp_simple_kv",
            "adapter_kind",
        )
    else:
        _exact(
            document.get("qualified_model_profile_id"),
            QUALIFIED_MODEL_PROFILE_ID,
            "qualified_model_profile_id",
        )
        _exact(
            document.get("qualified_model_sha256"),
            QUALIFIED_MODEL_SHA256,
            "qualified_model_sha256",
        )
    for field, expected in {
        "llama_cpp_commit": PINNED_LLAMA_CPP_COMMIT,
        "os": "Linux",
        "architecture": "x86_64",
        "abi": "System V AMD64",
        "memory_family": "simple_kv",
    }.items():
        _exact(document.get(field), expected, field)

    qualification = _object(document.get("qualification"), "qualification")
    if schema_version == "1.6":
        _exact(
            qualification.get("state"),
            "local_rehost_qualified",
            "qualification.state",
        )
        _exact(
            qualification.get("scope"),
            "rehosting_only",
            "qualification.scope",
        )
    else:
        _exact(
            qualification.get("state"),
            "linux_live_qualified",
            "qualification.state",
        )

    identity = _object(document.get("runtime_identity"), "runtime_identity")
    runtime_binary_digest = _digest(
        identity.get("runtime_binary_sha256"), "runtime_binary_sha256"
    )
    mapped_libllama_digest = _digest(
        identity.get("mapped_libllama_sha256"), "mapped_libllama_sha256"
    )
    build_id = _text(
        identity.get("mapped_libllama_elf_build_id"),
        "mapped_libllama_elf_build_id",
    )
    if _BUILD_ID.fullmatch(build_id) is None:
        raise RuntimeStateProfileError(
            "mapped_libllama_elf_build_id must be 40 lowercase hex"
        )
    if schema_version == "1.6":
        _absolute_posix_path(
            identity.get("runtime_binary_path"), "runtime_binary_path"
        )
    else:
        _exact(
            identity.get("runtime_binary_path"),
            "/opt/laris/llama-runtime/bin/llama-server",
            "runtime_binary_path",
        )

    policy = _object(document.get("runtime_policy"), "runtime_policy")
    for field, expected in {
        "cache_prompt": False,
        "cache_idle_slots": False,
        "context_shift": False,
        "requires_full_sequence_rewrite_per_lease": True,
    }.items():
        _exact(policy.get(field), expected, f"runtime_policy.{field}")

    cache_types = _object(document.get("kv_cache_types"), "kv_cache_types")
    _exact(cache_types.get("key"), "f16", "kv_cache_types.key")
    _exact(cache_types.get("value"), "f16", "kv_cache_types.value")
    _exact(cache_types.get("key_ggml_type_id"), 1, "key_ggml_type_id")
    _exact(cache_types.get("value_ggml_type_id"), 1, "value_ggml_type_id")
    key_ggml_type_id = 1
    value_ggml_type_id = 1

    if schema_version in {"1.4", "1.6"}:
        topology = _object(
            document.get("topology_expectations"), "topology_expectations"
        )
        n_layer = 0
        v_trans = None
        max_n_layer = _positive(topology.get("max_n_layer"), "max_n_layer")
        _positive(topology.get("max_n_pos_per_embd"), "max_n_pos_per_embd")
        _exact(topology.get("n_stream"), 4, "n_stream")
        _exact(topology.get("n_seq_max"), 4, "n_seq_max")
    else:
        topology = _object(document.get("topology"), "topology")
        n_layer = _positive(topology.get("n_layer"), "n_layer")
        raw_v_trans = topology.get("v_trans")
        if not isinstance(raw_v_trans, bool):
            raise RuntimeStateProfileError("v_trans must be a boolean")
        v_trans = raw_v_trans
        if schema_version == "1.5":
            topology_policy = _object(
                document.get("topology_policy"), "topology_policy"
            )
            for field, expected in {
                "max_n_layer": 64,
                "max_n_pos_per_embd": 16,
                "max_tensor_entries": 1024,
                "wire_version": 4,
            }.items():
                _exact(topology_policy.get(field), expected, field)
            _exact(
                topology_policy.get("supported_n_stream"),
                [4, 8],
                "supported_n_stream",
            )
            max_n_layer = 64
            n_pos_per_embd = _positive(
                topology.get("n_pos_per_embd"), "n_pos_per_embd"
            )
            if n_pos_per_embd > 16:
                raise RuntimeStateProfileError(
                    "n_pos_per_embd exceeds max_n_pos_per_embd"
                )
            required_tensor_entries = _positive(
                topology.get("required_tensor_entries"),
                "required_tensor_entries",
            )
            policy_n_stream = _positive(topology.get("n_stream"), "n_stream")
            if required_tensor_entries != 2 * n_layer * policy_n_stream:
                raise RuntimeStateProfileError(
                    "required_tensor_entries does not match topology"
                )
        else:
            for field, expected in {
                "n_stream": 4,
                "n_seq_max": 4,
                "n_layer": 24,
                "n_pos_per_embd": 1,
                "v_trans": True,
            }.items():
                _exact(topology.get(field), expected, field)
            max_n_layer = 24
    n_stream = _positive(topology.get("n_stream"), "n_stream")
    n_seq_max = _positive(topology.get("n_seq_max"), "n_seq_max")
    if n_stream != n_seq_max:
        raise RuntimeStateProfileError("n_seq_max must equal n_stream")
    if schema_version == "1.5" and n_stream not in {4, 8}:
        raise RuntimeStateProfileError("n_stream is outside the bounded policy")
    if n_layer and n_layer > max_n_layer:
        raise RuntimeStateProfileError("n_layer exceeds max_n_layer")

    return RehostRuntimeProfile(
        profile_digest=_canonical_digest(document),
        runtime_binary_digest=runtime_binary_digest,
        mapped_libllama_digest=mapped_libllama_digest,
        mapped_libllama_elf_build_id=build_id,
        n_seq_max=n_seq_max,
        n_stream=n_stream,
        n_layer=n_layer,
        v_trans=v_trans,
        key_ggml_type_id=key_ggml_type_id,
        value_ggml_type_id=value_ggml_type_id,
        max_n_layer=max_n_layer,
    )
