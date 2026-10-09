from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Final


PINNED_LLAMA_CPP_COMMIT: Final[str] = (
    "daf6bc9f2d3516d7bda73d191c6a01388aff09c5"
)
QUALIFIED_MODEL_PROFILE_ID: Final[str] = "qwen2.5-0.5b-q4"
QUALIFIED_MODEL_SHA256: Final[str] = (
    "sha256:6eb923e7d26e9cea28811e1a8e852009"
    "b21242fb157b26149d3b188f3a8c8653"
)
GENERIC_REQUIRED_LAUNCH_ARGUMENTS: Final[tuple[str, ...]] = (
    "--parallel",
    "4",
    "--ctx-size",
    "8192",
    "--no-kv-unified",
    "--flash-attn",
    "off",
    "--cache-type-k",
    "f16",
    "--cache-type-v",
    "f16",
    "--batch-size",
    "128",
    "--ubatch-size",
    "128",
    "--n-gpu-layers",
    "0",
    "--no-cache-idle-slots",
    "--no-context-shift",
)
REQUIRED_LAUNCH_ARGUMENTS: Final[tuple[str, ...]] = (
    *GENERIC_REQUIRED_LAUNCH_ARGUMENTS,
    "--log-disable",
)
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_BUILD_ID = re.compile(r"[0-9a-f]{40}\Z")
_FORBIDDEN_MUTATIONS: Final[tuple[str, ...]] = (
    "cache_shift",
    "context_shift",
    "kv_defragmentation",
    "multimodal_position",
    "sequence_copy",
    "sequence_remove",
    "sequence_stream_remap",
    "speculative_context",
)
_BASE_PROBES: Final[tuple[str, ...]] = (
    "kv_constructor",
    "n_pos_per_embd",
    "view_2d",
    "buffer_base",
    "buffer_size",
    "apply_ubatch",
)
_DYNAMIC_SEQUENCE_PROBES: Final[tuple[str, ...]] = (
    "sequence_copy",
    "sequence_remove",
)
_STATIC_FORBIDDEN_MUTATION_ATTESTATIONS: Final[tuple[str, ...]] = (
    "cache_shift",
    "context_shift",
    "kv_defragmentation",
    "multimodal_position",
    "sequence_stream_remap",
    "speculative_context",
)
_REQUIRED_PROBES: Final[frozenset[str]] = frozenset(
    (*_BASE_PROBES, *_DYNAMIC_SEQUENCE_PROBES)
)
_PROCESS_UBATCH_PROBE_ID: Final[str] = "process_ubatch"
_STATIC_QUALIFICATION_RECEIPT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "abi_probe_binary_sha256",
        "abi_probe_stdout_sha256",
        "pending_profile_digest",
        "phase",
        "qualification_material",
        "receipt_digest",
        "receipt_type",
        "runtime_expected_contract_digest",
        "runtime_input_contract_digest",
        "runtime_manifest_digest",
        "schema_version",
    }
)
_STATIC_QUALIFICATION_MATERIAL_FIELDS: Final[frozenset[str]] = frozenset(
    {"abi_manifest", "probe_targets", "qualified_toolchain_profile"}
)
_MAXIMUM_QUALIFICATION_RECEIPT_BYTES: Final[int] = 1024 * 1024
_PROBE_TARGET_CONTRACT: Final[dict[str, tuple[str, str, str, str]]] = {
    "kv_constructor": (
        "libllama",
        "/opt/laris/llama-runtime/lib/libllama.so.0",
        "_ZN14llama_kv_cacheC2ERK11llama_modelRK13llama_hparams9ggml_typeS6_bbbjjjj14llama_swa_typeP14llama_memory_iRKSt8functionIFbiEERKSA_IFiiEESI_",
        "entry_return",
    ),
    "n_pos_per_embd": (
        "libllama",
        "/opt/laris/llama-runtime/lib/libllama.so.0",
        "_ZNK13llama_hparams14n_pos_per_embdEv",
        "entry_return",
    ),
    "view_2d": (
        "libggml",
        "/opt/laris/llama-runtime/lib/libggml-base.so.0",
        "ggml_view_2d",
        "entry_return",
    ),
    "buffer_base": (
        "libggml",
        "/opt/laris/llama-runtime/lib/libggml-base.so.0",
        "ggml_backend_buffer_get_base",
        "entry_return",
    ),
    "buffer_size": (
        "libggml",
        "/opt/laris/llama-runtime/lib/libggml-base.so.0",
        "ggml_backend_buffer_get_size",
        "entry_return",
    ),
    "apply_ubatch": (
        "libllama",
        "/opt/laris/llama-runtime/lib/libllama.so.0",
        "_ZN14llama_kv_cache12apply_ubatchERKNS_9slot_infoERK12llama_ubatch",
        "entry_return",
    ),
    "process_ubatch": (
        "libllama",
        "/opt/laris/llama-runtime/lib/libllama.so.0",
        (
            "_ZN13llama_context14process_ubatchERK12llama_ubatch"
            "14llm_graph_typeP22llama_memory_context_iR11ggml_status"
        ),
        "entry_return",
    ),
    "sequence_copy": (
        "libllama",
        "/opt/laris/llama-runtime/lib/libllama.so.0",
        "llama_memory_seq_cp",
        "entry",
    ),
    "sequence_remove": (
        "libllama",
        "/opt/laris/llama-runtime/lib/libllama.so.0",
        "llama_memory_seq_rm",
        "entry",
    ),
}


class RuntimeStateProfileError(ValueError):
    """Raised when a runtime-state observer profile is not qualified."""


@dataclass(frozen=True)
class RuntimeStateAbiOffsets:
    slot_info_strm: int
    slot_info_idxs: int
    ubatch_n_tokens: int
    ubatch_pos: int
    ubatch_n_seq_id: int
    ubatch_seq_id: int
    tensor_data: int
    tensor_type: int
    cache_hparams: int
    cache_v_trans: int
    cache_n_seq_max: int
    cache_n_stream: int
    cache_layers: int
    kv_layer_size: int
    kv_layer_k_stream: int
    kv_layer_v_stream: int
    hparams_rope_type: int
    vector_begin: int
    vector_end: int
    tensor_ne: int
    tensor_nb: int
    tensor_view_src: int
    tensor_view_offs: int


@dataclass(frozen=True)
class RuntimeStateSymbols:
    kv_constructor: str
    apply_ubatch: str
    kv_buffer: str


@dataclass(frozen=True)
class RuntimeStateProbeTarget:
    probe_id: str
    owner_role: str
    path: PurePosixPath
    sha256: str
    elf_build_id: str
    symbol: str
    resolved_file_offset: int
    attachment: str


@dataclass(frozen=True)
class RuntimeStateAbiManifest:
    compiler_id: str
    compiler_flags: tuple[str, ...]
    compiler_flags_sha256: str
    pinned_headers_sha256: str
    measured_offsets: RuntimeStateAbiOffsets


@dataclass(frozen=True)
class RuntimeStateObserverProfile:
    schema_version: str
    profile_id: str
    qualified_model_profile_id: str | None
    qualified_model_sha256: str | None
    qualified_model_path: PurePosixPath | None
    adapter_kind: str
    llama_cpp_commit: str
    os: str
    architecture: str
    abi: str
    memory_family: str
    qualification_state: str
    qualification_reason: str
    runtime_binary_path: PurePosixPath
    required_launch_arguments: tuple[str, ...]
    runtime_binary_sha256: str
    mapped_libllama_sha256: str
    mapped_libllama_elf_build_id: str
    symbols: RuntimeStateSymbols
    offsets: RuntimeStateAbiOffsets
    max_apply_tokens: int
    supported_n_stream: tuple[int, ...]
    max_tensor_entries: int
    wire_version: int
    n_stream: int
    n_seq_max: int
    n_layer: int
    n_pos_per_embd: int
    v_trans: bool | None
    required_tensor_entries: int
    max_n_layer: int
    max_n_pos_per_embd: int
    cache_prompt: bool
    cache_idle_slots: bool
    context_shift: bool
    requires_full_sequence_rewrite_per_lease: bool
    key_cache_type: str
    value_cache_type: str
    key_ggml_type_id: int
    value_ggml_type_id: int
    forbidden_mutations: tuple[str, ...]
    static_forbidden_mutation_attestations: tuple[str, ...]
    probe_targets: tuple[RuntimeStateProbeTarget, ...]
    abi_manifest: RuntimeStateAbiManifest | None
    profile_digest: str

    @property
    def is_linux_live_qualified(self) -> bool:
        return self.qualification_state == "linux_live_qualified"

    @property
    def is_linux_live_attachable(self) -> bool:
        return self.qualification_state in {
            "linux_live_capture",
            "linux_live_qualified",
        }

    @property
    def is_model_generic(self) -> bool:
        return self.schema_version in {"1.4", "1.5"}

    @property
    def has_dynamic_topology(self) -> bool:
        return self.schema_version == "1.4" or (
            self.schema_version == "1.5"
            and self.qualification_state == "linux_live_capture"
        )


@dataclass(frozen=True)
class RuntimeStateQualificationProbeAuthority:
    """One in-memory probe profile bound to its validated static receipt."""

    profile: RuntimeStateObserverProfile
    static_receipt_digest: str


_TOP_LEVEL_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "profile_id",
        "qualified_model_profile_id",
        "qualified_model_sha256",
        "qualified_model_path",
        "llama_cpp_commit",
        "os",
        "architecture",
        "abi",
        "memory_family",
        "qualification",
        "runtime_identity",
        "symbols",
        "abi_offsets",
        "limits",
        "topology",
        "runtime_policy",
        "kv_cache_types",
        "forbidden_mutations",
        "static_forbidden_mutation_attestations",
        "probe_targets",
        "abi_manifest",
    }
)
_TOP_LEVEL_FIELDS_14: Final[frozenset[str]] = (
    _TOP_LEVEL_FIELDS
    - {
        "qualified_model_profile_id",
        "qualified_model_sha256",
        "qualified_model_path",
        "topology",
    }
    | {"adapter_kind", "topology_expectations"}
)
_TOP_LEVEL_FIELDS_15: Final[frozenset[str]] = (
    _TOP_LEVEL_FIELDS_14
    - {"topology_expectations"}
    | {"topology_policy", "topology"}
)


def _require_exact_fields(
    value: object, expected: frozenset[str], field: str
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise RuntimeStateProfileError(f"{field} must be an object")
    actual = set(value)
    if actual != expected:
        raise RuntimeStateProfileError(
            f"{field} fields differ: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )
    return value


def _require_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeStateProfileError(f"{field} must be a non-empty string")
    return value


def _require_digest(value: object, field: str) -> str:
    text = _require_text(value, field)
    if _SHA256.fullmatch(text) is None:
        raise RuntimeStateProfileError(
            f"{field} must be lowercase sha256:<64 hex>"
        )
    return text


def _require_exact(value: object, expected: object, field: str) -> object:
    if type(value) is not type(expected) or value != expected:
        raise RuntimeStateProfileError(f"{field} must equal {expected!r}")
    return value


def _require_nonnegative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeStateProfileError(f"{field} must be a non-negative integer")
    return value


def _require_positive_int(value: object, field: str) -> int:
    result = _require_nonnegative_int(value, field)
    if result == 0:
        raise RuntimeStateProfileError(f"{field} must be positive")
    return result


def _bounded_launch_arguments(n_stream: int) -> tuple[str, ...]:
    arguments = list(GENERIC_REQUIRED_LAUNCH_ARGUMENTS)
    arguments[arguments.index("--parallel") + 1] = str(n_stream)
    arguments[arguments.index("--ctx-size") + 1] = str(n_stream * 1024)
    return tuple(arguments)


def _require_absolute_path(value: object, field: str) -> PurePosixPath:
    path = PurePosixPath(_require_text(value, field))
    if not path.is_absolute():
        raise RuntimeStateProfileError(f"{field} must be an absolute path")
    return path


def _require_string_list(value: object, field: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
        or len(set(value)) != len(value)
    ):
        raise RuntimeStateProfileError(
            f"{field} must be a non-empty unique string list"
        )
    return tuple(value)


def _require_launch_argument_list(
    value: object, field: str
) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise RuntimeStateProfileError(
            f"{field} must be a non-empty split-form argv list"
        )
    arguments = tuple(value)
    options: set[str] = set()
    index = 0
    while index < len(arguments):
        option = arguments[index]
        if not option.startswith("--") or "=" in option or option in options:
            raise RuntimeStateProfileError(
                f"{field} must contain unique canonical split-form options"
            )
        options.add(option)
        if index + 1 < len(arguments) and not arguments[index + 1].startswith("-"):
            index += 1
        index += 1
    return arguments


def _canonical_digest(document: dict[str, object]) -> str:
    encoded = json.dumps(
        document,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def runtime_state_profile_from_document(
    document: object,
) -> RuntimeStateObserverProfile:
    if not isinstance(document, dict):
        raise RuntimeStateProfileError("profile must be an object")
    schema_version = document.get("schema_version")
    expected_top_level = {
        "1.4": _TOP_LEVEL_FIELDS_14,
        "1.5": _TOP_LEVEL_FIELDS_15,
    }.get(str(schema_version), _TOP_LEVEL_FIELDS)
    top = _require_exact_fields(
        document,
        expected_top_level,
        "profile",
    )
    qualification = _require_exact_fields(
        top["qualification"], frozenset({"state", "reason"}), "qualification"
    )
    qualification_state = _require_text(
        qualification["state"], "qualification.state"
    )
    allowed_qualification_states = {
        "linux_live_pending",
        "linux_live_qualified",
    }
    if schema_version == "1.5":
        allowed_qualification_states.add("linux_live_capture")
    if qualification_state not in allowed_qualification_states:
        raise RuntimeStateProfileError(
            "qualification.state is invalid for this profile schema"
        )
    qualification_reason = _require_text(
        qualification["reason"], "qualification.reason"
    )
    identity = _require_exact_fields(
        top["runtime_identity"],
        frozenset(
            {
                "runtime_binary_path",
                "runtime_binary_sha256",
                "mapped_libllama_sha256",
                "mapped_libllama_elf_build_id",
                "required_launch_arguments",
            }
        ),
        "runtime_identity",
    )
    symbols = _require_exact_fields(
        top["symbols"],
        frozenset({"kv_constructor", "apply_ubatch", "kv_buffer"}),
        "symbols",
    )
    offsets = _require_exact_fields(
        top["abi_offsets"],
        frozenset(
            {
                "slot_info_strm",
                "slot_info_idxs",
                "ubatch_n_tokens",
                "ubatch_pos",
                "ubatch_n_seq_id",
                "ubatch_seq_id",
                "tensor_data",
                "tensor_type",
                "cache_hparams",
                "cache_v_trans",
                "cache_n_seq_max",
                "cache_n_stream",
                "cache_layers",
                "kv_layer_size",
                "kv_layer_k_stream",
                "kv_layer_v_stream",
                "hparams_rope_type",
                "vector_begin",
                "vector_end",
                "tensor_ne",
                "tensor_nb",
                "tensor_view_src",
                "tensor_view_offs",
            }
        ),
        "abi_offsets",
    )
    limits = _require_exact_fields(
        top["limits"], frozenset({"max_apply_tokens"}), "limits"
    )
    if schema_version == "1.5":
        topology_policy = _require_exact_fields(
            top["topology_policy"],
            frozenset(
                {
                    "supported_n_stream",
                    "max_n_layer",
                    "max_n_pos_per_embd",
                    "max_tensor_entries",
                    "wire_version",
                }
            ),
            "topology_policy",
        )
        topology = _require_exact_fields(
            top["topology"],
            frozenset(
                {
                    "n_stream",
                    "n_seq_max",
                    "n_layer",
                    "n_pos_per_embd",
                    "v_trans",
                    "required_tensor_entries",
                }
            ),
            "topology",
        )
    elif schema_version == "1.4":
        topology = _require_exact_fields(
            top["topology_expectations"],
            frozenset(
                {
                    "n_stream",
                    "n_seq_max",
                    "max_n_layer",
                    "max_n_pos_per_embd",
                }
            ),
            "topology_expectations",
        )
    else:
        topology = _require_exact_fields(
            top["topology"],
            frozenset(
                {"n_stream", "n_seq_max", "n_layer", "n_pos_per_embd", "v_trans"}
            ),
            "topology",
        )
    policy = _require_exact_fields(
        top["runtime_policy"],
        frozenset(
            {
                "cache_prompt",
                "cache_idle_slots",
                "context_shift",
                "requires_full_sequence_rewrite_per_lease",
            }
        ),
        "runtime_policy",
    )
    cache_types = _require_exact_fields(
        top["kv_cache_types"],
        frozenset(
            {"key", "value", "key_ggml_type_id", "value_ggml_type_id"}
        ),
        "kv_cache_types",
    )

    fixed = {
        "schema_version": (
            "1.5" if schema_version == "1.5"
            else "1.4" if schema_version == "1.4"
            else "1.3"
        ),
        "profile_id": (
            "x86_64-cpu-simple-kv-bounded"
            if schema_version == "1.5"
            else "x86_64-cpu-simple-kv"
            if schema_version == "1.4"
            else "x86_64-cpu-simple-kv-qwen2.5-0.5b-q4"
        ),
        "llama_cpp_commit": PINNED_LLAMA_CPP_COMMIT,
        "os": "Linux",
        "architecture": "x86_64",
        "abi": "System V AMD64",
        "memory_family": "simple_kv",
    }
    for field, expected in fixed.items():
        _require_exact(top[field], expected, field)
    if schema_version in {"1.4", "1.5"}:
        _require_exact(
            top["adapter_kind"],
            "llama_cpp_simple_kv",
            "adapter_kind",
        )
        qualified_model_profile_id = None
        qualified_model_sha256 = None
        qualified_model_path = None
        adapter_kind = "llama_cpp_simple_kv"
    else:
        _require_exact(
            top["qualified_model_profile_id"],
            QUALIFIED_MODEL_PROFILE_ID,
            "qualified_model_profile_id",
        )
        qualified_model_profile_id = QUALIFIED_MODEL_PROFILE_ID
        qualified_model_sha256 = _require_digest(
            top["qualified_model_sha256"], "qualified_model_sha256"
        )
        _require_exact(
            qualified_model_sha256,
            QUALIFIED_MODEL_SHA256,
            "qualified_model_sha256",
        )
        qualified_model_path = _require_absolute_path(
            top["qualified_model_path"], "qualified_model_path"
        )
        adapter_kind = "llama_cpp_simple_kv"

    symbol_values = {
        "kv_constructor": "llama_kv_cache::llama_kv_cache",
        "apply_ubatch": "llama_kv_cache::apply_ubatch",
        "kv_buffer": "ggml_backend_buffer_get_base",
    }
    for field, expected in symbol_values.items():
        _require_exact(symbols[field], expected, field)

    offset_values = {
        "slot_info_strm": 8,
        "slot_info_idxs": 32,
        "ubatch_n_tokens": 4,
        "ubatch_pos": 40,
        "ubatch_n_seq_id": 48,
        "ubatch_seq_id": 56,
        "tensor_data": 248,
        "tensor_type": 0,
        "cache_hparams": 16,
        "cache_v_trans": 24,
        "cache_n_seq_max": 28,
        "cache_n_stream": 32,
        "cache_layers": 272,
        "kv_layer_size": 72,
        "kv_layer_k_stream": 24,
        "kv_layer_v_stream": 48,
        "hparams_rope_type": 20936,
        "vector_begin": 0,
        "vector_end": 8,
        "tensor_ne": 16,
        "tensor_nb": 48,
        "tensor_view_src": 232,
        "tensor_view_offs": 240,
    }
    for field, expected in offset_values.items():
        _require_exact(offsets[field], expected, field)

    max_apply_tokens = 2048 if schema_version in {"1.4", "1.5"} else 128
    _require_exact(
        limits["max_apply_tokens"],
        max_apply_tokens,
        "max_apply_tokens",
    )
    if schema_version == "1.5":
        supported_raw = topology_policy["supported_n_stream"]
        if supported_raw != [4, 8]:
            raise RuntimeStateProfileError(
                "supported_n_stream must equal the bounded 4/8 stream set"
            )
        supported_n_stream = (4, 8)
        max_n_layer = _require_positive_int(
            topology_policy["max_n_layer"], "max_n_layer"
        )
        max_n_pos_per_embd = _require_positive_int(
            topology_policy["max_n_pos_per_embd"], "max_n_pos_per_embd"
        )
        max_tensor_entries = _require_positive_int(
            topology_policy["max_tensor_entries"], "max_tensor_entries"
        )
        wire_version = _require_positive_int(
            topology_policy["wire_version"], "wire_version"
        )
        _require_exact(max_n_layer, 64, "max_n_layer")
        _require_exact(max_n_pos_per_embd, 16, "max_n_pos_per_embd")
        _require_exact(max_tensor_entries, 1024, "max_tensor_entries")
        _require_exact(wire_version, 4, "wire_version")

        n_stream = _require_positive_int(topology["n_stream"], "n_stream")
        if n_stream not in supported_n_stream:
            raise RuntimeStateProfileError("n_stream is outside the bounded policy")
        n_seq_max = _require_positive_int(topology["n_seq_max"], "n_seq_max")
        if n_seq_max != n_stream:
            raise RuntimeStateProfileError("n_seq_max must equal n_stream")
        n_layer = _require_nonnegative_int(topology["n_layer"], "n_layer")
        n_pos_per_embd = _require_nonnegative_int(
            topology["n_pos_per_embd"], "n_pos_per_embd"
        )
        v_trans_value = topology["v_trans"]
        required_tensor_entries = _require_nonnegative_int(
            topology["required_tensor_entries"], "required_tensor_entries"
        )
        if qualification_state == "linux_live_qualified":
            n_layer = _require_positive_int(topology["n_layer"], "n_layer")
            n_pos_per_embd = _require_positive_int(
                topology["n_pos_per_embd"], "n_pos_per_embd"
            )
            if not isinstance(v_trans_value, bool):
                raise RuntimeStateProfileError("v_trans must be a boolean")
            v_trans = v_trans_value
            if n_layer > max_n_layer:
                raise RuntimeStateProfileError("n_layer exceeds topology policy")
            if n_pos_per_embd > max_n_pos_per_embd:
                raise RuntimeStateProfileError(
                    "n_pos_per_embd exceeds topology policy"
                )
            expected_tensor_entries = 2 * n_stream * n_layer
            if required_tensor_entries != expected_tensor_entries:
                raise RuntimeStateProfileError(
                    "required_tensor_entries does not match the live tensor topology"
                )
            if required_tensor_entries > max_tensor_entries:
                raise RuntimeStateProfileError(
                    "required_tensor_entries exceeds max_tensor_entries"
                )
        else:
            if (
                n_layer != 0
                or n_pos_per_embd != 0
                or v_trans_value is not None
                or required_tensor_entries != 0
            ):
                raise RuntimeStateProfileError(
                    "unqualified topology must not publish live topology facts"
                )
            v_trans = None
    elif schema_version == "1.4":
        for field, expected in {"n_stream": 4, "n_seq_max": 4}.items():
            _require_exact(topology[field], expected, field)
        max_n_layer = _require_nonnegative_int(
            topology["max_n_layer"], "max_n_layer"
        )
        max_n_pos_per_embd = _require_nonnegative_int(
            topology["max_n_pos_per_embd"], "max_n_pos_per_embd"
        )
        if max_n_layer == 0 or max_n_pos_per_embd == 0:
            raise RuntimeStateProfileError(
                "dynamic topology maxima must be positive"
            )
        topology_values = {"n_stream": 4, "n_seq_max": 4}
        supported_n_stream = (4,)
        max_tensor_entries = 256
        wire_version = 3
        required_tensor_entries = 0
        n_stream = 4
        n_seq_max = 4
        n_layer = 0
        n_pos_per_embd = 0
        v_trans = None
    else:
        topology_values = {
            "n_stream": 4,
            "n_seq_max": 4,
            "n_layer": 24,
            "n_pos_per_embd": 1,
            "v_trans": True,
        }
        for field, expected in topology_values.items():
            _require_exact(topology[field], expected, field)
        max_n_layer = 24
        max_n_pos_per_embd = 1
        supported_n_stream = (4,)
        max_tensor_entries = 256
        wire_version = 3
        required_tensor_entries = 192
        n_stream = 4
        n_seq_max = 4
        n_layer = 24
        n_pos_per_embd = 1
        v_trans = True
    policy_values = {
        "cache_prompt": False,
        "cache_idle_slots": False,
        "context_shift": False,
        "requires_full_sequence_rewrite_per_lease": True,
    }
    for field, expected in policy_values.items():
        _require_exact(policy[field], expected, field)
    _require_exact(cache_types["key"], "f16", "key cache type")
    _require_exact(cache_types["value"], "f16", "value cache type")
    _require_exact(cache_types["key_ggml_type_id"], 1, "key_ggml_type_id")
    _require_exact(cache_types["value_ggml_type_id"], 1, "value_ggml_type_id")

    forbidden = top["forbidden_mutations"]
    if not isinstance(forbidden, list) or tuple(forbidden) != _FORBIDDEN_MUTATIONS:
        raise RuntimeStateProfileError(
            "forbidden_mutations must equal the qualified mutation set"
        )
    static_attestations = top["static_forbidden_mutation_attestations"]
    if (
        not isinstance(static_attestations, list)
        or tuple(static_attestations)
        != _STATIC_FORBIDDEN_MUTATION_ATTESTATIONS
    ):
        raise RuntimeStateProfileError(
            "static_forbidden_mutation_attestations must equal the qualified static set"
        )
    runtime_binary = _require_digest(
        identity["runtime_binary_sha256"], "runtime_binary_sha256"
    )
    _require_exact(
        identity["runtime_binary_path"],
        "/opt/laris/llama-runtime/bin/llama-server",
        "runtime_binary_path",
    )
    runtime_binary_path = _require_absolute_path(
        identity["runtime_binary_path"], "runtime_binary_path"
    )
    required_launch_arguments = _require_launch_argument_list(
        identity["required_launch_arguments"], "required_launch_arguments"
    )
    expected_launch_arguments = (
        _bounded_launch_arguments(n_stream)
        if schema_version == "1.5"
        else GENERIC_REQUIRED_LAUNCH_ARGUMENTS
        if schema_version == "1.4"
        else REQUIRED_LAUNCH_ARGUMENTS
    )
    if required_launch_arguments != expected_launch_arguments:
        raise RuntimeStateProfileError(
            "required_launch_arguments must equal the qualified launch policy"
        )
    mapped_lib = _require_digest(
        identity["mapped_libllama_sha256"], "mapped_libllama_sha256"
    )
    build_id = _require_text(
        identity["mapped_libllama_elf_build_id"],
        "mapped_libllama_elf_build_id",
    )
    if _BUILD_ID.fullmatch(build_id) is None:
        raise RuntimeStateProfileError(
            "mapped_libllama_elf_build_id must be 40 lowercase hex characters"
        )

    raw_targets = top["probe_targets"]
    if not isinstance(raw_targets, list):
        raise RuntimeStateProfileError("probe_targets must be an array")
    probe_targets: list[RuntimeStateProbeTarget] = []
    for index, value in enumerate(raw_targets):
        fields = _require_exact_fields(
            value,
            frozenset(
                {
                    "probe_id",
                    "owner_role",
                    "path",
                    "sha256",
                    "elf_build_id",
                    "symbol",
                    "resolved_file_offset",
                    "attachment",
                }
            ),
            f"probe_targets[{index}]",
        )
        probe_id = _require_text(fields["probe_id"], f"probe_targets[{index}].probe_id")
        role = _require_text(fields["owner_role"], f"probe_targets[{index}].owner_role")
        if role not in {"libllama", "libggml"}:
            raise RuntimeStateProfileError("probe target owner_role is invalid")
        target_build_id = _require_text(
            fields["elf_build_id"], f"probe_targets[{index}].elf_build_id"
        )
        if _BUILD_ID.fullmatch(target_build_id) is None:
            raise RuntimeStateProfileError("probe target elf_build_id is invalid")
        attachment = _require_text(
            fields["attachment"], f"probe_targets[{index}].attachment"
        )
        if attachment not in {"entry", "entry_return"}:
            raise RuntimeStateProfileError("probe target attachment is invalid")
        probe_targets.append(
            RuntimeStateProbeTarget(
                probe_id=probe_id,
                owner_role=role,
                path=_require_absolute_path(
                    fields["path"], f"probe_targets[{index}].path"
                ),
                sha256=_require_digest(
                    fields["sha256"], f"probe_targets[{index}].sha256"
                ),
                elf_build_id=target_build_id,
                symbol=_require_text(
                    fields["symbol"], f"probe_targets[{index}].symbol"
                ),
                resolved_file_offset=_require_nonnegative_int(
                    fields["resolved_file_offset"],
                    f"probe_targets[{index}].resolved_file_offset",
                ),
                attachment=attachment,
            )
        )
    target_ids = [target.probe_id for target in probe_targets]
    if len(target_ids) != len(set(target_ids)):
        raise RuntimeStateProfileError("probe_targets probe_id values must be unique")
    for target in probe_targets:
        expected = _PROBE_TARGET_CONTRACT.get(target.probe_id)
        observed = (
            target.owner_role,
            str(target.path),
            target.symbol,
            target.attachment,
        )
        if expected is None or observed != expected:
            raise RuntimeStateProfileError(
                f"probe target {target.probe_id} differs from the pinned ELF symbol contract"
            )

    raw_abi = top["abi_manifest"]
    abi_manifest: RuntimeStateAbiManifest | None = None
    if raw_abi is not None:
        abi = _require_exact_fields(
            raw_abi,
            frozenset(
                {
                    "compiler_id",
                    "compiler_flags",
                    "compiler_flags_sha256",
                    "pinned_headers_sha256",
                    "measured_offsets",
                }
            ),
            "abi_manifest",
        )
        measured = _require_exact_fields(
            abi["measured_offsets"], frozenset(offset_values), "abi_manifest.measured_offsets"
        )
        for field, expected in offset_values.items():
            if measured[field] != expected:
                raise RuntimeStateProfileError(
                    f"ABI measured offset mismatch for {field}"
                )
        compiler_flags = _require_string_list(
            abi["compiler_flags"], "abi_manifest.compiler_flags"
        )
        compiler_flags_sha256 = _require_digest(
            abi["compiler_flags_sha256"], "abi_manifest.compiler_flags_sha256"
        )
        expected_flags_digest = "sha256:" + hashlib.sha256(
            "|".join(compiler_flags).encode("utf-8")
        ).hexdigest()
        if compiler_flags_sha256 != expected_flags_digest:
            raise RuntimeStateProfileError(
                "abi_manifest.compiler_flags_sha256 does not cover compiler_flags"
            )
        abi_manifest = RuntimeStateAbiManifest(
            compiler_id=_require_text(abi["compiler_id"], "abi_manifest.compiler_id"),
            compiler_flags=compiler_flags,
            compiler_flags_sha256=compiler_flags_sha256,
            pinned_headers_sha256=_require_digest(
                abi["pinned_headers_sha256"], "abi_manifest.pinned_headers_sha256"
            ),
            measured_offsets=RuntimeStateAbiOffsets(**offset_values),
        )
    if qualification_state in {"linux_live_capture", "linux_live_qualified"}:
        allowed_probe_sets = {
            _REQUIRED_PROBES,
            _REQUIRED_PROBES | {_PROCESS_UBATCH_PROBE_ID},
        }
        if set(target_ids) not in allowed_probe_sets:
            raise RuntimeStateProfileError(
                "probe_targets must be complete for linux_live_qualified"
            )
        if abi_manifest is None:
            raise RuntimeStateProfileError(
                "abi_manifest is required for linux_live_qualified"
            )
        role_identities: dict[str, set[tuple[str, str, PurePosixPath]]] = {}
        for target in probe_targets:
            role_identities.setdefault(target.owner_role, set()).add(
                (target.sha256, target.elf_build_id, target.path)
            )
        if any(len(values) != 1 for values in role_identities.values()):
            raise RuntimeStateProfileError(
                "each probe owner_role must resolve to one ELF identity"
            )
        if (
            "libllama" not in role_identities
            or next(iter(role_identities["libllama"]))[:2]
            != (mapped_lib, build_id)
        ):
            raise RuntimeStateProfileError(
                "libllama probe identity contradicts runtime_identity"
            )
    elif probe_targets or abi_manifest is not None:
        raise RuntimeStateProfileError(
            "linux_live_pending must not publish unverified probe or ABI facts"
        )

    return RuntimeStateObserverProfile(
        schema_version=str(schema_version),
        profile_id=str(top["profile_id"]),
        qualified_model_profile_id=qualified_model_profile_id,
        qualified_model_sha256=qualified_model_sha256,
        qualified_model_path=qualified_model_path,
        adapter_kind=adapter_kind,
        llama_cpp_commit=PINNED_LLAMA_CPP_COMMIT,
        os="Linux",
        architecture="x86_64",
        abi="System V AMD64",
        memory_family="simple_kv",
        qualification_state=qualification_state,
        qualification_reason=qualification_reason,
        runtime_binary_path=runtime_binary_path,
        required_launch_arguments=required_launch_arguments,
        runtime_binary_sha256=runtime_binary,
        mapped_libllama_sha256=mapped_lib,
        mapped_libllama_elf_build_id=build_id,
        symbols=RuntimeStateSymbols(**symbol_values),
        offsets=RuntimeStateAbiOffsets(**offset_values),
        max_apply_tokens=max_apply_tokens,
        supported_n_stream=supported_n_stream,
        max_tensor_entries=max_tensor_entries,
        wire_version=wire_version,
        n_stream=n_stream,
        n_seq_max=n_seq_max,
        n_layer=n_layer,
        n_pos_per_embd=n_pos_per_embd,
        v_trans=v_trans,
        required_tensor_entries=required_tensor_entries,
        max_n_layer=max_n_layer,
        max_n_pos_per_embd=max_n_pos_per_embd,
        cache_prompt=False,
        cache_idle_slots=False,
        context_shift=False,
        requires_full_sequence_rewrite_per_lease=True,
        key_cache_type="f16",
        value_cache_type="f16",
        key_ggml_type_id=1,
        value_ggml_type_id=1,
        forbidden_mutations=_FORBIDDEN_MUTATIONS,
        static_forbidden_mutation_attestations=(
            _STATIC_FORBIDDEN_MUTATION_ATTESTATIONS
        ),
        probe_targets=tuple(probe_targets),
        abi_manifest=abi_manifest,
        profile_digest=_canonical_digest(top),
    )


def load_runtime_state_observer_profile(
    path: Path,
) -> RuntimeStateObserverProfile:
    if not isinstance(path, Path):
        raise RuntimeStateProfileError("profile path must be a pathlib.Path")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeStateProfileError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        document = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates
        )
    except RuntimeStateProfileError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeStateProfileError("invalid runtime-state profile JSON") from error
    return runtime_state_profile_from_document(document)


def validate_runtime_state_capture_mode(
    profile: RuntimeStateObserverProfile,
    capture_mode: str,
) -> None:
    """Fail closed when an opt-in capture mode lacks qualified probe authority."""

    if capture_mode not in {
        "metadata",
        "writer-boundary",
        "gather-writer-boundary",
        "compact-metadata",
    }:
        raise RuntimeStateProfileError("capture_mode is invalid")
    if capture_mode in {"metadata", "compact-metadata"}:
        return
    process_targets = tuple(
        target
        for target in getattr(profile, "probe_targets", ())
        if target.probe_id == _PROCESS_UBATCH_PROBE_ID
    )
    if len(process_targets) != 1:
        raise RuntimeStateProfileError(
            "writer-boundary requires one qualified process_ubatch probe target"
        )


def _qualification_json_document(
    path: Path,
    *,
    label: str,
    maximum_bytes: int,
) -> tuple[bytes, dict[str, object]]:
    if not isinstance(path, Path) or not path.is_absolute():
        raise RuntimeStateProfileError(f"{label} path must be absolute")

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
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > maximum_bytes
        ):
            raise RuntimeStateProfileError(f"{label} identity is invalid")
        chunks: list[bytes] = []
        size = 0
        while True:
            block = os.read(
                descriptor, min(64 * 1024, maximum_bytes + 1 - size)
            )
            if not block:
                break
            chunks.append(block)
            size += len(block)
            if size > maximum_bytes:
                raise RuntimeStateProfileError(f"{label} size is invalid")
        after = os.fstat(descriptor)
        current = os.stat(path, follow_symlinks=False)
        if (
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or (current.st_dev, current.st_ino)
            != (after.st_dev, after.st_ino)
        ):
            raise RuntimeStateProfileError(f"{label} changed while read")
        payload = b"".join(chunks)
        document = json.loads(
            payload.decode("ascii"), object_pairs_hook=reject_duplicates
        )
    except RuntimeStateProfileError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeStateProfileError(f"invalid {label} JSON") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if not isinstance(document, dict):
        raise RuntimeStateProfileError(f"{label} must be a JSON object")
    return payload, document


def runtime_state_qualification_probe_profile_from_documents(
    pending_profile_document: object,
    static_receipt_document: object,
) -> RuntimeStateObserverProfile:
    """Derive the exact future profile only in memory for one live probe.

    This is an explicit probe-authority boundary: the inert checked-in profile
    remains ``linux_live_pending`` and the static receipt remains
    ``linux_live_probe_pending``.  No qualified profile document is returned or
    written; callers receive only the validated in-memory profile required to
    attach the native observer and bind its evidence to the eventual candidate.
    """

    pending = runtime_state_profile_from_document(pending_profile_document)
    if pending.qualification_state != "linux_live_pending":
        raise RuntimeStateProfileError(
            "qualification probe source must be linux_live_pending"
        )
    receipt = _require_exact_fields(
        static_receipt_document,
        _STATIC_QUALIFICATION_RECEIPT_FIELDS,
        "static qualification receipt",
    )
    if (
        receipt["schema_version"] != "1.0"
        or receipt["receipt_type"] != "runtime_state_static_qualification"
        or receipt["phase"] != "linux_live_probe_pending"
    ):
        raise RuntimeStateProfileError(
            "static qualification receipt type or phase is invalid"
        )
    for field in (
        "abi_probe_binary_sha256",
        "abi_probe_stdout_sha256",
        "pending_profile_digest",
        "receipt_digest",
        "runtime_expected_contract_digest",
        "runtime_input_contract_digest",
        "runtime_manifest_digest",
    ):
        _require_digest(receipt[field], f"static qualification receipt {field}")
    body = dict(receipt)
    supplied_receipt_digest = body.pop("receipt_digest")
    if supplied_receipt_digest != _canonical_digest(body):
        raise RuntimeStateProfileError(
            "static qualification receipt digest is invalid"
        )
    if receipt["pending_profile_digest"] != pending.profile_digest:
        raise RuntimeStateProfileError(
            "static qualification receipt pending profile digest differs"
        )
    material = _require_exact_fields(
        receipt["qualification_material"],
        _STATIC_QUALIFICATION_MATERIAL_FIELDS,
        "static qualification material",
    )
    toolchain = material["qualified_toolchain_profile"]
    if not isinstance(toolchain, str) or not toolchain:
        raise RuntimeStateProfileError(
            "static qualification toolchain profile is invalid"
        )
    if not isinstance(pending_profile_document, dict):
        raise RuntimeStateProfileError("pending profile must be a JSON object")
    candidate = copy.deepcopy(pending_profile_document)
    candidate["qualification"] = {
        "reason": (
            "operator-selected pinned build live ELF and ABI qualification"
        ),
        "state": (
            "linux_live_capture"
            if pending.schema_version == "1.5"
            else "linux_live_qualified"
        ),
    }
    candidate["abi_manifest"] = copy.deepcopy(material["abi_manifest"])
    candidate["probe_targets"] = copy.deepcopy(material["probe_targets"])
    return runtime_state_profile_from_document(candidate)


def load_runtime_state_qualification_probe_authority(
    pending_profile_path: Path,
    static_receipt_path: Path,
) -> RuntimeStateQualificationProbeAuthority:
    """Load one root-private profile plus its exact validated receipt digest."""

    _, pending_document = _qualification_json_document(
        pending_profile_path,
        label="pending runtime-state profile",
        maximum_bytes=_MAXIMUM_QUALIFICATION_RECEIPT_BYTES,
    )
    receipt_payload, receipt_document = _qualification_json_document(
        static_receipt_path,
        label="static qualification receipt",
        maximum_bytes=_MAXIMUM_QUALIFICATION_RECEIPT_BYTES,
    )
    canonical_receipt = (
        json.dumps(
            receipt_document,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )
    if receipt_payload != canonical_receipt:
        raise RuntimeStateProfileError(
            "static qualification receipt is not canonical"
        )
    profile = runtime_state_qualification_probe_profile_from_documents(
        pending_document, receipt_document
    )
    receipt_digest = receipt_document.get("receipt_digest")
    if not isinstance(receipt_digest, str):
        raise RuntimeStateProfileError(
            "static qualification receipt digest is invalid"
        )
    return RuntimeStateQualificationProbeAuthority(
        profile=profile,
        static_receipt_digest=receipt_digest,
    )


def load_runtime_state_qualification_probe_profile(
    pending_profile_path: Path,
    static_receipt_path: Path,
) -> RuntimeStateObserverProfile:
    """Load a root-private static receipt and derive its in-memory probe profile."""

    return load_runtime_state_qualification_probe_authority(
        pending_profile_path, static_receipt_path
    ).profile


__all__ = [
    "PINNED_LLAMA_CPP_COMMIT",
    "QUALIFIED_MODEL_PROFILE_ID",
    "QUALIFIED_MODEL_SHA256",
    "REQUIRED_LAUNCH_ARGUMENTS",
    "RuntimeStateAbiOffsets",
    "RuntimeStateAbiManifest",
    "RuntimeStateObserverProfile",
    "RuntimeStateQualificationProbeAuthority",
    "RuntimeStateProbeTarget",
    "RuntimeStateProfileError",
    "RuntimeStateSymbols",
    "load_runtime_state_qualification_probe_authority",
    "load_runtime_state_qualification_probe_profile",
    "load_runtime_state_observer_profile",
    "runtime_state_qualification_probe_profile_from_documents",
    "runtime_state_profile_from_document",
]
