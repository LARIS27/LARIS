from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import re

from laris_onpremise.kv_state.observation.runtime_state_profile import (
    RuntimeStateObserverProfile,
)


class RehostRuntimeMetadataError(ValueError):
    """Raised when source runtime settings cannot be projected safely."""


_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")


@dataclass(frozen=True)
class RehostRuntimeMetadata:
    source_runtime_profile_digest: str
    llama_cpp_commit: str
    memory_family: str
    n_batch: int
    n_ubatch: int
    cache_type_k: str
    cache_type_v: str
    flash_attention: bool
    n_gpu_layers: int
    context_shift: bool
    cache_prompt: bool
    cache_idle_slots: bool

    def __post_init__(self) -> None:
        if _SHA256.fullmatch(self.source_runtime_profile_digest) is None:
            raise RehostRuntimeMetadataError("runtime profile digest is invalid")
        if _COMMIT.fullmatch(self.llama_cpp_commit) is None:
            raise RehostRuntimeMetadataError("llama.cpp commit is invalid")
        if self.memory_family != "simple_kv":
            raise RehostRuntimeMetadataError("memory family must be simple_kv")
        for field in ("n_batch", "n_ubatch"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise RehostRuntimeMetadataError(f"{field} must be positive")
        if (
            isinstance(self.n_gpu_layers, bool)
            or not isinstance(self.n_gpu_layers, int)
            or self.n_gpu_layers < 0
        ):
            raise RehostRuntimeMetadataError("n_gpu_layers must be non-negative")
        for field in ("cache_type_k", "cache_type_v"):
            if getattr(self, field) not in {"f16", "bf16", "q8_0", "q4_0"}:
                raise RehostRuntimeMetadataError(f"{field} is unsupported")
        for field in (
            "flash_attention",
            "context_shift",
            "cache_prompt",
            "cache_idle_slots",
        ):
            if type(getattr(self, field)) is not bool:
                raise RehostRuntimeMetadataError(f"{field} must be a boolean")

    @classmethod
    def from_observer_profile(
        cls,
        profile: RuntimeStateObserverProfile,
    ) -> "RehostRuntimeMetadata":
        if not isinstance(profile, RuntimeStateObserverProfile):
            raise RehostRuntimeMetadataError("observer profile is invalid")
        arguments = profile.required_launch_arguments

        def value_after(flag: str) -> str:
            if arguments.count(flag) != 1:
                raise RehostRuntimeMetadataError(
                    f"qualified launch argument {flag} is unavailable"
                )
            index = arguments.index(flag)
            if index + 1 >= len(arguments):
                raise RehostRuntimeMetadataError(
                    f"qualified launch argument {flag} has no value"
                )
            return arguments[index + 1]

        try:
            n_batch = int(value_after("--batch-size"))
            n_ubatch = int(value_after("--ubatch-size"))
            n_gpu_layers = int(value_after("--n-gpu-layers"))
        except ValueError as error:
            raise RehostRuntimeMetadataError(
                "qualified launch integer is invalid"
            ) from error
        cache_type_k = value_after("--cache-type-k")
        cache_type_v = value_after("--cache-type-v")
        flash_attention = value_after("--flash-attn") == "on"
        if (
            cache_type_k != profile.key_cache_type
            or cache_type_v != profile.value_cache_type
            or ("--no-context-shift" in arguments) != (not profile.context_shift)
            or ("--no-cache-idle-slots" in arguments)
            != (not profile.cache_idle_slots)
            or profile.cache_prompt
        ):
            raise RehostRuntimeMetadataError(
                "qualified launch policy differs from observer profile"
            )
        return cls(
            source_runtime_profile_digest=profile.profile_digest,
            llama_cpp_commit=profile.llama_cpp_commit,
            memory_family=profile.memory_family,
            n_batch=n_batch,
            n_ubatch=n_ubatch,
            cache_type_k=cache_type_k,
            cache_type_v=cache_type_v,
            flash_attention=flash_attention,
            n_gpu_layers=n_gpu_layers,
            context_shift=profile.context_shift,
            cache_prompt=profile.cache_prompt,
            cache_idle_slots=profile.cache_idle_slots,
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": "1.0",
            "contract_kind": "laris_rehost_runtime",
            "prompt_contract": {
                "api": "llama_cpp_chat",
                "template_policy": "gguf_default",
            },
            "runtime": asdict(self),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "RehostRuntimeMetadata":
        if (
            not isinstance(value, Mapping)
            or set(value)
            != {"schema_version", "contract_kind", "prompt_contract", "runtime"}
            or value.get("schema_version") != "1.0"
            or value.get("contract_kind") != "laris_rehost_runtime"
        ):
            raise RehostRuntimeMetadataError("rehost metadata document is invalid")
        prompt_contract = value.get("prompt_contract")
        if (
            not isinstance(prompt_contract, Mapping)
            or dict(prompt_contract)
            != {"api": "llama_cpp_chat", "template_policy": "gguf_default"}
        ):
            raise RehostRuntimeMetadataError("prompt contract is invalid")
        runtime = value.get("runtime")
        expected_fields = {field.name for field in cls.__dataclass_fields__.values()}
        if not isinstance(runtime, Mapping) or set(runtime) != expected_fields:
            raise RehostRuntimeMetadataError("runtime metadata field set is invalid")
        try:
            return cls(**dict(runtime))
        except TypeError as error:
            raise RehostRuntimeMetadataError("runtime metadata types are invalid") from error


__all__ = [
    "RehostRuntimeMetadata",
    "RehostRuntimeMetadataError",
]
