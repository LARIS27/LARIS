from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import math
from pathlib import Path
import re


_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_RUN_ID = re.compile(r"run-[A-Za-z0-9._-]{1,191}\Z")
_BUNDLE_ID = re.compile(r"rb_[0-9a-f]{64}\Z")
_USER_HMAC = re.compile(r"[A-Za-z0-9_-]{43}\Z")
_SESSION_REF = re.compile(r"sr_[0-9a-f]{64}\Z")
_SAFE_REF = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,191}\Z")
_BUILD_ID = re.compile(r"[0-9a-f]{40}\Z")


class RehostValidationError(ValueError):
    """Raised when rehosting evidence does not satisfy the frozen contract."""


class RehostSubjectKind(StrEnum):
    USER_HMAC = "USER_HMAC"
    ENDPOINT_REF = "ENDPOINT_REF"


@dataclass(frozen=True)
class InvestigatorSampler:
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.95
    min_p: float = 0.05
    seed: int = 0

    def __post_init__(self) -> None:
        if (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or not math.isfinite(self.temperature)
            or self.temperature <= 0
        ):
            raise RehostValidationError("temperature must be finite and positive")
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int) or self.top_k < 1:
            raise RehostValidationError("top_k must be a positive integer")
        for name in ("top_p", "min_p"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 < value <= 1
            ):
                raise RehostValidationError(f"{name} must be in (0, 1]")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not 0 <= self.seed <= 0xFFFFFFFF
        ):
            raise RehostValidationError("seed must be an unsigned 32-bit integer")

    def to_mapping(self) -> dict[str, object]:
        return {
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "min_p": self.min_p,
            "seed": self.seed,
        }


def _safe_text(name: str, value: object) -> str:
    if not isinstance(value, str) or _SAFE_REF.fullmatch(value) is None:
        raise RehostValidationError(f"{name} must be canonical path-safe text")
    return value


def _digest(name: str, value: object) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise RehostValidationError(f"{name} must be lowercase sha256:<64 hex>")
    return value


def _positive(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RehostValidationError(f"{name} must be a positive integer")
    return value


def _nonnegative(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RehostValidationError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class RehostRuntimeProfile:
    profile_digest: str
    runtime_binary_digest: str
    mapped_libllama_digest: str
    mapped_libllama_elf_build_id: str
    n_seq_max: int
    n_stream: int
    n_layer: int
    v_trans: bool | None
    key_ggml_type_id: int
    value_ggml_type_id: int
    max_n_layer: int = 0

    def __post_init__(self) -> None:
        _digest("profile_digest", self.profile_digest)
        _digest("runtime_binary_digest", self.runtime_binary_digest)
        _digest("mapped_libllama_digest", self.mapped_libllama_digest)
        if (
            not isinstance(self.mapped_libllama_elf_build_id, str)
            or _BUILD_ID.fullmatch(self.mapped_libllama_elf_build_id) is None
        ):
            raise RehostValidationError(
                "mapped_libllama_elf_build_id must be 40 lowercase hex"
            )
        _positive("n_seq_max", self.n_seq_max)
        _positive("n_stream", self.n_stream)
        if self.n_layer == 0:
            _positive("max_n_layer", self.max_n_layer)
        else:
            _positive("n_layer", self.n_layer)
            if self.max_n_layer and self.max_n_layer < self.n_layer:
                raise RehostValidationError(
                    "max_n_layer must cover the fixed n_layer"
                )
        if self.n_seq_max != self.n_stream:
            raise RehostValidationError("runtime profile n_seq_max must equal n_stream")
        if self.v_trans is not None and not isinstance(self.v_trans, bool):
            raise RehostValidationError(
                "runtime profile v_trans must be a boolean or dynamically observed"
            )
        _nonnegative("key_ggml_type_id", self.key_ggml_type_id)
        _nonnegative("value_ggml_type_id", self.value_ggml_type_id)

    def verify_case(self, case: RehostCase) -> None:
        if not isinstance(case, RehostCase):
            raise RehostValidationError("case must be a RehostCase")
        comparisons = (
            ("n_seq_max", case.n_seq_max, self.n_seq_max),
            ("n_stream", case.n_stream, self.n_stream),
            ("key_ggml_type_id", case.key_ggml_type_id, self.key_ggml_type_id),
            (
                "value_ggml_type_id",
                case.value_ggml_type_id,
                self.value_ggml_type_id,
            ),
        )
        for name, observed, expected in comparisons:
            if type(observed) is not type(expected) or observed != expected:
                raise RehostValidationError(
                    f"case {name} differs from the qualified runtime profile"
                )
        if self.n_layer:
            if case.n_layer != self.n_layer:
                raise RehostValidationError(
                    "case n_layer differs from the qualified runtime profile"
                )
        elif not 0 < case.n_layer <= self.max_n_layer:
            raise RehostValidationError(
                "case n_layer exceeds the qualified runtime profile limit"
            )
        if self.v_trans is not None and case.v_trans is not self.v_trans:
            raise RehostValidationError(
                "case v_trans differs from the qualified runtime profile"
            )


@dataclass(frozen=True)
class RehostCase:
    run_id: str
    bundle_id: str
    user_hmac: str
    identity_key_id: str
    session_ref: str
    payload_path: Path
    payload_digest: str
    payload_size_bytes: int
    next_position: int
    destination_seq_id: int
    model_profile_id: str
    model_digest: str
    runtime_profile_digest: str
    memory_family: str
    n_ctx: int
    n_seq_max: int
    n_stream: int
    n_layer: int
    v_trans: bool
    key_ggml_type_id: int
    value_ggml_type_id: int
    source_seq_id: int
    source_stream_index: int
    position_range: tuple[int, int]
    subject_kind: RehostSubjectKind = RehostSubjectKind.USER_HMAC

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or _RUN_ID.fullmatch(self.run_id) is None:
            raise RehostValidationError("run_id is invalid")
        if not isinstance(self.bundle_id, str) or _BUNDLE_ID.fullmatch(self.bundle_id) is None:
            raise RehostValidationError("bundle_id is invalid")
        if not isinstance(self.subject_kind, RehostSubjectKind):
            raise RehostValidationError("subject_kind is invalid")
        if not isinstance(self.user_hmac, str) or _USER_HMAC.fullmatch(self.user_hmac) is None:
            raise RehostValidationError("subject reference is invalid")
        _safe_text("identity_key_id", self.identity_key_id)
        if not isinstance(self.session_ref, str) or _SESSION_REF.fullmatch(self.session_ref) is None:
            raise RehostValidationError("session_ref is invalid")
        if not isinstance(self.payload_path, Path) or not self.payload_path.is_absolute():
            raise RehostValidationError("payload_path must be an absolute pathlib.Path")
        if self.payload_path.name != "runtime_state_payload.bin":
            raise RehostValidationError("payload_path filename is invalid")
        _digest("payload_digest", self.payload_digest)
        _positive("payload_size_bytes", self.payload_size_bytes)
        _positive("next_position", self.next_position)
        _nonnegative("destination_seq_id", self.destination_seq_id)
        _safe_text("model_profile_id", self.model_profile_id)
        _digest("model_digest", self.model_digest)
        _digest("runtime_profile_digest", self.runtime_profile_digest)
        if self.memory_family != "simple_kv":
            raise RehostValidationError("memory_family must be simple_kv")
        _positive("n_ctx", self.n_ctx)
        _positive("n_seq_max", self.n_seq_max)
        _positive("n_stream", self.n_stream)
        _positive("n_layer", self.n_layer)
        if not isinstance(self.v_trans, bool):
            raise RehostValidationError("v_trans must be a boolean")
        _nonnegative("key_ggml_type_id", self.key_ggml_type_id)
        _nonnegative("value_ggml_type_id", self.value_ggml_type_id)
        _nonnegative("source_seq_id", self.source_seq_id)
        _nonnegative("source_stream_index", self.source_stream_index)
        if self.n_seq_max != self.n_stream:
            raise RehostValidationError("n_seq_max must equal n_stream")
        if self.n_ctx % self.n_stream:
            raise RehostValidationError("n_ctx must divide evenly across n_stream")
        if self.destination_seq_id >= self.n_seq_max:
            raise RehostValidationError("destination_seq_id exceeds the runtime profile")
        if self.source_seq_id >= self.n_seq_max:
            raise RehostValidationError("source_seq_id exceeds the runtime profile")
        if self.source_stream_index >= self.n_stream:
            raise RehostValidationError("source_stream_index exceeds the runtime profile")
        if (
            not isinstance(self.position_range, tuple)
            or len(self.position_range) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in self.position_range)
            or self.position_range != (0, self.next_position - 1)
        ):
            raise RehostValidationError("position_range must be dense from zero to next_position - 1")
        if self.next_position > self.n_ctx // self.n_stream:
            raise RehostValidationError("next_position exceeds per-stream context capacity")

    def to_mapping(self) -> dict[str, object]:
        mapping: dict[str, object] = {
            "run_id": self.run_id,
            "bundle_id": self.bundle_id,
            "session_ref": self.session_ref,
            "payload_path": str(self.payload_path),
            "payload_digest": self.payload_digest,
            "payload_size_bytes": self.payload_size_bytes,
            "next_position": self.next_position,
            "destination_seq_id": self.destination_seq_id,
            "model_profile_id": self.model_profile_id,
            "model_digest": self.model_digest,
            "runtime_profile_digest": self.runtime_profile_digest,
            "memory_family": self.memory_family,
            "n_ctx": self.n_ctx,
            "n_seq_max": self.n_seq_max,
            "n_stream": self.n_stream,
            "n_layer": self.n_layer,
            "v_trans": self.v_trans,
            "key_ggml_type_id": self.key_ggml_type_id,
            "value_ggml_type_id": self.value_ggml_type_id,
            "source_seq_id": self.source_seq_id,
            "source_stream_index": self.source_stream_index,
            "position_range": list(self.position_range),
            "subject_kind": self.subject_kind.value,
            "subject_ref": self.user_hmac,
        }
        if self.subject_kind is RehostSubjectKind.USER_HMAC:
            mapping.update(
                {
                    "user_hmac": self.user_hmac,
                    "identity_key_id": self.identity_key_id,
                }
            )
        else:
            mapping.update(
                {
                    "endpoint_ref": self.user_hmac,
                    "correlation_key_id": self.identity_key_id,
                }
            )
        return mapping


@dataclass(frozen=True)
class HelperReceipt:
    exit_code: int
    response: str
    stderr: str
    mapped_libllama_device: int | None
    mapped_libllama_inode: int | None
    restored_bytes: int | None
    restored_pos_min: int | None
    restored_pos_max: int | None
    appended_token_count: int | None
    logical_batch_count: int | None
    effective_n_batch: int | None
    effective_n_ubatch: int | None
    generated_token_count: int | None
    prompt_contract: str | None
    sampler: InvestigatorSampler | None
    duration_ms: int

    def to_mapping(self) -> dict[str, object]:
        return {
            "exit_code": self.exit_code,
            "response": self.response,
            "stderr": self.stderr,
            "mapped_libllama_device": self.mapped_libllama_device,
            "mapped_libllama_inode": self.mapped_libllama_inode,
            "restored_bytes": self.restored_bytes,
            "restored_pos_min": self.restored_pos_min,
            "restored_pos_max": self.restored_pos_max,
            "appended_token_count": self.appended_token_count,
            "logical_batch_count": self.logical_batch_count,
            "effective_n_batch": self.effective_n_batch,
            "effective_n_ubatch": self.effective_n_ubatch,
            "generated_token_count": self.generated_token_count,
            "prompt_contract": self.prompt_contract,
            "sampler": self.sampler.to_mapping() if self.sampler else None,
            "duration_ms": self.duration_ms,
        }


__all__ = [
    "HelperReceipt",
    "InvestigatorSampler",
    "RehostCase",
    "RehostRuntimeProfile",
    "RehostSubjectKind",
    "RehostValidationError",
]
