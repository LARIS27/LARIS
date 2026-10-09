from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from laris_onpremise.experiments.source_contract import digest_json


_PROFILE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "profile_id",
        "llama_cpp_commit",
        "os",
        "architecture",
        "abi",
        "target_kind",
        "symbol_name",
        "batch_size_bytes",
        "batch_argument_location",
        "n_tokens_offset",
        "n_seq_id_offset",
        "seq_id_offset",
        "max_batch_tokens",
        "allowed_seq_ids",
        "server_batch_allocation_n_seq_max",
        "configured_context_ref",
        "decode_return_type",
    }
)
_PROFILE_CONSTANTS: Final[dict[str, object]] = {
    "schema_version": "1.0",
    "profile_id": "x86_64-cpu-llama-decode",
    "llama_cpp_commit": "daf6bc9f2d3516d7bda73d191c6a01388aff09c5",
    "os": "Linux",
    "architecture": "x86_64",
    "abi": "System V AMD64",
    "target_kind": "mapped_shared_library",
    "symbol_name": "llama_decode",
    "batch_size_bytes": 56,
    "batch_argument_location": "rsp_plus_8",
    "n_tokens_offset": 0,
    "n_seq_id_offset": 32,
    "seq_id_offset": 40,
    "max_batch_tokens": 128,
    "allowed_seq_ids": (0, 1, 2, 3),
    "server_batch_allocation_n_seq_max": 1,
    "configured_context_ref": "context-main",
    "decode_return_type": "signed_int32",
}
_PROFILE_SEQUENCE_VARIANTS: Final[dict[str, tuple[int, ...]]] = {
    "x86_64-cpu-llama-decode": (0, 1, 2, 3),
    "x86_64-cpu-llama-decode-8slot": (0, 1, 2, 3, 4, 5, 6, 7),
}


class ObserverProfileError(ValueError):
    """Raised when a runtime observer profile is malformed or unsupported."""


def _require_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ObserverProfileError(f"{field} must be a non-empty string")
    return value


def _require_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ObserverProfileError(f"{field} must be a non-negative integer")
    return value


def _decode_json(path: Path) -> dict[str, object]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ObserverProfileError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
        )
    except ObserverProfileError:
        raise
    except (OSError, json.JSONDecodeError) as error:
        raise ObserverProfileError("invalid observer profile JSON") from error
    if not isinstance(value, dict):
        raise ObserverProfileError("observer profile must be an object")
    return value


@dataclass(frozen=True)
class RuntimeObserverProfile:
    schema_version: str
    profile_id: str
    llama_cpp_commit: str
    operating_system: str
    architecture: str
    abi: str
    target_kind: str
    symbol_name: str
    batch_size_bytes: int
    batch_argument_location: str
    n_tokens_offset: int
    n_seq_id_offset: int
    seq_id_offset: int
    max_batch_tokens: int
    allowed_seq_ids: tuple[int, ...]
    server_batch_allocation_n_seq_max: int
    configured_context_ref: str
    decode_return_type: str

    @classmethod
    def supported(cls) -> RuntimeObserverProfile:
        """Return the immutable profile compiled into this evaluator."""
        value = dict(_PROFILE_CONSTANTS)
        value["allowed_seq_ids"] = list(_PROFILE_CONSTANTS["allowed_seq_ids"])
        return cls.from_mapping(value)

    @classmethod
    def load(cls, path: Path) -> RuntimeObserverProfile:
        return cls.from_mapping(_decode_json(path))

    @classmethod
    def from_mapping(
        cls, value: dict[str, object]
    ) -> RuntimeObserverProfile:
        extra = set(value) - _PROFILE_KEYS
        if extra:
            raise ObserverProfileError(
                f"unexpected profile fields: {sorted(extra)}"
            )
        missing = _PROFILE_KEYS - set(value)
        if missing:
            raise ObserverProfileError(
                f"missing profile fields: {sorted(missing)}"
            )

        raw_sequences = value["allowed_seq_ids"]
        if not isinstance(raw_sequences, list) or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in raw_sequences
        ):
            raise ObserverProfileError(
                "allowed_seq_ids must contain non-negative integers"
            )
        sequences = tuple(raw_sequences)
        if tuple(sorted(set(sequences))) != sequences:
            raise ObserverProfileError(
                "allowed_seq_ids must be sorted and unique"
            )

        parsed: dict[str, object] = {
            "schema_version": _require_string(
                value["schema_version"], "schema_version"
            ),
            "profile_id": _require_string(value["profile_id"], "profile_id"),
            "llama_cpp_commit": _require_string(
                value["llama_cpp_commit"], "llama_cpp_commit"
            ),
            "os": _require_string(value["os"], "os"),
            "architecture": _require_string(
                value["architecture"], "architecture"
            ),
            "abi": _require_string(value["abi"], "abi"),
            "target_kind": _require_string(
                value["target_kind"], "target_kind"
            ),
            "symbol_name": _require_string(
                value["symbol_name"], "symbol_name"
            ),
            "batch_size_bytes": _require_integer(
                value["batch_size_bytes"], "batch_size_bytes"
            ),
            "batch_argument_location": _require_string(
                value["batch_argument_location"], "batch_argument_location"
            ),
            "n_tokens_offset": _require_integer(
                value["n_tokens_offset"], "n_tokens_offset"
            ),
            "n_seq_id_offset": _require_integer(
                value["n_seq_id_offset"], "n_seq_id_offset"
            ),
            "seq_id_offset": _require_integer(
                value["seq_id_offset"], "seq_id_offset"
            ),
            "max_batch_tokens": _require_integer(
                value["max_batch_tokens"], "max_batch_tokens"
            ),
            "allowed_seq_ids": sequences,
            "server_batch_allocation_n_seq_max": _require_integer(
                value["server_batch_allocation_n_seq_max"],
                "server_batch_allocation_n_seq_max",
            ),
            "configured_context_ref": _require_string(
                value["configured_context_ref"], "configured_context_ref"
            ),
            "decode_return_type": _require_string(
                value["decode_return_type"], "decode_return_type"
            ),
        }
        for field, expected in _PROFILE_CONSTANTS.items():
            if field in {"profile_id", "allowed_seq_ids"}:
                continue
            if parsed[field] != expected:
                raise ObserverProfileError(
                    f"{field} does not match the supported profile"
                )
        expected_sequences = _PROFILE_SEQUENCE_VARIANTS.get(
            str(parsed["profile_id"])
        )
        if expected_sequences is None or sequences != expected_sequences:
            raise ObserverProfileError(
                "profile_id and allowed_seq_ids do not match a supported profile"
            )

        return cls(
            schema_version=parsed["schema_version"],
            profile_id=parsed["profile_id"],
            llama_cpp_commit=parsed["llama_cpp_commit"],
            operating_system=parsed["os"],
            architecture=parsed["architecture"],
            abi=parsed["abi"],
            target_kind=parsed["target_kind"],
            symbol_name=parsed["symbol_name"],
            batch_size_bytes=parsed["batch_size_bytes"],
            batch_argument_location=parsed["batch_argument_location"],
            n_tokens_offset=parsed["n_tokens_offset"],
            n_seq_id_offset=parsed["n_seq_id_offset"],
            seq_id_offset=parsed["seq_id_offset"],
            max_batch_tokens=parsed["max_batch_tokens"],
            allowed_seq_ids=parsed["allowed_seq_ids"],
            server_batch_allocation_n_seq_max=parsed[
                "server_batch_allocation_n_seq_max"
            ],
            configured_context_ref=parsed["configured_context_ref"],
            decode_return_type=parsed["decode_return_type"],
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "llama_cpp_commit": self.llama_cpp_commit,
            "os": self.operating_system,
            "architecture": self.architecture,
            "abi": self.abi,
            "target_kind": self.target_kind,
            "symbol_name": self.symbol_name,
            "batch_size_bytes": self.batch_size_bytes,
            "batch_argument_location": self.batch_argument_location,
            "n_tokens_offset": self.n_tokens_offset,
            "n_seq_id_offset": self.n_seq_id_offset,
            "seq_id_offset": self.seq_id_offset,
            "max_batch_tokens": self.max_batch_tokens,
            "allowed_seq_ids": list(self.allowed_seq_ids),
            "server_batch_allocation_n_seq_max": (
                self.server_batch_allocation_n_seq_max
            ),
            "configured_context_ref": self.configured_context_ref,
            "decode_return_type": self.decode_return_type,
        }

    def canonical_digest(self) -> str:
        return digest_json(self.to_mapping())


def derive_runtime_build_profile_digest(
    *,
    source_commit: str,
    server_binary_digest: str,
    mapped_libllama_digest: str,
    model_digest: str,
    server_config_digest: str,
    build_command_digest: str,
    probe_profile_digest: str,
    abi_layout_manifest_digest: str,
) -> str:
    return digest_json(
        {
            "source_commit": source_commit,
            "server_binary_digest": server_binary_digest,
            "mapped_libllama_digest": mapped_libllama_digest,
            "model_digest": model_digest,
            "server_config_digest": server_config_digest,
            "build_command_digest": build_command_digest,
            "probe_profile_digest": probe_profile_digest,
            "abi_layout_manifest_digest": abi_layout_manifest_digest,
        }
    )


def derive_observer_artifact_profile_digest(
    *,
    bpf_object_digest: str,
    native_collector_digest: str,
    native_dependency_manifest_digest: str,
    supervisor_source_bundle_digest: str,
    supervisor_interpreter_digest: str,
    wire_protocol_version: str,
    runtime_oracle_schema_version: str,
    qualified_toolchain_profile: str,
) -> str:
    return digest_json(
        {
            "bpf_object_digest": bpf_object_digest,
            "native_collector_digest": native_collector_digest,
            "native_dependency_manifest_digest": (
                native_dependency_manifest_digest
            ),
            "supervisor_source_bundle_digest": supervisor_source_bundle_digest,
            "supervisor_interpreter_digest": supervisor_interpreter_digest,
            "wire_protocol_version": wire_protocol_version,
            "runtime_oracle_schema_version": runtime_oracle_schema_version,
            "qualified_toolchain_profile": qualified_toolchain_profile,
        }
    )
