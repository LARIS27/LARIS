from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Mapping

from laris_onpremise.core.records import canonical_json
from laris_onpremise.bundle.rehost_metadata import (
    RehostRuntimeMetadata,
    RehostRuntimeMetadataError,
)
from laris_onpremise.rehosting.contracts import RehostCase, RehostSubjectKind


class PortableBundleError(ValueError):
    """Raised when a portable rehosting bundle is incomplete or inconsistent."""


_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_RUN_ID = re.compile(r"run-[A-Za-z0-9._-]{1,191}\Z")
_BUNDLE_ID = re.compile(r"rb_[0-9a-f]{64}\Z")
_USER_HMAC = re.compile(r"[A-Za-z0-9_-]{43}\Z")
_ENDPOINT_REF = _USER_HMAC
_SESSION_REF = re.compile(r"sr_[0-9a-f]{64}\Z")
_SAFE_TEXT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,191}\Z")
_PACKAGE_FIELDS_V1 = frozenset(
    {"schema_version", "package_type", "run_id", "source", "bundle_index"}
)
_PACKAGE_FIELDS_V1_1 = _PACKAGE_FIELDS_V1 | {"rehost_metadata"}
_PACKAGE_FIELDS_V2 = _PACKAGE_FIELDS_V1_1 | {"subject_kind"}
_INDEX_FIELDS = frozenset({"schema_version", "run_id", "entries"})
_ENDPOINT_INDEX_FIELDS = _INDEX_FIELDS | {"subject_kind"}
_ENTRY_FIELDS = frozenset(
    {
        "bundle_id",
        "user_hmac",
        "session_ref",
        "state",
        "rehost_manifest_path",
        "rehost_manifest_digest",
        "payload_path",
        "payload_digest",
        "payload_size_bytes",
    }
)
_ENDPOINT_ENTRY_FIELDS = frozenset(
    {
        "bundle_id",
        "endpoint_ref",
        "endpoint_type",
        "correlation_key_id",
        "attribution_decision",
        "session_ref",
        "state",
        "rehost_manifest_path",
        "rehost_manifest_digest",
        "payload_path",
        "payload_digest",
        "payload_size_bytes",
    }
)
_SESSION_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "bundle_id",
        "user_hmac",
        "identity_key_id",
        "session_ref",
        "application_context_mode",
        "last_turn_index",
        "source_request_ref",
        "source_lineage_digest",
        "model_profile_id",
        "model_digest",
        "runtime_profile_digest",
        "memory_family",
        "n_ctx",
        "n_seq_max",
        "n_stream",
        "n_layer",
        "v_trans",
        "key_ggml_type_id",
        "value_ggml_type_id",
        "source_seq_id",
        "source_stream_index",
        "destination_sequence_policy",
        "position_range",
        "kv_cell_ranges",
        "next_position",
        "payload_format",
        "payload_file",
        "payload_size_bytes",
        "payload_digest",
        "acquisition_manifest_digest",
        "gate_a_result_digest",
        "evidence_root_digest",
        "contains_raw_prompt",
        "contains_raw_response",
        "contains_explicit_token_history",
        "restore_api",
    }
)
_ENDPOINT_SESSION_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "bundle_id",
        "subject_kind",
        "endpoint_ref",
        "endpoint_type",
        "correlation_key_id",
        "attribution_decision",
        "session_ref",
        "request_refs",
        "attribution_receipt_digest",
        "source_evidence_package_digest",
        "source_unit_digests",
        "model_profile_id",
        "model_digest",
        "runtime_profile_digest",
        "memory_family",
        "n_ctx",
        "n_seq_max",
        "n_stream",
        "n_layer",
        "v_trans",
        "key_ggml_type_id",
        "value_ggml_type_id",
        "source_seq_id",
        "source_stream_index",
        "destination_sequence_policy",
        "position_range",
        "next_position",
        "payload_format",
        "payload_file",
        "payload_size_bytes",
        "payload_digest",
        "contains_raw_prompt",
        "contains_raw_response",
        "contains_explicit_token_history",
        "restore_api",
    }
)
_MAX_METADATA_BYTES = 2 * 1024 * 1024


@dataclass(frozen=True)
class PortableBundleEntry:
    bundle_id: str
    user_hmac: str | None
    identity_key_id: str
    session_ref: str
    manifest_path: Path
    manifest_digest: str
    payload_path: Path
    payload_digest: str
    payload_size_bytes: int
    model_profile_id: str
    model_digest: str
    runtime_profile_digest: str
    last_turn_index: int
    source_request_ref: str
    n_ctx: int
    n_seq_max: int
    n_stream: int
    n_layer: int
    v_trans: bool
    key_ggml_type_id: int
    value_ggml_type_id: int
    next_position: int
    source_seq_id: int
    source_stream_index: int
    position_range: tuple[int, int]
    subject_kind: RehostSubjectKind = RehostSubjectKind.USER_HMAC
    endpoint_ref: str | None = None

    @property
    def subject_ref(self) -> str:
        value = (
            self.user_hmac
            if self.subject_kind is RehostSubjectKind.USER_HMAC
            else self.endpoint_ref
        )
        if value is None:
            raise PortableBundleError("bundle subject reference is unavailable")
        return value

    def to_rehost_case(
        self,
        *,
        run_id: str,
        destination_seq_id: int,
    ) -> RehostCase:
        return RehostCase(
            run_id=run_id,
            bundle_id=self.bundle_id,
            user_hmac=self.subject_ref,
            identity_key_id=self.identity_key_id,
            session_ref=self.session_ref,
            payload_path=self.payload_path,
            payload_digest=self.payload_digest,
            payload_size_bytes=self.payload_size_bytes,
            next_position=self.next_position,
            destination_seq_id=destination_seq_id,
            model_profile_id=self.model_profile_id,
            model_digest=self.model_digest,
            runtime_profile_digest=self.runtime_profile_digest,
            memory_family="simple_kv",
            n_ctx=self.n_ctx,
            n_seq_max=self.n_seq_max,
            n_stream=self.n_stream,
            n_layer=self.n_layer,
            v_trans=self.v_trans,
            key_ggml_type_id=self.key_ggml_type_id,
            value_ggml_type_id=self.value_ggml_type_id,
            source_seq_id=self.source_seq_id,
            source_stream_index=self.source_stream_index,
            position_range=self.position_range,
            subject_kind=self.subject_kind,
        )


@dataclass(frozen=True)
class PortableBundleCatalog:
    bundle_root: Path
    source_run_id: str
    manifest_digest: str
    index_digest: str
    entries: tuple[PortableBundleEntry, ...]
    runtime_metadata: RehostRuntimeMetadata | None = None
    prompt_contract: str = "legacy_raw_completion"
    subject_kind: RehostSubjectKind = RehostSubjectKind.USER_HMAC

    @classmethod
    def open(
        cls,
        bundle_root: Path,
        expected_manifest_digest: str | None = None,
    ) -> "PortableBundleCatalog":
        root = _absolute_directory(bundle_root, "bundle root")
        root_inventory = _inventory(root)
        if "bundle_manifest.json" not in root_inventory:
            raise PortableBundleError("bundle completion manifest is unavailable")
        legacy_inventory = {"bundle_manifest.json", "bundle_index.json", "users"}
        current_inventory = legacy_inventory | {"rehost_metadata.json"}
        endpoint_inventory = {
            "bundle_manifest.json",
            "bundle_index.json",
            "rehost_metadata.json",
            "endpoints",
        }
        if frozenset(root_inventory) not in {
            frozenset(legacy_inventory),
            frozenset(current_inventory),
            frozenset(endpoint_inventory),
        }:
            raise PortableBundleError("bundle root inventory is invalid")

        package, package_digest = _canonical_mapping(
            root / "bundle_manifest.json",
            "bundle completion manifest",
        )
        if expected_manifest_digest is not None:
            _digest("expected manifest digest", expected_manifest_digest)
            if not hmac.compare_digest(package_digest, expected_manifest_digest):
                raise PortableBundleError("bundle completion manifest digest differs")
        (
            run_id,
            expected_index_digest,
            metadata_binding,
            subject_kind,
        ) = _validate_package(package)
        if metadata_binding is None:
            if root_inventory != legacy_inventory:
                raise PortableBundleError("legacy bundle root inventory is invalid")
            runtime_metadata = None
            prompt_contract = "legacy_raw_completion"
        else:
            expected_inventory = (
                endpoint_inventory
                if subject_kind is RehostSubjectKind.ENDPOINT_REF
                else current_inventory
            )
            if root_inventory != expected_inventory:
                raise PortableBundleError("current bundle root inventory is invalid")
            metadata_path = root / "rehost_metadata.json"
            raw_metadata, observed_metadata_digest = _read_regular_file(
                metadata_path,
                limit=_MAX_METADATA_BYTES,
            )
            if (
                len(raw_metadata) != metadata_binding[1]
                or not hmac.compare_digest(
                    observed_metadata_digest, metadata_binding[0]
                )
            ):
                raise PortableBundleError("rehost metadata digest or size differs")
            metadata_document, canonical_digest = _canonical_mapping(
                metadata_path,
                "rehost metadata",
            )
            if not hmac.compare_digest(canonical_digest, observed_metadata_digest):
                raise PortableBundleError("rehost metadata changed while validating")
            try:
                runtime_metadata = RehostRuntimeMetadata.from_mapping(
                    metadata_document
                )
            except RehostRuntimeMetadataError as error:
                raise PortableBundleError("rehost metadata is invalid") from error
            prompt_contract = "gguf_default"

        index, index_digest = _canonical_mapping(
            root / "bundle_index.json",
            "bundle index",
        )
        if not hmac.compare_digest(index_digest, expected_index_digest):
            raise PortableBundleError("bundle index digest differs")
        entries = _validate_index(root, run_id, index, subject_kind=subject_kind)
        if runtime_metadata is not None and any(
            entry.runtime_profile_digest
            != runtime_metadata.source_runtime_profile_digest
            for entry in entries
        ):
            raise PortableBundleError(
                "session runtime profile differs from root metadata"
            )
        return cls(
            root,
            run_id,
            package_digest,
            index_digest,
            entries,
            runtime_metadata,
            prompt_contract,
            subject_kind,
        )


def _validate_package(
    package: Mapping[str, object],
) -> tuple[str, str, tuple[str, int] | None, RehostSubjectKind]:
    source = package.get("source")
    binding = package.get("bundle_index")
    run_id = package.get("run_id")
    if (
        package.get("schema_version") not in {"1.0", "1.1", "2.0"}
        or package.get("package_type") != "laris_rehost_bundle"
        or not isinstance(run_id, str)
        or _RUN_ID.fullmatch(run_id) is None
        or not isinstance(source, dict)
        or not isinstance(binding, dict)
        or set(binding) != {"path", "digest"}
        or binding.get("path") != "bundle_index.json"
    ):
        raise PortableBundleError("bundle completion manifest is invalid")
    version = package["schema_version"]
    expected_fields = (
        _PACKAGE_FIELDS_V1
        if version == "1.0"
        else _PACKAGE_FIELDS_V1_1
        if version == "1.1"
        else _PACKAGE_FIELDS_V2
    )
    if set(package) != expected_fields:
        raise PortableBundleError("bundle completion manifest is invalid")
    subject_kind = (
        RehostSubjectKind.ENDPOINT_REF
        if version == "2.0"
        else RehostSubjectKind.USER_HMAC
    )
    expected_source_fields = (
        {"evidence_package_manifest_digest"}
        if subject_kind is RehostSubjectKind.ENDPOINT_REF
        else {"evidence_root_digest", "gate_a_result_digest"}
    )
    if set(source) != expected_source_fields:
        raise PortableBundleError("bundle completion source is invalid")
    if (
        subject_kind is RehostSubjectKind.ENDPOINT_REF
        and package.get("subject_kind") != subject_kind.value
    ):
        raise PortableBundleError("bundle subject kind is invalid")
    for name, value in source.items():
        _digest(str(name), value)
    metadata_binding: tuple[str, int] | None = None
    if version in {"1.1", "2.0"}:
        raw_metadata = package.get("rehost_metadata")
        if (
            not isinstance(raw_metadata, dict)
            or set(raw_metadata) != {"path", "digest", "size_bytes"}
            or raw_metadata.get("path") != "rehost_metadata.json"
        ):
            raise PortableBundleError("rehost metadata binding is invalid")
        metadata_binding = (
            _digest("rehost metadata digest", raw_metadata["digest"]),
            _positive_int("rehost metadata size", raw_metadata["size_bytes"]),
        )
    return (
        run_id,
        _digest("bundle index digest", binding["digest"]),
        metadata_binding,
        subject_kind,
    )


def _validate_index(
    root: Path,
    run_id: str,
    index: Mapping[str, object],
    *,
    subject_kind: RehostSubjectKind = RehostSubjectKind.USER_HMAC,
) -> tuple[PortableBundleEntry, ...]:
    if subject_kind is RehostSubjectKind.ENDPOINT_REF:
        return _validate_endpoint_index(root, run_id, index)
    raw_entries = index.get("entries")
    if (
        set(index) != _INDEX_FIELDS
        or index.get("schema_version") != "1.0"
        or index.get("run_id") != run_id
        or not isinstance(raw_entries, list)
        or not raw_entries
    ):
        raise PortableBundleError("bundle index is invalid")

    entries: list[PortableBundleEntry] = []
    seen_bundle_ids: set[str] = set()
    seen_sessions: set[tuple[str, str]] = set()
    expected_sessions: dict[str, set[str]] = {}
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, dict) or set(raw_entry) != _ENTRY_FIELDS:
            raise PortableBundleError("bundle index entry is invalid")
        bundle_id = _pattern("bundle_id", raw_entry["bundle_id"], _BUNDLE_ID)
        user_hmac = _pattern("user_hmac", raw_entry["user_hmac"], _USER_HMAC)
        session_ref = _pattern("session_ref", raw_entry["session_ref"], _SESSION_REF)
        if raw_entry["state"] != "READY":
            raise PortableBundleError("bundle index entry is not ready")
        if bundle_id in seen_bundle_ids or (user_hmac, session_ref) in seen_sessions:
            raise PortableBundleError("bundle index identity is duplicated")
        seen_bundle_ids.add(bundle_id)
        seen_sessions.add((user_hmac, session_ref))

        manifest_relative = _relative_path(
            raw_entry["rehost_manifest_path"],
            "rehost manifest path",
        )
        payload_relative = _relative_path(raw_entry["payload_path"], "payload path")
        manifest_path = root.joinpath(*manifest_relative.parts)
        payload_path = root.joinpath(*payload_relative.parts)
        session_root = manifest_path.parent
        session_digest = hashlib.sha256(session_ref.encode("utf-8")).hexdigest()
        expected_user_root = root / "users" / f"uh_{user_hmac}"
        if (
            payload_path.parent != session_root
            or session_root.parent.name != "sessions"
            or session_root.parent.parent != expected_user_root
            or manifest_path.name != "rehost_manifest.json"
            or payload_path.name != "runtime_state_payload.bin"
            or not session_root.name.startswith("s_")
            or not 16 <= len(session_root.name[2:]) <= 64
            or not session_digest.startswith(session_root.name[2:])
        ):
            raise PortableBundleError("bundle session path is invalid")
        expected_sessions.setdefault(user_hmac, set()).add(session_root.name)
        if _inventory(session_root) != {
            "rehost_manifest.json",
            "runtime_state_payload.bin",
        }:
            raise PortableBundleError("bundle session inventory is invalid")

        manifest, manifest_digest = _canonical_mapping(
            manifest_path,
            "rehost manifest",
        )
        expected_manifest_digest = _digest(
            "rehost manifest digest",
            raw_entry["rehost_manifest_digest"],
        )
        if not hmac.compare_digest(manifest_digest, expected_manifest_digest):
            raise PortableBundleError("rehost manifest digest differs")
        payload_size = _positive_int(
            "payload_size_bytes",
            raw_entry["payload_size_bytes"],
        )
        payload_digest = _digest("payload digest", raw_entry["payload_digest"])
        payload, observed_payload_digest = _read_regular_file(payload_path)
        if len(payload) != payload_size or not hmac.compare_digest(
            observed_payload_digest,
            payload_digest,
        ):
            raise PortableBundleError("bundle payload digest or size differs")
        entries.append(
            _entry_from_manifest(
                manifest,
                run_id=run_id,
                bundle_id=bundle_id,
                user_hmac=user_hmac,
                session_ref=session_ref,
                manifest_path=manifest_path,
                manifest_digest=manifest_digest,
                payload_path=payload_path,
                payload_digest=payload_digest,
                payload_size=payload_size,
            )
        )

    _validate_user_inventory(root / "users", expected_sessions)
    entries.sort(key=lambda item: (item.user_hmac, item.session_ref, item.bundle_id))
    return tuple(entries)


def _validate_endpoint_index(
    root: Path,
    run_id: str,
    index: Mapping[str, object],
) -> tuple[PortableBundleEntry, ...]:
    raw_entries = index.get("entries")
    if (
        set(index) != _ENDPOINT_INDEX_FIELDS
        or index.get("schema_version") != "2.0"
        or index.get("run_id") != run_id
        or index.get("subject_kind") != RehostSubjectKind.ENDPOINT_REF.value
        or not isinstance(raw_entries, list)
        or not raw_entries
    ):
        raise PortableBundleError("endpoint bundle index is invalid")
    expected_sessions: dict[str, set[str]] = {}
    entries: list[PortableBundleEntry] = []
    seen: set[tuple[str, str]] = set()
    seen_bundle_ids: set[str] = set()
    for raw_entry in raw_entries:
        if (
            not isinstance(raw_entry, dict)
            or set(raw_entry) != _ENDPOINT_ENTRY_FIELDS
        ):
            raise PortableBundleError("endpoint bundle index entry is invalid")
        bundle_id = _pattern("bundle_id", raw_entry["bundle_id"], _BUNDLE_ID)
        endpoint_ref = _pattern(
            "endpoint_ref", raw_entry["endpoint_ref"], _ENDPOINT_REF
        )
        endpoint_type = raw_entry["endpoint_type"]
        correlation_key_id = _safe_text(
            "correlation_key_id", raw_entry["correlation_key_id"]
        )
        attribution_decision = raw_entry["attribution_decision"]
        session_ref = _pattern(
            "session_ref", raw_entry["session_ref"], _SESSION_REF
        )
        if (
            endpoint_type not in {"IPV4", "IPV6"}
            or attribution_decision not in {"CERTIFIED", "RANKED"}
            or raw_entry["state"] != "READY"
        ):
            raise PortableBundleError("endpoint bundle index entry is invalid")
        if bundle_id in seen_bundle_ids or (endpoint_ref, session_ref) in seen:
            raise PortableBundleError("endpoint bundle identity is duplicated")
        seen_bundle_ids.add(bundle_id)
        seen.add((endpoint_ref, session_ref))

        manifest_relative = _relative_path(
            raw_entry["rehost_manifest_path"], "endpoint rehost manifest path"
        )
        payload_relative = _relative_path(
            raw_entry["payload_path"], "endpoint payload path"
        )
        manifest_path = root.joinpath(*manifest_relative.parts)
        payload_path = root.joinpath(*payload_relative.parts)
        session_root = manifest_path.parent
        session_digest = hashlib.sha256(session_ref.encode("utf-8")).hexdigest()
        endpoint_root = root / "endpoints" / f"ep_{endpoint_ref}"
        if (
            payload_path.parent != session_root
            or session_root.parent.name != "sessions"
            or session_root.parent.parent != endpoint_root
            or manifest_path.name != "log_linked_rehost_manifest.json"
            or payload_path.name != "runtime_state_payload.bin"
            or not session_root.name.startswith("s_")
            or not 16 <= len(session_root.name[2:]) <= 64
            or not session_digest.startswith(session_root.name[2:])
        ):
            raise PortableBundleError("endpoint bundle session path is invalid")
        expected_sessions.setdefault(endpoint_ref, set()).add(session_root.name)
        if _inventory(session_root) != {
            "log_linked_rehost_manifest.json",
            "runtime_state_payload.bin",
        }:
            raise PortableBundleError("endpoint bundle session inventory is invalid")
        manifest, manifest_digest = _canonical_mapping(
            manifest_path, "endpoint rehost manifest"
        )
        if not hmac.compare_digest(
            manifest_digest,
            _digest("endpoint manifest digest", raw_entry["rehost_manifest_digest"]),
        ):
            raise PortableBundleError("endpoint rehost manifest digest differs")
        payload_size = _positive_int(
            "payload_size_bytes", raw_entry["payload_size_bytes"]
        )
        payload_digest = _digest("payload digest", raw_entry["payload_digest"])
        payload, observed_payload_digest = _read_regular_file(payload_path)
        if len(payload) != payload_size or not hmac.compare_digest(
            observed_payload_digest, payload_digest
        ):
            raise PortableBundleError("endpoint payload digest or size differs")
        entries.append(
            _entry_from_endpoint_manifest(
                manifest,
                run_id=run_id,
                bundle_id=bundle_id,
                endpoint_ref=endpoint_ref,
                endpoint_type=str(endpoint_type),
                correlation_key_id=correlation_key_id,
                attribution_decision=str(attribution_decision),
                session_ref=session_ref,
                manifest_path=manifest_path,
                manifest_digest=manifest_digest,
                payload_path=payload_path,
                payload_digest=payload_digest,
                payload_size=payload_size,
            )
        )
    _validate_endpoint_inventory(root / "endpoints", expected_sessions)
    entries.sort(key=lambda item: (item.subject_ref, item.session_ref, item.bundle_id))
    return tuple(entries)


def _entry_from_endpoint_manifest(
    manifest: Mapping[str, object],
    *,
    run_id: str,
    bundle_id: str,
    endpoint_ref: str,
    endpoint_type: str,
    correlation_key_id: str,
    attribution_decision: str,
    session_ref: str,
    manifest_path: Path,
    manifest_digest: str,
    payload_path: Path,
    payload_digest: str,
    payload_size: int,
) -> PortableBundleEntry:
    fixed = {
        "schema_version": "1.0",
        "run_id": run_id,
        "bundle_id": bundle_id,
        "subject_kind": RehostSubjectKind.ENDPOINT_REF.value,
        "endpoint_ref": endpoint_ref,
        "endpoint_type": endpoint_type,
        "correlation_key_id": correlation_key_id,
        "attribution_decision": attribution_decision,
        "session_ref": session_ref,
        "memory_family": "simple_kv",
        "destination_sequence_policy": "caller_selected_empty_sequence",
        "payload_format": "llama-simple-kv-seq-payload-v1",
        "payload_file": "runtime_state_payload.bin",
        "payload_size_bytes": payload_size,
        "payload_digest": payload_digest,
        "contains_raw_prompt": False,
        "contains_raw_response": False,
        "contains_explicit_token_history": False,
        "restore_api": "llama_state_seq_set_data_ext",
    }
    if set(manifest) != _ENDPOINT_SESSION_MANIFEST_FIELDS or any(
        type(manifest.get(name)) is not type(value) or manifest.get(name) != value
        for name, value in fixed.items()
    ):
        raise PortableBundleError("endpoint rehost manifest binding is invalid")
    request_refs = manifest["request_refs"]
    source_unit_digests = manifest["source_unit_digests"]
    if (
        not isinstance(request_refs, list)
        or not request_refs
        or request_refs != sorted(set(request_refs))
        or any(_SAFE_TEXT.fullmatch(str(value)) is None for value in request_refs)
        or not isinstance(source_unit_digests, list)
        or not source_unit_digests
        or len(source_unit_digests) != len(set(source_unit_digests))
    ):
        raise PortableBundleError("endpoint source references are invalid")
    for digest in source_unit_digests:
        _digest("source unit digest", digest)
    for field in (
        "attribution_receipt_digest",
        "source_evidence_package_digest",
        "model_digest",
        "runtime_profile_digest",
    ):
        _digest(field, manifest[field])
    next_position = _positive_int("next_position", manifest["next_position"])
    return PortableBundleEntry(
        bundle_id=bundle_id,
        user_hmac=None,
        identity_key_id=correlation_key_id,
        session_ref=session_ref,
        manifest_path=manifest_path,
        manifest_digest=manifest_digest,
        payload_path=payload_path,
        payload_digest=payload_digest,
        payload_size_bytes=payload_size,
        model_profile_id=_safe_text("model_profile_id", manifest["model_profile_id"]),
        model_digest=str(manifest["model_digest"]),
        runtime_profile_digest=str(manifest["runtime_profile_digest"]),
        last_turn_index=len(request_refs),
        source_request_ref=str(request_refs[-1]),
        n_ctx=_positive_int("n_ctx", manifest["n_ctx"]),
        n_seq_max=_positive_int("n_seq_max", manifest["n_seq_max"]),
        n_stream=_positive_int("n_stream", manifest["n_stream"]),
        n_layer=_positive_int("n_layer", manifest["n_layer"]),
        v_trans=_boolean("v_trans", manifest["v_trans"]),
        key_ggml_type_id=_nonnegative_int(
            "key_ggml_type_id", manifest["key_ggml_type_id"]
        ),
        value_ggml_type_id=_nonnegative_int(
            "value_ggml_type_id", manifest["value_ggml_type_id"]
        ),
        next_position=next_position,
        source_seq_id=_nonnegative_int("source_seq_id", manifest["source_seq_id"]),
        source_stream_index=_nonnegative_int(
            "source_stream_index", manifest["source_stream_index"]
        ),
        position_range=_position_range(manifest["position_range"], next_position),
        subject_kind=RehostSubjectKind.ENDPOINT_REF,
        endpoint_ref=endpoint_ref,
    )


def _validate_endpoint_inventory(
    endpoints_root: Path, expected_sessions: Mapping[str, set[str]]
) -> None:
    _directory(endpoints_root, "bundle endpoints root")
    if _inventory(endpoints_root) != {f"ep_{value}" for value in expected_sessions}:
        raise PortableBundleError("bundle endpoint inventory is invalid")
    for endpoint_ref, sessions in expected_sessions.items():
        endpoint_root = endpoints_root / f"ep_{endpoint_ref}"
        if _inventory(endpoint_root) != {"sessions"}:
            raise PortableBundleError("bundle endpoint inventory is invalid")
        sessions_root = endpoint_root / "sessions"
        _directory(sessions_root, "bundle endpoint sessions root")
        if _inventory(sessions_root) != sessions:
            raise PortableBundleError("bundle endpoint sessions inventory is invalid")


def _entry_from_manifest(
    manifest: Mapping[str, object],
    *,
    run_id: str,
    bundle_id: str,
    user_hmac: str,
    session_ref: str,
    manifest_path: Path,
    manifest_digest: str,
    payload_path: Path,
    payload_digest: str,
    payload_size: int,
) -> PortableBundleEntry:
    bindings = {
        "schema_version": "1.0",
        "run_id": run_id,
        "bundle_id": bundle_id,
        "user_hmac": user_hmac,
        "session_ref": session_ref,
        "application_context_mode": "cumulative_context",
        "payload_file": "runtime_state_payload.bin",
        "payload_digest": payload_digest,
        "payload_size_bytes": payload_size,
        "contains_raw_prompt": False,
        "contains_raw_response": False,
        "contains_explicit_token_history": False,
        "restore_api": "llama_state_seq_set_data_ext",
    }
    if set(manifest) != _SESSION_MANIFEST_FIELDS or any(
        type(manifest.get(name)) is not type(value) or manifest.get(name) != value
        for name, value in bindings.items()
    ):
        raise PortableBundleError("rehost manifest binding is invalid")
    identity_key_id = _safe_text("identity_key_id", manifest["identity_key_id"])
    model_profile_id = _safe_text("model_profile_id", manifest["model_profile_id"])
    source_request_ref = _safe_text("source_request_ref", manifest["source_request_ref"])
    model_digest = _digest("model_digest", manifest["model_digest"])
    runtime_profile_digest = _digest(
        "runtime_profile_digest",
        manifest["runtime_profile_digest"],
    )
    if manifest["memory_family"] != "simple_kv":
        raise PortableBundleError("memory_family is invalid")
    next_position = _positive_int("next_position", manifest["next_position"])
    position_range = _position_range(manifest["position_range"], next_position)
    return PortableBundleEntry(
        bundle_id=bundle_id,
        user_hmac=user_hmac,
        identity_key_id=identity_key_id,
        session_ref=session_ref,
        manifest_path=manifest_path,
        manifest_digest=manifest_digest,
        payload_path=payload_path,
        payload_digest=payload_digest,
        payload_size_bytes=payload_size,
        model_profile_id=model_profile_id,
        model_digest=model_digest,
        runtime_profile_digest=runtime_profile_digest,
        last_turn_index=_positive_int("last_turn_index", manifest["last_turn_index"]),
        source_request_ref=source_request_ref,
        n_ctx=_positive_int("n_ctx", manifest["n_ctx"]),
        n_seq_max=_positive_int("n_seq_max", manifest["n_seq_max"]),
        n_stream=_positive_int("n_stream", manifest["n_stream"]),
        n_layer=_positive_int("n_layer", manifest["n_layer"]),
        v_trans=_boolean("v_trans", manifest["v_trans"]),
        key_ggml_type_id=_nonnegative_int(
            "key_ggml_type_id", manifest["key_ggml_type_id"]
        ),
        value_ggml_type_id=_nonnegative_int(
            "value_ggml_type_id", manifest["value_ggml_type_id"]
        ),
        next_position=next_position,
        source_seq_id=_nonnegative_int("source_seq_id", manifest["source_seq_id"]),
        source_stream_index=_nonnegative_int(
            "source_stream_index", manifest["source_stream_index"]
        ),
        position_range=position_range,
    )


def _validate_user_inventory(
    users_root: Path,
    expected_sessions: Mapping[str, set[str]],
) -> None:
    _directory(users_root, "bundle users root")
    expected_users = {f"uh_{value}" for value in expected_sessions}
    if _inventory(users_root) != expected_users:
        raise PortableBundleError("bundle user inventory is invalid")
    for user_hmac, sessions in expected_sessions.items():
        user_root = users_root / f"uh_{user_hmac}"
        if _inventory(user_root) != {"sessions"}:
            raise PortableBundleError("bundle user inventory is invalid")
        sessions_root = user_root / "sessions"
        _directory(sessions_root, "bundle sessions root")
        if _inventory(sessions_root) != sessions:
            raise PortableBundleError("bundle sessions inventory is invalid")


def _canonical_mapping(path: Path, description: str) -> tuple[dict[str, object], str]:
    raw, digest = _read_regular_file(path, limit=_MAX_METADATA_BYTES)

    def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise PortableBundleError(f"{description} has duplicate fields")
            result[key] = value
        return result

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=unique)
    except PortableBundleError:
        raise
    except (UnicodeError, json.JSONDecodeError) as error:
        raise PortableBundleError(f"{description} is invalid") from error
    if not isinstance(value, dict) or raw != canonical_json(value) + b"\n":
        raise PortableBundleError(f"{description} is not canonical")
    return value, digest


def _read_regular_file(path: Path, limit: int | None = None) -> tuple[bytes, str]:
    try:
        before = path.lstat()
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise PortableBundleError("bundle component is not a regular file")
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
    except PortableBundleError:
        raise
    except OSError as error:
        raise PortableBundleError("bundle component is unavailable") from error
    try:
        opened = os.fstat(descriptor)
        if opened.st_dev != before.st_dev or opened.st_ino != before.st_ino:
            raise PortableBundleError("bundle component identity changed")
        chunks: list[bytes] = []
        size = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if limit is not None and size > limit:
                raise PortableBundleError("bundle component exceeds size limit")
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        opened.st_size != after.st_size
        or opened.st_mtime_ns != after.st_mtime_ns
        or len(b"".join(chunks)) != after.st_size
    ):
        raise PortableBundleError("bundle component changed while reading")
    payload = b"".join(chunks)
    return payload, "sha256:" + hashlib.sha256(payload).hexdigest()


def _absolute_directory(path: Path, description: str) -> Path:
    if not isinstance(path, Path) or not path.is_absolute():
        raise PortableBundleError(f"{description} must be absolute")
    candidate = Path(os.path.abspath(os.fspath(path)))
    _directory(candidate, description)
    return candidate


def _directory(path: Path, description: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise PortableBundleError(f"{description} is unavailable") from error
    is_reparse = bool(
        getattr(metadata, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )
    if stat.S_ISLNK(metadata.st_mode) or is_reparse or not stat.S_ISDIR(metadata.st_mode):
        raise PortableBundleError(f"{description} is not a regular directory")


def _inventory(path: Path) -> set[str]:
    _directory(path, "bundle directory")
    try:
        return {item.name for item in path.iterdir()}
    except OSError as error:
        raise PortableBundleError("bundle directory inventory is unavailable") from error


def _relative_path(value: object, description: str) -> PurePosixPath:
    if not isinstance(value, str) or "\\" in value:
        raise PortableBundleError(f"{description} is invalid")
    candidate = PurePosixPath(value)
    if (
        candidate.is_absolute()
        or not candidate.parts
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        raise PortableBundleError(f"{description} is invalid")
    return candidate


def _pattern(name: str, value: object, pattern: re.Pattern[str]) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise PortableBundleError(f"{name} is invalid")
    return value


def _safe_text(name: str, value: object) -> str:
    return _pattern(name, value, _SAFE_TEXT)


def _digest(name: str, value: object) -> str:
    return _pattern(name, value, _SHA256)


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PortableBundleError(f"{name} is invalid")
    return value


def _nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PortableBundleError(f"{name} is invalid")
    return value


def _boolean(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise PortableBundleError(f"{name} is invalid")
    return value


def _position_range(value: object, next_position: int) -> tuple[int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise PortableBundleError("position_range is invalid")
    result = (value[0], value[1])
    if result != (0, next_position - 1):
        raise PortableBundleError("position_range is invalid")
    return result
