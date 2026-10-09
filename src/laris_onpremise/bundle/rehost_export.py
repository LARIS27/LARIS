from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import hmac
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import uuid

from laris_onpremise.kv_state.acquisition.contracts import RuntimeStatePayloadCommit
from laris_onpremise.kv_state.evidence.manifest import (
    AcquisitionManifestError,
    parse_acquisition_manifest_bytes,
)
from laris_onpremise.kv_state.acquisition.payload import (
    PayloadValidationError,
    inspect_simple_kv_payload,
)
from laris_onpremise.bundle.rehost_metadata import RehostRuntimeMetadata


class RehostExportError(RuntimeError):
    """Raised when a minimal continuation bundle cannot be sealed safely."""


_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_USER_HMAC = re.compile(r"[A-Za-z0-9_-]{43}\Z")
_SESSION_REF = re.compile(r"sr_[0-9a-f]{64}\Z")
_SAFE_REF = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,191}\Z")
_RUN_ID = re.compile(r"run-[A-Za-z0-9._-]{1,191}\Z")
_PAYLOAD_FILE = "runtime_state_payload.bin"
_MANIFEST_FILE = "rehost_manifest.json"
_INDEX_FILE = "bundle_index.json"
_PACKAGE_MANIFEST_FILE = "bundle_manifest.json"
_RUNTIME_METADATA_FILE = "rehost_metadata.json"


def _text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value or "\0" in value:
        raise RehostExportError(f"{name} must be non-empty text")
    return value


def _safe_ref(name: str, value: object) -> str:
    text = _text(name, value)
    if _SAFE_REF.fullmatch(text) is None or text in {".", ".."}:
        raise RehostExportError(f"{name} is not path-safe")
    return text


def _digest(name: str, value: object) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise RehostExportError(f"{name} must be lowercase sha256:<64 hex>")
    return value


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RehostExportError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RehostExportError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class ModelCompatibility:
    run_id: str
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

    def __post_init__(self) -> None:
        if _RUN_ID.fullmatch(self.run_id) is None:
            raise RehostExportError("run_id is invalid")
        _safe_ref("model_profile_id", self.model_profile_id)
        _digest("model_digest", self.model_digest)
        _digest("runtime_profile_digest", self.runtime_profile_digest)
        if self.memory_family != "simple_kv":
            raise RehostExportError("memory_family must be simple_kv")
        _positive_int("n_ctx", self.n_ctx)
        _positive_int("n_seq_max", self.n_seq_max)
        _positive_int("n_stream", self.n_stream)
        _positive_int("n_layer", self.n_layer)
        if not isinstance(self.v_trans, bool):
            raise RehostExportError("v_trans must be a boolean")
        _nonnegative_int("key_ggml_type_id", self.key_ggml_type_id)
        _nonnegative_int("value_ggml_type_id", self.value_ggml_type_id)
        if self.n_seq_max != self.n_stream:
            raise RehostExportError(
                "n_seq_max must equal n_stream for the qualified profile"
            )
        if self.n_ctx % self.n_stream:
            raise RehostExportError("n_ctx must divide evenly across n_stream")


@dataclass(frozen=True)
class AttributedRuntimeStatePayload:
    user_hmac: str
    identity_key_id: str
    session_ref: str
    application_context_mode: str
    turn_index: int
    request_ref: str
    lease_id: str
    slot_id: int
    slot_epoch: int
    observed_seq_id: int
    source_lineage_digest: str
    acquisition_manifest_path: Path
    acquisition_manifest_digest: str
    commit: RuntimeStatePayloadCommit

    def __post_init__(self) -> None:
        if _USER_HMAC.fullmatch(self.user_hmac) is None:
            raise RehostExportError("user_hmac is invalid")
        _safe_ref("identity_key_id", self.identity_key_id)
        if _SESSION_REF.fullmatch(self.session_ref) is None:
            raise RehostExportError("session_ref is invalid")
        if self.application_context_mode not in {
            "cumulative_context",
            "turn_only",
        }:
            raise RehostExportError("application_context_mode is invalid")
        _positive_int("turn_index", self.turn_index)
        _safe_ref("request_ref", self.request_ref)
        _safe_ref("lease_id", self.lease_id)
        _nonnegative_int("slot_id", self.slot_id)
        _positive_int("slot_epoch", self.slot_epoch)
        _nonnegative_int("observed_seq_id", self.observed_seq_id)
        _digest("source_lineage_digest", self.source_lineage_digest)
        if not isinstance(self.acquisition_manifest_path, Path):
            raise RehostExportError(
                "acquisition_manifest_path must be a pathlib.Path"
            )
        _digest("acquisition_manifest_digest", self.acquisition_manifest_digest)
        if not isinstance(self.commit, RuntimeStatePayloadCommit):
            raise RehostExportError("commit must be a RuntimeStatePayloadCommit")
        bindings = {
            "request_ref": self.request_ref,
            "lease_id": self.lease_id,
            "slot_id": self.slot_id,
            "slot_epoch": self.slot_epoch,
            "observed_seq_id": self.observed_seq_id,
        }
        for field, expected in bindings.items():
            if getattr(self.commit, field) != expected:
                raise RehostExportError(
                    f"attributed {field} does not match the payload commit"
                )


@dataclass(frozen=True)
class RehostBundleIndexEntry:
    bundle_id: str
    user_hmac: str
    session_ref: str
    state: str
    rehost_manifest_path: str
    rehost_manifest_digest: str
    payload_path: str
    payload_digest: str
    payload_size_bytes: int

    def to_mapping(self) -> dict[str, object]:
        return {
            "bundle_id": self.bundle_id,
            "user_hmac": self.user_hmac,
            "session_ref": self.session_ref,
            "state": self.state,
            "rehost_manifest_path": self.rehost_manifest_path,
            "rehost_manifest_digest": self.rehost_manifest_digest,
            "payload_path": self.payload_path,
            "payload_digest": self.payload_digest,
            "payload_size_bytes": self.payload_size_bytes,
        }


@dataclass(frozen=True)
class RehostBundleIndex:
    run_id: str
    path: Path
    digest: str
    entries: tuple[RehostBundleIndexEntry, ...]


@dataclass(frozen=True)
class _VerifiedSource:
    record: AttributedRuntimeStatePayload
    payload: bytes
    acquisition_manifest: dict[str, object]
    acquisition_manifest_digest: str
    source_stream_index: int


def export_cumulative_rehost_bundles(
    attributed: Sequence[AttributedRuntimeStatePayload],
    *,
    evidence_root: Path,
    rehost_root: Path,
    index_path: Path,
    model: ModelCompatibility,
    runtime_metadata: RehostRuntimeMetadata,
    gate_a_digest: str,
    evidence_root_digest: str,
) -> RehostBundleIndex:
    """Publish one latest cumulative payload per user/session as one package."""

    if not isinstance(model, ModelCompatibility):
        raise RehostExportError("model must be ModelCompatibility")
    if not isinstance(runtime_metadata, RehostRuntimeMetadata):
        raise RehostExportError(
            "runtime_metadata must be RehostRuntimeMetadata"
        )
    if runtime_metadata.source_runtime_profile_digest != model.runtime_profile_digest:
        raise RehostExportError(
            "runtime metadata profile differs from model compatibility"
        )
    gate_a_digest = _digest("gate_a_digest", gate_a_digest)
    evidence_root_digest = _digest(
        "evidence_root_digest", evidence_root_digest
    )
    if not isinstance(index_path, Path) or not index_path.is_absolute():
        raise RehostExportError("index_path must be absolute")
    index_path = Path(os.path.abspath(os.fspath(index_path)))
    if index_path.name != _INDEX_FILE:
        raise RehostExportError(f"index_path must end with {_INDEX_FILE}")
    package_root = index_path.parent
    run_root = _absolute_root(package_root.parent, "run_root", create=False)
    expected_evidence_root = run_root / "evidence"
    expected_rehost_root = run_root / "bundle"
    if _absolute_name(evidence_root) != expected_evidence_root:
        raise RehostExportError("evidence_root must be the current run evidence root")
    if _absolute_name(rehost_root) != expected_rehost_root:
        raise RehostExportError("rehost_root must be the current run rehost root")
    evidence_root = _absolute_root(evidence_root, "evidence_root", create=False)
    rehost_root = _absolute_root(rehost_root, "rehost_root", create=True)
    if package_root != rehost_root:
        raise RehostExportError("index_path must be inside the current run bundle root")
    if index_path.exists() or index_path.is_symlink():
        raise RehostExportError("completion index already exists")
    package_manifest_path = rehost_root / _PACKAGE_MANIFEST_FILE
    if package_manifest_path.exists() or package_manifest_path.is_symlink():
        raise RehostExportError("bundle manifest already exists")
    runtime_metadata_path = rehost_root / _RUNTIME_METADATA_FILE
    if runtime_metadata_path.exists() or runtime_metadata_path.is_symlink():
        raise RehostExportError("rehost metadata already exists")
    if model.run_id != run_root.name:
        raise RehostExportError("model run_id does not match the run directory")
    if not isinstance(attributed, Sequence) or not attributed:
        raise RehostExportError("attributed records must not be empty")
    if any(not isinstance(item, AttributedRuntimeStatePayload) for item in attributed):
        raise RehostExportError(
            "attributed records must contain AttributedRuntimeStatePayload"
        )
    _validate_global_attribution(attributed)

    groups: dict[tuple[str, str], list[AttributedRuntimeStatePayload]] = {}
    for record in attributed:
        if record.application_context_mode != "cumulative_context":
            raise RehostExportError(
                "only cumulative_context can produce a continuation bundle"
            )
        groups.setdefault((record.user_hmac, record.session_ref), []).append(record)

    selected: list[_VerifiedSource] = []
    for key, records in sorted(groups.items()):
        ordered = sorted(records, key=lambda item: item.turn_index)
        turns = [item.turn_index for item in ordered]
        if turns != list(range(1, len(ordered) + 1)):
            raise RehostExportError("session turn lineage is incomplete or duplicated")
        if len({item.identity_key_id for item in ordered}) != 1:
            raise RehostExportError("session identity_key_id changed within a run")
        selected.append(
            _verify_source(
                ordered[-1],
                evidence_root=evidence_root,
                model=model,
            )
        )

    session_prefixes = _unique_session_prefixes(
        tuple(item.record.session_ref for item in selected)
    )
    users_root = rehost_root / "users"
    _create_private_directory(users_root, "bundle users root")
    entries: list[RehostBundleIndexEntry] = []
    for verified in selected:
        record = verified.record
        session_digest = session_prefixes[record.session_ref]
        relative_root = PurePosixPath(
            "users",
            f"uh_{record.user_hmac}",
            "sessions",
            f"s_{session_digest}",
        )
        bundle_root = rehost_root / Path(*relative_root.parts)
        _create_private_directory(bundle_root.parent.parent, "bundle user root")
        _create_private_directory(bundle_root.parent, "bundle sessions root")
        if bundle_root.exists() or bundle_root.is_symlink():
            raise RehostExportError("rehost bundle directory already exists")
        bundle_id = derive_rehost_bundle_id(
            run_id=model.run_id,
            user_hmac=record.user_hmac,
            identity_key_id=record.identity_key_id,
            session_ref=record.session_ref,
            last_turn_index=record.turn_index,
            source_request_ref=record.request_ref,
            source_lineage_digest=record.source_lineage_digest,
            model_profile_id=model.model_profile_id,
            model_digest=model.model_digest,
            runtime_profile_digest=model.runtime_profile_digest,
            payload_digest=record.commit.payload_digest,
            acquisition_manifest_digest=verified.acquisition_manifest_digest,
            gate_a_digest=gate_a_digest,
            evidence_root_digest=evidence_root_digest,
        )
        manifest = _rehost_manifest(
            verified,
            model=model,
            bundle_id=bundle_id,
            gate_a_digest=gate_a_digest,
            evidence_root_digest=evidence_root_digest,
        )
        manifest_bytes = _canonical_json_bytes(manifest)
        _publish_bundle_directory(
            bundle_root,
            payload=verified.payload,
            manifest_bytes=manifest_bytes,
        )
        entries.append(
            RehostBundleIndexEntry(
                bundle_id=bundle_id,
                user_hmac=record.user_hmac,
                session_ref=record.session_ref,
                state="READY",
                rehost_manifest_path=str(relative_root / _MANIFEST_FILE),
                rehost_manifest_digest=_sha256(manifest_bytes),
                payload_path=str(relative_root / _PAYLOAD_FILE),
                payload_digest=record.commit.payload_digest,
                payload_size_bytes=record.commit.payload_size_bytes,
            )
        )

    runtime_metadata_bytes = _canonical_json_bytes(runtime_metadata.to_mapping())
    _publish_file_no_clobber(runtime_metadata_path, runtime_metadata_bytes)
    runtime_metadata_digest = _sha256(runtime_metadata_bytes)
    index_document = {
        "schema_version": "1.0",
        "run_id": model.run_id,
        "entries": [entry.to_mapping() for entry in entries],
    }
    index_bytes = _canonical_json_bytes(index_document)
    _publish_file_no_clobber(index_path, index_bytes)
    index_digest = _sha256(index_bytes)
    package_manifest = {
        "schema_version": "1.1",
        "package_type": "laris_rehost_bundle",
        "run_id": model.run_id,
        "source": {
            "evidence_root_digest": evidence_root_digest,
            "gate_a_result_digest": gate_a_digest,
        },
        "bundle_index": {
            "path": _INDEX_FILE,
            "digest": index_digest,
        },
        "rehost_metadata": {
            "path": _RUNTIME_METADATA_FILE,
            "digest": runtime_metadata_digest,
            "size_bytes": len(runtime_metadata_bytes),
        },
    }
    _publish_file_no_clobber(
        package_manifest_path,
        _canonical_json_bytes(package_manifest),
    )
    return RehostBundleIndex(
        model.run_id,
        index_path,
        index_digest,
        tuple(entries),
    )


def derive_rehost_bundle_id(
    *,
    run_id: str,
    user_hmac: str,
    identity_key_id: str,
    session_ref: str,
    last_turn_index: int,
    source_request_ref: str,
    source_lineage_digest: str,
    model_profile_id: str,
    model_digest: str,
    runtime_profile_digest: str,
    payload_digest: str,
    acquisition_manifest_digest: str,
    gate_a_digest: str,
    evidence_root_digest: str,
) -> str:
    """Derive a bundle identity from attribution, model, and evidence seals."""

    components = (
        run_id,
        user_hmac,
        identity_key_id,
        session_ref,
        str(last_turn_index),
        source_request_ref,
        source_lineage_digest,
        model_profile_id,
        model_digest,
        runtime_profile_digest,
        payload_digest,
        acquisition_manifest_digest,
        gate_a_digest,
        evidence_root_digest,
    )
    if any(not isinstance(value, str) or not value for value in components):
        raise RehostExportError("bundle identity components are invalid")
    return "rb_" + hashlib.sha256("\0".join(components).encode("utf-8")).hexdigest()


def _validate_global_attribution(
    attributed: Sequence[AttributedRuntimeStatePayload],
) -> None:
    unique_values: dict[str, set[object]] = {
        "request_ref": set(),
        "lease_id": set(),
        "capture_id": set(),
        "payload_path": set(),
        "acquisition_manifest_path": set(),
    }
    user_keys: dict[str, str] = {}
    session_owners: dict[str, tuple[str, str]] = {}
    for record in attributed:
        values: dict[str, object] = {
            "request_ref": record.request_ref,
            "lease_id": record.lease_id,
            "capture_id": record.commit.capture_id,
            "payload_path": _absolute_name(record.commit.payload_path),
            "acquisition_manifest_path": _absolute_name(
                record.acquisition_manifest_path
            ),
        }
        for name, value in values.items():
            if value in unique_values[name]:
                raise RehostExportError(
                    f"global attribution reuses {name} across records"
                )
            unique_values[name].add(value)

        prior_key = user_keys.setdefault(
            record.user_hmac, record.identity_key_id
        )
        if prior_key != record.identity_key_id:
            raise RehostExportError(
                "user_hmac maps to multiple identity_key_id values"
            )
        owner = (record.user_hmac, record.identity_key_id)
        prior_owner = session_owners.setdefault(record.session_ref, owner)
        if prior_owner != owner:
            raise RehostExportError(
                "session_ref maps to multiple user identities"
            )


def _verify_source(
    record: AttributedRuntimeStatePayload,
    *,
    evidence_root: Path,
    model: ModelCompatibility,
) -> _VerifiedSource:
    expected_root = evidence_root / "acquisition" / record.request_ref
    expected_payload = expected_root / _PAYLOAD_FILE
    expected_manifest = expected_root / "acquisition_manifest.json"
    _assert_regular_directory(
        evidence_root / "acquisition", "acquisition directory"
    )
    _assert_regular_directory(expected_root, "acquisition request directory")
    if _absolute_name(record.commit.payload_path) != _absolute_name(expected_payload):
        raise RehostExportError("payload path escapes the acquisition evidence root")
    if _absolute_name(record.acquisition_manifest_path) != _absolute_name(
        expected_manifest
    ):
        raise RehostExportError(
            "acquisition manifest path escapes the evidence root"
        )
    payload = _read_regular_file(expected_payload)
    observed_payload_digest = _sha256(payload)
    if (
        len(payload) != record.commit.payload_size_bytes
        or not hmac.compare_digest(
            observed_payload_digest, record.commit.payload_digest
        )
    ):
        raise RehostExportError("payload digest or size does not match the commit")
    manifest_bytes = _read_regular_file(expected_manifest)
    try:
        manifest = parse_acquisition_manifest_bytes(manifest_bytes)
    except AcquisitionManifestError as error:
        raise RehostExportError("acquisition manifest is invalid") from error
    acquisition_manifest_digest = _sha256(manifest_bytes)
    if not hmac.compare_digest(
        acquisition_manifest_digest, record.acquisition_manifest_digest
    ):
        raise RehostExportError("acquisition manifest digest does not match its seal")
    bindings = {
        "capture_id": record.commit.capture_id,
        "lease_id": record.lease_id,
        "request_ref": record.request_ref,
        "runtime_instance_id": record.commit.runtime_instance_id,
        "slot_id": record.slot_id,
        "slot_epoch": record.slot_epoch,
        "observed_seq_id": record.observed_seq_id,
        "memory_family": record.commit.memory_family,
        "payload_format": record.commit.payload_format,
        "payload_file": _PAYLOAD_FILE,
        "payload_size_bytes": record.commit.payload_size_bytes,
        "payload_digest": record.commit.payload_digest,
        "next_position": record.commit.next_position,
    }
    for field, expected in bindings.items():
        if type(manifest[field]) is not type(expected) or manifest[field] != expected:
            raise RehostExportError(
                f"acquisition manifest {field} does not match attribution"
            )
    if tuple(manifest["position_range"]) != record.commit.position_range:
        raise RehostExportError("acquisition manifest position_range mismatch")
    if tuple(tuple(item) for item in manifest["kv_cell_ranges"]) != record.commit.kv_cell_ranges:
        raise RehostExportError("acquisition manifest kv_cell_ranges mismatch")
    if manifest["layout_profile_digest"] != model.runtime_profile_digest:
        raise RehostExportError(
            "acquisition layout profile does not match the runtime profile"
        )
    try:
        summary = inspect_simple_kv_payload(payload)
    except PayloadValidationError as error:
        raise RehostExportError("runtime-state payload is invalid") from error
    if summary.n_stream != model.n_stream:
        raise RehostExportError("payload stream count is incompatible with model")
    if summary.n_layer != model.n_layer:
        raise RehostExportError("payload layer count is incompatible with model")
    if summary.v_trans is not model.v_trans:
        raise RehostExportError("payload v_trans is incompatible with runtime")
    if set(summary.key_ggml_type_ids) != {model.key_ggml_type_id}:
        raise RehostExportError("payload K cache type is incompatible with runtime")
    if set(summary.value_ggml_type_ids) != {model.value_ggml_type_id}:
        raise RehostExportError("payload V cache type is incompatible with runtime")
    if summary.nonempty_streams != (manifest["observed_stream_index"],):
        raise RehostExportError("payload does not contain exactly the observed stream")
    if summary.source_seq_ids != (record.observed_seq_id,):
        raise RehostExportError("payload source sequence is misattributed")
    expected_positions = tuple(range(record.commit.next_position))
    if tuple(sorted(summary.positions)) != expected_positions:
        raise RehostExportError(
            "latest cumulative payload does not have dense position coverage"
        )
    if record.commit.position_range != (0, record.commit.next_position - 1):
        raise RehostExportError("payload position range is not dense from zero")
    cell_count = sum(
        last - first + 1 for first, last in record.commit.kv_cell_ranges
    )
    if cell_count != summary.cell_count:
        raise RehostExportError("payload KV cell cardinality is inconsistent")
    if record.commit.next_position > model.n_ctx // model.n_stream:
        raise RehostExportError("payload exceeds the model context capacity")
    if record.observed_seq_id >= model.n_seq_max:
        raise RehostExportError("source sequence exceeds the runtime profile")
    if int(manifest["observed_stream_index"]) >= model.n_stream:
        raise RehostExportError("source stream exceeds the runtime profile")
    return _VerifiedSource(
        record=record,
        payload=payload,
        acquisition_manifest=manifest,
        acquisition_manifest_digest=acquisition_manifest_digest,
        source_stream_index=int(manifest["observed_stream_index"]),
    )


def _rehost_manifest(
    source: _VerifiedSource,
    *,
    model: ModelCompatibility,
    bundle_id: str,
    gate_a_digest: str,
    evidence_root_digest: str,
) -> dict[str, object]:
    record = source.record
    return {
        "schema_version": "1.0",
        "run_id": model.run_id,
        "bundle_id": bundle_id,
        "user_hmac": record.user_hmac,
        "identity_key_id": record.identity_key_id,
        "session_ref": record.session_ref,
        "application_context_mode": record.application_context_mode,
        "last_turn_index": record.turn_index,
        "source_request_ref": record.request_ref,
        "source_lineage_digest": record.source_lineage_digest,
        "model_profile_id": model.model_profile_id,
        "model_digest": model.model_digest,
        "runtime_profile_digest": model.runtime_profile_digest,
        "memory_family": model.memory_family,
        "n_ctx": model.n_ctx,
        "n_seq_max": model.n_seq_max,
        "n_stream": model.n_stream,
        "n_layer": model.n_layer,
        "v_trans": model.v_trans,
        "key_ggml_type_id": model.key_ggml_type_id,
        "value_ggml_type_id": model.value_ggml_type_id,
        "source_seq_id": record.observed_seq_id,
        "source_stream_index": source.source_stream_index,
        "destination_sequence_policy": "caller_selected_empty_sequence",
        "position_range": list(record.commit.position_range),
        "kv_cell_ranges": [list(item) for item in record.commit.kv_cell_ranges],
        "next_position": record.commit.next_position,
        "payload_format": record.commit.payload_format,
        "payload_file": _PAYLOAD_FILE,
        "payload_size_bytes": record.commit.payload_size_bytes,
        "payload_digest": record.commit.payload_digest,
        "acquisition_manifest_digest": source.acquisition_manifest_digest,
        "gate_a_result_digest": gate_a_digest,
        "evidence_root_digest": evidence_root_digest,
        "contains_raw_prompt": False,
        "contains_raw_response": False,
        "contains_explicit_token_history": False,
        "restore_api": "llama_state_seq_set_data_ext",
    }


def _canonical_json_bytes(document: Mapping[str, object]) -> bytes:
    try:
        return (
            json.dumps(
                dict(document),
                ensure_ascii=True,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise RehostExportError("bundle metadata is not canonical JSON") from error


def _sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _absolute_name(path: Path) -> Path:
    if not isinstance(path, Path) or not path.is_absolute():
        raise RehostExportError("artifact paths must be absolute")
    return Path(os.path.abspath(os.fspath(path)))


def _absolute_root(path: Path, name: str, *, create: bool) -> Path:
    candidate = _absolute_name(path)
    if create and not candidate.exists():
        try:
            _assert_regular_directory(candidate.parent, f"{name} parent")
            candidate.mkdir(mode=0o700)
        except RehostExportError:
            raise
        except OSError as error:
            raise RehostExportError(f"{name} cannot be created") from error
    try:
        metadata = candidate.lstat()
    except OSError as error:
        raise RehostExportError(f"{name} is unavailable") from error
    reparse = bool(
        getattr(metadata, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    )
    if stat.S_ISLNK(metadata.st_mode) or reparse or not stat.S_ISDIR(metadata.st_mode):
        raise RehostExportError(f"{name} must be a regular directory")
    return candidate


def _assert_regular_directory(path: Path, name: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise RehostExportError(f"{name} is unavailable") from error
    reparse = bool(
        getattr(metadata, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    )
    if stat.S_ISLNK(metadata.st_mode) or reparse or not stat.S_ISDIR(metadata.st_mode):
        raise RehostExportError(f"{name} must be a regular directory")


def _create_private_directory(path: Path, name: str) -> None:
    if path.exists() or path.is_symlink():
        _assert_regular_directory(path, name)
        return
    _assert_regular_directory(path.parent, f"{name} parent")
    try:
        path.mkdir(mode=0o700)
    except OSError as error:
        raise RehostExportError(f"{name} cannot be created") from error
    _assert_regular_directory(path, name)


def _unique_session_prefixes(session_refs: tuple[str, ...]) -> dict[str, str]:
    digests = {
        value: hashlib.sha256(value.encode("utf-8")).hexdigest()
        for value in session_refs
    }
    result: dict[str, str] = {}
    for session_ref, digest in digests.items():
        for length in range(16, 65):
            prefix = digest[:length]
            if all(
                other_ref == session_ref or not other_digest.startswith(prefix)
                for other_ref, other_digest in digests.items()
            ):
                result[session_ref] = prefix
                break
        else:
            raise RehostExportError("session digest collision cannot be resolved")
    return result


def _read_regular_file(path: Path) -> bytes:
    try:
        native = _native_path(path)
        before = os.lstat(native)
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise RehostExportError(f"artifact is not a regular file: {path.name}")
        flags = (
            os.O_RDONLY
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(native, flags)
    except RehostExportError:
        raise
    except OSError as error:
        raise RehostExportError(f"artifact cannot be opened: {path.name}") from error
    try:
        observed = os.fstat(descriptor)
        if not stat.S_ISREG(observed.st_mode):
            raise RehostExportError(f"artifact is not regular: {path.name}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        before.st_dev != observed.st_dev
        or before.st_ino != observed.st_ino
        or observed.st_size != after.st_size
        or observed.st_mtime_ns != after.st_mtime_ns
        or len(payload) != after.st_size
    ):
        raise RehostExportError(f"artifact changed while read: {path.name}")
    return payload


def _publish_bundle_directory(
    final_root: Path,
    *,
    payload: bytes,
    manifest_bytes: bytes,
) -> None:
    sessions_root = final_root.parent
    _assert_regular_directory(sessions_root, "bundle sessions root")
    staging = sessions_root / f".t-{uuid.uuid4().hex[:8]}"
    try:
        staging.mkdir(mode=0o700)
        _write_private_file(staging / _PAYLOAD_FILE, payload)
        _write_private_file(staging / _MANIFEST_FILE, manifest_bytes)
        _fsync_directory(staging)
        try:
            os.rename(_native_path(staging), _native_path(final_root))
        except OSError as error:
            raise RehostExportError(
                "rehost bundle directory already exists or cannot be published"
            ) from error
        _fsync_directory(sessions_root)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _write_private_file(path: Path, payload: bytes) -> None:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            _native_path(path),
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        view = memoryview(payload)
        offset = 0
        while offset < len(view):
            offset += os.write(descriptor, view[offset:])
        os.fsync(descriptor)
    except OSError as error:
        raise RehostExportError(f"artifact cannot be written: {path.name}") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _publish_file_no_clobber(path: Path, payload: bytes) -> None:
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        _write_private_file(temporary, payload)
        try:
            os.link(temporary, path)
        except OSError as error:
            raise RehostExportError("completion index already exists or cannot be published") from error
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()
            _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        if os.name == "nt":
            return
        raise RehostExportError("artifact directory cannot be opened for fsync")
    try:
        os.fsync(descriptor)
    except OSError:
        if os.name != "nt":
            raise RehostExportError("artifact directory fsync failed")
    finally:
        os.close(descriptor)


def _native_path(path: Path) -> str:
    value = os.fspath(path)
    if os.name == "nt" and path.is_absolute() and not value.startswith("\\\\?\\"):
        return "\\\\?\\" + value
    return value


__all__ = [
    "AttributedRuntimeStatePayload",
    "ModelCompatibility",
    "RehostBundleIndex",
    "RehostBundleIndexEntry",
    "RehostExportError",
    "derive_rehost_bundle_id",
    "export_cumulative_rehost_bundles",
]
