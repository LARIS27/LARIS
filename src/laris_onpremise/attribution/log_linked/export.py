from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile

from laris_onpremise.kv_state.acquisition.payload import inspect_simple_kv_payload
from laris_onpremise.bundle.rehost_metadata import RehostRuntimeMetadata
from laris_onpremise.core.records import canonical_json

from .contracts import AttributionDecision
from .correlation import AttributionResult
from .materialize import MaterializedEndpointState


class EndpointExportError(RuntimeError):
    """Raised when endpoint-correlated state cannot be exported safely."""


_RUN_ID = re.compile(r"run-[A-Za-z0-9._-]{1,191}\Z")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,191}\Z")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class EndpointBundleExportInputs:
    run_id: str
    attribution_result: AttributionResult
    materialized_state: MaterializedEndpointState
    evidence_package_manifest_digest: str
    attribution_receipt_digest: str
    model_profile_id: str
    model_digest: str
    runtime_profile_digest: str
    n_ctx: int
    n_seq_max: int
    request_refs: tuple[str, ...]
    runtime_metadata: RehostRuntimeMetadata


@dataclass(frozen=True)
class ExportedEndpointBundle:
    root: Path
    manifest_path: Path
    index_path: Path
    rehost_manifest_path: Path
    payload_path: Path
    bundle_id: str
    endpoint_ref: str
    session_ref: str


class EndpointBundleExporter:
    def export(
        self, root: Path | str, inputs: EndpointBundleExportInputs
    ) -> ExportedEndpointBundle:
        destination = Path(root)
        if destination.exists() or _is_link_like(destination):
            raise EndpointExportError("bundle root already exists")
        _validate_inputs(inputs)
        result = inputs.attribution_result
        state = inputs.materialized_state
        endpoint_ref = str(result.endpoint_ref)
        session_material = {
            "endpoint_ref": endpoint_ref,
            "runtime_instance_id": str(result.runtime_instance_id),
            "request_refs": list(inputs.request_refs),
            "state_unit_chain_digests": list(state.source_unit_digests),
        }
        session_ref = "sr_" + hashlib.sha256(
            _canonical(session_material)
        ).hexdigest()
        bundle_id = "rb_" + hashlib.sha256(
            _canonical(
                {
                    "attribution_receipt_digest": inputs.attribution_receipt_digest,
                    "payload_digest": state.payload_digest,
                    "session_ref": session_ref,
                }
            )
        ).hexdigest()
        session_directory = "s_" + hashlib.sha256(
            session_ref.encode("utf-8")
        ).hexdigest()[:16]
        relative_root = (
            Path("endpoints")
            / f"ep_{endpoint_ref}"
            / "sessions"
            / session_directory
        )
        manifest_relative = relative_root / "log_linked_rehost_manifest.json"
        payload_relative = relative_root / "runtime_state_payload.bin"

        rehost_manifest = {
            "schema_version": "1.0",
            "run_id": inputs.run_id,
            "bundle_id": bundle_id,
            "subject_kind": "ENDPOINT_REF",
            "endpoint_ref": endpoint_ref,
            "endpoint_type": str(result.endpoint_type),
            "correlation_key_id": str(result.correlation_key_id),
            "attribution_decision": result.final_decision.value,
            "session_ref": session_ref,
            "request_refs": list(inputs.request_refs),
            "attribution_receipt_digest": inputs.attribution_receipt_digest,
            "source_evidence_package_digest": (
                inputs.evidence_package_manifest_digest
            ),
            "source_unit_digests": list(state.source_unit_digests),
            "model_profile_id": inputs.model_profile_id,
            "model_digest": inputs.model_digest,
            "runtime_profile_digest": inputs.runtime_profile_digest,
            "memory_family": "simple_kv",
            "n_ctx": inputs.n_ctx,
            "n_seq_max": inputs.n_seq_max,
            "n_stream": state.n_stream,
            "n_layer": state.n_layer,
            "v_trans": state.v_trans,
            "key_ggml_type_id": state.key_ggml_type_id,
            "value_ggml_type_id": state.value_ggml_type_id,
            "source_seq_id": state.source_seq_id,
            "source_stream_index": state.source_stream_index,
            "destination_sequence_policy": "caller_selected_empty_sequence",
            "position_range": list(state.position_range),
            "next_position": state.next_position,
            "payload_format": "llama-simple-kv-seq-payload-v1",
            "payload_file": "runtime_state_payload.bin",
            "payload_size_bytes": len(state.payload),
            "payload_digest": state.payload_digest,
            "contains_raw_prompt": False,
            "contains_raw_response": False,
            "contains_explicit_token_history": False,
            "restore_api": "llama_state_seq_set_data_ext",
        }
        _validate_rehost_manifest(rehost_manifest)
        rehost_bytes = _canonical(rehost_manifest)
        index = {
            "schema_version": "2.0",
            "run_id": inputs.run_id,
            "subject_kind": "ENDPOINT_REF",
            "entries": [
                {
                    "bundle_id": bundle_id,
                    "endpoint_ref": endpoint_ref,
                    "endpoint_type": str(result.endpoint_type),
                    "correlation_key_id": str(result.correlation_key_id),
                    "attribution_decision": result.final_decision.value,
                    "session_ref": session_ref,
                    "state": "READY",
                    "rehost_manifest_path": manifest_relative.as_posix(),
                    "rehost_manifest_digest": _digest(rehost_bytes),
                    "payload_path": payload_relative.as_posix(),
                    "payload_digest": state.payload_digest,
                    "payload_size_bytes": len(state.payload),
                }
            ],
        }
        index_bytes = _canonical(index)
        metadata_bytes = _canonical(inputs.runtime_metadata.to_mapping())
        package = {
            "schema_version": "2.0",
            "package_type": "laris_rehost_bundle",
            "subject_kind": "ENDPOINT_REF",
            "run_id": inputs.run_id,
            "source": {
                "evidence_package_manifest_digest": (
                    inputs.evidence_package_manifest_digest
                )
            },
            "bundle_index": {
                "path": "bundle_index.json",
                "digest": _digest(index_bytes),
            },
            "rehost_metadata": {
                "path": "rehost_metadata.json",
                "digest": _digest(metadata_bytes),
                "size_bytes": len(metadata_bytes),
            },
        }
        package_bytes = _canonical(package)

        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name}.staging-", dir=str(destination.parent)
            )
        )
        try:
            session_root = staging / relative_root
            session_root.mkdir(mode=0o700, parents=True)
            _private_directories(staging)
            _write(session_root / "runtime_state_payload.bin", state.payload)
            _write(session_root / "log_linked_rehost_manifest.json", rehost_bytes)
            _write(staging / "rehost_metadata.json", metadata_bytes)
            _write(staging / "bundle_index.json", index_bytes)
            _write(staging / "bundle_manifest.json", package_bytes)
            if destination.exists() or _is_link_like(destination):
                raise EndpointExportError("bundle root already exists")
            os.rename(staging, destination)
        except Exception:
            if staging.exists() and staging.parent == destination.parent:
                shutil.rmtree(staging)
            raise
        return ExportedEndpointBundle(
            root=destination,
            manifest_path=destination / "bundle_manifest.json",
            index_path=destination / "bundle_index.json",
            rehost_manifest_path=destination / manifest_relative,
            payload_path=destination / payload_relative,
            bundle_id=bundle_id,
            endpoint_ref=endpoint_ref,
            session_ref=session_ref,
        )


def _validate_inputs(inputs: EndpointBundleExportInputs) -> None:
    if not isinstance(inputs, EndpointBundleExportInputs):
        raise EndpointExportError("endpoint export inputs are invalid")
    if _RUN_ID.fullmatch(inputs.run_id) is None:
        raise EndpointExportError("run_id is invalid")
    result = inputs.attribution_result
    if not isinstance(result, AttributionResult) or result.final_decision not in {
        AttributionDecision.CERTIFIED,
        AttributionDecision.RANKED,
    }:
        raise EndpointExportError("only a resolved attribution can be exported")
    if result.endpoint_ref is None or result.endpoint_type not in {"IPV4", "IPV6"}:
        raise EndpointExportError("resolved attribution endpoint is unavailable")
    state = inputs.materialized_state
    if not isinstance(state, MaterializedEndpointState):
        raise EndpointExportError("materialized endpoint state is invalid")
    if _digest(state.payload) != state.payload_digest:
        raise EndpointExportError("materialized payload digest differs")
    try:
        summary = inspect_simple_kv_payload(state.payload)
    except ValueError as error:
        raise EndpointExportError("materialized payload is invalid") from error
    if summary.source_seq_ids != (state.source_seq_id,) or summary.positions != state.positions:
        raise EndpointExportError("materialized payload metadata differs")
    for name in (
        "evidence_package_manifest_digest",
        "attribution_receipt_digest",
        "model_digest",
        "runtime_profile_digest",
    ):
        if _DIGEST.fullmatch(str(getattr(inputs, name))) is None:
            raise EndpointExportError(f"{name} is invalid")
    if _SAFE_ID.fullmatch(inputs.model_profile_id) is None:
        raise EndpointExportError("model_profile_id is invalid")
    if (
        isinstance(inputs.n_ctx, bool)
        or not isinstance(inputs.n_ctx, int)
        or inputs.n_ctx <= 0
        or isinstance(inputs.n_seq_max, bool)
        or not isinstance(inputs.n_seq_max, int)
        or inputs.n_seq_max != state.n_stream
        or inputs.n_ctx % inputs.n_seq_max
    ):
        raise EndpointExportError("runtime dimensions are invalid")
    if (
        not isinstance(inputs.request_refs, tuple)
        or not inputs.request_refs
        or inputs.request_refs != tuple(sorted(set(inputs.request_refs)))
        or any(_SAFE_ID.fullmatch(value) is None for value in inputs.request_refs)
    ):
        raise EndpointExportError("request_refs must be sorted and unique")
    if result.request_ref not in inputs.request_refs:
        raise EndpointExportError("attributed request is absent from request_refs")
    if not isinstance(inputs.runtime_metadata, RehostRuntimeMetadata):
        raise EndpointExportError("runtime metadata is invalid")
    if inputs.runtime_metadata.source_runtime_profile_digest != inputs.runtime_profile_digest:
        raise EndpointExportError("runtime metadata profile differs")


def _validate_rehost_manifest(value: dict[str, object]) -> None:
    forbidden = {"user_hmac", "prompt", "response", "token_ids"}
    if forbidden.intersection(value):
        raise EndpointExportError("endpoint manifest contains a forbidden field")
    if value["position_range"] != [0, value["next_position"] - 1]:
        raise EndpointExportError("endpoint payload positions must be dense")


def _canonical(value: dict[str, object]) -> bytes:
    return canonical_json(value) + b"\n"


def _digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _write(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise EndpointExportError("bundle file write failed")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if os.name == "posix":
        os.chmod(path, 0o600, follow_symlinks=False)


def _private_directories(root: Path) -> None:
    if os.name != "posix":
        return
    os.chmod(root, 0o700, follow_symlinks=False)
    for path in root.rglob("*"):
        if path.is_dir():
            os.chmod(path, 0o700, follow_symlinks=False)


def _is_link_like(path: Path) -> bool:
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(metadata.st_mode):
        return True
    return bool(
        getattr(metadata, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    )


__all__ = [
    "EndpointBundleExportInputs",
    "EndpointBundleExporter",
    "EndpointExportError",
    "ExportedEndpointBundle",
]
