from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Mapping

from laris_onpremise.kv_state.acquisition.payload import (
    PayloadValidationError,
    inspect_simple_kv_payload,
)
from laris_onpremise.bundle.rehost_export import derive_rehost_bundle_id

from .contracts import RehostCase, RehostRuntimeProfile, RehostValidationError


class RehostPreparationError(RuntimeError):
    """Raised when a rehost bundle cannot be selected and verified safely."""


_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_RUN_ID = re.compile(r"run-[A-Za-z0-9._-]{1,191}\Z")
_BUNDLE_ID = re.compile(r"rb_[0-9a-f]{64}\Z")
_USER_HMAC = re.compile(r"[A-Za-z0-9_-]{43}\Z")
_SESSION_REF = re.compile(r"sr_[0-9a-f]{64}\Z")
_SAFE_REF = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,191}\Z")
_INDEX_FIELDS = {"schema_version", "run_id", "entries"}
_ENTRY_FIELDS = {
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
_MANIFEST_FIELDS = {
    "schema_version", "run_id", "bundle_id", "user_hmac",
    "identity_key_id", "session_ref", "application_context_mode",
    "last_turn_index", "source_request_ref", "source_lineage_digest",
    "model_profile_id", "model_digest", "runtime_profile_digest",
    "memory_family", "n_ctx", "n_seq_max", "n_stream", "n_layer",
    "v_trans", "key_ggml_type_id", "value_ggml_type_id",
    "source_seq_id", "source_stream_index", "destination_sequence_policy",
    "position_range", "kv_cell_ranges", "next_position", "payload_format",
    "payload_file", "payload_size_bytes", "payload_digest",
    "acquisition_manifest_digest", "gate_a_result_digest",
    "evidence_root_digest", "contains_raw_prompt", "contains_raw_response",
    "contains_explicit_token_history", "restore_api",
}


def _reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise RehostPreparationError("JSON contains a duplicate key")
        result[key] = value
    return result


def _canonical_bytes(value: Mapping[str, object]) -> bytes:
    try:
        return (
            json.dumps(
                dict(value),
                ensure_ascii=True,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise RehostPreparationError("JSON artifact is not canonical") from error


def _read_regular_bytes(path: Path) -> bytes:
    try:
        before = path.lstat()
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise RehostPreparationError(f"{path.name} is not a regular file")
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
    except RehostPreparationError:
        raise
    except OSError as error:
        raise RehostPreparationError(f"cannot open {path.name}") from error
    try:
        observed = os.fstat(descriptor)
        if not stat.S_ISREG(observed.st_mode):
            raise RehostPreparationError(f"{path.name} is not a regular file")
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
        (before.st_dev, before.st_ino) != (observed.st_dev, observed.st_ino)
        or (observed.st_size, observed.st_mtime_ns, observed.st_ctime_ns)
        != (after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        or len(payload) != after.st_size
    ):
        raise RehostPreparationError(f"{path.name} changed while read")
    return payload


def _load_canonical_object(path: Path) -> tuple[dict[str, object], bytes]:
    raw = _read_regular_bytes(path)
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RehostPreparationError(f"cannot parse {path.name}") from error
    if not isinstance(value, dict):
        raise RehostPreparationError(f"{path.name} must contain an object")
    if raw != _canonical_bytes(value):
        raise RehostPreparationError(f"{path.name} is not canonical JSON")
    return value, raw


def load_json_object(path: Path) -> dict[str, object]:
    """Compatibility helper that now requires canonical, duplicate-free JSON."""

    return _load_canonical_object(path)[0]


def _text(name: str, value: object, pattern: re.Pattern[str] = _SAFE_REF) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise RehostPreparationError(f"{name} is invalid")
    return value


def _digest(name: str, value: object) -> str:
    return _text(name, value, _SHA256)


def _integer(name: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RehostPreparationError(f"{name} is invalid")
    return value


def _range(name: str, value: object) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise RehostPreparationError(f"{name} is invalid")
    first = _integer(f"{name}[0]", value[0])
    last = _integer(f"{name}[1]", value[1])
    if first > last:
        raise RehostPreparationError(f"{name} is invalid")
    return first, last


def _safe_directory(path: Path, name: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise RehostPreparationError(f"{name} is unavailable") from error
    reparse = bool(
        getattr(metadata, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    )
    if stat.S_ISLNK(metadata.st_mode) or reparse or not stat.S_ISDIR(metadata.st_mode):
        raise RehostPreparationError(f"{name} is not a regular directory")


def _entry_path(run_root: Path, raw: object, expected: PurePosixPath) -> Path:
    if not isinstance(raw, str) or "\\" in raw:
        raise RehostPreparationError("index path is invalid")
    relative = PurePosixPath(raw)
    if relative.is_absolute() or ".." in relative.parts or relative != expected:
        raise RehostPreparationError("index path does not match its identity")
    return run_root.joinpath(*relative.parts)


def _validate_manifest(value: dict[str, object]) -> None:
    if set(value) != _MANIFEST_FIELDS:
        raise RehostPreparationError("rehost manifest field set is invalid")
    if value["schema_version"] != "1.0":
        raise RehostPreparationError("rehost manifest schema version is invalid")
    _text("run_id", value["run_id"], _RUN_ID)
    _text("bundle_id", value["bundle_id"], _BUNDLE_ID)
    _text("user_hmac", value["user_hmac"], _USER_HMAC)
    _text("identity_key_id", value["identity_key_id"])
    _text("session_ref", value["session_ref"], _SESSION_REF)
    _text("source_request_ref", value["source_request_ref"])
    _text("model_profile_id", value["model_profile_id"])
    for field in (
        "source_lineage_digest", "model_digest", "runtime_profile_digest",
        "payload_digest", "acquisition_manifest_digest",
        "gate_a_result_digest", "evidence_root_digest",
    ):
        _digest(field, value[field])
    for field in ("last_turn_index", "n_ctx", "n_seq_max", "n_stream", "n_layer", "next_position", "payload_size_bytes"):
        _integer(field, value[field], minimum=1)
    for field in ("source_seq_id", "source_stream_index"):
        _integer(field, value[field])
    if not isinstance(value["v_trans"], bool):
        raise RehostPreparationError("v_trans is invalid")
    _integer("key_ggml_type_id", value["key_ggml_type_id"])
    _integer("value_ggml_type_id", value["value_ggml_type_id"])
    if value["application_context_mode"] != "cumulative_context":
        raise RehostPreparationError("manifest is not cumulative_context")
    if value["memory_family"] != "simple_kv":
        raise RehostPreparationError("manifest memory family is unsupported")
    if value["destination_sequence_policy"] != "caller_selected_empty_sequence":
        raise RehostPreparationError("destination sequence policy is invalid")
    if value["payload_format"] != "llama-simple-kv-seq-payload-v1":
        raise RehostPreparationError("payload format is unsupported")
    if value["payload_file"] != "runtime_state_payload.bin":
        raise RehostPreparationError("payload filename is invalid")
    if value["restore_api"] != "llama_state_seq_set_data_ext":
        raise RehostPreparationError("restore API is invalid")
    if any(
        value[field] is not False
        for field in (
            "contains_raw_prompt", "contains_raw_response",
            "contains_explicit_token_history",
        )
    ):
        raise RehostPreparationError("rehost privacy policy is invalid")
    position_range = _range("position_range", value["position_range"])
    next_position = int(value["next_position"])
    if position_range != (0, next_position - 1):
        raise RehostPreparationError("position range is not dense from zero")
    ranges = value["kv_cell_ranges"]
    if not isinstance(ranges, list) or not ranges:
        raise RehostPreparationError("kv_cell_ranges is invalid")
    parsed = tuple(_range("kv_cell_ranges", item) for item in ranges)
    if tuple(sorted(parsed)) != parsed:
        raise RehostPreparationError("kv_cell_ranges is not ordered")
    for previous, current in zip(parsed, parsed[1:]):
        if current[0] <= previous[1]:
            raise RehostPreparationError("kv_cell_ranges overlap")


def prepare_rehost_cases(
    run_root: Path,
    *,
    user_hmac: str | None,
    session_ref: str | None,
    bundle_id: str | None,
    destination_seq_id: int,
    expected_model_profile_id: str,
    expected_model_digest: str,
    expected_runtime_profile: RehostRuntimeProfile,
    expected_index_digest: str,
) -> tuple[RehostCase, ...]:
    """Select and verify READY cumulative payloads from the completion index."""

    run_root = Path(os.path.abspath(os.fspath(run_root)))
    if not run_root.is_absolute():
        raise RehostPreparationError("run_root must be absolute")
    _safe_directory(run_root, "run_root")
    if _RUN_ID.fullmatch(run_root.name) is None:
        raise RehostPreparationError("run_root name is invalid")
    if user_hmac is not None:
        _text("user_hmac", user_hmac, _USER_HMAC)
    if session_ref is not None:
        _text("session_ref", session_ref, _SESSION_REF)
    if bundle_id is not None:
        _text("bundle_id", bundle_id, _BUNDLE_ID)
    if user_hmac is None and bundle_id is None:
        raise RehostPreparationError("user_hmac or bundle_id is required")
    _integer("destination_seq_id", destination_seq_id)
    _text("expected_model_profile_id", expected_model_profile_id)
    _digest("expected_model_digest", expected_model_digest)
    if not isinstance(expected_runtime_profile, RehostRuntimeProfile):
        raise RehostPreparationError(
            "expected_runtime_profile must be a qualified runtime profile"
        )
    _digest("expected_index_digest", expected_index_digest)

    index, index_bytes = _load_canonical_object(
        run_root / "rehost_bundle_index.json"
    )
    observed_index_digest = "sha256:" + hashlib.sha256(index_bytes).hexdigest()
    if not hmac.compare_digest(observed_index_digest, expected_index_digest):
        raise RehostPreparationError("rehost bundle index digest is not anchored")
    if set(index) != _INDEX_FIELDS or index.get("schema_version") != "1.0":
        raise RehostPreparationError("rehost bundle index field set is invalid")
    if index.get("run_id") != run_root.name:
        raise RehostPreparationError("index run_id does not match run_root")
    entries = index.get("entries")
    if not isinstance(entries, list) or not entries:
        raise RehostPreparationError("rehost bundle index has no entries")

    selected: list[dict[str, object]] = []
    identities: set[tuple[str, str, str]] = set()
    for raw_entry in entries:
        if not isinstance(raw_entry, dict) or set(raw_entry) != _ENTRY_FIELDS:
            raise RehostPreparationError("index entry field set is invalid")
        current_bundle = _text("bundle_id", raw_entry["bundle_id"], _BUNDLE_ID)
        current_user = _text("user_hmac", raw_entry["user_hmac"], _USER_HMAC)
        current_session = _text("session_ref", raw_entry["session_ref"], _SESSION_REF)
        identity = (current_bundle, current_user, current_session)
        if identity in identities:
            raise RehostPreparationError("index contains a duplicate entry")
        identities.add(identity)
        if raw_entry["state"] != "READY":
            raise RehostPreparationError("index contains a non-READY entry")
        _digest("rehost_manifest_digest", raw_entry["rehost_manifest_digest"])
        _digest("payload_digest", raw_entry["payload_digest"])
        _integer("payload_size_bytes", raw_entry["payload_size_bytes"], minimum=1)
        if bundle_id is not None and current_bundle != bundle_id:
            continue
        if user_hmac is not None and current_user != user_hmac:
            continue
        if session_ref is not None and current_session != session_ref:
            continue
        selected.append(raw_entry)
    if not selected:
        raise RehostPreparationError("no matching READY rehost bundle exists")
    if user_hmac is not None and session_ref is None and bundle_id is None and len(selected) > 1:
        raise RehostPreparationError("user has multiple sessions; session_ref is required")
    if bundle_id is not None and len(selected) != 1:
        raise RehostPreparationError("bundle_id is not unique")

    cases: list[RehostCase] = []
    for entry in selected:
        current_user = str(entry["user_hmac"])
        current_session = str(entry["session_ref"])
        raw_manifest_path = entry["rehost_manifest_path"]
        if not isinstance(raw_manifest_path, str) or "\\" in raw_manifest_path:
            raise RehostPreparationError("index path is invalid")
        manifest_relative = PurePosixPath(raw_manifest_path)
        if len(manifest_relative.parts) != 4:
            raise RehostPreparationError("index path does not match its identity")
        session_component = manifest_relative.parts[2]
        if not session_component.startswith("s_"):
            raise RehostPreparationError("index session path is invalid")
        session_digest = session_component.removeprefix("s_")
        full_session_digest = hashlib.sha256(
            current_session.encode("utf-8")
        ).hexdigest()
        if (
            not 16 <= len(session_digest) <= 64
            or not full_session_digest.startswith(session_digest)
        ):
            raise RehostPreparationError("index session path digest is invalid")
        relative_root = PurePosixPath(
            "rehost", f"uh_{current_user}", f"s_{session_digest}"
        )
        manifest_path = _entry_path(
            run_root,
            entry["rehost_manifest_path"],
            relative_root / "rehost_manifest.json",
        )
        payload_path = _entry_path(
            run_root,
            entry["payload_path"],
            relative_root / "runtime_state_payload.bin",
        )
        _safe_directory(run_root / "rehost", "rehost root")
        _safe_directory(manifest_path.parent.parent, "user rehost root")
        _safe_directory(manifest_path.parent, "session rehost root")
        try:
            names = {path.name for path in manifest_path.parent.iterdir()}
        except OSError as error:
            raise RehostPreparationError("cannot inspect rehost bundle") from error
        if names != {"runtime_state_payload.bin", "rehost_manifest.json"}:
            raise RehostPreparationError("rehost bundle must contain exactly two files")

        manifest, manifest_bytes = _load_canonical_object(manifest_path)
        observed_manifest_digest = (
            "sha256:" + hashlib.sha256(manifest_bytes).hexdigest()
        )
        if not hmac.compare_digest(
            observed_manifest_digest, str(entry["rehost_manifest_digest"])
        ):
            raise RehostPreparationError("rehost manifest digest is invalid")
        _validate_manifest(manifest)
        cross = {
            "run_id": run_root.name,
            "bundle_id": entry["bundle_id"],
            "user_hmac": current_user,
            "session_ref": current_session,
        }
        for field, expected in cross.items():
            if manifest[field] != expected:
                raise RehostPreparationError(f"manifest {field} does not match index")
        if manifest["model_profile_id"] != expected_model_profile_id:
            raise RehostPreparationError("model profile does not match the bundle")
        if manifest["model_digest"] != expected_model_digest:
            raise RehostPreparationError("model digest does not match the bundle")
        expected_bundle_id = derive_rehost_bundle_id(
            run_id=run_root.name,
            user_hmac=current_user,
            identity_key_id=str(manifest["identity_key_id"]),
            session_ref=current_session,
            last_turn_index=int(manifest["last_turn_index"]),
            source_request_ref=str(manifest["source_request_ref"]),
            source_lineage_digest=str(manifest["source_lineage_digest"]),
            model_profile_id=str(manifest["model_profile_id"]),
            model_digest=str(manifest["model_digest"]),
            runtime_profile_digest=str(manifest["runtime_profile_digest"]),
            payload_digest=str(manifest["payload_digest"]),
            acquisition_manifest_digest=str(
                manifest["acquisition_manifest_digest"]
            ),
            gate_a_digest=str(manifest["gate_a_result_digest"]),
            evidence_root_digest=str(manifest["evidence_root_digest"]),
        )
        if expected_bundle_id != entry["bundle_id"]:
            raise RehostPreparationError("bundle identity derivation is invalid")

        payload = _read_regular_bytes(payload_path)
        observed_digest = "sha256:" + hashlib.sha256(payload).hexdigest()
        if (
            len(payload) != manifest["payload_size_bytes"]
            or not hmac.compare_digest(observed_digest, str(manifest["payload_digest"]))
            or len(payload) != entry["payload_size_bytes"]
            or not hmac.compare_digest(observed_digest, str(entry["payload_digest"]))
        ):
            raise RehostPreparationError("payload digest or size is invalid")
        try:
            summary = inspect_simple_kv_payload(payload)
        except PayloadValidationError as error:
            raise RehostPreparationError("payload structure is invalid") from error
        n_ctx = int(manifest["n_ctx"])
        n_stream = int(manifest["n_stream"])
        n_seq_max = int(manifest["n_seq_max"])
        n_layer = int(manifest["n_layer"])
        v_trans = bool(manifest["v_trans"])
        key_ggml_type_id = int(manifest["key_ggml_type_id"])
        value_ggml_type_id = int(manifest["value_ggml_type_id"])
        next_position = int(manifest["next_position"])
        source_seq_id = int(manifest["source_seq_id"])
        source_stream_index = int(manifest["source_stream_index"])
        if n_seq_max != n_stream:
            raise RehostPreparationError("runtime stream topology is invalid")
        if n_ctx % n_stream:
            raise RehostPreparationError("runtime context topology is invalid")
        if next_position > n_ctx // n_stream:
            raise RehostPreparationError("payload exceeds per-stream context capacity")
        if summary.n_stream != n_stream or summary.n_layer != n_layer:
            raise RehostPreparationError("payload layout is incompatible with the runtime")
        if summary.v_trans is not v_trans:
            raise RehostPreparationError("payload v_trans is incompatible with the runtime")
        if set(summary.key_ggml_type_ids) != {key_ggml_type_id}:
            raise RehostPreparationError("payload K type is incompatible with the runtime")
        if set(summary.value_ggml_type_ids) != {value_ggml_type_id}:
            raise RehostPreparationError("payload V type is incompatible with the runtime")
        if summary.nonempty_streams != (source_stream_index,):
            raise RehostPreparationError("payload stream attribution is invalid")
        if summary.source_seq_ids != (source_seq_id,):
            raise RehostPreparationError("payload sequence attribution is invalid")
        if tuple(sorted(summary.positions)) != tuple(range(next_position)):
            raise RehostPreparationError("payload position lineage is invalid")
        cell_ranges = tuple(_range("kv_cell_ranges", item) for item in manifest["kv_cell_ranges"])
        if sum(last - first + 1 for first, last in cell_ranges) != summary.cell_count:
            raise RehostPreparationError("payload KV cell cardinality is invalid")
        try:
            case = RehostCase(
                run_id=run_root.name,
                bundle_id=str(entry["bundle_id"]),
                user_hmac=current_user,
                identity_key_id=str(manifest["identity_key_id"]),
                session_ref=current_session,
                payload_path=payload_path,
                payload_digest=observed_digest,
                payload_size_bytes=len(payload),
                next_position=next_position,
                destination_seq_id=destination_seq_id,
                model_profile_id=str(manifest["model_profile_id"]),
                model_digest=str(manifest["model_digest"]),
                runtime_profile_digest=str(manifest["runtime_profile_digest"]),
                memory_family=str(manifest["memory_family"]),
                n_ctx=n_ctx,
                n_seq_max=n_seq_max,
                n_stream=n_stream,
                n_layer=n_layer,
                v_trans=v_trans,
                key_ggml_type_id=key_ggml_type_id,
                value_ggml_type_id=value_ggml_type_id,
                source_seq_id=source_seq_id,
                source_stream_index=source_stream_index,
                position_range=_range("position_range", manifest["position_range"]),
            )
            expected_runtime_profile.verify_case(case)
            cases.append(case)
        except RehostValidationError as error:
            raise RehostPreparationError(
                f"verified rehost case is incompatible with runtime: {error}"
            ) from error
    return tuple(sorted(cases, key=lambda item: (item.user_hmac, item.session_ref)))


__all__ = [
    "RehostPreparationError",
    "load_json_object",
    "prepare_rehost_cases",
]
