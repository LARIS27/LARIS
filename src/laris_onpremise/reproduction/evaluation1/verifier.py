"""Offline, fail-closed verification for the public Evaluation1 release."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Mapping, Sequence

from .archive import (
    ArchiveError,
    hash_safe_tar_members,
    inspect_safe_tar,
    read_safe_tar_member,
    safe_relative_path,
    sha256_file,
)
from .contracts import (
    ContractError,
    PublicBundleIndex,
    load_model_license_registry,
)


_CHECKSUM_RE = re.compile(r"([0-9a-f]{64})  (.+)\Z")
_FORBIDDEN_SUFFIXES = {
    ".gguf",
    ".safetensors",
    ".ckpt",
    ".pth",
    ".pt",
    ".vmdk",
    ".vmem",
    ".vmsn",
    ".vmx",
    ".nvram",
}


class ReleaseVerificationError(ValueError):
    """Raised when a public release cannot be accepted."""


def _load_json_mapping(path: Path) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReleaseVerificationError(f"cannot read JSON: {path}") from exc
    if not isinstance(value, Mapping):
        raise ReleaseVerificationError(f"JSON root must be an object: {path}")
    return value


def verify_checksums(root: Path, *, filename: str = "checksums.sha256") -> int:
    checksum_path = root / filename
    try:
        lines = checksum_path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ReleaseVerificationError("checksum file is missing or unreadable") from exc
    if not lines:
        raise ReleaseVerificationError("checksum file is empty")

    seen: set[str] = set()
    previous = ""
    for line in lines:
        match = _CHECKSUM_RE.fullmatch(line)
        if match is None:
            raise ReleaseVerificationError("checksum row is not canonical")
        expected, relative = match.groups()
        try:
            safe = safe_relative_path(relative, "checksum path")
        except ArchiveError as exc:
            raise ReleaseVerificationError(str(exc)) from exc
        normalized = safe.as_posix()
        if normalized in seen or normalized <= previous:
            raise ReleaseVerificationError("checksum paths must be unique and sorted")
        seen.add(normalized)
        previous = normalized
        target = root.joinpath(*safe.parts)
        if not target.is_file() or target.is_symlink():
            raise ReleaseVerificationError(f"checksum target is missing: {normalized}")
        actual = sha256_file(target).removeprefix("sha256:")
        if actual != expected:
            raise ReleaseVerificationError(f"checksum mismatch: {normalized}")
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != checksum_path
    }
    if seen != actual_files:
        missing = sorted(actual_files.difference(seen))
        stale = sorted(seen.difference(actual_files))
        detail = missing[0] if missing else stale[0]
        raise ReleaseVerificationError(f"checksum inventory mismatch: {detail}")
    return len(lines)


def verify_exact_release_indexes(
    documents: Sequence[Mapping[str, object]],
) -> tuple[PublicBundleIndex, ...]:
    if len(documents) != 4:
        raise ReleaseVerificationError("release must contain exactly four mode indexes")
    indexes = tuple(PublicBundleIndex.from_mapping(value) for value in documents)
    model_ids = {index.model_id for index in indexes}
    combinations = {(index.model_id, index.mode.value) for index in indexes}
    expected = {
        (model_id, mode)
        for model_id in model_ids
        for mode in ("PROXY", "LOG_LINKED")
    }
    if len(model_ids) != 2 or combinations != expected:
        raise ReleaseVerificationError(
            "release indexes must cover two models and two modes"
        )
    count = sum(len(index.entries) for index in indexes)
    if count != 240:
        raise ReleaseVerificationError(f"release must contain exactly 240 payloads, got {count}")
    keys = [entry.tuple_key for index in indexes for entry in index.entries]
    if len(keys) != len(set(keys)):
        raise ReleaseVerificationError("release contains a duplicate payload tuple")
    return indexes


def verify_required_notices(text: str, registry_path: Path) -> None:
    records = load_model_license_registry(registry_path)
    for record in records.values():
        for notice in record.required_notices:
            if notice not in text:
                raise ReleaseVerificationError(f"required notice is missing: {notice}")


def reject_forbidden_release_files(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ReleaseVerificationError(f"forbidden symlink in release: {path}")
        if path.is_file() and path.suffix.lower() in _FORBIDDEN_SUFFIXES:
            raise ReleaseVerificationError(f"forbidden model or VM file: {path.name}")


def _verify_bundle_members(archive_path: Path, index: PublicBundleIndex) -> None:
    inspected = inspect_safe_tar(archive_path)
    for member in inspected:
        if (
            member.mode != 0o644
            or member.mtime != 0
            or member.uid != 0
            or member.gid != 0
            or member.uname
            or member.gname
        ):
            raise ReleaseVerificationError(
                f"tar member metadata is not deterministic: {member.name}"
            )
    member_names = {member.name for member in inspected}
    selected: list[tuple[object, str, str]] = []
    for entry in index.entries:
        prefix = entry.bundle_path.rstrip("/") + "/"
        manifest_name = prefix + "bundle_manifest.json"
        subject_directory = (
            f"users/uh_{entry.subject_ref}"
            if entry.mode.value == "PROXY"
            else f"endpoints/ep_{entry.subject_ref}"
        )
        session_prefix = prefix + subject_directory + "/sessions/s_"
        session_digest = hashlib.sha256(entry.session_ref.encode("utf-8")).hexdigest()
        candidates = [
            name
            for name in member_names
            if name.startswith(session_prefix)
            and name.endswith("/runtime_state_payload.bin")
            and session_digest.startswith(
                name[len(session_prefix) :].split("/", 1)[0]
            )
        ]
        if not candidates:
            all_payloads = [
                name
                for name in member_names
                if name.startswith(prefix)
                and name.endswith("/runtime_state_payload.bin")
            ]
            if len(all_payloads) == 1:
                candidates = all_payloads
        if manifest_name not in member_names:
            raise ReleaseVerificationError(f"bundle manifest is missing: {manifest_name}")
        if len(candidates) != 1:
            raise ReleaseVerificationError(
                f"bundle must contain one runtime payload: {entry.bundle_path}"
            )
        selected.append((entry, manifest_name, candidates[0]))

    names = {
        member_name
        for _, manifest_name, payload_name in selected
        for member_name in (manifest_name, payload_name)
    }
    digests = hash_safe_tar_members(archive_path, names)
    for entry, manifest_name, payload_name in selected:
        manifest_digest, _ = digests[manifest_name]
        if manifest_digest != entry.source_bundle_manifest_sha256:
            raise ReleaseVerificationError(
                f"source bundle manifest digest mismatch: {entry.bundle_path}"
            )
        payload_digest, payload_size = digests[payload_name]
        if payload_digest != entry.payload_sha256 or payload_size != entry.payload_size_bytes:
            raise ReleaseVerificationError(f"payload digest mismatch: {entry.bundle_path}")


def verify_public_release(root: Path, registry_path: Path) -> dict[str, object]:
    """Verify the external release without extracting or executing its content."""

    root = root.resolve()
    if not root.is_dir():
        raise ReleaseVerificationError(f"release root is not a directory: {root}")
    reject_forbidden_release_files(root)
    checksum_count = verify_checksums(root)
    notices_path = root / "THIRD_PARTY_NOTICES.md"
    try:
        notices = notices_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ReleaseVerificationError("THIRD_PARTY_NOTICES.md is missing") from exc
    verify_required_notices(notices, registry_path)
    registry = load_model_license_registry(registry_path)
    if not (root / "shared-data.tar.gz").is_file():
        raise ReleaseVerificationError("shared-data.tar.gz is missing")
    identity_fixtures = _load_json_mapping(
        root / "fixtures" / "identity-fixtures.json"
    )
    if identity_fixtures.get("synthetic_only") is not True:
        raise ReleaseVerificationError("identity fixtures are not marked synthetic-only")
    key_provenance = _load_json_mapping(root / "fixtures" / "key-provenance.json")
    if key_provenance.get("synthetic_only") is not True:
        raise ReleaseVerificationError("key provenance is not marked synthetic-only")
    raw_keys = key_provenance.get("keys")
    if not isinstance(raw_keys, list) or not raw_keys:
        raise ReleaseVerificationError("public synthetic fixture keys are missing")
    verified_key_ids: set[str] = set()
    for raw in raw_keys:
        if not isinstance(raw, Mapping):
            raise ReleaseVerificationError("key provenance row must be an object")
        key_id = raw.get("key_id")
        if not isinstance(key_id, str) or not key_id or key_id in verified_key_ids:
            raise ReleaseVerificationError("key provenance key_id is invalid or duplicated")
        if raw.get("status") != "VERIFIED_SYNTHETIC_ONLY":
            raise ReleaseVerificationError(f"key is not synthetic-only: {key_id}")
        try:
            key_rel = safe_relative_path(str(raw["key_path"]), "key_path")
        except (KeyError, ArchiveError) as exc:
            raise ReleaseVerificationError(f"key path is invalid: {key_id}") from exc
        key_path = root.joinpath(*key_rel.parts)
        if not key_path.is_file() or key_path.is_symlink() or key_path.stat().st_size != 32:
            raise ReleaseVerificationError(f"synthetic fixture key is invalid: {key_id}")
        if sha256_file(key_path) != raw.get("key_sha256"):
            raise ReleaseVerificationError(f"synthetic fixture key digest mismatch: {key_id}")
        verified_key_ids.add(key_id)

    manifest = _load_json_mapping(root / "release-manifest.json")
    if manifest.get("schema_version") != "1.0":
        raise ReleaseVerificationError("release manifest schema_version must be 1.0")
    if manifest.get("source_evaluation") != "Evaluation1":
        raise ReleaseVerificationError("release is not bound to Evaluation1")
    if manifest.get("weights_included") is not False:
        raise ReleaseVerificationError("release must declare weights_included=false")
    raw_license_rows = manifest.get("model_licenses")
    if not isinstance(raw_license_rows, list) or len(raw_license_rows) != len(registry):
        raise ReleaseVerificationError("release manifest has an incomplete license review")
    seen_licenses: set[str] = set()
    for raw in raw_license_rows:
        if not isinstance(raw, Mapping):
            raise ReleaseVerificationError("model license review must be an object")
        model_id = raw.get("model_id")
        if not isinstance(model_id, str) or model_id in seen_licenses:
            raise ReleaseVerificationError("model license review is duplicated or invalid")
        seen_licenses.add(model_id)
        expected = registry.get(model_id)
        if expected is None:
            raise ReleaseVerificationError(f"unknown model license review: {model_id}")
        if (
            raw.get("source_revision") != expected.source_revision
            or raw.get("license_id") != expected.license_id
            or raw.get("license_url") != expected.license_url
            or raw.get("redistributes_weights") is not False
        ):
            raise ReleaseVerificationError(f"model license review differs: {model_id}")
        if raw.get("kv_redistribution_decision") != "PUBLIC_APPROVED":
            raise ReleaseVerificationError(f"KV redistribution is not approved: {model_id}")
        reviewer = raw.get("reviewer")
        if not isinstance(reviewer, str) or not reviewer.strip():
            raise ReleaseVerificationError(f"license reviewer is missing: {model_id}")
        try:
            license_rel = safe_relative_path(str(raw["license_path"]), "license_path")
        except (KeyError, ArchiveError) as exc:
            raise ReleaseVerificationError(f"license path is invalid: {model_id}") from exc
        license_path = root.joinpath(*license_rel.parts)
        if not license_path.is_file() or license_path.is_symlink():
            raise ReleaseVerificationError(f"license file is missing: {model_id}")
        if sha256_file(license_path) != raw.get("license_sha256"):
            raise ReleaseVerificationError(f"license digest mismatch: {model_id}")
    if seen_licenses != set(registry):
        raise ReleaseVerificationError("model license review set is incomplete")
    raw_artifacts = manifest.get("artifacts")
    if not isinstance(raw_artifacts, list) or len(raw_artifacts) != 4:
        raise ReleaseVerificationError("release manifest must contain four artifacts")

    documents: list[Mapping[str, object]] = []
    archive_rows: list[tuple[Path, Mapping[str, object]]] = []
    for raw in raw_artifacts:
        if not isinstance(raw, Mapping):
            raise ReleaseVerificationError("artifact record must be an object")
        try:
            archive_rel = safe_relative_path(str(raw["archive_path"]), "archive_path")
        except (KeyError, ArchiveError) as exc:
            raise ReleaseVerificationError("artifact archive_path is invalid") from exc
        archive_path = root.joinpath(*archive_rel.parts)
        if not archive_path.is_file() or archive_path.suffix != ".tar":
            raise ReleaseVerificationError(f"artifact archive is missing: {archive_rel}")
        if sha256_file(archive_path) != raw.get("archive_sha256"):
            raise ReleaseVerificationError(f"artifact archive digest mismatch: {archive_rel}")
        if archive_path.stat().st_size != raw.get("archive_size_bytes"):
            raise ReleaseVerificationError(f"artifact archive size mismatch: {archive_rel}")
        index_bytes = read_safe_tar_member(
            archive_path,
            "bundle-index.json",
            limit_bytes=16 * 1024 * 1024,
        )
        actual_index_digest = "sha256:" + hashlib.sha256(index_bytes).hexdigest()
        if actual_index_digest != raw.get("index_sha256"):
            raise ReleaseVerificationError(f"bundle index digest mismatch: {archive_rel}")
        try:
            document = json.loads(index_bytes.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ReleaseVerificationError(f"bundle index is invalid: {archive_rel}") from exc
        if not isinstance(document, Mapping):
            raise ReleaseVerificationError(f"bundle index root is invalid: {archive_rel}")
        documents.append(document)
        archive_rows.append((archive_path, raw))

    indexes = verify_exact_release_indexes(documents)
    indexed_key_ids = {
        entry.key_id for index in indexes for entry in index.entries
    }
    if indexed_key_ids != verified_key_ids:
        raise ReleaseVerificationError("public index key IDs differ from key provenance")
    by_key = {(index.model_id, index.mode.value): index for index in indexes}
    for archive_path, raw in archive_rows:
        key = (raw.get("model_id"), raw.get("mode"))
        index = by_key.get(key)
        if index is None:
            raise ReleaseVerificationError("artifact does not match an embedded index")
        _verify_bundle_members(archive_path, index)
    return {
        "schema_version": "1.0",
        "release_id": manifest.get("release_id"),
        "verification_status": "VERIFIED",
        "artifact_count": 4,
        "payload_count": 240,
        "checksum_count": checksum_count,
        "license_review_count": len(seen_licenses),
    }
