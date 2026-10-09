"""Digest-bound lookup and narrow materialization for public Evaluation1 bundles."""

from __future__ import annotations

import hashlib
import hmac
import json
import tarfile
from pathlib import Path
from typing import Mapping

from laris_onpremise.attribution.log_linked.endpoint import derive_endpoint_ref
from laris_onpremise.core.records import IdentityKey, IdentityKeyRegistry, derive_user_hmac

from .archive import (
    ArchiveError,
    inspect_safe_tar,
    read_safe_tar_member,
    safe_relative_path,
    sha256_file,
)
from .contracts import CaptureMode, ContractError, PublicBundleEntry, PublicBundleIndex


class ReleaseResolutionError(ValueError):
    """Raised when a public tuple cannot be proven and resolved exactly once."""


def _json_mapping(path: Path) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReleaseResolutionError(f"cannot read release JSON: {path}") from exc
    if not isinstance(value, Mapping):
        raise ReleaseResolutionError(f"release JSON root must be an object: {path}")
    return value


def _entry_document(entry: PublicBundleEntry) -> dict[str, object]:
    return {
        "model_id": entry.model_id,
        "mode": entry.mode.value,
        "checkpoint_id": entry.checkpoint_id,
        "public_user_ref": entry.public_user_ref,
        "subject_ref_kind": entry.subject_ref_kind,
        "subject_ref": entry.subject_ref,
        "key_id": entry.key_id,
        "session_ref": entry.session_ref,
        "bundle_id": entry.bundle_id,
        "bundle_path": entry.bundle_path,
        "payload_size_bytes": entry.payload_size_bytes,
        "payload_sha256": entry.payload_sha256,
        "source_bundle_manifest_sha256": entry.source_bundle_manifest_sha256,
    }


class Evaluation1Release:
    def __init__(
        self,
        root: Path,
        manifest: Mapping[str, object],
        fixtures: Mapping[str, object],
        key_provenance: Mapping[str, object],
    ) -> None:
        self.root = root
        self.manifest = manifest
        self.fixtures = fixtures
        self.key_provenance = key_provenance
        self._indexes: dict[tuple[str, str], PublicBundleIndex] = {}

    @classmethod
    def open(cls, root: Path) -> "Evaluation1Release":
        root = root.resolve()
        if not root.is_dir():
            raise ReleaseResolutionError(f"release root is not a directory: {root}")
        manifest = _json_mapping(root / "release-manifest.json")
        if (
            manifest.get("schema_version") != "1.0"
            or manifest.get("source_evaluation") != "Evaluation1"
            or manifest.get("weights_included") is not False
        ):
            raise ReleaseResolutionError("release manifest contract is invalid")
        fixtures = _json_mapping(root / "fixtures" / "identity-fixtures.json")
        key_provenance = _json_mapping(root / "fixtures" / "key-provenance.json")
        if fixtures.get("synthetic_only") is not True:
            raise ReleaseResolutionError("identity fixtures are not synthetic-only")
        if key_provenance.get("synthetic_only") is not True:
            raise ReleaseResolutionError("key provenance is not synthetic-only")
        return cls(root, manifest, fixtures, key_provenance)

    def _artifact(self, model_id: str, mode: CaptureMode) -> Mapping[str, object]:
        raw_artifacts = self.manifest.get("artifacts")
        if not isinstance(raw_artifacts, list):
            raise ReleaseResolutionError("release artifact list is invalid")
        matches = [
            raw
            for raw in raw_artifacts
            if isinstance(raw, Mapping)
            and raw.get("model_id") == model_id
            and raw.get("mode") == mode.value
        ]
        if len(matches) != 1:
            raise ReleaseResolutionError("model and mode must resolve to one archive")
        return matches[0]

    def _archive(self, model_id: str, mode: CaptureMode) -> tuple[Path, Mapping[str, object]]:
        artifact = self._artifact(model_id, mode)
        try:
            relative = safe_relative_path(str(artifact["archive_path"]), "archive_path")
        except (KeyError, ArchiveError) as exc:
            raise ReleaseResolutionError("release archive path is invalid") from exc
        archive = self.root.joinpath(*relative.parts)
        if not archive.is_file() or archive.is_symlink():
            raise ReleaseResolutionError("release archive is missing")
        if sha256_file(archive) != artifact.get("archive_sha256"):
            raise ReleaseResolutionError("release archive digest mismatch")
        if archive.stat().st_size != artifact.get("archive_size_bytes"):
            raise ReleaseResolutionError("release archive size mismatch")
        return archive, artifact

    def index(self, model_id: str, mode: str | CaptureMode) -> PublicBundleIndex:
        try:
            capture_mode = mode if isinstance(mode, CaptureMode) else CaptureMode(mode)
        except ValueError as exc:
            raise ReleaseResolutionError("capture mode is invalid") from exc
        key = (model_id, capture_mode.value)
        cached = self._indexes.get(key)
        if cached is not None:
            return cached
        archive, artifact = self._archive(model_id, capture_mode)
        try:
            raw = read_safe_tar_member(
                archive,
                "bundle-index.json",
                limit_bytes=16 * 1024 * 1024,
            )
        except ArchiveError as exc:
            raise ReleaseResolutionError(str(exc)) from exc
        digest = "sha256:" + hashlib.sha256(raw).hexdigest()
        if digest != artifact.get("index_sha256"):
            raise ReleaseResolutionError("bundle index digest mismatch")
        try:
            document = json.loads(raw.decode("utf-8"))
            if not isinstance(document, Mapping):
                raise ReleaseResolutionError("bundle index root is invalid")
            index = PublicBundleIndex.from_mapping(document)
        except (UnicodeError, json.JSONDecodeError, ContractError) as exc:
            raise ReleaseResolutionError("bundle index contract is invalid") from exc
        if index.model_id != model_id or index.mode is not capture_mode:
            raise ReleaseResolutionError("bundle index differs from selected artifact")
        self._indexes[key] = index
        return index

    def list_entries(
        self,
        model_id: str,
        mode: str | CaptureMode,
    ) -> tuple[PublicBundleEntry, ...]:
        return self.index(model_id, mode).entries

    def resolve(
        self,
        model_id: str,
        mode: str | CaptureMode,
        checkpoint_id: str,
        public_user_ref: str,
    ) -> PublicBundleEntry:
        matches = [
            entry
            for entry in self.index(model_id, mode).entries
            if entry.checkpoint_id == checkpoint_id
            and entry.public_user_ref == public_user_ref
        ]
        if len(matches) != 1:
            raise ReleaseResolutionError("public tuple must resolve to exactly one bundle")
        return matches[0]

    def _key_bytes(self, key_id: str) -> bytes:
        raw_keys = self.key_provenance.get("keys")
        if not isinstance(raw_keys, list):
            raise ReleaseResolutionError("key provenance list is invalid")
        matches = [
            raw
            for raw in raw_keys
            if isinstance(raw, Mapping) and raw.get("key_id") == key_id
        ]
        if len(matches) != 1 or matches[0].get("status") != "VERIFIED_SYNTHETIC_ONLY":
            raise ReleaseResolutionError("synthetic key must resolve exactly once")
        raw = matches[0]
        try:
            relative = safe_relative_path(str(raw["key_path"]), "key_path")
        except (KeyError, ArchiveError) as exc:
            raise ReleaseResolutionError("synthetic key path is invalid") from exc
        path = self.root.joinpath(*relative.parts)
        if not path.is_file() or path.is_symlink() or path.stat().st_size != 32:
            raise ReleaseResolutionError("synthetic key file is invalid")
        if sha256_file(path) != raw.get("key_sha256"):
            raise ReleaseResolutionError("synthetic key digest mismatch")
        return path.read_bytes()

    def subject_id(self, public_user_ref: str) -> str:
        raw_subjects = self.fixtures.get("subjects")
        if not isinstance(raw_subjects, list):
            raise ReleaseResolutionError("synthetic subject fixtures are invalid")
        matches = [
            raw.get("authenticated_subject_id")
            for raw in raw_subjects
            if isinstance(raw, Mapping) and raw.get("public_user_ref") == public_user_ref
        ]
        if len(matches) != 1 or not isinstance(matches[0], str) or not matches[0]:
            raise ReleaseResolutionError("synthetic subject fixture must resolve once")
        return matches[0]

    def _endpoint(self, public_user_ref: str) -> tuple[str, str]:
        raw_endpoints = self.fixtures.get("endpoints")
        if not isinstance(raw_endpoints, list):
            raise ReleaseResolutionError("synthetic endpoint fixtures are invalid")
        matches = [
            raw
            for raw in raw_endpoints
            if isinstance(raw, Mapping) and raw.get("public_user_ref") == public_user_ref
        ]
        if len(matches) != 1:
            raise ReleaseResolutionError("synthetic endpoint fixture must resolve once")
        endpoint_type = matches[0].get("endpoint_type")
        endpoint_value = matches[0].get("endpoint_value")
        if not isinstance(endpoint_type, str) or not isinstance(endpoint_value, str):
            raise ReleaseResolutionError("synthetic endpoint fixture is invalid")
        return endpoint_type, endpoint_value

    def verify_identifier(self, entry: PublicBundleEntry) -> dict[str, object]:
        key = self._key_bytes(entry.key_id)
        if entry.mode is CaptureMode.PROXY:
            tenant_id = self.fixtures.get("tenant_id")
            if not isinstance(tenant_id, str) or not tenant_id:
                raise ReleaseResolutionError("synthetic tenant fixture is invalid")
            calculated = derive_user_hmac(
                IdentityKeyRegistry((IdentityKey(entry.key_id, key),)),
                entry.key_id,
                tenant_id,
                self.subject_id(entry.public_user_ref),
            )
        else:
            endpoint_type, endpoint_value = self._endpoint(entry.public_user_ref)
            calculated = derive_endpoint_ref(
                correlation_key=key,
                endpoint_type=endpoint_type,
                endpoint_value=endpoint_value,
            )
        if not hmac.compare_digest(calculated, entry.subject_ref):
            raise ReleaseResolutionError("public identifier recomputation mismatch")
        return {
            "status": "VERIFIED",
            "subject_ref_kind": entry.subject_ref_kind,
            "subject_ref": entry.subject_ref,
            "key_id": entry.key_id,
        }

    def _shared_member(self, name: str, limit_bytes: int = 16 * 1024 * 1024) -> bytes:
        try:
            safe_relative_path(name, "shared member")
            with tarfile.open(self.root / "shared-data.tar.gz", mode="r:*") as archive:
                seen: set[str] = set()
                selected: tarfile.TarInfo | None = None
                for member in archive:
                    safe_relative_path(member.name, "shared tar member")
                    if member.name in seen:
                        raise ReleaseResolutionError("duplicate shared-data member")
                    seen.add(member.name)
                    if member.issym() or member.islnk() or member.isdev() or member.isfifo():
                        raise ReleaseResolutionError("unsafe shared-data member")
                    if member.name == name:
                        selected = member
                if selected is None or not selected.isreg() or selected.size > limit_bytes:
                    raise ReleaseResolutionError(f"shared-data member is unavailable: {name}")
                stream = archive.extractfile(selected)
                if stream is None:
                    raise ReleaseResolutionError(f"shared-data member is unreadable: {name}")
                payload = stream.read(limit_bytes + 1)
        except (OSError, tarfile.TarError, ArchiveError) as exc:
            if isinstance(exc, ReleaseResolutionError):
                raise
            raise ReleaseResolutionError("shared-data archive is invalid") from exc
        if len(payload) != selected.size:
            raise ReleaseResolutionError(f"shared-data member is truncated: {name}")
        return payload

    def validate_record(self, checkpoint_id: str, record_id: str) -> None:
        try:
            schedule = json.loads(self._shared_member("checkpoint-schedule.json"))
        except json.JSONDecodeError as exc:
            raise ReleaseResolutionError("checkpoint schedule is invalid") from exc
        if not isinstance(schedule, Mapping) or not isinstance(schedule.get("checkpoints"), list):
            raise ReleaseResolutionError("checkpoint schedule is invalid")
        matches = [
            raw
            for raw in schedule["checkpoints"]
            if isinstance(raw, Mapping) and raw.get("checkpoint_id") == checkpoint_id
        ]
        if len(matches) != 1 or record_id not in matches[0].get("record_ids", []):
            raise ReleaseResolutionError("record is outside the selected checkpoint cohort")

    def question(self, checkpoint_id: str, record_id: str) -> str:
        self.validate_record(checkpoint_id, record_id)
        rows = self._jsonl("questions.jsonl")
        matches = [
            row.get("question")
            for row in rows
            if row.get("checkpoint_id") == checkpoint_id and row.get("record_id") == record_id
        ]
        if len(matches) != 1 or not isinstance(matches[0], str):
            raise ReleaseResolutionError("published question must resolve exactly once")
        return matches[0]

    def expected_answer(
        self,
        checkpoint_id: str,
        public_user_ref: str,
        record_id: str,
    ) -> str:
        self.validate_record(checkpoint_id, record_id)
        rows = self._jsonl("oracle.jsonl")
        matches = [
            row.get("expected")
            for row in rows
            if row.get("checkpoint_id") == checkpoint_id
            and row.get("public_user_ref") == public_user_ref
            and row.get("record_id") == record_id
        ]
        if len(matches) != 1 or not isinstance(matches[0], str):
            raise ReleaseResolutionError("published oracle must resolve exactly once")
        return matches[0]

    def _jsonl(self, name: str) -> tuple[Mapping[str, object], ...]:
        rows: list[Mapping[str, object]] = []
        for line in self._shared_member(name).decode("utf-8").splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ReleaseResolutionError(f"published JSONL is invalid: {name}") from exc
            if not isinstance(value, Mapping):
                raise ReleaseResolutionError(f"published JSONL row is invalid: {name}")
            rows.append(value)
        return tuple(rows)

    def materialize(
        self,
        model_id: str,
        mode: str | CaptureMode,
        checkpoint_id: str,
        public_user_ref: str,
        destination: Path,
    ) -> PublicBundleEntry:
        entry = self.resolve(model_id, mode, checkpoint_id, public_user_ref)
        self.verify_identifier(entry)
        archive, _ = self._archive(entry.model_id, entry.mode)
        inspected = inspect_safe_tar(archive)
        prefix = entry.bundle_path.rstrip("/") + "/"
        selected = [member for member in inspected if member.name.startswith(prefix)]
        if not selected:
            raise ReleaseResolutionError("selected bundle has no archive members")
        if destination.exists() and any(destination.iterdir()):
            raise ReleaseResolutionError("bundle destination must be empty")
        destination.mkdir(parents=True, exist_ok=True)
        try:
            with tarfile.open(archive, mode="r:") as source:
                source_members = {member.name: member for member in source}
                for member in selected:
                    relative_text = member.name[len(prefix) :]
                    relative = safe_relative_path(relative_text, "bundle member")
                    target = destination.joinpath(*relative.parts)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    stream = source.extractfile(source_members[member.name])
                    if stream is None:
                        raise ReleaseResolutionError("selected bundle member is unreadable")
                    with target.open("xb") as output:
                        while True:
                            chunk = stream.read(1024 * 1024)
                            if not chunk:
                                break
                            output.write(chunk)
        except (OSError, tarfile.TarError, ArchiveError) as exc:
            if isinstance(exc, ReleaseResolutionError):
                raise
            raise ReleaseResolutionError("selected bundle materialization failed") from exc
        manifests = [path for path in destination.rglob("bundle_manifest.json")]
        session_digest = hashlib.sha256(entry.session_ref.encode("utf-8")).hexdigest()
        subject_root = (
            destination / "users" / f"uh_{entry.subject_ref}"
            if entry.mode is CaptureMode.PROXY
            else destination / "endpoints" / f"ep_{entry.subject_ref}"
        )
        payloads = [
            path
            for path in subject_root.glob("sessions/s_*/runtime_state_payload.bin")
            if session_digest.startswith(path.parent.name[2:])
        ]
        if not payloads:
            all_payloads = list(destination.rglob("runtime_state_payload.bin"))
            if len(all_payloads) == 1:
                payloads = all_payloads
        if len(manifests) != 1 or len(payloads) != 1:
            raise ReleaseResolutionError("selected bundle inventory is incomplete")
        if sha256_file(manifests[0]) != entry.source_bundle_manifest_sha256:
            raise ReleaseResolutionError("materialized bundle manifest digest mismatch")
        if (
            sha256_file(payloads[0]) != entry.payload_sha256
            or payloads[0].stat().st_size != entry.payload_size_bytes
        ):
            raise ReleaseResolutionError("materialized payload digest mismatch")
        return entry

    @staticmethod
    def entry_document(entry: PublicBundleEntry) -> dict[str, object]:
        return _entry_document(entry)
