"""Streaming, fail-closed packager for the external Evaluation1 KV release."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import tarfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Callable, Iterable, Mapping

from .archive import (
    ArchiveError,
    inspect_safe_tar,
    safe_relative_path,
    sha256_file,
    write_checksums,
)
from .contracts import (
    CaptureMode,
    ContractError,
    PublicBundleEntry,
    PublicBundleIndex,
    load_model_license_registry,
)
from .verifier import verify_public_release


class PackageError(ValueError):
    """Raised when source evidence cannot produce a complete public release."""


@dataclass(frozen=True)
class SourceMember:
    source_path: str
    bundle_relative_path: str
    role: str

    def __post_init__(self) -> None:
        try:
            safe_relative_path(self.source_path, "source member")
            safe_relative_path(self.bundle_relative_path, "bundle member")
        except ArchiveError as exc:
            raise PackageError(str(exc)) from exc
        if self.role not in {"BUNDLE_MANIFEST", "RUNTIME_PAYLOAD", "BUNDLE_FILE"}:
            raise PackageError(f"unsupported source member role: {self.role}")


@dataclass(frozen=True)
class BundleSelection:
    entry: PublicBundleEntry
    members: tuple[SourceMember, ...]

    def __post_init__(self) -> None:
        roles = [member.role for member in self.members]
        if roles.count("BUNDLE_MANIFEST") != 1:
            raise PackageError("bundle selection needs one BUNDLE_MANIFEST")
        if roles.count("RUNTIME_PAYLOAD") != 1:
            raise PackageError("bundle selection needs one RUNTIME_PAYLOAD")
        source_paths = [member.source_path for member in self.members]
        destination_paths = [member.bundle_relative_path for member in self.members]
        if len(source_paths) != len(set(source_paths)):
            raise PackageError("bundle selection has duplicate source members")
        if len(destination_paths) != len(set(destination_paths)):
            raise PackageError("bundle selection has duplicate destination members")


class _HashingReader:
    def __init__(self, stream: BinaryIO) -> None:
        self._stream = stream
        self._digest = hashlib.sha256()
        self.size = 0

    def read(self, size: int = -1) -> bytes:
        payload = self._stream.read(size)
        if payload:
            self._digest.update(payload)
            self.size += len(payload)
        return payload

    @property
    def digest(self) -> str:
        return "sha256:" + self._digest.hexdigest()


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


def index_document(index: PublicBundleIndex) -> dict[str, object]:
    return {
        "schema_version": index.schema_version,
        "release_id": index.release_id,
        "model_id": index.model_id,
        "mode": index.mode.value,
        "subject_ref_kind": index.subject_ref_kind,
        "entries": [_entry_document(entry) for entry in index.entries],
    }


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"


def _normalized_info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name=name)
    info.size = size
    info.mode = 0o644
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info


def package_mode_archive(
    *,
    source_archive: Path,
    expected_source_sha256: str,
    destination: Path,
    index: PublicBundleIndex,
    selections: tuple[BundleSelection, ...],
) -> dict[str, object]:
    """Copy only explicitly selected members while streaming the source once."""

    if not source_archive.is_file() or source_archive.is_symlink():
        raise PackageError(f"source archive is not a regular file: {source_archive}")
    source_stat = source_archive.stat()
    source_signature = (source_stat.st_size, source_stat.st_mtime_ns)
    actual_source_sha256 = sha256_file(source_archive)
    if actual_source_sha256 != expected_source_sha256:
        raise PackageError("source archive digest does not match the sealed manifest")
    if len(selections) != 60:
        raise PackageError("mode package must select exactly 60 bundles")
    selection_keys = [selection.entry.tuple_key for selection in selections]
    if set(selection_keys) != {entry.tuple_key for entry in index.entries}:
        raise PackageError("bundle selections do not match the public index")

    by_source: dict[str, tuple[PublicBundleEntry, SourceMember, str]] = {}
    public_names: set[str] = set()
    for selection in selections:
        for member in selection.members:
            public_name = (
                selection.entry.bundle_path.rstrip("/")
                + "/"
                + member.bundle_relative_path
            )
            try:
                safe_relative_path(public_name, "public member")
            except ArchiveError as exc:
                raise PackageError(str(exc)) from exc
            existing = by_source.get(member.source_path)
            if existing is not None:
                existing_entry, existing_member, existing_public_name = existing
                same_binding = (
                    existing_member == member
                    and existing_public_name == public_name
                    and (
                        member.role != "BUNDLE_MANIFEST"
                        or existing_entry.source_bundle_manifest_sha256
                        == selection.entry.source_bundle_manifest_sha256
                    )
                    and (
                        member.role != "RUNTIME_PAYLOAD"
                        or (
                            existing_entry.payload_sha256
                            == selection.entry.payload_sha256
                            and existing_entry.payload_size_bytes
                            == selection.entry.payload_size_bytes
                        )
                    )
                )
                if not same_binding:
                    raise PackageError(
                        f"source member has conflicting selections: {member.source_path}"
                    )
                continue
            if public_name in public_names:
                raise PackageError(f"public member selected twice: {public_name}")
            public_names.add(public_name)
            by_source[member.source_path] = (selection.entry, member, public_name)

    index_bytes = canonical_json_bytes(index_document(index))
    destination.parent.mkdir(parents=True, exist_ok=True)
    incomplete = destination.with_name(destination.name + ".incomplete")
    found: set[str] = set()
    source_names: set[str] = set()
    try:
        with tarfile.open(incomplete, mode="w", format=tarfile.USTAR_FORMAT) as target:
            target.addfile(
                _normalized_info("bundle-index.json", len(index_bytes)),
                io.BytesIO(index_bytes),
            )
            try:
                source = tarfile.open(source_archive, mode="r:*")
            except (OSError, tarfile.TarError) as exc:
                raise PackageError(f"cannot open source archive: {source_archive}") from exc
            with source:
                for info in source:
                    try:
                        safe_relative_path(info.name, "source tar member")
                    except ArchiveError as exc:
                        raise PackageError(str(exc)) from exc
                    if info.name in source_names:
                        raise PackageError(f"duplicate source tar member: {info.name}")
                    source_names.add(info.name)
                    if info.issym() or info.islnk() or info.isdev() or info.isfifo():
                        raise PackageError(f"unsafe source tar member: {info.name}")
                    selected = by_source.get(info.name)
                    if selected is None:
                        continue
                    if not info.isreg():
                        raise PackageError(f"selected source member is not regular: {info.name}")
                    stream = source.extractfile(info)
                    if stream is None:
                        raise PackageError(f"cannot read selected source member: {info.name}")
                    entry, member, public_name = selected
                    reader = _HashingReader(stream)
                    target.addfile(_normalized_info(public_name, info.size), reader)
                    if reader.size != info.size:
                        raise PackageError(f"selected member was truncated: {info.name}")
                    if member.role == "BUNDLE_MANIFEST" and (
                        reader.digest != entry.source_bundle_manifest_sha256
                    ):
                        raise PackageError(f"bundle manifest digest mismatch: {info.name}")
                    if member.role == "RUNTIME_PAYLOAD" and (
                        reader.digest != entry.payload_sha256
                        or reader.size != entry.payload_size_bytes
                    ):
                        raise PackageError(f"runtime payload digest mismatch: {info.name}")
                    found.add(info.name)
        missing = sorted(set(by_source).difference(found))
        if missing:
            raise PackageError(f"selected source member is missing: {missing[0]}")
        final_stat = source_archive.stat()
        if (final_stat.st_size, final_stat.st_mtime_ns) != source_signature:
            raise PackageError("source archive metadata changed during packaging")
        os.replace(incomplete, destination)
    except Exception:
        incomplete.unlink(missing_ok=True)
        raise

    inspected = inspect_safe_tar(destination)
    return {
        "schema_version": "1.0",
        "model_id": index.model_id,
        "mode": index.mode.value,
        "bundle_count": len(selections),
        "member_count": len(inspected),
        "source_archive_sha256": actual_source_sha256,
        "archive_sha256": sha256_file(destination),
        "archive_size_bytes": destination.stat().st_size,
        "index_sha256": "sha256:" + hashlib.sha256(index_bytes).hexdigest(),
    }


def validate_publication_gates(
    *,
    model_ids: Iterable[str],
    key_ids: Iterable[str],
    license_reviews: Mapping[str, str],
    key_provenance: Mapping[str, str],
) -> None:
    for model_id in set(model_ids):
        if license_reviews.get(model_id) != "PUBLIC_APPROVED":
            raise PackageError(f"license review is not PUBLIC_APPROVED: {model_id}")
    for key_id in set(key_ids):
        if key_provenance.get(key_id) != "VERIFIED_SYNTHETIC_ONLY":
            raise PackageError(f"key provenance is not synthetic-only: {key_id}")


def audit_release_sources_from_spec(spec_path: Path) -> dict[str, object]:
    """Verify four sealed source archives without creating release artifacts."""

    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PackageError(f"cannot read release specification: {spec_path}") from exc
    if not isinstance(spec, Mapping) or spec.get("schema_version") != "1.0":
        raise PackageError("release specification schema_version must be 1.0")
    raw_sources = spec.get("sources")
    if not isinstance(raw_sources, list) or len(raw_sources) != 4:
        raise PackageError("release specification must name exactly four sources")

    rows: list[dict[str, object]] = []
    tuples: set[tuple[str, str]] = set()
    for raw in raw_sources:
        if not isinstance(raw, Mapping):
            raise PackageError("source record must be an object")
        model_id = raw.get("model_id")
        mode = raw.get("mode")
        if not isinstance(model_id, str) or not model_id:
            raise PackageError("source model_id must be non-empty")
        if mode not in {CaptureMode.PROXY.value, CaptureMode.LOG_LINKED.value}:
            raise PackageError(f"source mode is not public: {mode}")
        tuple_key = (model_id, str(mode))
        if tuple_key in tuples:
            raise PackageError("release specification has a duplicate model/mode source")
        tuples.add(tuple_key)

        archive = Path(str(raw.get("archive_path", "")))
        expected_digest = raw.get("archive_sha256")
        if not archive.is_file() or archive.is_symlink():
            raise PackageError(f"source archive is not a regular file: {archive}")
        before = archive.stat()
        signature = (before.st_size, before.st_mtime_ns)
        actual_digest = sha256_file(archive)
        if actual_digest != expected_digest:
            raise PackageError("source archive digest does not match the sealed manifest")
        after = archive.stat()
        if (after.st_size, after.st_mtime_ns) != signature:
            raise PackageError("source archive metadata changed during inventory audit")
        rows.append(
            {
                "model_id": model_id,
                "mode": mode,
                "archive_path": str(archive),
                "archive_size_bytes": before.st_size,
                "archive_sha256": actual_digest,
                "digest_matches": True,
            }
        )
    return {
        "schema_version": "1.0",
        "verification_status": "VERIFIED",
        "source_count": len(rows),
        "total_source_bytes": sum(
            int(row["archive_size_bytes"]) for row in rows
        ),
        "sources": sorted(
            rows,
            key=lambda row: (str(row["model_id"]), str(row["mode"])),
        ),
    }


def atomic_publish(destination: Path, builder: Callable[[Path], None]) -> Path:
    """Build beside the destination and rename only after complete verification."""

    if destination.exists():
        raise PackageError(f"release destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    candidate = destination.with_name(
        f"{destination.name}.incomplete-{uuid.uuid4().hex}"
    )
    try:
        candidate.mkdir()
        builder(candidate)
        os.replace(candidate, destination)
    except Exception:
        if candidate.exists():
            shutil.rmtree(candidate)
        raise
    return destination


def _selection_from_mapping(
    raw: Mapping[str, object],
    entries: Mapping[tuple[str, str], PublicBundleEntry],
) -> BundleSelection:
    checkpoint_id = raw.get("checkpoint_id")
    public_user_ref = raw.get("public_user_ref")
    entry = entries.get((str(checkpoint_id), str(public_user_ref)))
    if entry is None:
        raise PackageError("selection does not match an index entry")
    raw_members = raw.get("members")
    if not isinstance(raw_members, list):
        raise PackageError("selection members must be a list")
    members: list[SourceMember] = []
    for value in raw_members:
        if not isinstance(value, Mapping):
            raise PackageError("source member must be an object")
        try:
            members.append(
                SourceMember(
                    source_path=str(value["source_path"]),
                    bundle_relative_path=str(value["bundle_relative_path"]),
                    role=str(value["role"]),
                )
            )
        except KeyError as exc:
            raise PackageError(f"source member is missing {exc.args[0]}") from exc
    return BundleSelection(entry=entry, members=tuple(members))


def prepare_release_from_spec(
    spec_path: Path,
    destination: Path,
    registry_path: Path,
) -> dict[str, object]:
    """Package four sealed sources described by an explicit, reviewable JSON spec."""

    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PackageError(f"cannot read release specification: {spec_path}") from exc
    if not isinstance(spec, Mapping) or spec.get("schema_version") != "1.0":
        raise PackageError("release specification schema_version must be 1.0")
    raw_sources = spec.get("sources")
    raw_license_reviews = spec.get("license_reviews")
    raw_key_provenance = spec.get("key_provenance")
    if not isinstance(raw_sources, list) or len(raw_sources) != 4:
        raise PackageError("release specification must name exactly four sources")
    if not isinstance(raw_license_reviews, list) or not isinstance(raw_key_provenance, list):
        raise PackageError("release specification gates are missing")

    registry = load_model_license_registry(registry_path)
    license_status = {
        str(row.get("model_id")): str(row.get("kv_redistribution_decision"))
        for row in raw_license_reviews
        if isinstance(row, Mapping)
    }
    key_status = {
        str(row.get("key_id")): str(row.get("status"))
        for row in raw_key_provenance
        if isinstance(row, Mapping)
    }
    parsed: list[tuple[Mapping[str, object], PublicBundleIndex, tuple[BundleSelection, ...]]] = []
    for raw in raw_sources:
        if not isinstance(raw, Mapping):
            raise PackageError("source record must be an object")
        raw_index = raw.get("index")
        raw_selections = raw.get("selections")
        if not isinstance(raw_index, Mapping) or not isinstance(raw_selections, list):
            raise PackageError("source record is missing index or selections")
        try:
            index = PublicBundleIndex.from_mapping(raw_index)
        except ContractError as exc:
            raise PackageError(str(exc)) from exc
        by_tuple = {
            (entry.checkpoint_id, entry.public_user_ref): entry
            for entry in index.entries
        }
        selections = tuple(
            _selection_from_mapping(value, by_tuple)
            for value in raw_selections
            if isinstance(value, Mapping)
        )
        parsed.append((raw, index, selections))
    validate_publication_gates(
        model_ids=(index.model_id for _, index, _ in parsed),
        key_ids=(entry.key_id for _, index, _ in parsed for entry in index.entries),
        license_reviews=license_status,
        key_provenance=key_status,
    )

    release_id = str(spec.get("release_id", "evaluation1-release-v1"))
    notices_source = Path(str(spec.get("third_party_notices_path", "")))
    if not notices_source.is_file():
        raise PackageError("third_party_notices_path is missing")
    identity_fixtures_source = Path(str(spec.get("identity_fixtures_path", "")))
    if (
        not identity_fixtures_source.is_file()
        or sha256_file(identity_fixtures_source)
        != spec.get("identity_fixtures_sha256")
    ):
        raise PackageError("identity fixture digest does not match")
    shared_data_source = Path(str(spec.get("shared_data_path", "")))
    if (
        not shared_data_source.is_file()
        or sha256_file(shared_data_source) != spec.get("shared_data_sha256")
    ):
        raise PackageError("shared data digest does not match")

    final_receipt: dict[str, object] = {}

    def build(candidate: Path) -> None:
        artifacts: list[dict[str, object]] = []
        for raw, index, selections in parsed:
            archive_rel = (
                Path("models")
                / index.model_id
                / ("proxy.tar" if index.mode is CaptureMode.PROXY else "log-linked.tar")
            )
            try:
                source_archive = Path(str(raw["archive_path"]))
                source_digest = str(raw["archive_sha256"])
            except KeyError as exc:
                raise PackageError(f"source record is missing {exc.args[0]}") from exc
            receipt = package_mode_archive(
                source_archive=source_archive,
                expected_source_sha256=source_digest,
                destination=candidate / archive_rel,
                index=index,
                selections=selections,
            )
            artifacts.append(
                {
                    "model_id": index.model_id,
                    "mode": index.mode.value,
                    "archive_path": archive_rel.as_posix(),
                    "archive_sha256": receipt["archive_sha256"],
                    "archive_size_bytes": receipt["archive_size_bytes"],
                    "index_sha256": receipt["index_sha256"],
                    "bundle_count": 60,
                }
            )

        shutil.copyfile(notices_source, candidate / "THIRD_PARTY_NOTICES.md")
        fixtures_root = candidate / "fixtures"
        fixtures_root.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(
            identity_fixtures_source,
            fixtures_root / "identity-fixtures.json",
        )
        shutil.copyfile(shared_data_source, candidate / "shared-data.tar.gz")
        public_key_rows: list[dict[str, object]] = []
        keys_root = fixtures_root / "keys"
        keys_root.mkdir()
        for raw in raw_key_provenance:
            if not isinstance(raw, Mapping):
                raise PackageError("key provenance row must be an object")
            key_id = str(raw.get("key_id", ""))
            if key_status.get(key_id) != "VERIFIED_SYNTHETIC_ONLY":
                raise PackageError(f"key provenance is not synthetic-only: {key_id}")
            source_path = Path(str(raw.get("key_path", "")))
            if not source_path.is_file() or source_path.stat().st_size != 32:
                raise PackageError(f"synthetic fixture key must be exactly 32 bytes: {key_id}")
            if sha256_file(source_path) != raw.get("key_sha256"):
                raise PackageError(f"synthetic fixture key digest mismatch: {key_id}")
            safe_key_name = hashlib.sha256(key_id.encode("utf-8")).hexdigest()[:16]
            key_rel = Path("fixtures") / "keys" / f"{safe_key_name}.key"
            shutil.copyfile(source_path, candidate / key_rel)
            public_key_rows.append(
                {
                    "key_id": key_id,
                    "status": "VERIFIED_SYNTHETIC_ONLY",
                    "fingerprint": str(raw.get("fingerprint", "")),
                    "key_path": key_rel.as_posix(),
                    "key_sha256": sha256_file(candidate / key_rel),
                    "warning": "Synthetic Evaluation1 fixture only; never reuse for deployment.",
                }
            )
        (fixtures_root / "key-provenance.json").write_bytes(
            canonical_json_bytes(
                {
                    "schema_version": "1.0",
                    "synthetic_only": True,
                    "keys": sorted(public_key_rows, key=lambda row: str(row["key_id"])),
                }
            )
        )
        license_rows: list[dict[str, object]] = []
        by_model = {
            str(row.get("model_id")): row
            for row in raw_license_reviews
            if isinstance(row, Mapping)
        }
        for model_id, record in registry.items():
            review = by_model.get(model_id)
            if review is None:
                raise PackageError(f"license review is missing: {model_id}")
            source_path = Path(str(review.get("license_source_path", "")))
            if not source_path.is_file():
                raise PackageError(f"upstream license file is missing: {model_id}")
            if sha256_file(source_path) != review.get("license_source_sha256"):
                raise PackageError(f"upstream license digest mismatch: {model_id}")
            license_rel = Path("licenses") / record.license_file_name
            license_path = candidate / license_rel
            license_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path, license_path)
            license_rows.append(
                {
                    "model_id": model_id,
                    "source_revision": record.source_revision,
                    "license_id": record.license_id,
                    "license_url": record.license_url,
                    "license_path": license_rel.as_posix(),
                    "license_sha256": sha256_file(license_path),
                    "redistributes_weights": False,
                    "kv_redistribution_decision": "PUBLIC_APPROVED",
                    "reviewer": str(review.get("reviewer", "")),
                }
            )
        manifest = {
            "schema_version": "1.0",
            "release_id": release_id,
            "source_evaluation": "Evaluation1",
            "weights_included": False,
            "artifacts": sorted(
                artifacts,
                key=lambda row: (str(row["model_id"]), str(row["mode"])),
            ),
            "model_licenses": sorted(
                license_rows,
                key=lambda row: str(row["model_id"]),
            ),
        }
        (candidate / "release-manifest.json").write_bytes(canonical_json_bytes(manifest))
        paths = sorted(
            path.relative_to(candidate).as_posix()
            for path in candidate.rglob("*")
            if path.is_file()
        )
        write_checksums(candidate, paths)
        final_receipt.update(verify_public_release(candidate, registry_path))

    atomic_publish(destination, build)
    return final_receipt
