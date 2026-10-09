"""Strict contracts for the public Evaluation1 KV release.

The public release is intentionally narrower than the private evaluation
archive.  Each index represents exactly one model and one capture mode and
contains the complete 6 checkpoint by 10 user matrix.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import yaml


_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_REVISION_RE = re.compile(r"[0-9a-f]{40}\Z")
_CHECKPOINT_RE = re.compile(r"C[1-6]\Z")
_USER_RE = re.compile(r"U(?:0[1-9]|10)\Z")
_EXPECTED_CHECKPOINTS = tuple(f"C{value}" for value in range(1, 7))
_EXPECTED_USERS = tuple(f"U{value:02d}" for value in range(1, 11))


class ContractError(ValueError):
    """Raised when a public release object violates its sealed contract."""


class CaptureMode(str, Enum):
    PROXY = "PROXY"
    LOG_LINKED = "LOG_LINKED"

    @property
    def subject_kind(self) -> str:
        if self is CaptureMode.PROXY:
            return "USER_HMAC"
        return "ENDPOINT_REF"


def _require_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{field} must be a non-empty string")
    return value


def _require_sha256(value: object, field: str) -> str:
    text = _require_text(value, field)
    if _SHA256_RE.fullmatch(text) is None:
        raise ContractError(f"{field} must be a canonical sha256 digest")
    return text


def _require_safe_relative_path(value: object, field: str) -> str:
    text = _require_text(value, field)
    if "\\" in text:
        raise ContractError(f"{field} must use a safe POSIX relative path")
    path = PurePosixPath(text)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ContractError(f"{field} must use a safe POSIX relative path")
    return text


@dataclass(frozen=True)
class PublicBundleEntry:
    model_id: str
    mode: CaptureMode
    checkpoint_id: str
    public_user_ref: str
    subject_ref_kind: str
    subject_ref: str
    key_id: str
    session_ref: str
    bundle_id: str
    bundle_path: str
    payload_sha256: str
    payload_size_bytes: int
    source_bundle_manifest_sha256: str

    def __post_init__(self) -> None:
        _require_text(self.model_id, "model_id")
        mode = self.mode
        if not isinstance(mode, CaptureMode):
            try:
                mode = CaptureMode(mode)
            except (TypeError, ValueError) as exc:
                raise ContractError("mode is not supported") from exc
            object.__setattr__(self, "mode", mode)
        if _CHECKPOINT_RE.fullmatch(self.checkpoint_id) is None:
            raise ContractError("checkpoint_id must be C1 through C6")
        if _USER_RE.fullmatch(self.public_user_ref) is None:
            raise ContractError("public_user_ref must be U01 through U10")
        if self.subject_ref_kind != mode.subject_kind:
            raise ContractError(
                f"subject_ref_kind must be {mode.subject_kind} for {mode.value}"
            )
        _require_text(self.subject_ref, "subject_ref")
        _require_text(self.key_id, "key_id")
        _require_text(self.session_ref, "session_ref")
        _require_text(self.bundle_id, "bundle_id")
        _require_safe_relative_path(self.bundle_path, "bundle_path")
        _require_sha256(self.payload_sha256, "payload_sha256")
        _require_sha256(
            self.source_bundle_manifest_sha256,
            "source_bundle_manifest_sha256",
        )
        if not isinstance(self.payload_size_bytes, int) or self.payload_size_bytes <= 0:
            raise ContractError("payload_size_bytes must be a positive integer")

    @property
    def tuple_key(self) -> tuple[str, str, str, str]:
        return (
            self.model_id,
            self.mode.value,
            self.checkpoint_id,
            self.public_user_ref,
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "PublicBundleEntry":
        try:
            return cls(
                model_id=value["model_id"],  # type: ignore[arg-type]
                mode=CaptureMode(value["mode"]),
                checkpoint_id=value["checkpoint_id"],  # type: ignore[arg-type]
                public_user_ref=value["public_user_ref"],  # type: ignore[arg-type]
                subject_ref_kind=value["subject_ref_kind"],  # type: ignore[arg-type]
                subject_ref=value["subject_ref"],  # type: ignore[arg-type]
                key_id=value["key_id"],  # type: ignore[arg-type]
                session_ref=value["session_ref"],  # type: ignore[arg-type]
                bundle_id=value["bundle_id"],  # type: ignore[arg-type]
                bundle_path=value["bundle_path"],  # type: ignore[arg-type]
                payload_sha256=value["payload_sha256"],  # type: ignore[arg-type]
                payload_size_bytes=value["payload_size_bytes"],  # type: ignore[arg-type]
                source_bundle_manifest_sha256=value[
                    "source_bundle_manifest_sha256"
                ],  # type: ignore[arg-type]
            )
        except KeyError as exc:
            raise ContractError(f"bundle entry is missing {exc.args[0]}") from exc
        except (TypeError, ValueError) as exc:
            if isinstance(exc, ContractError):
                raise
            raise ContractError("bundle entry has an invalid mode") from exc


@dataclass(frozen=True)
class PublicBundleIndex:
    schema_version: str
    release_id: str
    model_id: str
    mode: CaptureMode
    subject_ref_kind: str
    entries: tuple[PublicBundleEntry, ...]

    def __post_init__(self) -> None:
        if self.schema_version != "1.0":
            raise ContractError("schema_version must be 1.0")
        _require_text(self.release_id, "release_id")
        _require_text(self.model_id, "model_id")
        mode = self.mode
        if not isinstance(mode, CaptureMode):
            try:
                mode = CaptureMode(mode)
            except (TypeError, ValueError) as exc:
                raise ContractError("mode is not supported") from exc
            object.__setattr__(self, "mode", mode)
        if self.subject_ref_kind != mode.subject_kind:
            raise ContractError(
                f"subject_ref_kind must be {mode.subject_kind} for {mode.value}"
            )
        if len(self.entries) != 60:
            raise ContractError("public bundle index must contain exactly 60 entries")

        keys = [entry.tuple_key for entry in self.entries]
        if len(keys) != len(set(keys)):
            raise ContractError("duplicate bundle tuple in public index")
        for entry in self.entries:
            if entry.model_id != self.model_id or entry.mode is not mode:
                raise ContractError("bundle entry does not match index model and mode")
            if entry.subject_ref_kind != self.subject_ref_kind:
                raise ContractError("bundle entry does not match index subject_ref_kind")

        expected = {
            (self.model_id, mode.value, checkpoint, user)
            for checkpoint in _EXPECTED_CHECKPOINTS
            for user in _EXPECTED_USERS
        }
        if set(keys) != expected:
            raise ContractError("public bundle index does not cover the complete matrix")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "PublicBundleIndex":
        try:
            raw_entries = value["entries"]
            if not isinstance(raw_entries, list):
                raise ContractError("entries must be a list")
            return cls(
                schema_version=value["schema_version"],  # type: ignore[arg-type]
                release_id=value["release_id"],  # type: ignore[arg-type]
                model_id=value["model_id"],  # type: ignore[arg-type]
                mode=CaptureMode(value["mode"]),
                subject_ref_kind=value["subject_ref_kind"],  # type: ignore[arg-type]
                entries=tuple(
                    PublicBundleEntry.from_mapping(entry)
                    for entry in raw_entries
                    if isinstance(entry, Mapping)
                ),
            )
        except KeyError as exc:
            raise ContractError(f"public index is missing {exc.args[0]}") from exc
        except (TypeError, ValueError) as exc:
            if isinstance(exc, ContractError):
                raise
            raise ContractError("public index has an invalid mode") from exc


@dataclass(frozen=True)
class ModelLicenseRecord:
    model_id: str
    source_repository: str
    source_revision: str
    license_id: str
    license_name: str
    license_url: str
    license_file_name: str
    redistributes_weights: bool
    required_notices: tuple[str, ...]
    acceptable_use_url: str | None = None

    def __post_init__(self) -> None:
        for field in (
            "model_id",
            "source_repository",
            "license_id",
            "license_name",
            "license_url",
            "license_file_name",
        ):
            _require_text(getattr(self, field), field)
        if _REVISION_RE.fullmatch(self.source_revision) is None:
            raise ContractError("source_revision must be a pinned 40-hex revision")
        if not self.license_url.startswith("https://"):
            raise ContractError("license_url must use https")
        if self.redistributes_weights:
            raise ContractError("the public KV release must not redistribute weights")
        if any(not isinstance(notice, str) or not notice.strip() for notice in self.required_notices):
            raise ContractError("required_notices must contain non-empty strings")


def _load_yaml_mapping(path: Path) -> Mapping[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ContractError(f"cannot load YAML registry: {path}") from exc
    if not isinstance(value, Mapping):
        raise ContractError(f"YAML registry must contain a mapping: {path}")
    return value


def load_model_license_registry(path: Path) -> dict[str, ModelLicenseRecord]:
    document = _load_yaml_mapping(path)
    if document.get("schema_version") != "1.0":
        raise ContractError("model license schema_version must be 1.0")
    raw_models = document.get("models")
    if not isinstance(raw_models, list):
        raise ContractError("model license registry must contain a models list")

    records: dict[str, ModelLicenseRecord] = {}
    for raw in raw_models:
        if not isinstance(raw, Mapping):
            raise ContractError("model license record must be a mapping")
        try:
            notices = raw.get("required_notices", [])
            if not isinstance(notices, list):
                raise ContractError("required_notices must be a list")
            record = ModelLicenseRecord(
                model_id=raw["model_id"],
                source_repository=raw["source_repository"],
                source_revision=raw["source_revision"],
                license_id=raw["license_id"],
                license_name=raw["license_name"],
                license_url=raw["license_url"],
                license_file_name=raw["license_file_name"],
                redistributes_weights=raw["redistributes_weights"],
                required_notices=tuple(notices),
                acceptable_use_url=raw.get("acceptable_use_url"),
            )
        except KeyError as exc:
            raise ContractError(f"model license record is missing {exc.args[0]}") from exc
        if record.model_id in records:
            raise ContractError(f"duplicate model license record: {record.model_id}")
        records[record.model_id] = record

    models_path = path.with_name("models.yaml")
    if models_path.is_file():
        source_document = _load_yaml_mapping(models_path)
        source_models = source_document.get("models")
        if not isinstance(source_models, list):
            raise ContractError("models.yaml must contain a models list")
        pinned = {
            raw.get("model_id"): raw
            for raw in source_models
            if isinstance(raw, Mapping)
        }
        if set(records) != set(pinned):
            raise ContractError("license registry model set differs from models.yaml")
        for model_id, record in records.items():
            source = pinned[model_id]
            if (
                source.get("source_repository") != record.source_repository
                or source.get("source_revision") != record.source_revision
            ):
                raise ContractError(f"license registry pin differs for {model_id}")
    return records
