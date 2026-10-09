"""Provider-neutral locations and offline verification for Evaluation1 artifacts."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .archive import ArchiveError, safe_relative_path, sha256_file


PENDING_PUBLICATION = "PENDING_PUBLICATION"
REQUIRED_EXTERNAL_PATHS = tuple(
    sorted(
        (
            "models/llama-3.2-3b-instruct-q4-k-m/log-linked.tar",
            "models/llama-3.2-3b-instruct-q4-k-m/proxy.tar",
            "models/qwen2.5-7b-instruct-q4-k-m/log-linked.tar",
            "models/qwen2.5-7b-instruct-q4-k-m/proxy.tar",
            "shared-data.tar.gz",
            "fixtures/keys/75cbd8e47c923053.key",
            "fixtures/keys/da75811916eb78a6.key",
        )
    )
)
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_HTTPS_URL = re.compile(r"https://\S+\Z")
_ROOT_KEYS = frozenset(("schema_version", "release_id", "artifacts"))
_ENTRY_KEYS = frozenset(("relative_path", "size_bytes", "sha256", "download_url"))


class ArtifactLocationError(ValueError):
    """Raised when location metadata or a local external artifact is unsafe."""


@dataclass(frozen=True)
class ExternalArtifactLocation:
    relative_path: PurePosixPath
    size_bytes: int
    sha256: str
    download_url: str

    @property
    def is_published(self) -> bool:
        return self.download_url != PENDING_PUBLICATION


@dataclass(frozen=True)
class ArtifactLocationManifest:
    schema_version: str
    release_id: str
    artifacts: tuple[ExternalArtifactLocation, ...]

    @property
    def is_published(self) -> bool:
        return all(item.is_published for item in self.artifacts)


def _object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ArtifactLocationError(f"{field} must be an object")
    return value


def _exact_keys(value: dict[str, Any], expected: frozenset[str], field: str) -> None:
    if frozenset(value) != expected:
        raise ArtifactLocationError(f"{field} has unexpected or missing properties")


def _parse_entry(value: Any, index: int) -> ExternalArtifactLocation:
    row = _object(value, f"artifacts[{index}]")
    _exact_keys(row, _ENTRY_KEYS, f"artifacts[{index}]")
    try:
        relative_path = safe_relative_path(row["relative_path"], "artifact path")
    except ArchiveError as exc:
        raise ArtifactLocationError(str(exc)) from exc
    size = row["size_bytes"]
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ArtifactLocationError("artifact size_bytes must be a positive integer")
    digest = row["sha256"]
    if not isinstance(digest, str) or _DIGEST.fullmatch(digest) is None:
        raise ArtifactLocationError("artifact sha256 must be a lowercase sha256 digest")
    url = row["download_url"]
    if not isinstance(url, str) or (
        url != PENDING_PUBLICATION and _HTTPS_URL.fullmatch(url) is None
    ):
        raise ArtifactLocationError("artifact download_url must be pending or HTTPS")
    return ExternalArtifactLocation(relative_path, size, digest, url)


def load_artifact_locations(path: Path) -> ArtifactLocationManifest:
    if not path.is_file() or path.is_symlink():
        raise ArtifactLocationError(f"location manifest is not a regular file: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactLocationError(f"cannot read location manifest: {path}") from exc
    root = _object(document, "location manifest")
    _exact_keys(root, _ROOT_KEYS, "location manifest")
    if root["schema_version"] != "1.0":
        raise ArtifactLocationError("unsupported location schema_version")
    if root["release_id"] != "evaluation1-release-v1":
        raise ArtifactLocationError("unsupported location release_id")
    rows = root["artifacts"]
    if not isinstance(rows, list):
        raise ArtifactLocationError("artifacts must be an array")
    artifacts = tuple(_parse_entry(row, index) for index, row in enumerate(rows))
    names = tuple(item.relative_path.as_posix() for item in artifacts)
    if len(names) != len(set(names)):
        raise ArtifactLocationError("duplicate external artifact path")
    if names != REQUIRED_EXTERNAL_PATHS:
        raise ArtifactLocationError("manifest does not contain the exact external artifact set")
    pending = tuple(item.download_url == PENDING_PUBLICATION for item in artifacts)
    if pending and any(pending) and not all(pending):
        raise ArtifactLocationError("location manifest cannot mix pending and published URLs")
    return ArtifactLocationManifest("1.0", "evaluation1-release-v1", artifacts)


def _reject_symlink_path(root: Path, relative_path: PurePosixPath) -> Path:
    if root.is_symlink():
        raise ArtifactLocationError("external release root is a symlink")
    candidate = root
    for part in relative_path.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise ArtifactLocationError(
                f"external artifact path contains a symlink: {relative_path.as_posix()}"
            )
    return candidate


def verify_external_files(
    manifest: ArtifactLocationManifest,
    release_root: Path,
) -> tuple[Path, ...]:
    verified: list[Path] = []
    for item in manifest.artifacts:
        path = _reject_symlink_path(release_root, item.relative_path)
        if not path.is_file():
            raise ArtifactLocationError(
                f"external artifact is missing: {item.relative_path.as_posix()}"
            )
        if path.stat().st_size != item.size_bytes:
            raise ArtifactLocationError(
                f"external artifact size mismatch: {item.relative_path.as_posix()}"
            )
        if sha256_file(path) != item.sha256:
            raise ArtifactLocationError(
                f"external artifact digest mismatch: {item.relative_path.as_posix()}"
            )
        verified.append(path)
    return tuple(verified)


__all__ = [
    "PENDING_PUBLICATION",
    "REQUIRED_EXTERNAL_PATHS",
    "ArtifactLocationError",
    "ArtifactLocationManifest",
    "ExternalArtifactLocation",
    "load_artifact_locations",
    "verify_external_files",
]
