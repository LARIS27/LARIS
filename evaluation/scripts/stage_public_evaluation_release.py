from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Mapping

from laris_onpremise.reproduction.evaluation1.archive import (
    ArchiveError,
    safe_relative_path,
)
from laris_onpremise.reproduction.evaluation1.locations import (
    PENDING_PUBLICATION,
    REQUIRED_EXTERNAL_PATHS,
    ArtifactLocationError,
    load_artifact_locations,
    verify_external_files,
)
from laris_onpremise.reproduction.evaluation1.verifier import verify_public_release


_CHECKSUM_ROW = re.compile(r"[0-9a-f]{64}  (.+)\Z")


def _locations_path(metadata_root: Path) -> Path:
    return metadata_root / "artifact-locations.json"


def _license_registry_path() -> Path:
    return Path(__file__).resolve().parents[1] / "experiment_setup" / "model_licenses.yaml"


def _checksum_paths(metadata_root: Path) -> tuple[str, ...]:
    checksum_path = metadata_root / "checksums.sha256"
    if not checksum_path.is_file() or checksum_path.is_symlink():
        raise ArtifactLocationError("checksums.sha256 is missing or unsafe")
    try:
        lines = checksum_path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ArtifactLocationError("cannot read checksums.sha256") from exc
    paths: list[str] = []
    for line in lines:
        match = _CHECKSUM_ROW.fullmatch(line)
        if match is None:
            raise ArtifactLocationError("checksums.sha256 contains a noncanonical row")
        try:
            path = safe_relative_path(match.group(1), "checksum path")
        except ArchiveError as exc:
            raise ArtifactLocationError(str(exc)) from exc
        paths.append(path.as_posix())
    if not paths or paths != sorted(set(paths)):
        raise ArtifactLocationError("checksum paths must be nonempty, unique, and sorted")
    return tuple(paths)


def _ensure_regular_path(root: Path, relative: str) -> Path:
    path = root
    for part in safe_relative_path(relative).parts:
        path = path / part
        if path.is_symlink():
            raise ArtifactLocationError(f"path contains a symlink: {relative}")
    return path


def locate_download_release_root(download_root: Path) -> Path:
    download_root = download_root.resolve()
    if not download_root.is_dir() or download_root.is_symlink():
        raise ArtifactLocationError("download root must be a regular directory")
    candidates = (
        download_root,
        download_root / "evaluation1-release-v1",
        download_root / "LARIS_Evaluation" / "evaluation1-release-v1",
    )
    matches: list[Path] = []
    for candidate in candidates:
        if not candidate.is_dir() or candidate.is_symlink():
            continue
        if all(
            _ensure_regular_path(candidate, relative).is_file()
            for relative in REQUIRED_EXTERNAL_PATHS
        ):
            matches.append(candidate.resolve())
    if len(matches) != 1:
        raise ArtifactLocationError(
            "download root must contain exactly one complete evaluation1-release-v1 tree"
        )
    return matches[0]


def initialize_release(metadata_root: Path, external_root: Path) -> tuple[Path, ...]:
    metadata_root = metadata_root.resolve()
    external_root = external_root.resolve()
    if metadata_root == external_root:
        raise ArtifactLocationError("metadata and external roots must be different")
    manifest = load_artifact_locations(_locations_path(metadata_root))
    excluded = {item.relative_path.as_posix() for item in manifest.artifacts}
    metadata_paths = tuple(path for path in _checksum_paths(metadata_root) if path not in excluded)
    copy_paths = ("checksums.sha256",) + metadata_paths

    sources: list[tuple[str, Path, Path]] = []
    for relative in copy_paths:
        source = _ensure_regular_path(metadata_root, relative)
        if not source.is_file():
            raise ArtifactLocationError(f"tracked metadata is missing: {relative}")
        destination = _ensure_regular_path(external_root, relative)
        if destination.exists() and (
            not destination.is_file() or destination.read_bytes() != source.read_bytes()
        ):
            raise ArtifactLocationError(f"conflicting existing metadata: {relative}")
        sources.append((relative, source, destination))

    if external_root.exists() and not external_root.is_dir():
        raise ArtifactLocationError("external root is not a directory")
    allowed_existing = set(copy_paths).union(excluded)
    if external_root.exists():
        for path in external_root.rglob("*"):
            if path.is_symlink():
                raise ArtifactLocationError("external root contains a symlink")
            if path.is_file() and path.relative_to(external_root).as_posix() not in allowed_existing:
                raise ArtifactLocationError("external root contains an unexpected file")

    copied: list[Path] = []
    for _relative, source, destination in sources:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(source, destination)
        copied.append(destination)
    return tuple(copied)


def artifact_checklist(metadata_root: Path) -> tuple[dict[str, object], ...]:
    manifest = load_artifact_locations(_locations_path(metadata_root.resolve()))
    return tuple(
        {
            "relative_path": item.relative_path.as_posix(),
            "size_bytes": item.size_bytes,
            "sha256": item.sha256,
            "publication_state": (
                "PENDING" if item.download_url == PENDING_PUBLICATION else "PUBLISHED"
            ),
        }
        for item in manifest.artifacts
    )


def verify_staged_release(
    metadata_root: Path,
    external_root: Path,
) -> dict[str, object]:
    metadata_root = metadata_root.resolve()
    external_root = external_root.resolve()
    manifest = load_artifact_locations(_locations_path(metadata_root))
    verify_external_files(manifest, external_root)
    aggregate = verify_public_release(external_root, _license_registry_path())
    allowed = {
        "schema_version",
        "release_id",
        "verification_status",
        "artifact_count",
        "payload_count",
        "checksum_count",
        "license_review_count",
    }
    receipt = {key: value for key, value in aggregate.items() if key in allowed}
    receipt["external_files"] = list(artifact_checklist(metadata_root))
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage and verify Evaluation1 external files.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    locate = subparsers.add_parser("locate")
    locate.add_argument("--download-root", type=Path, required=True)
    for command in ("initialize", "verify"):
        child = subparsers.add_parser(command)
        child.add_argument("--metadata-root", type=Path, required=True)
        child.add_argument("--external-root", type=Path, required=True)
    checklist = subparsers.add_parser("checklist")
    checklist.add_argument("--metadata-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "locate":
        output: Mapping[str, object] = {
            "status": "LOCATED",
            "external_root": str(locate_download_release_root(args.download_root)),
        }
    elif args.command == "initialize":
        copied = initialize_release(args.metadata_root, args.external_root)
        output = {
            "status": "INITIALIZED",
            "metadata_file_count": len(copied),
        }
    elif args.command == "checklist":
        output = {"artifacts": list(artifact_checklist(args.metadata_root))}
    else:
        output = verify_staged_release(args.metadata_root, args.external_root)
    print(json.dumps(output, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
