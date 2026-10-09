"""Assemble the deterministic paper source tree without changing its source."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath


class PaperReleaseAssemblyError(ValueError):
    """The requested paper source assembly is unsafe or inconsistent."""


def _safe_relative_path(value: object) -> PurePosixPath:
    if not isinstance(value, str):
        raise PaperReleaseAssemblyError("inventory path is invalid")
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if (
        not normalized
        or normalized.startswith("/")
        or path.is_absolute()
        or ".." in path.parts
        or normalized != path.as_posix()
    ):
        raise PaperReleaseAssemblyError("inventory path is unsafe")
    return path


def _verified_rows(inventory: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    if inventory.get("schema_version") != "1.0":
        raise PaperReleaseAssemblyError("inventory schema is unsupported")
    rows = inventory.get("files")
    if not isinstance(rows, list) or inventory.get("file_count") != len(rows):
        raise PaperReleaseAssemblyError("inventory file count is invalid")
    verified: list[Mapping[str, object]] = []
    paths: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise PaperReleaseAssemblyError("inventory row is invalid")
        path = _safe_relative_path(row.get("path")).as_posix()
        role = row.get("role")
        size = row.get("size_bytes")
        digest = row.get("sha256")
        if (
            path in paths
            or not isinstance(role, str)
            or not role
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not isinstance(digest, str)
            or len(digest) != len("sha256:") + 64
            or not digest.startswith("sha256:")
        ):
            raise PaperReleaseAssemblyError("inventory row is invalid")
        try:
            int(digest.removeprefix("sha256:"), 16)
        except ValueError as error:
            raise PaperReleaseAssemblyError("inventory digest is invalid") from error
        paths.add(path)
        verified.append(row)
    if not verified or [str(row["path"]) for row in verified] != sorted(paths):
        raise PaperReleaseAssemblyError("inventory paths are not canonical")
    return tuple(verified)


def _resolve_child(root: Path, relative: PurePosixPath) -> Path:
    candidate = root.joinpath(*relative.parts)
    resolved_parent = candidate.parent.resolve()
    try:
        resolved_parent.relative_to(root)
    except ValueError as error:
        raise PaperReleaseAssemblyError("inventory path escapes its root") from error
    return candidate


def _validate_destination(destination: Path) -> Path:
    if destination.is_symlink():
        raise PaperReleaseAssemblyError("destination must not be a symlink")
    destination.mkdir(parents=True, exist_ok=True)
    resolved = destination.resolve()
    unexpected = [item for item in destination.iterdir() if item.name != ".git"]
    if unexpected:
        raise PaperReleaseAssemblyError("destination is not empty")
    marker = destination / ".git"
    if marker.exists() and (marker.is_symlink() or not marker.is_file()):
        raise PaperReleaseAssemblyError("destination Git marker is invalid")
    return resolved


def assemble_source_tree(
    source_root: Path,
    destination_root: Path,
    inventory: Mapping[str, object],
) -> tuple[Path, ...]:
    """Copy exactly the verified inventory into an empty destination tree."""

    source = source_root.resolve(strict=True)
    if source_root.is_symlink() or not source.is_dir():
        raise PaperReleaseAssemblyError("source root is invalid")
    destination = _validate_destination(destination_root)
    if source == destination:
        raise PaperReleaseAssemblyError("source and destination must differ")

    copied: list[Path] = []
    for row in _verified_rows(inventory):
        relative = _safe_relative_path(row["path"])
        source_path = _resolve_child(source, relative)
        if source_path.is_symlink() or not source_path.is_file():
            raise PaperReleaseAssemblyError(
                f"source is not a regular file: {relative.as_posix()}"
            )
        payload = source_path.read_bytes()
        digest = "sha256:" + hashlib.sha256(payload).hexdigest()
        if len(payload) != row["size_bytes"] or digest != row["sha256"]:
            raise PaperReleaseAssemblyError(
                f"source differs from inventory: {relative.as_posix()}"
            )
        destination_path = _resolve_child(destination, relative)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)
        if destination_path.is_symlink() or destination_path.read_bytes() != payload:
            raise PaperReleaseAssemblyError(
                f"copied file verification failed: {relative.as_posix()}"
            )
        copied.append(destination_path)

    expected = {Path(*_safe_relative_path(row["path"]).parts) for row in _verified_rows(inventory)}
    actual = {
        path.relative_to(destination)
        for path in destination.rglob("*")
        if path.is_file() and path.name != ".git"
    }
    if actual != expected:
        raise PaperReleaseAssemblyError("destination file set differs from inventory")
    return tuple(copied)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--destination-root", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    arguments = parser.parse_args(argv)
    inventory = json.loads(arguments.inventory.read_text(encoding="utf-8"))
    copied = assemble_source_tree(
        arguments.source_root,
        arguments.destination_root,
        inventory,
    )
    print(
        json.dumps(
            {"schema_version": "1.0", "copied_file_count": len(copied)},
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
