"""Build the deterministic, curated paper-source inventory."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath

import yaml


class SourceBoundaryError(ValueError):
    """The proposed public source boundary is unsafe or ambiguous."""


_TEXT_SUFFIXES = {
    "",
    ".c",
    ".cc",
    ".cpp",
    ".css",
    ".h",
    ".html",
    ".in",
    ".ini",
    ".js",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

_POLICY_PATH = "evaluation/publication/source-allowlist.yaml"


def load_allowlist(path: Path) -> dict[str, object]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("schema_version") != "1.0":
        raise SourceBoundaryError("source allowlist schema is unsupported")
    roles = document.get("roles")
    deny = document.get("deny")
    if not isinstance(roles, list) or not roles or not isinstance(deny, dict):
        raise SourceBoundaryError("source allowlist is incomplete")
    names: list[str] = []
    for role in roles:
        if not isinstance(role, dict):
            raise SourceBoundaryError("source allowlist role is invalid")
        name = role.get("name")
        includes = role.get("include")
        excludes = role.get("exclude", [])
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(includes, list)
            or not includes
            or not all(isinstance(value, str) for value in includes)
            or not isinstance(excludes, list)
            or not all(isinstance(value, str) for value in excludes)
        ):
            raise SourceBoundaryError("source allowlist role is invalid")
        names.append(name)
    if len(names) != len(set(names)):
        raise SourceBoundaryError("source allowlist role names are duplicated")
    for field in ("prefixes", "suffixes", "content_markers"):
        values = deny.get(field)
        if not isinstance(values, list) or not all(
            isinstance(value, str) and value for value in values
        ):
            raise SourceBoundaryError(f"source allowlist deny.{field} is invalid")
    return document


def _matches(path: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)


def _safe_tracked_path(value: str) -> str:
    normalized = value.replace("\\", "/")
    candidate = PurePosixPath(normalized)
    if (
        not normalized
        or normalized.startswith("/")
        or candidate.is_absolute()
        or ".." in candidate.parts
        or normalized != candidate.as_posix()
    ):
        raise SourceBoundaryError("tracked source path is unsafe")
    return normalized


def _role_for(path: str, roles: Sequence[Mapping[str, object]]) -> str | None:
    matches: list[str] = []
    for role in roles:
        includes = role["include"]
        excludes = role.get("exclude", [])
        if _matches(path, includes) and not _matches(path, excludes):  # type: ignore[arg-type]
            matches.append(str(role["name"]))
    if len(matches) > 1:
        raise SourceBoundaryError(f"source path has multiple roles: {path}")
    return matches[0] if matches else None


def _validate_content(path: Path, markers: Sequence[str]) -> None:
    if path.suffix.lower() not in _TEXT_SUFFIXES:
        return
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise SourceBoundaryError(f"public text source is not UTF-8: {path.name}") from error
    for marker in markers:
        if marker in text:
            raise SourceBoundaryError(
                f"public source contains a forbidden host marker: {path.name}"
            )


def build_inventory(
    root: Path,
    policy: Mapping[str, object],
    tracked_paths: Iterable[str],
) -> dict[str, object]:
    resolved_root = root.resolve()
    roles = policy["roles"]
    deny = policy["deny"]
    prefixes = tuple(deny["prefixes"])  # type: ignore[index]
    suffixes = tuple(value.lower() for value in deny["suffixes"])  # type: ignore[index]
    markers = tuple(deny["content_markers"])  # type: ignore[index]
    files: list[dict[str, object]] = []
    for raw_path in sorted(tracked_paths):
        relative = _safe_tracked_path(raw_path)
        lowered = relative.lower()
        if any(relative.startswith(prefix) for prefix in prefixes):
            continue
        if any(lowered.endswith(suffix) for suffix in suffixes):
            continue
        role = _role_for(relative, roles)  # type: ignore[arg-type]
        if role is None:
            continue
        source = (resolved_root / Path(*PurePosixPath(relative).parts)).resolve()
        try:
            source.relative_to(resolved_root)
        except ValueError as error:
            raise SourceBoundaryError("public source escapes the repository") from error
        if source.is_symlink() or not source.is_file():
            raise SourceBoundaryError(f"public source is not a regular file: {relative}")
        # The policy necessarily names the forbidden markers it enforces.  It is
        # the only public text file exempt from scanning its own rule values.
        if relative != _POLICY_PATH:
            _validate_content(source, markers)
        payload = source.read_bytes()
        files.append(
            {
                "path": relative,
                "role": role,
                "size_bytes": len(payload),
                "sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
            }
        )
    if not files:
        raise SourceBoundaryError("public source inventory is empty")
    return {
        "schema_version": "1.0",
        "file_count": len(files),
        "files": files,
    }


def canonical_inventory_bytes(document: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            document,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _git_tracked_paths(root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return [
        item.decode("utf-8")
        for item in result.stdout.split(b"\0")
        if item
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    default_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--root", type=Path, default=default_root)
    parser.add_argument(
        "--allowlist",
        type=Path,
        default=default_root / "evaluation" / "publication" / "source-allowlist.yaml",
    )
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    inventory = build_inventory(
        arguments.root,
        load_allowlist(arguments.allowlist),
        _git_tracked_paths(arguments.root),
    )
    payload = canonical_inventory_bytes(inventory)
    if arguments.output is None:
        print(payload.decode("utf-8"), end="")
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_bytes(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
