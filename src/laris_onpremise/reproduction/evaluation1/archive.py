"""Deterministic tar and checksum primitives for Evaluation1 publication."""

from __future__ import annotations

import hashlib
import io
import os
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Iterable, Mapping


class ArchiveError(ValueError):
    """Raised when an archive or checksum path is unsafe."""


@dataclass(frozen=True)
class SafeTarMember:
    name: str
    size: int
    mode: int
    mtime: int
    uid: int
    gid: int
    uname: str
    gname: str


def safe_relative_path(value: str, field: str = "path") -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ArchiveError(f"unsafe {field}: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ArchiveError(f"unsafe {field}: {value!r}")
    if path.parts and ":" in path.parts[0]:
        raise ArchiveError(f"unsafe {field}: {value!r}")
    return path


def sha256_stream(stream: BinaryIO) -> str:
    digest = hashlib.sha256()
    while True:
        chunk = stream.read(1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return sha256_stream(stream)


def write_deterministic_tar(destination: Path, members: Mapping[str, bytes]) -> Path:
    """Write sorted regular files using normalized USTAR metadata."""

    normalized: list[tuple[str, bytes]] = []
    seen: set[str] = set()
    for name, payload in members.items():
        safe_relative_path(name, "tar member")
        if name in seen:
            raise ArchiveError(f"duplicate tar member: {name}")
        if not isinstance(payload, bytes):
            raise ArchiveError(f"tar member payload must be bytes: {name}")
        seen.add(name)
        normalized.append((name, payload))

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".incomplete")
    try:
        with tarfile.open(temporary, mode="w", format=tarfile.USTAR_FORMAT) as archive:
            for name, payload in sorted(normalized):
                info = tarfile.TarInfo(name=name)
                info.size = len(payload)
                info.mode = 0o644
                info.mtime = 0
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                archive.addfile(info, io.BytesIO(payload))
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def inspect_safe_tar(path: Path) -> tuple[SafeTarMember, ...]:
    """Inspect a public tar without extracting it and reject dangerous members."""

    seen: set[str] = set()
    inspected: list[SafeTarMember] = []
    try:
        with tarfile.open(path, mode="r:") as archive:
            for member in archive:
                safe_relative_path(member.name, "tar member")
                if member.name in seen:
                    raise ArchiveError(f"duplicate tar member: {member.name}")
                seen.add(member.name)
                if not member.isreg():
                    raise ArchiveError(
                        f"public tar member must be a regular file: {member.name}"
                    )
                inspected.append(
                    SafeTarMember(
                        name=member.name,
                        size=member.size,
                        mode=member.mode,
                        mtime=member.mtime,
                        uid=member.uid,
                        gid=member.gid,
                        uname=member.uname,
                        gname=member.gname,
                    )
                )
    except (OSError, tarfile.TarError) as exc:
        raise ArchiveError(f"cannot inspect tar: {path}") from exc
    return tuple(inspected)


def read_safe_tar_member(path: Path, name: str, *, limit_bytes: int) -> bytes:
    safe_relative_path(name, "tar member")
    members = {member.name: member for member in inspect_safe_tar(path)}
    selected = members.get(name)
    if selected is None:
        raise ArchiveError(f"tar member is missing: {name}")
    if selected.size > limit_bytes:
        raise ArchiveError(f"tar member exceeds read limit: {name}")
    try:
        with tarfile.open(path, mode="r:") as archive:
            stream = archive.extractfile(name)
            if stream is None:
                raise ArchiveError(f"tar member cannot be read: {name}")
            payload = stream.read(limit_bytes + 1)
    except (OSError, tarfile.TarError) as exc:
        raise ArchiveError(f"cannot read tar member: {name}") from exc
    if len(payload) != selected.size:
        raise ArchiveError(f"tar member size changed while reading: {name}")
    return payload


def hash_safe_tar_member(path: Path, name: str) -> tuple[str, int]:
    safe_relative_path(name, "tar member")
    members = {member.name: member for member in inspect_safe_tar(path)}
    selected = members.get(name)
    if selected is None:
        raise ArchiveError(f"tar member is missing: {name}")
    try:
        with tarfile.open(path, mode="r:") as archive:
            stream = archive.extractfile(name)
            if stream is None:
                raise ArchiveError(f"tar member cannot be read: {name}")
            digest = sha256_stream(stream)
    except (OSError, tarfile.TarError) as exc:
        raise ArchiveError(f"cannot hash tar member: {name}") from exc
    return digest, selected.size


def hash_safe_tar_members(
    path: Path,
    names: Iterable[str],
) -> dict[str, tuple[str, int]]:
    """Hash selected regular members while opening the large tar only once."""

    requested = tuple(names)
    if len(requested) != len(set(requested)):
        raise ArchiveError("duplicate requested tar member")
    for name in requested:
        safe_relative_path(name, "tar member")
    inspected = {member.name: member for member in inspect_safe_tar(path)}
    missing = sorted(set(requested).difference(inspected))
    if missing:
        raise ArchiveError(f"tar member is missing: {missing[0]}")

    result: dict[str, tuple[str, int]] = {}
    try:
        with tarfile.open(path, mode="r:") as archive:
            archive_members = {member.name: member for member in archive}
            for name in requested:
                stream = archive.extractfile(archive_members[name])
                if stream is None:
                    raise ArchiveError(f"tar member cannot be read: {name}")
                result[name] = (sha256_stream(stream), inspected[name].size)
    except (OSError, tarfile.TarError) as exc:
        raise ArchiveError(f"cannot hash selected tar members: {path}") from exc
    return result


def write_checksums(
    root: Path,
    relative_paths: Iterable[str],
    *,
    filename: str = "checksums.sha256",
) -> Path:
    safe_relative_path(filename, "checksum filename")
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    for value in relative_paths:
        path = safe_relative_path(value, "checksum path")
        normalized = path.as_posix()
        if normalized == filename:
            raise ArchiveError("checksum file cannot digest itself")
        if normalized in seen:
            raise ArchiveError(f"duplicate checksum path: {normalized}")
        seen.add(normalized)
        source = root.joinpath(*path.parts)
        if not source.is_file() or source.is_symlink():
            raise ArchiveError(f"checksum source is not a regular file: {normalized}")
        rows.append((normalized, sha256_file(source).removeprefix("sha256:")))
    payload = "".join(f"{digest}  {name}\n" for name, digest in sorted(rows))
    destination = root / filename
    destination.write_text(payload, encoding="ascii", newline="\n")
    return destination
