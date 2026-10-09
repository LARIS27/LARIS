from __future__ import annotations

import hashlib
import hmac
import os
from pathlib import Path
import stat

from laris_onpremise.rehosting.catalog import PortableBundleEntry
from laris_onpremise.rehosting.host_profile import InvestigatorProfile


class RuntimeCatalogError(ValueError):
    """Raised when a bundle model cannot be resolved exactly once."""


def resolve_model(
    profile: InvestigatorProfile,
    entry: PortableBundleEntry,
) -> Path:
    if not isinstance(profile, InvestigatorProfile):
        raise TypeError("profile must be an InvestigatorProfile")
    if not isinstance(entry, PortableBundleEntry):
        raise TypeError("entry must be a PortableBundleEntry")
    matches: list[Path] = []
    for root in profile.model_roots:
        for candidate in _model_files(root):
            if hmac.compare_digest(_file_digest(candidate), entry.model_digest):
                matches.append(candidate)
    if not matches:
        raise RuntimeCatalogError("no model matches the bundle digest")
    if len(matches) != 1:
        raise RuntimeCatalogError("multiple models match the bundle digest")
    return matches[0]


def _model_files(root: Path) -> tuple[Path, ...]:
    matches: list[Path] = []
    try:
        for directory, names, files in os.walk(root, topdown=True, followlinks=False):
            current = Path(directory)
            safe_names: list[str] = []
            for name in sorted(names):
                child = current / name
                metadata = child.lstat()
                is_reparse = bool(
                    getattr(metadata, "st_file_attributes", 0)
                    & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
                )
                if stat.S_ISLNK(metadata.st_mode) or is_reparse:
                    continue
                if stat.S_ISDIR(metadata.st_mode):
                    safe_names.append(name)
            names[:] = safe_names
            for name in sorted(files):
                if name.lower().endswith(".gguf"):
                    matches.append(current / name)
    except OSError as error:
        raise RuntimeCatalogError("model root inventory is unavailable") from error
    return tuple(matches)


def _file_digest(path: Path) -> str:
    try:
        before = path.lstat()
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise RuntimeCatalogError("model is not a regular file")
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
    except RuntimeCatalogError:
        raise
    except OSError as error:
        raise RuntimeCatalogError("model cannot be opened") from error
    digest = hashlib.sha256()
    try:
        opened = os.fstat(descriptor)
        if opened.st_dev != before.st_dev or opened.st_ino != before.st_ino:
            raise RuntimeCatalogError("model identity changed")
        size = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if opened.st_size != after.st_size or opened.st_mtime_ns != after.st_mtime_ns or size != after.st_size:
        raise RuntimeCatalogError("model changed while hashing")
    return "sha256:" + digest.hexdigest()
