from __future__ import annotations

import ctypes
import errno
import hashlib
import os
import platform
import secrets
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


RENAME_NOREPLACE = 1
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0o400000)
_O_CLOEXEC = getattr(os, "O_CLOEXEC", 0o2000000)


class PublicationError(RuntimeError):
    """Base class for evidence publication failures."""


class PublicationConflictError(PublicationError):
    """Raised when the final evidence basename already exists."""


class PublicationCrossFilesystemError(PublicationError):
    """Raised when a same-directory publication invariant is violated."""


class PublicationUnsupportedError(PublicationError):
    """Raised when no-clobber publication is unavailable."""


class PublicationDurabilityError(PublicationError):
    """Raised when rename succeeded but durable final visibility is unknown."""

    final_state = "unknown"


@dataclass(frozen=True)
class PublishedArtifact:
    basename: str
    size_bytes: int
    mode: int
    payload_digest: str


class _PublicationBackend(Protocol):
    def open_temp(self, directory_fd: int, name: str, flags: int, mode: int) -> int: ...
    def fchmod(self, file_fd: int, mode: int) -> None: ...
    def fstat(self, file_fd: int) -> tuple[bool, int]: ...
    def write_all(self, file_fd: int, payload: bytes) -> None: ...
    def fsync_file(self, file_fd: int) -> None: ...
    def close_file(self, file_fd: int) -> None: ...
    def rename_noreplace(self, old_directory_fd: int, old_name: str, new_directory_fd: int, new_name: str) -> None: ...
    def fsync_directory(self, directory_fd: int) -> None: ...
    def unlink(self, directory_fd: int, name: str) -> None: ...


class _SystemPublicationBackend:
    def open_temp(self, directory_fd: int, name: str, flags: int, mode: int) -> int:
        return os.open(name, flags, mode, dir_fd=directory_fd)

    def fchmod(self, file_fd: int, mode: int) -> None:
        os.fchmod(file_fd, mode)

    def fstat(self, file_fd: int) -> tuple[bool, int]:
        result = os.fstat(file_fd)
        return stat.S_ISREG(result.st_mode), result.st_nlink

    def write_all(self, file_fd: int, payload: bytes) -> None:
        view = memoryview(payload)
        while view:
            written = os.write(file_fd, view)
            if written <= 0:
                raise OSError(errno.EIO, "short write during publication")
            view = view[written:]

    def fsync_file(self, file_fd: int) -> None:
        os.fsync(file_fd)

    def close_file(self, file_fd: int) -> None:
        os.close(file_fd)

    def rename_noreplace(self, old_directory_fd: int, old_name: str, new_directory_fd: int, new_name: str) -> None:
        _renameat2(old_directory_fd, old_name, new_directory_fd, new_name)

    def fsync_directory(self, directory_fd: int) -> None:
        os.fsync(directory_fd)

    def unlink(self, directory_fd: int, name: str) -> None:
        os.unlink(name, dir_fd=directory_fd)


def _renameat2(old_directory_fd: int, old_name: str, new_directory_fd: int, new_name: str) -> None:
    try:
        renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    except AttributeError as error:
        raise OSError(errno.ENOSYS, "renameat2 is unavailable") from error
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        old_directory_fd,
        os.fsencode(old_name),
        new_directory_fd,
        os.fsencode(new_name),
        RENAME_NOREPLACE,
    )
    if result != 0:
        raise OSError(ctypes.get_errno(), "renameat2 failed")


def _validate_basename(basename: str) -> None:
    if (
        not isinstance(basename, str)
        or not basename
        or basename in {".", ".."}
        or "/" in basename
        or "\\" in basename
        or "\x00" in basename
    ):
        raise ValueError("basename must be one safe path component")


def _validate_mode(mode: int) -> None:
    if isinstance(mode, bool) or not isinstance(mode, int) or mode < 0 or mode > 0o777:
        raise ValueError("mode must be a file permission mode")


def _map_rename_error(error: OSError) -> PublicationError:
    if error.errno == errno.EEXIST:
        return PublicationConflictError("final basename already exists")
    if error.errno == errno.EXDEV:
        return PublicationCrossFilesystemError(
            "same-directory no-clobber publication crossed filesystems"
        )
    unsupported = {errno.ENOSYS, errno.EINVAL, errno.EOPNOTSUPP}
    if hasattr(errno, "ENOTSUP"):
        unsupported.add(errno.ENOTSUP)
    if error.errno in unsupported:
        return PublicationUnsupportedError("renameat2 RENAME_NOREPLACE is unsupported")
    return PublicationError("renameat2 no-clobber publication failed")


def _publish_with_backend(
    directory_fd: int,
    basename: str,
    payload: bytes,
    *,
    mode: int = 0o600,
    backend: _PublicationBackend,
) -> PublishedArtifact:
    """Publish through a narrow backend seam used by portable boundary tests."""
    if isinstance(directory_fd, bool) or not isinstance(directory_fd, int) or directory_fd < 0:
        raise ValueError("directory_fd must be a non-negative file descriptor")
    _validate_basename(basename)
    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    _validate_mode(mode)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | _O_NOFOLLOW | _O_CLOEXEC
    temporary_name = f".{basename}.tmp-{secrets.token_hex(16)}"
    file_fd: int | None = None
    temporary_entry_owned = False
    renamed = False
    try:
        file_fd = backend.open_temp(directory_fd, temporary_name, flags, mode)
        temporary_entry_owned = True
        backend.fchmod(file_fd, mode)
        regular, link_count = backend.fstat(file_fd)
        if not regular or link_count != 1:
            raise PublicationError("temporary evidence entry is not a single-link regular file")
        backend.write_all(file_fd, payload)
        backend.fsync_file(file_fd)
        backend.close_file(file_fd)
        file_fd = None
        try:
            backend.rename_noreplace(directory_fd, temporary_name, directory_fd, basename)
        except OSError as error:
            raise _map_rename_error(error) from error
        renamed = True
        try:
            backend.fsync_directory(directory_fd)
        except OSError as error:
            raise PublicationDurabilityError(
                "directory fsync failed after rename; final visibility is unknown"
            ) from error
    except BaseException:
        if file_fd is not None:
            try:
                backend.close_file(file_fd)
            except OSError:
                pass
        if temporary_entry_owned and not renamed:
            try:
                backend.unlink(directory_fd, temporary_name)
            except OSError:
                pass
        raise
    return PublishedArtifact(
        basename=basename,
        size_bytes=len(payload),
        mode=mode,
        payload_digest="sha256:" + hashlib.sha256(payload).hexdigest(),
    )


def open_trusted_directory(path: Path | str) -> int:
    """Open one trusted Linux directory for relative evidence publication."""
    if platform.system() != "Linux":
        raise PublicationUnsupportedError("no-clobber publication requires Linux")
    flags = os.O_RDONLY | os.O_DIRECTORY | _O_CLOEXEC | _O_NOFOLLOW
    directory_fd = os.open(os.fspath(path), flags)
    try:
        if not stat.S_ISDIR(os.fstat(directory_fd).st_mode):
            raise PublicationError("trusted publication target is not a directory")
    except BaseException:
        os.close(directory_fd)
        raise
    return directory_fd


def publish_noreplace(
    directory_fd: int,
    basename: str,
    payload: bytes,
    *,
    mode: int = 0o600,
) -> PublishedArtifact:
    """Atomically publish bytes without replacing an existing final basename."""
    if platform.system() != "Linux":
        raise PublicationUnsupportedError("no-clobber publication requires Linux")
    return _publish_with_backend(
        directory_fd,
        basename,
        payload,
        mode=mode,
        backend=_SystemPublicationBackend(),
    )
