from __future__ import annotations

import hashlib
import hmac
import os
import re
import stat
import struct
import uuid
from dataclasses import dataclass
from pathlib import Path


IO_MAGIC = 0xAF143CD8
_PAYLOAD_NAME = "runtime_state_payload.bin"
_OUTPUT_NAME = "laris_wrapped_seq_state.bin"
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


class DerivedStateError(RuntimeError):
    """Raised when a derived sequence-state wrapper cannot be materialized."""


@dataclass(frozen=True)
class DerivedRuntimeState:
    path: Path
    destination_seq_id: int
    size_bytes: int
    digest: str
    payload_size_bytes: int
    payload_digest: str


def _fsync_directory(path: Path) -> None:
    if os.name != "posix":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _write_all(descriptor: int, data: bytes | memoryview) -> None:
    view = memoryview(data)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise DerivedStateError("derived state write was incomplete")
        view = view[written:]


def materialize_derived_wrapper(
    *,
    payload_path: Path,
    expected_payload_digest: str,
    destination_seq_id: int,
    derived_dir: Path,
) -> DerivedRuntimeState:
    """Create an analysis-only llama sequence wrapper without modifying evidence."""

    payload_path = Path(payload_path)
    derived_dir = Path(derived_dir)
    if payload_path.name != _PAYLOAD_NAME:
        raise DerivedStateError(f"payload filename must be {_PAYLOAD_NAME}")
    if not _DIGEST.fullmatch(expected_payload_digest):
        raise DerivedStateError("expected payload digest is invalid")
    if not isinstance(destination_seq_id, int) or isinstance(
        destination_seq_id, bool
    ):
        raise DerivedStateError("destination_seq_id must be an integer")
    if destination_seq_id < 0 or destination_seq_id > 0x7FFFFFFF:
        raise DerivedStateError("destination_seq_id is outside int32 range")
    if not derived_dir.is_absolute():
        raise DerivedStateError("derived directory must be absolute")
    try:
        if payload_path.is_symlink():
            raise DerivedStateError("payload is a symbolic link")
        evidence_root = payload_path.parent.resolve(strict=True)
        derived_root = derived_dir.resolve(strict=False)
        if derived_root == evidence_root or evidence_root in derived_root.parents:
            raise DerivedStateError(
                "derived directory must remain outside the evidence bundle"
            )
    except OSError as error:
        raise DerivedStateError("payload identity cannot be inspected") from error

    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        payload_fd = os.open(payload_path, flags)
    except OSError as error:
        raise DerivedStateError("payload cannot be opened") from error

    derived_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    if derived_dir.is_symlink() or not derived_dir.is_dir():
        os.close(payload_fd)
        raise DerivedStateError("derived directory is not a regular directory")

    output_path = derived_dir / _OUTPUT_NAME
    if output_path.exists() or output_path.is_symlink():
        os.close(payload_fd)
        raise DerivedStateError("derived state already exists")
    temp_path = derived_dir / f".{_OUTPUT_NAME}.{uuid.uuid4().hex}.tmp"
    temp_fd = -1
    linked = False
    try:
        source_before = os.fstat(payload_fd)
        if not stat.S_ISREG(source_before.st_mode):
            raise DerivedStateError("payload is not a regular file")

        temp_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            temp_flags |= os.O_NOFOLLOW
        temp_fd = os.open(temp_path, temp_flags, 0o600)
        wrapper_digest = hashlib.sha256()
        payload_digest = hashlib.sha256()
        header = struct.pack("<Ii", IO_MAGIC, destination_seq_id)
        _write_all(temp_fd, header)
        wrapper_digest.update(header)

        payload_size = 0
        while True:
            chunk = os.read(payload_fd, 1024 * 1024)
            if not chunk:
                break
            payload_digest.update(chunk)
            wrapper_digest.update(chunk)
            payload_size += len(chunk)
            _write_all(temp_fd, chunk)

        source_after = os.fstat(payload_fd)
        if _stat_identity(source_before) != _stat_identity(source_after):
            raise DerivedStateError("payload changed while being read")
        observed_payload_digest = "sha256:" + payload_digest.hexdigest()
        if not hmac.compare_digest(
            observed_payload_digest, expected_payload_digest
        ):
            raise DerivedStateError("payload digest mismatch")

        os.fsync(temp_fd)
        os.close(temp_fd)
        temp_fd = -1
        try:
            os.link(temp_path, output_path, follow_symlinks=False)
            linked = True
        except FileExistsError as error:
            raise DerivedStateError("derived state already exists") from error
        except OSError as error:
            raise DerivedStateError(
                "derived state requires a same-filesystem no-clobber commit"
            ) from error
        _fsync_directory(derived_dir)
        temp_path.unlink()
        _fsync_directory(derived_dir)
        return DerivedRuntimeState(
            path=output_path,
            destination_seq_id=destination_seq_id,
            size_bytes=len(header) + payload_size,
            digest="sha256:" + wrapper_digest.hexdigest(),
            payload_size_bytes=payload_size,
            payload_digest=observed_payload_digest,
        )
    except Exception:
        if linked:
            output_path.unlink(missing_ok=True)
        raise
    finally:
        if temp_fd >= 0:
            os.close(temp_fd)
        os.close(payload_fd)
        temp_path.unlink(missing_ok=True)
