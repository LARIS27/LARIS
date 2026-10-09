from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import re
import secrets
import stat
from dataclasses import dataclass
from pathlib import Path


_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "model_profile_id",
        "canonical_path",
        "gguf_digest",
        "size_bytes",
        "device",
        "inode",
        "mtime_ns",
    }
)
_MAX_RECEIPT_BYTES = 16 * 1024


class ModelRegistrationError(RuntimeError):
    """Raised when a registered GGUF artifact is missing or has changed."""


@dataclass(frozen=True)
class ModelRegistrationReceipt:
    model_profile_id: str
    canonical_path: Path
    gguf_digest: str
    size_bytes: int
    device: int
    inode: int
    mtime_ns: int

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": "1.0",
            "model_profile_id": self.model_profile_id,
            "canonical_path": str(self.canonical_path),
            "gguf_digest": self.gguf_digest,
            "size_bytes": self.size_bytes,
            "device": self.device,
            "inode": self.inode,
            "mtime_ns": self.mtime_ns,
        }


@dataclass(frozen=True)
class RegisteredModelHandle:
    path: Path
    digest: str
    size_bytes: int
    device: int
    inode: int
    mtime_ns: int
    descriptor: int
    receipt_path: Path | None = None


def _canonical_bytes(value: dict[str, object]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        + b"\n"
    )


def _require_nonnegative_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ModelRegistrationError(f"registration receipt {label} is invalid")
    return value


def _receipt_from_bytes(raw: bytes) -> ModelRegistrationReceipt:
    if not raw or len(raw) > _MAX_RECEIPT_BYTES:
        raise ModelRegistrationError("model registration receipt is invalid")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ModelRegistrationError("model registration receipt has duplicate fields")
            result[key] = value
        return result

    try:
        value = json.loads(raw.decode("ascii"), object_pairs_hook=reject_duplicates)
    except ModelRegistrationError:
        raise
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ModelRegistrationError("model registration receipt is invalid") from error
    if (
        not isinstance(value, dict)
        or set(value) != _RECEIPT_FIELDS
        or _canonical_bytes(value) != raw
        or value["schema_version"] != "1.0"
    ):
        raise ModelRegistrationError("model registration receipt is invalid")
    profile_id = value["model_profile_id"]
    digest = value["gguf_digest"]
    canonical_path = value["canonical_path"]
    if not isinstance(profile_id, str) or _IDENTIFIER.fullmatch(profile_id) is None:
        raise ModelRegistrationError("registration receipt model_profile_id is invalid")
    if not isinstance(digest, str) or _DIGEST.fullmatch(digest) is None:
        raise ModelRegistrationError("registration receipt gguf_digest is invalid")
    if not isinstance(canonical_path, str):
        raise ModelRegistrationError("registration receipt canonical_path is invalid")
    path = Path(canonical_path)
    if not path.is_absolute() or ".." in path.parts:
        raise ModelRegistrationError("registration receipt canonical_path is invalid")
    return ModelRegistrationReceipt(
        model_profile_id=profile_id,
        canonical_path=path,
        gguf_digest=digest,
        size_bytes=_require_nonnegative_integer(value["size_bytes"], "size_bytes"),
        device=_require_nonnegative_integer(value["device"], "device"),
        inode=_require_nonnegative_integer(value["inode"], "inode"),
        mtime_ns=_require_nonnegative_integer(value["mtime_ns"], "mtime_ns"),
    )


def load_model_registration_receipt(path: Path) -> ModelRegistrationReceipt:
    try:
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ModelRegistrationError("model registration receipt is not a regular file")
        if os.name != "nt" and (
            before.st_uid != os.geteuid() or before.st_mode & 0o077
        ):
            raise ModelRegistrationError("model registration receipt is unsafe")
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        try:
            opened = os.fstat(descriptor)
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_nlink != 1
                or (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)
            ):
                raise ModelRegistrationError("model registration receipt changed during open")
            raw = bytearray()
            while block := os.read(descriptor, 4096):
                raw.extend(block)
                if len(raw) > _MAX_RECEIPT_BYTES:
                    raise ModelRegistrationError("model registration receipt is invalid")
            after = os.fstat(descriptor)
            if (
                opened.st_size,
                opened.st_mtime_ns,
                opened.st_ctime_ns,
            ) != (
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ):
                raise ModelRegistrationError("model registration receipt changed during read")
        finally:
            os.close(descriptor)
    except ModelRegistrationError:
        raise
    except OSError as error:
        raise ModelRegistrationError("model registration receipt is unavailable") from error
    return _receipt_from_bytes(bytes(raw))


def _open_model(path: Path) -> tuple[Path, os.stat_result, int]:
    try:
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ModelRegistrationError("model must be a single-link regular file")
        if os.name != "nt" and before.st_mode & 0o022:
            raise ModelRegistrationError("model must not be group/world writable")
        canonical = path.resolve(strict=True)
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)
        ):
            os.close(descriptor)
            raise ModelRegistrationError("model changed during open")
        return canonical, opened, descriptor
    except ModelRegistrationError:
        raise
    except OSError as error:
        raise ModelRegistrationError("model must be a single-link regular file") from error


def _stream_digest(descriptor: int) -> str:
    digest = hashlib.sha256()
    while block := os.read(descriptor, 1024 * 1024):
        digest.update(block)
    return "sha256:" + digest.hexdigest()


def _publish_noreplace(path: Path, raw: bytes) -> None:
    if path.exists() or path.is_symlink():
        existing = load_model_registration_receipt(path)
        if _canonical_bytes(existing.to_mapping()) == raw:
            return
        raise ModelRegistrationError("model registration receipt already differs")
    if not path.is_absolute() or not path.parent.is_dir() or path.parent.is_symlink():
        raise ModelRegistrationError("model registration receipt parent is unsafe")
    pending = path.parent / f".{path.name}.pending-{secrets.token_hex(16)}"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    published = False
    try:
        descriptor = os.open(pending, flags, 0o600)
        if os.name != "nt":
            os.fchmod(descriptor, 0o600)
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                raise OSError(errno.EIO, "model registration write made no progress")
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        if os.name == "nt":
            from ctypes import wintypes

            move_file = ctypes.WinDLL("kernel32", use_last_error=True).MoveFileExW
            move_file.argtypes = (wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD)
            move_file.restype = wintypes.BOOL
            if not move_file(str(pending), str(path), 0x00000008):
                error_code = ctypes.get_last_error()
                if error_code in {80, 183}:
                    raise FileExistsError(str(path))
                raise ctypes.WinError(error_code)
        else:
            libc = ctypes.CDLL(None, use_errno=True)
            renameat2 = getattr(libc, "renameat2", None)
            if renameat2 is None:
                raise ModelRegistrationError("atomic no-clobber receipt publication is unavailable")
            renameat2.argtypes = (
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            )
            renameat2.restype = ctypes.c_int
            if renameat2(-100, os.fsencode(pending), -100, os.fsencode(path), 1) != 0:
                error_code = ctypes.get_errno()
                if error_code == errno.EEXIST:
                    raise FileExistsError(str(path))
                raise OSError(error_code, os.strerror(error_code))
        published = True
        if os.name != "nt":
            parent = os.open(
                path.parent,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
            )
            try:
                os.fsync(parent)
            finally:
                os.close(parent)
    except FileExistsError:
        existing = load_model_registration_receipt(path)
        if _canonical_bytes(existing.to_mapping()) != raw:
            raise ModelRegistrationError("model registration receipt already differs")
    except ModelRegistrationError:
        raise
    except OSError as error:
        raise ModelRegistrationError("model registration receipt cannot be published") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if not published:
            try:
                pending.unlink()
            except OSError:
                pass


def register_model(
    *,
    model_profile_id: str,
    path: Path,
    expected_digest: str,
    receipt_path: Path,
) -> ModelRegistrationReceipt:
    if _IDENTIFIER.fullmatch(model_profile_id) is None:
        raise ModelRegistrationError("model_profile_id is invalid")
    if _DIGEST.fullmatch(expected_digest) is None:
        raise ModelRegistrationError("expected model digest is invalid")
    canonical, opened, descriptor = _open_model(path)
    try:
        observed_digest = _stream_digest(descriptor)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        opened.st_size,
        opened.st_mtime_ns,
        opened.st_ctime_ns,
    ) != (
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise ModelRegistrationError("model changed during registration")
    if observed_digest != expected_digest:
        raise ModelRegistrationError("model digest mismatch")
    receipt = ModelRegistrationReceipt(
        model_profile_id=model_profile_id,
        canonical_path=canonical,
        gguf_digest=observed_digest,
        size_bytes=int(opened.st_size),
        device=int(opened.st_dev),
        inode=int(opened.st_ino),
        mtime_ns=int(opened.st_mtime_ns),
    )
    _publish_noreplace(receipt_path, _canonical_bytes(receipt.to_mapping()))
    return receipt


def open_registered_model(
    *,
    path: Path,
    receipt: ModelRegistrationReceipt,
) -> RegisteredModelHandle:
    canonical, opened, descriptor = _open_model(path)
    if (
        canonical != receipt.canonical_path
        or opened.st_size != receipt.size_bytes
        or opened.st_dev != receipt.device
        or opened.st_ino != receipt.inode
        or opened.st_mtime_ns != receipt.mtime_ns
    ):
        os.close(descriptor)
        raise ModelRegistrationError("MODEL_REGISTRATION_STALE")
    return RegisteredModelHandle(
        path=canonical,
        digest=receipt.gguf_digest,
        size_bytes=receipt.size_bytes,
        device=receipt.device,
        inode=receipt.inode,
        mtime_ns=receipt.mtime_ns,
        descriptor=descriptor,
    )
