from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import re
import stat

from laris_onpremise.core.records import (
    IdentityKey,
    IdentityKeyRegistry,
    RecordValidationError,
    canonical_json,
)


class InvestigatorProfileError(ValueError):
    """Raised when the local investigator profile is unsafe or incomplete."""


_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_PROFILE_FIELDS = frozenset(
    {
        "analysis_root",
        "identity_keys",
        "model_roots",
        "schema_version",
        "tenant_id",
        "wsl",
    }
)
_WSL_FIELDS = frozenset(
    {
        "distribution",
        "helper",
        "helper_digest",
        "libllama",
        "python",
        "runtime_profile",
        "work_root",
    }
)
_KEY_FIELDS = frozenset({"identity_key_id", "secret_file"})
_MAX_PROFILE_BYTES = 1024 * 1024


@dataclass(frozen=True)
class WslProfile:
    distribution: str
    python: str
    runtime_profile: str
    helper: str
    helper_digest: str
    libllama: str
    work_root: str


@dataclass(frozen=True)
class InvestigatorProfile:
    tenant_id: str
    identity_registry: IdentityKeyRegistry = field(repr=False)
    model_roots: tuple[Path, ...]
    analysis_root: Path
    wsl: WslProfile

    @classmethod
    def load(cls, path: Path) -> "InvestigatorProfile":
        profile, _ = cls.load_with_bytes(path)
        return profile

    @classmethod
    def load_with_bytes(cls, path: Path) -> tuple["InvestigatorProfile", bytes]:
        document, raw = _canonical_profile(path)
        if set(document) != _PROFILE_FIELDS or document.get("schema_version") != "1.0":
            raise InvestigatorProfileError("investigator profile field set is invalid")
        tenant_id = _safe_id("tenant_id", document["tenant_id"])
        analysis_root = _absolute_path(
            "analysis_root",
            document["analysis_root"],
            require_directory=False,
        )
        model_roots = _model_roots(document["model_roots"])
        registry = _identity_registry(document["identity_keys"])
        wsl = _wsl_profile(document["wsl"])
        return cls(tenant_id, registry, model_roots, analysis_root, wsl), raw


def _canonical_profile(path: Path) -> tuple[dict[str, object], bytes]:
    if not isinstance(path, Path) or not path.is_absolute():
        raise InvestigatorProfileError("profile path must be absolute")
    raw = _read_regular(path, limit=_MAX_PROFILE_BYTES)

    def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise InvestigatorProfileError("investigator profile has duplicate fields")
            result[key] = value
        return result

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=unique)
    except InvestigatorProfileError:
        raise
    except (UnicodeError, json.JSONDecodeError) as error:
        raise InvestigatorProfileError("investigator profile is invalid") from error
    if not isinstance(value, dict) or raw != canonical_json(value) + b"\n":
        raise InvestigatorProfileError("investigator profile is not canonical")
    return value, raw


def _identity_registry(value: object) -> IdentityKeyRegistry:
    if not isinstance(value, list) or not value:
        raise InvestigatorProfileError("identity key registry is invalid")
    keys: list[IdentityKey] = []
    seen: set[str] = set()
    for raw in value:
        if not isinstance(raw, dict) or set(raw) != _KEY_FIELDS:
            raise InvestigatorProfileError("identity key entry is invalid")
        identity_key_id = _safe_id("identity_key_id", raw["identity_key_id"])
        if identity_key_id in seen:
            raise InvestigatorProfileError("identity key ID is duplicated")
        seen.add(identity_key_id)
        secret_path = _absolute_path(
            "identity key path",
            raw["secret_file"],
            require_directory=False,
        )
        secret = _read_regular(secret_path, limit=33)
        if len(secret) != 32:
            raise InvestigatorProfileError("identity key must contain exactly 32 bytes")
        keys.append(IdentityKey(identity_key_id, secret))
    try:
        return IdentityKeyRegistry(keys)
    except RecordValidationError as error:
        raise InvestigatorProfileError("identity key registry is invalid") from error


def _model_roots(value: object) -> tuple[Path, ...]:
    if not isinstance(value, list) or not value:
        raise InvestigatorProfileError("model_roots is invalid")
    roots = tuple(
        _absolute_path(f"model_roots[{index}]", raw, require_directory=True)
        for index, raw in enumerate(value)
    )
    if len(set(roots)) != len(roots):
        raise InvestigatorProfileError("model root is duplicated")
    return roots


def _wsl_profile(value: object) -> WslProfile:
    if not isinstance(value, dict) or set(value) != _WSL_FIELDS:
        raise InvestigatorProfileError("WSL profile field set is invalid")
    distribution = _safe_id("WSL distribution", value["distribution"])
    helper_digest = value["helper_digest"]
    if not isinstance(helper_digest, str) or _SHA256.fullmatch(helper_digest) is None:
        raise InvestigatorProfileError("WSL helper digest is invalid")
    paths = {
        name: _posix_absolute(name, value[name])
        for name in ("python", "runtime_profile", "helper", "libllama", "work_root")
    }
    if paths["work_root"] == "/":
        raise InvestigatorProfileError("WSL work_root is invalid")
    return WslProfile(
        distribution=distribution,
        python=paths["python"],
        runtime_profile=paths["runtime_profile"],
        helper=paths["helper"],
        helper_digest=helper_digest,
        libllama=paths["libllama"],
        work_root=paths["work_root"],
    )


def _absolute_path(name: str, value: object, *, require_directory: bool) -> Path:
    if not isinstance(value, str):
        raise InvestigatorProfileError(f"{name} is invalid")
    path = Path(value)
    if not path.is_absolute() or Path(os.path.abspath(value)) != path:
        raise InvestigatorProfileError(f"{name} must be a canonical absolute path")
    if require_directory:
        _regular_directory(path, name)
    return path


def _regular_directory(path: Path, name: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise InvestigatorProfileError(f"{name} is unavailable") from error
    is_reparse = bool(
        getattr(metadata, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )
    if stat.S_ISLNK(metadata.st_mode) or is_reparse or not stat.S_ISDIR(metadata.st_mode):
        raise InvestigatorProfileError(f"{name} is not a regular directory")


def _read_regular(path: Path, *, limit: int) -> bytes:
    try:
        before = path.lstat()
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise InvestigatorProfileError("profile component is not a regular file")
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
    except InvestigatorProfileError:
        raise
    except OSError as error:
        raise InvestigatorProfileError("profile component is unavailable") from error
    try:
        opened = os.fstat(descriptor)
        if opened.st_dev != before.st_dev or opened.st_ino != before.st_ino:
            raise InvestigatorProfileError("profile component identity changed")
        chunks: list[bytes] = []
        size = 0
        while True:
            chunk = os.read(descriptor, 64 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > limit:
                raise InvestigatorProfileError("profile component exceeds size limit")
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    if opened.st_size != after.st_size or opened.st_mtime_ns != after.st_mtime_ns or len(raw) != after.st_size:
        raise InvestigatorProfileError("profile component changed while reading")
    return raw


def _safe_id(name: str, value: object) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise InvestigatorProfileError(f"{name} is invalid")
    return value


def _posix_absolute(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith("/")
        or "\\" in value
        or any(ord(character) < 0x20 for character in value)
        or posixpath.normpath(value) != value
        or PurePosixPath(value).is_absolute() is False
    ):
        raise InvestigatorProfileError(f"WSL {name} path is invalid")
    return value
