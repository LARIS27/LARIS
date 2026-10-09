from __future__ import annotations

from dataclasses import asdict
import errno
import json
import os
from pathlib import Path
import stat
from typing import Final

from laris_onpremise.bundle.rehost_export import (
    ModelCompatibility,
    RehostExportError,
)
from laris_onpremise.kv_state.observation.runtime_state_profile import (
    load_runtime_state_observer_profile,
)


MODEL_COMPATIBILITY_FILENAME: Final[str] = (
    "runtime_state_model_compatibility.json"
)
_MODEL_COMPATIBILITY_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "run_id",
        "model_profile_id",
        "model_digest",
        "runtime_profile_digest",
        "memory_family",
        "n_ctx",
        "n_seq_max",
        "n_stream",
        "n_layer",
        "v_trans",
        "key_ggml_type_id",
        "value_ggml_type_id",
    }
)
_MAXIMUM_COMPATIBILITY_BYTES: Final[int] = 64 * 1024


class AcquisitionEvidenceRootError(ValueError):
    """Raised when the selective-acquisition root is not safely staged."""


def _is_link_like(metadata: os.stat_result) -> bool:
    if stat.S_ISLNK(metadata.st_mode):
        return True
    attributes = getattr(metadata, "st_file_attributes", 0)
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(reparse and attributes & reparse)


def _directory_identity(root: Path, *, require_empty: bool) -> tuple[int, int]:
    try:
        metadata = root.lstat()
    except OSError as error:
        raise AcquisitionEvidenceRootError(
            "acquisition evidence root is unavailable"
        ) from error
    if _is_link_like(metadata) or not stat.S_ISDIR(metadata.st_mode):
        raise AcquisitionEvidenceRootError("acquisition evidence root is unsafe")
    if os.name == "posix" and stat.S_IMODE(metadata.st_mode) != 0o700:
        raise AcquisitionEvidenceRootError(
            "acquisition evidence root mode is not 0700"
        )
    try:
        entries = tuple(root.iterdir())
    except OSError as error:
        raise AcquisitionEvidenceRootError(
            "acquisition evidence root cannot be inspected"
        ) from error
    if require_empty and entries:
        raise AcquisitionEvidenceRootError(
            "acquisition evidence root is not empty"
        )
    return metadata.st_dev, metadata.st_ino


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(
            path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
    except OSError as error:
        if os.name == "nt" and error.errno in {errno.EACCES, errno.EPERM}:
            return
        raise
    try:
        try:
            os.fsync(descriptor)
        except OSError as error:
            unsupported = {
                errno.EINVAL,
                getattr(errno, "ENOTSUP", errno.EINVAL),
                getattr(errno, "EOPNOTSUPP", errno.EINVAL),
            }
            if error.errno not in unsupported:
                raise
    finally:
        os.close(descriptor)


def prepare_acquisition_evidence_root(root: Path) -> None:
    """Create the one run-owned, initially empty selective evidence root."""

    if not isinstance(root, Path) or not root.is_absolute():
        raise AcquisitionEvidenceRootError(
            "acquisition evidence root must be an absolute pathlib.Path"
        )
    try:
        root.mkdir(mode=0o700, parents=False, exist_ok=False)
        if os.name == "posix":
            os.chmod(root, 0o700, follow_symlinks=False)
    except OSError as error:
        raise AcquisitionEvidenceRootError(
            "acquisition evidence root cannot be created exclusively"
        ) from error
    _directory_identity(root, require_empty=True)
    _fsync_directory(root.parent)


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")


def publish_runtime_state_model_compatibility(
    path: Path,
    compatibility: ModelCompatibility,
) -> None:
    """Publish the sole producer-owned compatibility file with O_EXCL."""

    if path.name != MODEL_COMPATIBILITY_FILENAME:
        raise AcquisitionEvidenceRootError(
            "model compatibility filename is not canonical"
        )
    root = path.parent
    identity = _directory_identity(root, require_empty=True)
    payload = _canonical_bytes(asdict(compatibility))
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= (
        getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_BINARY", 0)
    )
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as error:
        raise AcquisitionEvidenceRootError(
            "model compatibility evidence cannot be published exclusively"
        ) from error
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise AcquisitionEvidenceRootError(
                    "model compatibility publication made no progress"
                )
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if _directory_identity(root, require_empty=False) != identity:
        raise AcquisitionEvidenceRootError(
            "acquisition evidence root identity changed during publication"
        )
    if {entry.name for entry in root.iterdir()} != {
        MODEL_COMPATIBILITY_FILENAME
    }:
        raise AcquisitionEvidenceRootError(
            "acquisition evidence root contains foreign entries"
        )
    _fsync_directory(root)


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise AcquisitionEvidenceRootError(
                "duplicate model compatibility field"
            )
        value[key] = item
    return value


def _read_compatibility_file(path: Path) -> bytes:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise AcquisitionEvidenceRootError(
            "model compatibility evidence is unavailable"
        ) from error
    if _is_link_like(metadata) or not stat.S_ISREG(metadata.st_mode):
        raise AcquisitionEvidenceRootError("model compatibility evidence is unsafe")
    if os.name == "posix" and stat.S_IMODE(metadata.st_mode) != 0o600:
        raise AcquisitionEvidenceRootError(
            "model compatibility evidence mode is not 0600"
        )
    if metadata.st_size <= 0 or metadata.st_size > _MAXIMUM_COMPATIBILITY_BYTES:
        raise AcquisitionEvidenceRootError(
            "model compatibility evidence size is invalid"
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise AcquisitionEvidenceRootError(
            "model compatibility evidence cannot be opened safely"
        ) from error
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_dev != metadata.st_dev
            or (opened.st_ino and metadata.st_ino and opened.st_ino != metadata.st_ino)
        ):
            raise AcquisitionEvidenceRootError(
                "model compatibility evidence identity changed"
            )
        payload = os.read(descriptor, _MAXIMUM_COMPATIBILITY_BYTES + 1)
        if os.read(descriptor, 1):
            raise AcquisitionEvidenceRootError(
                "model compatibility evidence exceeds size limit"
            )
    finally:
        os.close(descriptor)
    return payload


def validate_prepared_acquisition_evidence_root(
    root: Path,
    *,
    runtime_state_profile: Path,
) -> ModelCompatibility:
    """Validate the exact producer handoff accepted by the Proxy."""

    identity = _directory_identity(root, require_empty=False)
    try:
        entries = tuple(root.iterdir())
    except OSError as error:
        raise AcquisitionEvidenceRootError(
            "acquisition evidence root cannot be inspected"
        ) from error
    if {path.name for path in entries} != {MODEL_COMPATIBILITY_FILENAME}:
        raise AcquisitionEvidenceRootError(
            "acquisition evidence root contains foreign entries"
        )
    payload = _read_compatibility_file(root / MODEL_COMPATIBILITY_FILENAME)
    try:
        if not payload.endswith(b"\n"):
            raise AcquisitionEvidenceRootError(
                "model compatibility evidence framing is invalid"
            )
        document = json.loads(
            payload[:-1].decode("ascii"), object_pairs_hook=_strict_object
        )
        if (
            not isinstance(document, dict)
            or set(document) != _MODEL_COMPATIBILITY_FIELDS
            or _canonical_bytes(document) != payload
        ):
            raise AcquisitionEvidenceRootError(
                "model compatibility evidence is not canonical"
            )
        compatibility = ModelCompatibility(**document)
        profile = load_runtime_state_observer_profile(runtime_state_profile)
    except AcquisitionEvidenceRootError:
        raise
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
        RehostExportError,
    ) as error:
        raise AcquisitionEvidenceRootError(
            "model compatibility evidence is invalid"
        ) from error
    launch_arguments = profile.required_launch_arguments
    expected_n_ctx = int(
        launch_arguments[launch_arguments.index("--ctx-size") + 1]
    )
    common_mismatch = (
        compatibility.runtime_profile_digest != profile.profile_digest
        or compatibility.memory_family != profile.memory_family
        or compatibility.n_ctx != expected_n_ctx
        or compatibility.n_seq_max != profile.n_seq_max
        or compatibility.n_stream != profile.n_stream
        or compatibility.key_ggml_type_id != profile.key_ggml_type_id
        or compatibility.value_ggml_type_id != profile.value_ggml_type_id
    )
    if profile.has_dynamic_topology:
        topology_mismatch = (
            compatibility.n_layer <= 0
            or compatibility.n_layer > profile.max_n_layer
            or type(compatibility.v_trans) is not bool
        )
        model_mismatch = False
    else:
        topology_mismatch = (
            compatibility.n_layer != profile.n_layer
            or compatibility.v_trans != profile.v_trans
        )
        model_mismatch = (not profile.is_model_generic) and (
            compatibility.model_profile_id != profile.qualified_model_profile_id
            or compatibility.model_digest != profile.qualified_model_sha256
        )
    if common_mismatch or topology_mismatch or model_mismatch:
        raise AcquisitionEvidenceRootError(
            "model compatibility evidence does not match the qualified profile"
        )
    if _directory_identity(root, require_empty=False) != identity:
        raise AcquisitionEvidenceRootError(
            "acquisition evidence root identity changed during validation"
        )
    return compatibility


__all__ = [
    "AcquisitionEvidenceRootError",
    "MODEL_COMPATIBILITY_FILENAME",
    "prepare_acquisition_evidence_root",
    "publish_runtime_state_model_compatibility",
    "validate_prepared_acquisition_evidence_root",
]
