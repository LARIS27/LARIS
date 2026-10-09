from __future__ import annotations

from argparse import Namespace
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import TextIO

from laris_onpremise.core.records import (
    IdentityKeyNotFoundError,
    canonical_json,
)
from laris_onpremise.rehosting.catalog import PortableBundleCatalog
from laris_onpremise.rehosting.host_profile import (
    InvestigatorProfile,
    InvestigatorProfileError,
)
from laris_onpremise.rehosting.runtime_catalog import resolve_model


class ProfileCommandError(ValueError):
    """Raised when a local investigator profile cannot be generated safely."""


_CONSTRUCTION_ARGUMENTS = (
    "tenant_id",
    "identity_key",
    "model_root",
    "analysis_root",
    "wsl_distribution",
    "wsl_python",
    "wsl_runtime_profile",
    "wsl_helper",
    "wsl_helper_digest",
    "wsl_libllama",
    "wsl_work_root",
)


def default_profile_path() -> Path:
    base = os.environ.get("APPDATA")
    if base:
        return Path(base) / "LARIS" / "rehost-profile.json"
    return Path.home() / ".config" / "laris" / "rehost-profile.json"


def default_analysis_root() -> Path:
    base = os.environ.get("APPDATA")
    if base:
        return Path(base) / "LARIS" / "analyses"
    return Path.home() / ".local" / "share" / "laris" / "analyses"


def required_identity_key_id(catalog: PortableBundleCatalog) -> str:
    identity_key_ids = {entry.identity_key_id for entry in catalog.entries}
    if len(identity_key_ids) != 1:
        raise ProfileCommandError("bundle must require a single identity key")
    return next(iter(identity_key_ids))


def create_main(
    arguments: Namespace,
    *,
    output_stream: TextIO | None = None,
) -> int:
    if output_stream is None:
        output_stream = sys.stdout
    catalog = PortableBundleCatalog.open(_absolute(Path(arguments.bundle_root)))
    target = _absolute(Path(arguments.profile or default_profile_path()))
    construction_requested = any(
        getattr(arguments, name, None) not in (None, [])
        for name in _CONSTRUCTION_ARGUMENTS
    )
    if not construction_requested:
        if not _entry_exists(target):
            raise ProfileCommandError(
                "profile components are required when the target profile does not exist"
            )
        profile = InvestigatorProfile.load(target)
        _validate_bundle(profile, catalog)
        return _report(
            "PROFILE_UNCHANGED",
            target,
            catalog,
            as_json=arguments.json,
            output_stream=output_stream,
        )

    document = _profile_document(arguments)
    if getattr(arguments, "preserve_existing_identity_keys", False) and _entry_exists(
        target
    ):
        document = _with_existing_identity_keys(target, document)
    raw = canonical_json(document) + b"\n"
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    _make_private_directory(parent)
    target_existed = _entry_exists(target)

    if target_existed:
        try:
            existing, existing_raw = InvestigatorProfile.load_with_bytes(target)
        except (InvestigatorProfileError, OSError):
            if not arguments.replace:
                raise
        else:
            if existing_raw == raw:
                _validate_bundle(existing, catalog)
                return _report(
                    "PROFILE_UNCHANGED",
                    target,
                    catalog,
                    as_json=arguments.json,
                    output_stream=output_stream,
                )
            if not arguments.replace:
                raise ProfileCommandError(
                    "profile differs; pass --replace to install the generated profile"
                )

    temporary = _write_temporary(parent, target.name, raw)
    try:
        candidate = InvestigatorProfile.load(temporary)
        _validate_bundle(candidate, catalog)
        if arguments.replace:
            os.replace(temporary, target)
        else:
            try:
                os.link(temporary, target)
            except FileExistsError as error:
                raise ProfileCommandError("profile appeared during publication") from error
            temporary.unlink()
        _sync_directory(parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass

    installed = InvestigatorProfile.load(target)
    _validate_bundle(installed, catalog)
    return _report(
        "PROFILE_REPLACED"
        if arguments.replace and target_existed
        else "PROFILE_CREATED",
        target,
        catalog,
        as_json=arguments.json,
        output_stream=output_stream,
    )


def add_identity_main(
    arguments: Namespace,
    *,
    output_stream: TextIO | None = None,
) -> int:
    """Append one identity-key epoch without coupling the update to a bundle."""
    if output_stream is None:
        output_stream = sys.stdout
    target = _absolute(Path(arguments.profile or default_profile_path()))
    if not _entry_exists(target):
        raise ProfileCommandError(
            "investigator profile is unavailable; install the full profile first"
        )
    _, existing_raw = InvestigatorProfile.load_with_bytes(target)
    document = json.loads(existing_raw.decode("utf-8"))
    candidate_key = _identity_key(arguments.identity_key)
    existing_by_id = {
        item["identity_key_id"]: item for item in document["identity_keys"]
    }
    identity_key_id = candidate_key["identity_key_id"]
    existing_key = existing_by_id.get(identity_key_id)
    if existing_key is not None:
        if existing_key != candidate_key:
            raise ProfileCommandError(
                "identity key ID already refers to a different secret file"
            )
        return _report_identity(
            "PROFILE_IDENTITY_UNCHANGED",
            target,
            identity_key_id,
            as_json=arguments.json,
            output_stream=output_stream,
        )

    existing_by_id[identity_key_id] = candidate_key
    document["identity_keys"] = [
        existing_by_id[key] for key in sorted(existing_by_id)
    ]
    raw = canonical_json(document) + b"\n"
    temporary = _write_temporary(target.parent, target.name, raw)
    try:
        InvestigatorProfile.load(temporary)
        os.replace(temporary, target)
        _sync_directory(target.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    InvestigatorProfile.load(target)
    return _report_identity(
        "PROFILE_IDENTITY_ADDED",
        target,
        identity_key_id,
        as_json=arguments.json,
        output_stream=output_stream,
    )


def _profile_document(arguments: Namespace) -> dict[str, object]:
    required = {
        "tenant_id": arguments.tenant_id,
        "identity_key": arguments.identity_key,
        "model_root": arguments.model_root,
        "wsl_distribution": arguments.wsl_distribution,
        "wsl_python": arguments.wsl_python,
        "wsl_runtime_profile": arguments.wsl_runtime_profile,
        "wsl_helper": arguments.wsl_helper,
        "wsl_helper_digest": arguments.wsl_helper_digest,
        "wsl_libllama": arguments.wsl_libllama,
        "wsl_work_root": arguments.wsl_work_root,
    }
    missing = [name.replace("_", "-") for name, value in required.items() if not value]
    if missing:
        raise ProfileCommandError(
            "profile component options are incomplete: " + ", ".join(sorted(missing))
        )
    identity_keys = [_identity_key(value) for value in arguments.identity_key]
    model_roots = [str(_absolute(Path(value))) for value in arguments.model_root]
    analysis_root = _absolute(
        Path(arguments.analysis_root or default_analysis_root())
    )
    return {
        "analysis_root": str(analysis_root),
        "identity_keys": identity_keys,
        "model_roots": model_roots,
        "schema_version": "1.0",
        "tenant_id": arguments.tenant_id,
        "wsl": {
            "distribution": arguments.wsl_distribution,
            "helper": arguments.wsl_helper,
            "helper_digest": arguments.wsl_helper_digest,
            "libllama": arguments.wsl_libllama,
            "python": arguments.wsl_python,
            "runtime_profile": arguments.wsl_runtime_profile,
            "work_root": arguments.wsl_work_root,
        },
    }


def _identity_key(value: str) -> dict[str, str]:
    identity_key_id, separator, raw_path = value.partition("=")
    if not separator or not identity_key_id or not raw_path:
        raise ProfileCommandError("--identity-key must use ID=PATH")
    return {
        "identity_key_id": identity_key_id,
        "secret_file": str(_absolute(Path(raw_path))),
    }


def _with_existing_identity_keys(
    target: Path,
    document: dict[str, object],
) -> dict[str, object]:
    _, existing_raw = InvestigatorProfile.load_with_bytes(target)
    existing_document = json.loads(existing_raw.decode("utf-8"))
    merged: dict[str, dict[str, str]] = {
        item["identity_key_id"]: item
        for item in existing_document["identity_keys"]
    }
    for item in document["identity_keys"]:
        merged[item["identity_key_id"]] = item
    return {
        **document,
        "identity_keys": [merged[key] for key in sorted(merged)],
    }


def _validate_bundle(
    profile: InvestigatorProfile,
    catalog: PortableBundleCatalog,
) -> None:
    for identity_key_id in sorted({entry.identity_key_id for entry in catalog.entries}):
        try:
            profile.identity_registry.resolve(identity_key_id)
        except IdentityKeyNotFoundError as error:
            raise ProfileCommandError(
                f"required identity key is unavailable: {identity_key_id}"
            ) from error
    entries_by_digest = {
        entry.model_digest: entry
        for entry in catalog.entries
    }
    for entry in entries_by_digest.values():
        resolve_model(profile, entry)


def _write_temporary(parent: Path, name: str, raw: bytes) -> Path:
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{name}.pending-",
        dir=parent,
    )
    path = Path(raw_path)
    try:
        try:
            os.fchmod(descriptor, 0o600)
        except (AttributeError, OSError):
            pass
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                raise ProfileCommandError("profile temporary write failed")
            offset += written
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise
    os.close(descriptor)
    return path


def _entry_exists(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _make_private_directory(path: Path) -> None:
    try:
        path.chmod(0o700)
    except OSError:
        pass


def _sync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _report(
    status: str,
    target: Path,
    catalog: PortableBundleCatalog,
    *,
    as_json: bool,
    output_stream: TextIO,
) -> int:
    result = {
        "bundle_run_id": catalog.source_run_id,
        "profile": str(target),
        "status": status,
    }
    if as_json:
        output_stream.write(
            json.dumps(result, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
            + "\n"
        )
    else:
        output_stream.write(
            f"{status} profile={target} bundle_run_id={catalog.source_run_id}\n"
        )
    output_stream.flush()
    return 0


def _report_identity(
    status: str,
    target: Path,
    identity_key_id: str,
    *,
    as_json: bool,
    output_stream: TextIO,
) -> int:
    result = {
        "identity_key_id": identity_key_id,
        "profile": str(target),
        "status": status,
    }
    if as_json:
        output_stream.write(
            json.dumps(result, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
            + "\n"
        )
    else:
        output_stream.write(
            f"{status} profile={target} identity_key_id={identity_key_id}\n"
        )
    output_stream.flush()
    return 0


__all__ = [
    "ProfileCommandError",
    "add_identity_main",
    "create_main",
    "default_analysis_root",
    "default_profile_path",
]
