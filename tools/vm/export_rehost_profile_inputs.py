from __future__ import annotations

import argparse
import hashlib
import hmac
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tarfile


_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


class ExportError(ValueError):
    """Raised when a privileged rehost export cannot remain receipt-bound."""


def _json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )


def _canonical(path: Path, label: str) -> tuple[dict[str, object], bytes]:
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise ExportError(f"{label} is not a single-link regular file")
    if metadata.st_size < 2 or metadata.st_size > 1024 * 1024:
        raise ExportError(f"{label} size is invalid")
    raw = path.read_bytes()

    def no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ExportError(f"{label} contains a duplicate key")
            result[key] = value
        return result

    value = json.loads(raw.decode("ascii"), object_pairs_hook=no_duplicates)
    if not isinstance(value, dict) or _json(value) != raw:
        raise ExportError(f"{label} is not canonical")
    return value, raw


def _physical(root: Path, canonical: str) -> Path:
    return root.joinpath(*PurePosixPath(canonical).parts[1:])


def _safe_source(root: Path, path_text: str, label: str) -> Path:
    canonical_path = PurePosixPath(path_text)
    if (
        not canonical_path.is_absolute()
        or ".." in PurePosixPath(path_text).parts
        or not path_text.startswith("/opt/laris/")
    ):
        raise ExportError(f"{label} path is invalid")
    path = _physical(root, path_text)
    before = path.lstat()
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_uid != 0
    ):
        raise ExportError(f"{label} object is invalid")
    return path


class _DigestingReader(io.RawIOBase):
    def __init__(self, descriptor: int, digest: object) -> None:
        self._descriptor = descriptor
        self._digest = digest

    def readable(self) -> bool:
        return True

    def readinto(self, target: bytearray) -> int:
        chunk = os.read(self._descriptor, len(target))
        target[: len(chunk)] = chunk
        self._digest.update(chunk)  # type: ignore[attr-defined]
        return len(chunk)


def _add_file(
    archive: tarfile.TarFile,
    source: Path,
    destination: str,
    expected_digest: str,
    expected_size: int,
    *,
    executable: bool = False,
) -> dict[str, object]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(source, flags)
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid != 0
            or before.st_size != expected_size
        ):
            raise ExportError(f"export source changed: {source}")
        info = tarfile.TarInfo(destination)
        info.size = before.st_size
        info.mode = 0o700 if executable else 0o600
        info.uid = 0
        info.gid = 0
        info.mtime = 0
        digest = hashlib.sha256()
        reader = io.BufferedReader(_DigestingReader(descriptor, digest))
        archive.addfile(info, reader)
        after = os.fstat(descriptor)
        if (
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        ):
            raise ExportError(f"export source changed while reading: {source}")
        observed = "sha256:" + digest.hexdigest()
        if not hmac.compare_digest(observed, expected_digest):
            raise ExportError(f"export source digest differs: {source}")
    finally:
        os.close(descriptor)
    return {"path": destination, "sha256": expected_digest, "size_bytes": expected_size}


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _component_inventory(
    receipt: dict[str, object], requirements: dict[str, object]
) -> list[dict[str, object]]:
    receipt_components = receipt.get("rehost_components")
    requested_components = requirements.get("rehost_components")
    if not isinstance(receipt_components, list) or requested_components != receipt_components:
        raise ExportError("rehost component inventory differs")
    if len(receipt_components) < 3 or not all(
        isinstance(item, dict) for item in receipt_components
    ):
        raise ExportError("rehost component inventory is incomplete")
    return [dict(item) for item in receipt_components]


def export(arguments: argparse.Namespace) -> tuple[Path, Path]:
    filesystem_root = Path(getattr(arguments, "filesystem_root", Path("/")))
    if not filesystem_root.is_absolute():
        raise ExportError("filesystem root is invalid")
    receipt, receipt_raw = _canonical(arguments.installation_receipt, "installation receipt")
    requirements, _ = _canonical(arguments.bundle_requirements, "bundle requirements")
    receipt_digest = "sha256:" + hashlib.sha256(receipt_raw).hexdigest()
    if (
        receipt.get("schema_version") != "1.0"
        or receipt.get("phase") != "COMPLETE"
        or requirements.get("installation_receipt_sha256") != receipt_digest
    ):
        raise ExportError("installation receipt binding differs")
    if (
        _IDENTIFIER.fullmatch(arguments.identity_key_id) is None
        or requirements.get("identity_key_id") != arguments.identity_key_id
        or receipt.get("identity_key_id") != arguments.identity_key_id
    ):
        raise ExportError("identity key binding differs")

    identity_only = bool(getattr(arguments, "identity_only", False))
    key_path = _physical(filesystem_root, "/etc/laris-lab/identity/identity.key")
    key_metadata = key_path.lstat()
    if (
        not stat.S_ISREG(key_metadata.st_mode)
        or key_metadata.st_nlink != 1
        or key_metadata.st_uid != 0
        or key_metadata.st_size != 32
    ):
        raise ExportError("identity key object is invalid")
    key_fingerprint = receipt.get("identity_key_fingerprint")
    if not isinstance(key_fingerprint, str) or _DIGEST.fullmatch(key_fingerprint) is None:
        raise ExportError("identity key fingerprint is invalid")

    model_name: str | None = None
    model_digest: str | None = None
    model_size: int | None = None
    components: list[dict[str, object]] = []
    if not identity_only:
        models = receipt.get("models")
        if not isinstance(models, list):
            raise ExportError("installed model inventory is invalid")
        model_matches = [
            item
            for item in models
            if isinstance(item, dict)
            and item.get("model_profile_id") == requirements.get("model_profile_id")
            and item.get("gguf_digest") == requirements.get("model_digest")
            and item.get("filename") == requirements.get("model_filename")
        ]
        if len(model_matches) != 1:
            raise ExportError("installed model binding differs")
        model = model_matches[0]
        raw_model_name = model.get("filename")
        raw_model_digest = model.get("gguf_digest")
        raw_model_size = model.get("size_bytes")
        if (
            not isinstance(raw_model_name, str)
            or PurePosixPath(raw_model_name).name != raw_model_name
            or not raw_model_name.endswith(".gguf")
            or not isinstance(raw_model_digest, str)
            or _DIGEST.fullmatch(raw_model_digest) is None
            or isinstance(raw_model_size, bool)
            or not isinstance(raw_model_size, int)
            or raw_model_size < 1
        ):
            raise ExportError("installed model record is invalid")
        model_name = raw_model_name
        model_digest = raw_model_digest
        model_size = raw_model_size
        components = _component_inventory(receipt, requirements)
    export_path = _physical(
        filesystem_root, f"/tmp/laris-rehost-export-{arguments.export_id}.tar"
    )
    digest_path = export_path.with_suffix(".tar.sha256")
    pending = export_path.with_suffix(".tar.pending")
    for path in (export_path, digest_path, pending):
        if path.exists() or path.is_symlink():
            raise ExportError("export destination already exists")

    entries: list[dict[str, object]] = []
    try:
        with tarfile.open(pending, "x", format=tarfile.USTAR_FORMAT) as archive:
            entries.append(_add_file(archive, key_path, "identity.key", key_fingerprint, 32))
            if not identity_only:
                assert model_name is not None
                assert model_digest is not None
                assert model_size is not None
                entries.append(
                    _add_file(
                        archive,
                        _safe_source(
                            filesystem_root,
                            f"/opt/laris/models/{model_name}",
                            "model",
                        ),
                        f"model/{model_name}",
                        model_digest,
                        model_size,
                    )
                )
            library_names: set[str] = set()
            for component in components:
                role = component.get("role")
                source_text = component.get("canonical_path")
                expected_digest = component.get("sha256")
                expected_size = component.get("size_bytes")
                if (
                    role not in {"helper", "runtime_profile", "shared_library"}
                    or not isinstance(source_text, str)
                    or not isinstance(expected_digest, str)
                    or _DIGEST.fullmatch(expected_digest) is None
                    or isinstance(expected_size, bool)
                    or not isinstance(expected_size, int)
                    or expected_size < 1
                ):
                    raise ExportError("rehost component record is invalid")
                basename = PurePosixPath(source_text).name
                if role == "helper":
                    destination = "native/llama-runtime-payload-rehost"
                elif role == "runtime_profile":
                    destination = "runtime/runtime-profile.json"
                else:
                    if basename in library_names:
                        raise ExportError("rehost library name is ambiguous")
                    library_names.add(basename)
                    destination = f"lib/{basename}"
                entry = _add_file(
                    archive,
                    _safe_source(
                        filesystem_root, source_text, f"{role} component"
                    ),
                    destination,
                    expected_digest,
                    expected_size,
                    executable=role == "helper",
                )
                entry["role"] = role
                entries.append(entry)
            manifest = {
                "entries": sorted(entries, key=lambda item: str(item["path"]).encode("ascii")),
                "identity_key_id": arguments.identity_key_id,
                "installation_receipt_sha256": receipt_digest,
                "schema_version": "1.0",
            }
            if identity_only:
                manifest["export_type"] = "identity_only"
            else:
                manifest["model_profile_id"] = requirements["model_profile_id"]
            raw_manifest = _json(manifest)
            info = tarfile.TarInfo("export-manifest.json")
            info.size = len(raw_manifest)
            info.mode = 0o600
            info.uid = 0
            info.gid = 0
            info.mtime = 0
            archive.addfile(info, io.BytesIO(raw_manifest))
        os.chmod(pending, 0o600)
        os.replace(pending, export_path)
        raw_digest = _hash_file(export_path)
        digest_path.write_text(f"{raw_digest}  {export_path.name}\n", encoding="ascii")
        os.chmod(digest_path, 0o600)
        os.chown(export_path, arguments.caller_uid, arguments.caller_gid)
        os.chown(digest_path, arguments.caller_uid, arguments.caller_gid)
    except BaseException:
        for path in (pending, export_path, digest_path):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise
    return export_path, digest_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("export_id")
    parser.add_argument("identity_key_id")
    parser.add_argument("installation_receipt", type=Path)
    parser.add_argument("bundle_requirements", type=Path)
    parser.add_argument("caller_uid", type=int)
    parser.add_argument("caller_gid", type=int)
    parser.add_argument("--identity-only", action="store_true")
    arguments = parser.parse_args()
    if os.getuid() != 0:
        raise SystemExit("root privileges are required")
    if re.fullmatch(r"[0-9a-f]{32}", arguments.export_id) is None:
        raise SystemExit("export identifier is invalid")
    if arguments.caller_uid < 1 or arguments.caller_gid < 1:
        raise SystemExit("sudo caller identity is invalid")
    try:
        archive, digest = export(arguments)
    except Exception as error:
        raise SystemExit(f"rehost profile export failed: {error}") from error
    print(f"REHOST_PROFILE_EXPORT_READY archive={archive} digest={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
