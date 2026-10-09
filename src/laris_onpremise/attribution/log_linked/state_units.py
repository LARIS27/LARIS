from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Mapping

from .contracts import (
    SequenceGenerationRef,
    canonical_document_bytes,
    parse_state_unit_manifest,
)
from .generation import GenerationSeal


class StateUnitStoreError(RuntimeError):
    """Raised when an append-only state unit cannot be published safely."""


_UNIT_DIRECTORY = re.compile(r"unit_([0-9]{8})\Z")
_SAFE_RUNTIME_INSTANCE_ID = re.compile(
    r"(?:[A-Za-z0-9][A-Za-z0-9._-]{0,191}|sha256:[0-9a-f]{64})\Z"
)


@dataclass(frozen=True, init=False)
class StateUnitManifest:
    _payload: bytes

    def __init__(self, document: Mapping[str, object]) -> None:
        try:
            payload = canonical_document_bytes(document)
            parse_state_unit_manifest(payload)
        except ValueError as error:
            raise StateUnitStoreError(f"state-unit manifest is invalid: {error}") from error
        object.__setattr__(self, "_payload", payload)

    def to_mapping(self) -> dict[str, object]:
        return json.loads(self._payload.decode("utf-8"))

    def to_bytes(self) -> bytes:
        return self._payload

    @property
    def unit_id(self) -> str:
        return str(self.to_mapping()["unit_id"])

    @property
    def runtime_instance_id(self) -> str:
        return str(self.to_mapping()["runtime_instance_id"])

    @property
    def seq_id(self) -> int:
        return int(self.to_mapping()["seq_id"])

    @property
    def sequence_generation(self) -> int:
        return int(self.to_mapping()["sequence_generation"])


@dataclass(frozen=True)
class StoredStateUnit:
    root: Path
    payload_path: Path
    manifest_path: Path
    unit_number: int
    ref: SequenceGenerationRef
    unit_id: str
    payload_digest: str
    manifest_digest: str


class StateUnitStore:
    """Private, manifest-last store for independent sequence generations."""

    def __init__(self, evidence_root: Path | str) -> None:
        self.evidence_root = Path(evidence_root)
        _ensure_private_directory(self.evidence_root)
        self.state_units_root = self.evidence_root / "state-units"
        _ensure_private_directory(self.state_units_root)
        self._root_identity = _directory_identity(self.evidence_root)
        self._state_units_identity = _directory_identity(self.state_units_root)

    def commit(
        self, *, manifest: StateUnitManifest, payload: bytes
    ) -> StoredStateUnit:
        self._verify_root_identity()
        if not isinstance(manifest, StateUnitManifest):
            raise StateUnitStoreError("manifest must be a StateUnitManifest")
        if not isinstance(payload, bytes):
            raise StateUnitStoreError("payload must be bytes")
        document = manifest.to_mapping()
        size = len(payload)
        digest = "sha256:" + hashlib.sha256(payload).hexdigest()
        if (
            document["payload_file"] != "runtime_state_payload.bin"
            or document["payload_size_bytes"] != size
            or not hmac.compare_digest(str(document["payload_digest"]), digest)
        ):
            raise StateUnitStoreError("manifest payload claim does not match payload")

        generation_root = self._generation_root(
            manifest.runtime_instance_id,
            manifest.seq_id,
            manifest.sequence_generation,
        )
        _ensure_private_directory(generation_root)
        unit_roots = self._verified_unit_roots(generation_root)
        if (generation_root / "generation_seal.json").exists():
            raise StateUnitStoreError("generation is already sealed")
        for unit_root in unit_roots:
            existing = StateUnitManifest(
                parse_state_unit_manifest(
                    _read_regular_file(unit_root / "state_unit_manifest.json")
                )
            )
            if existing.unit_id == manifest.unit_id:
                raise StateUnitStoreError("state unit is already committed")
        self._verify_append_chain(unit_roots, document)

        unit_number = len(unit_roots) + 1
        final_root = generation_root / f"unit_{unit_number:08d}"
        staging_root = Path(tempfile.mkdtemp(prefix=".tmp-unit-", dir=generation_root))
        _make_private(staging_root, directory=True)
        final_created = False
        try:
            staging_payload = staging_root / "runtime_state_payload.bin"
            staging_manifest = staging_root / "state_unit_manifest.json"
            _write_new_file(staging_payload, payload)
            _write_new_file(staging_manifest, manifest.to_bytes())
            _fsync_directory(staging_root)
            try:
                os.mkdir(final_root, mode=0o700)
                final_created = True
                _make_private(final_root, directory=True)
                final_payload = final_root / "runtime_state_payload.bin"
                final_manifest = final_root / "state_unit_manifest.json"
                os.link(staging_payload, final_payload, follow_symlinks=False)
                _fsync_directory(final_root)
                os.link(staging_manifest, final_manifest, follow_symlinks=False)
            except OSError as error:
                raise StateUnitStoreError("state-unit publication failed") from error
            _verify_regular_file(final_payload)
            _verify_regular_file(final_manifest)
            if _read_regular_file(final_payload) != payload:
                raise StateUnitStoreError("published state-unit payload differs")
            if _read_regular_file(final_manifest) != manifest.to_bytes():
                raise StateUnitStoreError("published state-unit manifest differs")
            _fsync_directory(final_root)
            _fsync_directory(generation_root)
        except Exception:
            if final_created:
                _remove_unit_root(final_root)
            raise
        finally:
            _remove_staging_root(staging_root)
            _fsync_directory(generation_root)

        manifest_digest = "sha256:" + hashlib.sha256(
            manifest.to_bytes()
        ).hexdigest()
        return StoredStateUnit(
            root=final_root,
            payload_path=final_root / "runtime_state_payload.bin",
            manifest_path=final_root / "state_unit_manifest.json",
            unit_number=unit_number,
            ref=SequenceGenerationRef(
                manifest.runtime_instance_id,
                manifest.seq_id,
                manifest.sequence_generation,
            ),
            unit_id=manifest.unit_id,
            payload_digest=digest,
            manifest_digest=manifest_digest,
        )

    def seal_generation(self, seal: GenerationSeal) -> Path:
        self._verify_root_identity()
        if not isinstance(seal, GenerationSeal):
            raise StateUnitStoreError("seal must be a GenerationSeal")
        generation_root = self._generation_root(
            seal.ref.runtime_instance_id,
            seal.ref.seq_id,
            seal.ref.sequence_generation,
        )
        _ensure_private_directory(generation_root)
        units = self._verified_unit_roots(generation_root)
        destination = generation_root / "generation_seal.json"
        if destination.exists() or _is_link_like(destination):
            raise StateUnitStoreError("generation is already sealed")
        if len(units) != seal.unit_count:
            raise StateUnitStoreError("generation seal unit_count differs")
        if units:
            latest = _read_regular_file(units[-1] / "state_unit_manifest.json")
            latest_digest = "sha256:" + hashlib.sha256(latest).hexdigest()
            if seal.last_unit_digest != latest_digest:
                raise StateUnitStoreError("generation seal last digest differs")
        elif seal.last_unit_digest is not None:
            raise StateUnitStoreError("empty generation has a last digest")
        payload = canonical_document_bytes(seal.to_mapping())
        staging = generation_root / (
            ".tmp-seal-" + hashlib.sha256(payload).hexdigest()[:16]
        )
        try:
            _write_new_file(staging, payload)
            try:
                os.link(staging, destination, follow_symlinks=False)
            except OSError as error:
                raise StateUnitStoreError("generation seal publication failed") from error
            _verify_regular_file(destination)
            _fsync_directory(generation_root)
        finally:
            if staging.exists() or _is_link_like(staging):
                _verify_regular_file(staging)
                staging.unlink()
        return destination

    def _generation_root(
        self, runtime_instance_id: str, seq_id: int, generation: int
    ) -> Path:
        if _SAFE_RUNTIME_INSTANCE_ID.fullmatch(runtime_instance_id) is None:
            raise StateUnitStoreError("runtime_instance_id is unsafe")
        if isinstance(seq_id, bool) or not isinstance(seq_id, int) or seq_id < 0:
            raise StateUnitStoreError("seq_id is invalid")
        if (
            isinstance(generation, bool)
            or not isinstance(generation, int)
            or generation <= 0
        ):
            raise StateUnitStoreError("sequence_generation is invalid")
        runtime_component = (
            "runtime_sha256_" + runtime_instance_id.removeprefix("sha256:")
            if runtime_instance_id.startswith("sha256:")
            else f"runtime_{runtime_instance_id}"
        )
        root = self.state_units_root
        for component in (
            runtime_component,
            f"seq_{seq_id}",
            f"generation_{generation}",
        ):
            root = root / component
            _ensure_private_directory(root)
        return root

    @staticmethod
    def _verify_append_chain(
        unit_roots: list[Path], document: Mapping[str, object]
    ) -> None:
        if not unit_roots:
            if document["unit_kind"] != "BASE":
                raise StateUnitStoreError("first state unit must be BASE")
            return
        if document["unit_kind"] != "DELTA":
            raise StateUnitStoreError("later state units must be DELTA")
        first = _read_regular_file(unit_roots[0] / "state_unit_manifest.json")
        previous = _read_regular_file(unit_roots[-1] / "state_unit_manifest.json")
        first_digest = "sha256:" + hashlib.sha256(first).hexdigest()
        previous_digest = "sha256:" + hashlib.sha256(previous).hexdigest()
        if (
            document["base_unit_digest"] != first_digest
            or document["previous_unit_digest"] != previous_digest
        ):
            raise StateUnitStoreError("state-unit digest chain is broken")

    def _verified_unit_roots(self, generation_root: Path) -> list[Path]:
        units: list[tuple[int, Path]] = []
        for path in generation_root.iterdir():
            if path.name.startswith(".tmp-"):
                raise StateUnitStoreError("stale state-unit staging directory exists")
            if path.name == "generation_seal.json":
                _verify_regular_file(path)
                continue
            match = _UNIT_DIRECTORY.fullmatch(path.name)
            if match is None or _is_link_like(path) or not path.is_dir():
                raise StateUnitStoreError("unexpected generation artifact")
            names = {item.name for item in path.iterdir()}
            if names != {"runtime_state_payload.bin", "state_unit_manifest.json"}:
                raise StateUnitStoreError("committed state-unit file set is invalid")
            _verify_regular_file(path / "runtime_state_payload.bin")
            _verify_regular_file(path / "state_unit_manifest.json")
            units.append((int(match.group(1)), path))
        units.sort()
        if [number for number, _ in units] != list(range(1, len(units) + 1)):
            raise StateUnitStoreError("state-unit sequence has a gap")
        return [path for _, path in units]

    def _verify_root_identity(self) -> None:
        if _directory_identity(self.evidence_root) != self._root_identity:
            raise StateUnitStoreError("evidence root identity changed")
        if _directory_identity(self.state_units_root) != self._state_units_identity:
            raise StateUnitStoreError("state-units root identity changed")


def _ensure_private_directory(path: Path) -> None:
    if path.exists() and (_is_link_like(path) or not path.is_dir()):
        if _is_link_like(path):
            raise StateUnitStoreError("state-unit root is a symbolic link")
        raise StateUnitStoreError("state-unit root is not a directory")
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if _is_link_like(path) or not path.is_dir():
        raise StateUnitStoreError("state-unit root is a symbolic link")
    _make_private(path, directory=True)


def _make_private(path: Path, *, directory: bool) -> None:
    if os.name == "posix":
        os.chmod(path, 0o700 if directory else 0o600, follow_symlinks=False)


def _directory_identity(path: Path) -> tuple[int, int]:
    if _is_link_like(path):
        raise StateUnitStoreError("state-unit directory is a symbolic link")
    metadata = os.stat(path, follow_symlinks=False)
    if not stat.S_ISDIR(metadata.st_mode):
        raise StateUnitStoreError("state-unit directory is invalid")
    return metadata.st_dev, metadata.st_ino


def _is_link_like(path: Path) -> bool:
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(metadata.st_mode):
        return True
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(reparse and getattr(metadata, "st_file_attributes", 0) & reparse)


def _write_new_file(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as error:
        raise StateUnitStoreError("state-unit staging file cannot be created") from error
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise StateUnitStoreError("state-unit write made no progress")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _make_private(path, directory=False)


def _verify_regular_file(path: Path) -> None:
    if _is_link_like(path):
        raise StateUnitStoreError("state-unit artifact is a symbolic link")
    metadata = os.stat(path, follow_symlinks=False)
    if not stat.S_ISREG(metadata.st_mode):
        raise StateUnitStoreError("state-unit artifact is not a regular file")


def _read_regular_file(path: Path) -> bytes:
    _verify_regular_file(path)
    before = os.stat(path, follow_symlinks=False)
    payload = path.read_bytes()
    after = os.stat(path, follow_symlinks=False)
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise StateUnitStoreError("state-unit artifact changed while reading")
    return payload


def _fsync_directory(path: Path) -> None:
    if os.name != "posix":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _remove_unit_root(root: Path) -> None:
    for name in ("state_unit_manifest.json", "runtime_state_payload.bin"):
        path = root / name
        if path.exists() or _is_link_like(path):
            _verify_regular_file(path)
            path.unlink()
    try:
        root.rmdir()
    except FileNotFoundError:
        pass


def _remove_staging_root(root: Path) -> None:
    if not root.exists() and not _is_link_like(root):
        return
    if _is_link_like(root) or not root.is_dir():
        raise StateUnitStoreError("state-unit staging root is unsafe")
    for path in root.iterdir():
        if path.name not in {
            "runtime_state_payload.bin",
            "state_unit_manifest.json",
        }:
            raise StateUnitStoreError("state-unit staging root contains unexpected files")
        _verify_regular_file(path)
        path.unlink()
    root.rmdir()


__all__ = [
    "GenerationSeal",
    "StateUnitManifest",
    "StateUnitStore",
    "StateUnitStoreError",
    "StoredStateUnit",
]
