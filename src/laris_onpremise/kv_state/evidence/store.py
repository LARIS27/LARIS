from __future__ import annotations

from collections.abc import Callable, Mapping
import hashlib
import hmac
import os
from pathlib import Path
import re
import stat
import tempfile
import time

from .manifest import (
    AcquisitionManifestError,
    canonical_acquisition_manifest_bytes,
    parse_acquisition_manifest_bytes,
)


_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_PAYLOAD_WRITE_BLOCK_BYTES = 1024 * 1024


class PayloadStoreError(RuntimeError):
    """Raised when selective acquisition evidence cannot be sealed safely."""


class PayloadStoreRootIntegrityError(PayloadStoreError):
    """Raised when the shared evidence-root identity is untrustworthy."""


class RuntimeStatePayloadStore:
    """No-clobber store for one payload and its acquisition manifest."""

    def __init__(self, evidence_root: Path | str) -> None:
        self.evidence_root = Path(evidence_root)
        _ensure_non_link_directory(self.evidence_root)
        self.acquisition_root = self.evidence_root / "acquisition"
        _ensure_non_link_directory(self.acquisition_root)
        self._evidence_root_identity = _directory_identity(self.evidence_root)
        self._acquisition_root_identity = _directory_identity(
            self.acquisition_root
        )

    def is_committed(self, request_ref: str) -> bool:
        """Return true only when the manifest-last completion barrier verifies."""

        _require_safe_request_ref(request_ref)
        self._verify_root_identity()
        root = self.acquisition_root / request_ref
        if not root.exists() and not _is_link_like(root):
            return False
        if _is_link_like(root) or not root.is_dir():
            raise PayloadStoreError("request publication root is unsafe")
        names = {path.name for path in root.iterdir()}
        if "acquisition_manifest.json" not in names:
            return False
        expected = {
            "runtime_state_payload.bin",
            "acquisition_manifest.json",
        }
        if names != expected:
            raise PayloadStoreError("corrupt committed publication file set")
        payload_path = root / "runtime_state_payload.bin"
        manifest_path = root / "acquisition_manifest.json"
        _verify_regular_file(payload_path)
        _verify_regular_file(manifest_path)
        try:
            document = parse_acquisition_manifest_bytes(
                _read_regular_file(manifest_path)
            )
        except (OSError, AcquisitionManifestError) as error:
            raise PayloadStoreError(
                "corrupt committed publication manifest"
            ) from error
        if document["request_ref"] != request_ref:
            raise PayloadStoreError(
                "corrupt committed publication manifest request_ref"
            )
        size, digest = _hash_file(payload_path)
        if (
            document.get("payload_file") != "runtime_state_payload.bin"
            or document.get("payload_size_bytes") != size
            or not isinstance(document.get("payload_digest"), str)
            or not hmac.compare_digest(document["payload_digest"], digest)
        ):
            raise PayloadStoreError("corrupt committed publication payload")
        return True

    def commit(
        self,
        *,
        request_ref: str,
        payload: bytes,
        manifest: Mapping[str, object],
    ) -> tuple[Path, Path, int, str]:
        if not isinstance(payload, bytes) or not payload:
            raise PayloadStoreError("runtime-state payload is empty")
        return self.commit_parts(
            request_ref=request_ref,
            payload_parts=(payload,),
            manifest=manifest,
        )

    def commit_parts(
        self,
        *,
        request_ref: str,
        payload_parts: tuple[bytes, ...],
        manifest: Mapping[str, object],
        phase_sink: Callable[[str, int, int], None] | None = None,
    ) -> tuple[Path, Path, int, str]:
        """Durably publish canonical payload chunks as one unchanged file."""

        _require_safe_request_ref(request_ref)
        self._verify_root_identity()
        if (
            not isinstance(payload_parts, tuple)
            or not payload_parts
            or any(not isinstance(part, bytes) or not part for part in payload_parts)
        ):
            raise PayloadStoreError("runtime-state payload parts are invalid")
        if not isinstance(manifest, Mapping):
            raise PayloadStoreError("manifest must be a mapping")

        final_root = self.acquisition_root / request_ref
        self._prepare_retryable_root(final_root, request_ref)
        staging_root = Path(
            tempfile.mkdtemp(prefix=".tmp-", dir=self.acquisition_root)
        )
        self._verify_root_identity()
        final_created = False
        try:
            staging_payload = staging_root / "runtime_state_payload.bin"
            staging_manifest = staging_root / "acquisition_manifest.json"
            size, digest = _write_new_file_parts(
                staging_payload,
                payload_parts,
                phase_sink=phase_sink,
                write_phase="payload_write_hash",
                fsync_phase="payload_file_fsync",
            )
            document = dict(manifest)
            if document.get("request_ref") != request_ref:
                raise PayloadStoreError(
                    "manifest request_ref does not match publication root"
                )
            _merge_payload_claim(
                document, "payload_file", "runtime_state_payload.bin"
            )
            _merge_payload_claim(document, "payload_size_bytes", size)
            _merge_payload_claim(document, "payload_digest", digest)
            for flag in (
                "contains_prompt_text",
                "contains_response_text",
                "contains_explicit_token_history",
            ):
                if document.get(flag) is not False:
                    raise PayloadStoreError(f"{flag} must be false")
            try:
                manifest_bytes = canonical_acquisition_manifest_bytes(document)
            except AcquisitionManifestError as error:
                raise PayloadStoreError(f"manifest is invalid: {error}") from error
            _write_new_file(
                staging_manifest,
                manifest_bytes,
                phase_sink=phase_sink,
                write_phase="manifest_write",
                fsync_phase="manifest_file_fsync",
            )
            started_ns = _phase_started(phase_sink)
            _fsync_directory(staging_root)
            _record_phase(
                phase_sink, "staging_directory_fsync", started_ns
            )

            try:
                started_ns = _phase_started(phase_sink)
                self._verify_root_identity()
                os.mkdir(final_root, mode=0o700)
                final_created = True
            except FileExistsError as error:
                raise PayloadStoreError(
                    "request acquisition directory already exists"
                ) from error
            final_payload = final_root / "runtime_state_payload.bin"
            final_manifest = final_root / "acquisition_manifest.json"
            try:
                os.link(staging_payload, final_payload, follow_symlinks=False)
                _record_phase(
                    phase_sink, "payload_publication", started_ns, size
                )
                started_ns = _phase_started(phase_sink)
                _fsync_directory(final_root)
                _record_phase(
                    phase_sink,
                    "payload_publication_directory_fsync",
                    started_ns,
                )
                started_ns = _phase_started(phase_sink)
                os.link(staging_manifest, final_manifest, follow_symlinks=False)
                _record_phase(
                    phase_sink,
                    "manifest_publication",
                    started_ns,
                    len(manifest_bytes),
                )
            except OSError as error:
                raise PayloadStoreError(
                    "payload publication requires same-filesystem hard links"
                ) from error

            started_ns = _phase_started(phase_sink)
            _verify_regular_file(final_payload)
            _verify_regular_file(final_manifest)
            staging_payload_stat = staging_payload.stat(follow_symlinks=False)
            final_payload_stat = final_payload.stat(follow_symlinks=False)
            staging_manifest_stat = staging_manifest.stat(follow_symlinks=False)
            final_manifest_stat = final_manifest.stat(follow_symlinks=False)
            if (
                staging_payload_stat.st_dev,
                staging_payload_stat.st_ino,
            ) != (
                final_payload_stat.st_dev,
                final_payload_stat.st_ino,
            ) or (
                staging_manifest_stat.st_dev,
                staging_manifest_stat.st_ino,
            ) != (
                final_manifest_stat.st_dev,
                final_manifest_stat.st_ino,
            ):
                raise PayloadStoreError("published hard-link identity mismatch")
            if final_payload_stat.st_size != size:
                raise PayloadStoreError("published payload size mismatch")
            if final_manifest.read_bytes() != manifest_bytes:
                raise PayloadStoreError("published manifest digest mismatch")
            _record_phase(
                phase_sink, "publication_validation", started_ns, size
            )
            started_ns = _phase_started(phase_sink)
            _fsync_directory(final_root)
            _record_phase(
                phase_sink, "publication_directory_fsync", started_ns
            )
            started_ns = _phase_started(phase_sink)
            _fsync_directory(self.acquisition_root)
            _record_phase(
                phase_sink, "acquisition_directory_fsync", started_ns
            )
        except Exception:
            if final_created:
                self._verify_root_identity()
                _remove_known_publication(final_root)
            raise
        finally:
            self._verify_root_identity()
            started_ns = _phase_started(phase_sink)
            _remove_staging_publication(staging_root)
            _record_phase(phase_sink, "staging_cleanup", started_ns)
            started_ns = _phase_started(phase_sink)
            _fsync_directory(self.acquisition_root)
            _record_phase(
                phase_sink, "cleanup_directory_fsync", started_ns
            )

        return final_payload, final_manifest, size, digest

    def _verify_root_identity(self) -> None:
        if _directory_identity(self.evidence_root) != self._evidence_root_identity:
            raise PayloadStoreRootIntegrityError("evidence root identity changed")
        if (
            _directory_identity(self.acquisition_root)
            != self._acquisition_root_identity
        ):
            raise PayloadStoreRootIntegrityError("acquisition root identity changed")

    def _prepare_retryable_root(
        self, final_root: Path, request_ref: str
    ) -> None:
        if not final_root.exists() and not _is_link_like(final_root):
            return
        if _is_link_like(final_root) or not final_root.is_dir():
            raise PayloadStoreError("request publication root is unsafe")
        names = {path.name for path in final_root.iterdir()}
        if "acquisition_manifest.json" in names:
            if self.is_committed(request_ref):
                raise PayloadStoreError(
                    "request acquisition directory already exists"
                )
            raise PayloadStoreError("corrupt committed publication")
        allowed_partial = {"runtime_state_payload.bin"}
        if not names.issubset(allowed_partial):
            raise PayloadStoreError("unrecognized partial publication")
        if "runtime_state_payload.bin" in names:
            payload_path = final_root / "runtime_state_payload.bin"
            _verify_regular_file(payload_path)
            payload_path.unlink()
        final_root.rmdir()
        _fsync_directory(self.acquisition_root)


def _require_safe_request_ref(request_ref: str) -> None:
    if not isinstance(request_ref, str) or _SAFE_ID.fullmatch(request_ref) is None:
        raise PayloadStoreError("request_ref is not a safe identifier")


def _merge_payload_claim(
    document: dict[str, object], name: str, expected: object
) -> None:
    if name in document and document[name] != expected:
        raise PayloadStoreError(f"manifest {name} does not match payload")
    document[name] = expected


def _ensure_non_link_directory(path: Path) -> None:
    if path.exists() and (_is_link_like(path) or not path.is_dir()):
        if _is_link_like(path):
            raise PayloadStoreRootIntegrityError(
                "evidence root is a symbolic link or reparse point"
            )
        raise PayloadStoreRootIntegrityError("evidence root is not a directory")
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if _is_link_like(path) or not path.is_dir():
        raise PayloadStoreRootIntegrityError(
            "evidence root is a symbolic link or reparse point"
        )
    if os.name == "posix":
        os.chmod(path, 0o700, follow_symlinks=False)


def _directory_identity(path: Path) -> tuple[int, int]:
    if _is_link_like(path):
        raise PayloadStoreRootIntegrityError(
            "evidence directory is a symbolic link"
        )
    try:
        metadata = os.stat(path, follow_symlinks=False)
    except OSError as error:
        raise PayloadStoreRootIntegrityError(
            "evidence directory identity is unavailable"
        ) from error
    if not stat.S_ISDIR(metadata.st_mode):
        raise PayloadStoreRootIntegrityError(
            "evidence directory is not a directory"
        )
    return metadata.st_dev, metadata.st_ino


def _is_link_like(path: Path) -> bool:
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(metadata.st_mode):
        return True
    attributes = getattr(metadata, "st_file_attributes", 0)
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(reparse and attributes & reparse)


def _write_new_file(
    path: Path,
    payload: bytes,
    *,
    phase_sink: Callable[[str, int, int], None] | None = None,
    write_phase: str | None = None,
    fsync_phase: str | None = None,
) -> None:
    _write_new_file_parts(
        path,
        (payload,),
        phase_sink=phase_sink,
        write_phase=write_phase,
        fsync_phase=fsync_phase,
    )


def _write_new_file_parts(
    path: Path,
    payload_parts: tuple[bytes, ...],
    *,
    phase_sink: Callable[[str, int, int], None] | None = None,
    write_phase: str | None = None,
    fsync_phase: str | None = None,
) -> tuple[int, str]:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as error:
        raise PayloadStoreError("staging file cannot be created") from error
    try:
        started_ns = _phase_started(phase_sink)
        digest = hashlib.sha256()
        size = 0

        def write_all(data: bytes | bytearray | memoryview) -> None:
            nonlocal size
            view = memoryview(data)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise PayloadStoreError("staging file write made no progress")
                digest.update(view[:written])
                size += written
                view = view[written:]

        pending = bytearray()
        for part in payload_parts:
            view = memoryview(part)
            if pending:
                taken = min(_PAYLOAD_WRITE_BLOCK_BYTES - len(pending), len(view))
                pending.extend(view[:taken])
                view = view[taken:]
                if len(pending) == _PAYLOAD_WRITE_BLOCK_BYTES:
                    write_all(pending)
                    pending.clear()
            while len(view) >= _PAYLOAD_WRITE_BLOCK_BYTES:
                write_all(view[:_PAYLOAD_WRITE_BLOCK_BYTES])
                view = view[_PAYLOAD_WRITE_BLOCK_BYTES:]
            if view:
                pending.extend(view)
        if pending:
            write_all(pending)
        if write_phase is not None:
            _record_phase(phase_sink, write_phase, started_ns, size)
        started_ns = _phase_started(phase_sink)
        os.fsync(descriptor)
        if fsync_phase is not None:
            _record_phase(phase_sink, fsync_phase, started_ns, size)
        return size, "sha256:" + digest.hexdigest()
    finally:
        os.close(descriptor)


def _verify_regular_file(path: Path) -> None:
    if _is_link_like(path):
        raise PayloadStoreError("published artifact is a symbolic link")
    metadata = os.stat(path, follow_symlinks=False)
    if not stat.S_ISREG(metadata.st_mode):
        raise PayloadStoreError("published artifact is not a regular file")


def _hash_file(path: Path) -> tuple[int, str]:
    flags = os.O_RDONLY
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        digest = hashlib.sha256()
        size = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
        return size, "sha256:" + digest.hexdigest()
    finally:
        os.close(descriptor)


def _read_regular_file(path: Path) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise PayloadStoreError("published artifact is not a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if identity_before != identity_after:
            raise PayloadStoreError("published artifact changed while reading")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    if os.name != "posix":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _phase_started(
    phase_sink: Callable[[str, int, int], None] | None,
) -> int:
    return time.monotonic_ns() if phase_sink is not None else 0


def _record_phase(
    phase_sink: Callable[[str, int, int], None] | None,
    phase: str,
    started_ns: int,
    byte_count: int = 0,
) -> None:
    if phase_sink is not None:
        phase_sink(phase, time.monotonic_ns() - started_ns, byte_count)


def _remove_known_publication(root: Path) -> None:
    for name in ("runtime_state_payload.bin", "acquisition_manifest.json"):
        try:
            (root / name).unlink()
        except FileNotFoundError:
            pass
    try:
        root.rmdir()
    except FileNotFoundError:
        pass


def _remove_staging_publication(root: Path) -> None:
    if not root.exists() and not _is_link_like(root):
        return
    if _is_link_like(root) or not root.is_dir():
        raise PayloadStoreError("staging publication root is unsafe")
    names = {path.name for path in root.iterdir()}
    allowed = {"runtime_state_payload.bin", "acquisition_manifest.json"}
    if not names.issubset(allowed):
        raise PayloadStoreError("staging publication contains unexpected files")
    for name in allowed:
        path = root / name
        if path.exists() or _is_link_like(path):
            _verify_regular_file(path)
            path.unlink()
    root.rmdir()
