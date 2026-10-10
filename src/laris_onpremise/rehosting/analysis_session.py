from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Literal
import uuid

from laris_onpremise.core.records import canonical_json
from laris_onpremise.rehosting.catalog import (
    PortableBundleCatalog,
    PortableBundleEntry,
)
from laris_onpremise.rehosting.contracts import (
    InvestigatorSampler,
    RehostSubjectKind,
)


class RehostAnalysisSessionError(RuntimeError):
    """Raised when an investigator analysis session cannot be recorded safely."""


_ANALYSIS_ID = re.compile(r"analysis-[A-Za-z0-9_-]{1,64}\Z")
_EVIDENCE_NAME = re.compile(r"(?:inquiry|turn)-([0-9]{4})\.json\Z")
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_SAFE_REF = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,191}\Z")
_RUN_ID = re.compile(r"run-[A-Za-z0-9._-]{1,191}\Z")
_BUNDLE_ID = re.compile(r"rb_[0-9a-f]{64}\Z")
_USER_HMAC = re.compile(r"[A-Za-z0-9_-]{43}\Z")
_SESSION_REF = re.compile(r"sr_[0-9a-f]{64}\Z")
_MAX_FAILURE_REASON = 256
_MAX_ANALYSIS_JSON_BYTES = 16 * 1024 * 1024


class RehostAnalysisSession:
    def __init__(
        self,
        *,
        directory: Path,
        lock_descriptor: int,
        context: dict[str, object],
        clock: Callable[[], datetime],
    ) -> None:
        self._directory = directory
        self._lock_descriptor: int | None = lock_descriptor
        self._context = context
        self._clock = clock
        self._active = True

    @classmethod
    def start(
        cls,
        analysis_root: Path,
        *,
        catalog: PortableBundleCatalog,
        entry: PortableBundleEntry,
        mode: str,
        sampler: InvestigatorSampler,
        predict: int,
        id_factory: Callable[[], str] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> "RehostAnalysisSession":
        root = _analysis_root(analysis_root)
        if not isinstance(catalog, PortableBundleCatalog):
            raise RehostAnalysisSessionError("bundle catalog is invalid")
        if not isinstance(entry, PortableBundleEntry) or sum(
            candidate == entry for candidate in catalog.entries
        ) != 1:
            raise RehostAnalysisSessionError("bundle entry is not selected from the catalog")
        if mode not in {"simple", "session"}:
            raise RehostAnalysisSessionError("analysis mode is invalid")
        if not isinstance(sampler, InvestigatorSampler):
            raise RehostAnalysisSessionError("investigator sampler is invalid")
        if isinstance(predict, bool) or not isinstance(predict, int) or predict <= 0:
            raise RehostAnalysisSessionError("predict must be a positive integer")
        actual_clock = clock or (lambda: datetime.now(timezone.utc))
        started_at = _timestamp(actual_clock())
        analysis_id = (id_factory or (lambda: "analysis-" + uuid.uuid4().hex))()
        if not isinstance(analysis_id, str) or _ANALYSIS_ID.fullmatch(analysis_id) is None:
            raise RehostAnalysisSessionError("analysis identifier is invalid")
        directory = root / analysis_id
        try:
            directory.mkdir(mode=0o700, exist_ok=False)
            evidence = directory / "evidence"
            evidence.mkdir(mode=0o700, exist_ok=False)
            descriptor = _create_and_lock(directory / ".session.lock")
        except Exception as error:
            _remove_failed_start(directory)
            if isinstance(error, RehostAnalysisSessionError):
                raise
            raise RehostAnalysisSessionError("analysis session cannot be created") from error
        context = _context_document(
            analysis_id=analysis_id,
            catalog=catalog,
            entry=entry,
            mode=mode,
            sampler=sampler,
            predict=predict,
            started_at=started_at,
        )
        try:
            _atomic_publish(
                directory / "analysis.pending.json",
                canonical_json(
                    {
                        **context,
                        "schema_version": _context_schema_version(context),
                        "status": "RUNNING",
                    }
                )
                + b"\n",
            )
        except Exception:
            _unlock_and_close(descriptor)
            _remove_failed_start(directory)
            raise
        return cls(
            directory=directory,
            lock_descriptor=descriptor,
            context=context,
            clock=actual_clock,
        )

    @classmethod
    def recover_stale(
        cls,
        analysis_root: Path,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> tuple[Path, ...]:
        root = _analysis_root(analysis_root)
        actual_clock = clock or (lambda: datetime.now(timezone.utc))
        try:
            candidates = sorted(root.iterdir(), key=lambda path: path.name)
        except OSError as error:
            raise RehostAnalysisSessionError("analysis root inventory is unavailable") from error
        recovered: list[Path] = []
        for directory in candidates:
            if directory.is_symlink() or not directory.is_dir():
                raise RehostAnalysisSessionError("analysis root inventory is invalid")
            if _ANALYSIS_ID.fullmatch(directory.name) is None:
                raise RehostAnalysisSessionError("analysis directory name is invalid")
            names = _inventory(directory)
            if "analysis.json" in names:
                if names != {"analysis.json", "evidence"}:
                    raise RehostAnalysisSessionError("final analysis inventory is invalid")
                continue
            if names != {".session.lock", "analysis.pending.json", "evidence"}:
                raise RehostAnalysisSessionError("pending analysis inventory is invalid")
            _validate_active_inventory(directory)
            descriptor = _try_lock_existing(directory / ".session.lock")
            if descriptor is None:
                continue
            session: RehostAnalysisSession | None = None
            try:
                _raw, pending = _read_canonical(directory / "analysis.pending.json")
                context = _pending_context(pending, directory.name)
                session = cls(
                    directory=directory,
                    lock_descriptor=descriptor,
                    context=context,
                    clock=actual_clock,
                )
                session.finalize(
                    status="INCOMPLETE",
                    reason="unclean_previous_termination",
                )
            except Exception:
                if session is None or session._lock_descriptor is not None:
                    _unlock_and_close(descriptor)
                raise
            recovered.append(directory)
        return tuple(recovered)

    @property
    def directory(self) -> Path:
        return self._directory

    def record_success(
        self,
        *,
        messages: tuple[Mapping[str, str], ...],
        result: Mapping[str, object],
        started_at: datetime | None = None,
    ) -> Path:
        self._require_active()
        normalized_messages = _messages(messages)
        normalized_result = _bound_result(result, self._context)
        ended = _timestamp(self._clock())
        started = _timestamp(started_at) if started_at is not None else ended
        sequence = self._next_sequence()
        path = self._evidence_path(sequence)
        _write_exclusive(
            path,
            canonical_json(
                {
                    "bundle_id": self._context["bundle_id"],
                    "ended_at": ended,
                    "messages": normalized_messages,
                    "model_digest": self._context["model_digest"],
                    "result": normalized_result,
                    "schema_version": "2.0",
                    "sequence": sequence,
                    "session_ref": self._context["session_ref"],
                    "started_at": started,
                    "status": "COMPLETE",
                }
            )
            + b"\n",
        )
        return path

    def record_failure(
        self,
        *,
        messages: tuple[Mapping[str, str], ...],
        reason: str,
        started_at: datetime | None = None,
    ) -> Path:
        self._require_active()
        normalized_messages = _messages(messages)
        safe_reason = _failure_reason(reason)
        ended = _timestamp(self._clock())
        started = _timestamp(started_at) if started_at is not None else ended
        sequence = self._next_sequence()
        path = self._evidence_path(sequence)
        _write_exclusive(
            path,
            canonical_json(
                {
                    "bundle_id": self._context["bundle_id"],
                    "ended_at": ended,
                    "failure_reason": safe_reason,
                    "messages": normalized_messages,
                    "model_digest": self._context["model_digest"],
                    "schema_version": "2.0",
                    "sequence": sequence,
                    "session_ref": self._context["session_ref"],
                    "started_at": started,
                    "status": "INCOMPLETE",
                }
            )
            + b"\n",
        )
        return path

    def finalize(
        self,
        *,
        status: Literal["COMPLETE", "INCOMPLETE"],
        reason: str | None = None,
    ) -> Path:
        self._require_active()
        if status not in {"COMPLETE", "INCOMPLETE"}:
            raise RehostAnalysisSessionError("final analysis status is invalid")
        if status == "COMPLETE" and reason is not None:
            raise RehostAnalysisSessionError("complete analysis cannot have a failure reason")
        if status == "INCOMPLETE":
            reason = _failure_reason(reason)
        evidence = self._read_evidence()
        successful = sum(item[1]["status"] == "COMPLETE" for item in evidence)
        final: dict[str, object] = {
            **self._context,
            "counts": {
                "failed": len(evidence) - successful,
                "successful": successful,
                "total": len(evidence),
            },
            "ended_at": _timestamp(self._clock()),
            "schema_version": _context_schema_version(self._context),
            "status": status,
        }
        summary_name = "inquiries" if self._context["mode"] == "simple" else "conversation"
        final[summary_name] = [
            _evidence_summary(path, document)
            for path, document, _raw in evidence
        ]
        if reason is not None:
            final["failure_reason"] = reason
        path = self._directory / "analysis.json"
        _atomic_publish(path, canonical_json(final) + b"\n")
        try:
            (self._directory / "analysis.pending.json").unlink()
        except OSError as error:
            raise RehostAnalysisSessionError("analysis pending state cannot be removed") from error
        self._release_lock()
        return path

    def _require_active(self) -> None:
        if not self._active or self._lock_descriptor is None:
            raise RehostAnalysisSessionError("analysis session is already finalized")
        _validate_active_inventory(self._directory)

    def _next_sequence(self) -> int:
        sequence = len(self._read_evidence()) + 1
        if sequence > 9999:
            raise RehostAnalysisSessionError("analysis evidence count exceeds the limit")
        return sequence

    def _evidence_path(self, sequence: int) -> Path:
        prefix = "inquiry" if self._context["mode"] == "simple" else "turn"
        return self._directory / "evidence" / f"{prefix}-{sequence:04d}.json"

    def _read_evidence(self) -> list[tuple[Path, dict[str, object], bytes]]:
        evidence_root = self._directory / "evidence"
        try:
            paths = sorted(evidence_root.iterdir(), key=lambda path: path.name)
        except OSError as error:
            raise RehostAnalysisSessionError("analysis evidence inventory is unavailable") from error
        expected_prefix = "inquiry" if self._context["mode"] == "simple" else "turn"
        records: list[tuple[Path, dict[str, object], bytes]] = []
        for expected_sequence, path in enumerate(paths, 1):
            match = _EVIDENCE_NAME.fullmatch(path.name)
            if (
                match is None
                or not path.name.startswith(expected_prefix + "-")
                or int(match.group(1)) != expected_sequence
            ):
                raise RehostAnalysisSessionError("analysis evidence inventory is invalid")
            raw, document = _read_canonical(path)
            if document.get("sequence") != expected_sequence:
                raise RehostAnalysisSessionError("analysis evidence sequence is invalid")
            _validate_evidence_document(document, self._context)
            records.append((path, document, raw))
        return records

    def _release_lock(self) -> None:
        descriptor = self._lock_descriptor
        if descriptor is None:
            return
        self._lock_descriptor = None
        self._active = False
        _unlock_and_close(descriptor)
        try:
            (self._directory / ".session.lock").unlink()
        except OSError as error:
            raise RehostAnalysisSessionError("analysis lock file cannot be removed") from error


def _analysis_root(path: Path) -> Path:
    if not isinstance(path, Path) or not path.is_absolute() or path.is_symlink():
        raise RehostAnalysisSessionError("analysis root must be an absolute directory")
    try:
        path.mkdir(parents=True, exist_ok=True)
        metadata = path.lstat()
    except OSError as error:
        raise RehostAnalysisSessionError("analysis root is unavailable") from error
    if not stat.S_ISDIR(metadata.st_mode):
        raise RehostAnalysisSessionError("analysis root must be a directory")
    return path


def _context_document(
    *,
    analysis_id: str,
    catalog: PortableBundleCatalog,
    entry: PortableBundleEntry,
    mode: str,
    sampler: InvestigatorSampler,
    predict: int,
    started_at: str,
) -> dict[str, object]:
    common = {
        "analysis_id": analysis_id,
        "bundle_id": entry.bundle_id,
        "bundle_index_digest": catalog.index_digest,
        "bundle_manifest_digest": catalog.manifest_digest,
        "inquiry_policy": {
            "predict": predict,
            "sampler": sampler.to_mapping(),
        },
        "mode": mode,
        "model_digest": entry.model_digest,
        "model_profile_id": entry.model_profile_id,
        "runtime": {
            "key_ggml_type_id": entry.key_ggml_type_id,
            "memory_family": "simple_kv",
            "n_ctx": entry.n_ctx,
            "n_layer": entry.n_layer,
            "n_seq_max": entry.n_seq_max,
            "n_stream": entry.n_stream,
            "position_range": list(entry.position_range),
            "v_trans": entry.v_trans,
            "value_ggml_type_id": entry.value_ggml_type_id,
        },
        "runtime_profile_digest": entry.runtime_profile_digest,
        "session_ref": entry.session_ref,
        "source_run_id": catalog.source_run_id,
        "started_at": started_at,
    }
    if entry.subject_kind is RehostSubjectKind.ENDPOINT_REF:
        return {
            **common,
            "correlation_key_id": entry.identity_key_id,
            "endpoint_ref": entry.subject_ref,
            "subject_kind": RehostSubjectKind.ENDPOINT_REF.value,
        }
    return {
        **common,
        "identity_key_id": entry.identity_key_id,
        "user_hmac": entry.subject_ref,
    }


def _context_schema_version(context: Mapping[str, object]) -> str:
    return (
        "2.1"
        if context.get("subject_kind") == RehostSubjectKind.ENDPOINT_REF.value
        else "2.0"
    )


def _pending_context(
    document: Mapping[str, object],
    expected_analysis_id: str,
) -> dict[str, object]:
    common_fields = {
        "analysis_id",
        "bundle_id",
        "bundle_index_digest",
        "bundle_manifest_digest",
        "inquiry_policy",
        "mode",
        "model_digest",
        "model_profile_id",
        "runtime",
        "runtime_profile_digest",
        "session_ref",
        "source_run_id",
        "started_at",
    }
    schema_version = document.get("schema_version")
    if schema_version == "2.1":
        context_fields = common_fields | {
            "correlation_key_id",
            "endpoint_ref",
            "subject_kind",
        }
        subject_invalid = (
            document.get("subject_kind") != RehostSubjectKind.ENDPOINT_REF.value
            or _SAFE_REF.fullmatch(str(document.get("correlation_key_id"))) is None
            or _USER_HMAC.fullmatch(str(document.get("endpoint_ref"))) is None
        )
    else:
        context_fields = common_fields | {"identity_key_id", "user_hmac"}
        subject_invalid = (
            schema_version != "2.0"
            or _SAFE_REF.fullmatch(str(document.get("identity_key_id"))) is None
            or _USER_HMAC.fullmatch(str(document.get("user_hmac"))) is None
        )
    if (
        set(document) != context_fields | {"schema_version", "status"}
        or subject_invalid
        or document.get("status") != "RUNNING"
        or document.get("analysis_id") != expected_analysis_id
        or _ANALYSIS_ID.fullmatch(str(document.get("analysis_id"))) is None
        or _BUNDLE_ID.fullmatch(str(document.get("bundle_id"))) is None
        or _SHA256.fullmatch(str(document.get("bundle_index_digest"))) is None
        or _SHA256.fullmatch(str(document.get("bundle_manifest_digest"))) is None
        or document.get("mode") not in {"simple", "session"}
        or _SHA256.fullmatch(str(document.get("model_digest"))) is None
        or _SAFE_REF.fullmatch(str(document.get("model_profile_id"))) is None
        or _SHA256.fullmatch(str(document.get("runtime_profile_digest"))) is None
        or _SESSION_REF.fullmatch(str(document.get("session_ref"))) is None
        or _RUN_ID.fullmatch(str(document.get("source_run_id"))) is None
        or not isinstance(document.get("inquiry_policy"), dict)
        or not isinstance(document.get("runtime"), dict)
        or not isinstance(document.get("started_at"), str)
    ):
        raise RehostAnalysisSessionError("analysis pending state is invalid")
    context = dict(document)
    del context["schema_version"]
    del context["status"]
    return context


def _messages(value: tuple[Mapping[str, str], ...]) -> list[dict[str, str]]:
    if not isinstance(value, tuple) or not value:
        raise RehostAnalysisSessionError("analysis messages are invalid")
    normalized: list[dict[str, str]] = []
    for item in value:
        if (
            not isinstance(item, Mapping)
            or set(item) != {"content", "role"}
            or item.get("role") not in {"user", "assistant"}
            or not isinstance(item.get("content"), str)
            or not item["content"].strip()
            or "\x00" in item["content"]
        ):
            raise RehostAnalysisSessionError("analysis messages are invalid")
        normalized.append({"content": item["content"], "role": item["role"]})
    return normalized


def _bound_result(
    value: Mapping[str, object],
    context: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise RehostAnalysisSessionError("analysis result is invalid")
    normalized = json.loads(canonical_json(dict(value)).decode("utf-8"))
    case = normalized.get("case")
    receipt = normalized.get("receipt")
    if (
        normalized.get("status") != "COMPLETE"
        or normalized.get("model_digest") != context["model_digest"]
        or not isinstance(case, dict)
        or case.get("bundle_id") != context["bundle_id"]
        or case.get("model_digest") != context["model_digest"]
        or not isinstance(receipt, dict)
        or receipt.get("exit_code") != 0
        or not isinstance(receipt.get("response"), str)
    ):
        raise RehostAnalysisSessionError("analysis result binding is invalid")
    return normalized


def _validate_evidence_document(
    document: Mapping[str, object],
    context: Mapping[str, object],
) -> None:
    common = {
        "bundle_id",
        "ended_at",
        "messages",
        "model_digest",
        "schema_version",
        "sequence",
        "session_ref",
        "started_at",
        "status",
    }
    status = document.get("status")
    expected = common | ({"result"} if status == "COMPLETE" else {"failure_reason"})
    if (
        status not in {"COMPLETE", "INCOMPLETE"}
        or set(document) != expected
        or document.get("schema_version") != "2.0"
        or document.get("bundle_id") != context["bundle_id"]
        or document.get("session_ref") != context["session_ref"]
        or document.get("model_digest") != context["model_digest"]
        or isinstance(document.get("sequence"), bool)
        or not isinstance(document.get("sequence"), int)
        or document["sequence"] <= 0
        or not isinstance(document.get("started_at"), str)
        or not isinstance(document.get("ended_at"), str)
        or not isinstance(document.get("messages"), list)
    ):
        raise RehostAnalysisSessionError("analysis evidence binding is invalid")
    _messages(tuple(document["messages"]))
    if status == "COMPLETE":
        _bound_result(document["result"], context)
    else:
        _failure_reason(document.get("failure_reason"))


def _evidence_summary(
    path: Path,
    document: Mapping[str, object],
) -> dict[str, object]:
    raw = path.read_bytes()
    messages = document["messages"]
    question = next(
        item["content"] for item in reversed(messages) if item.get("role") == "user"
    )
    summary: dict[str, object] = {
        "evidence_digest": "sha256:" + hashlib.sha256(raw).hexdigest(),
        "evidence_path": "evidence/" + path.name,
        "question": question,
        "sequence": document["sequence"],
        "status": document["status"],
    }
    if document["status"] == "COMPLETE":
        summary["response"] = document["result"]["receipt"]["response"]
    return summary


def _timestamp(value: datetime) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise RehostAnalysisSessionError("analysis timestamp must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _failure_reason(value: object) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise RehostAnalysisSessionError("incomplete analysis requires a failure reason")
    return value if len(value) <= _MAX_FAILURE_REASON else "analysis session failed"


def _read_canonical(path: Path) -> tuple[bytes, dict[str, object]]:
    descriptor: int | None = None
    try:
        before = path.lstat()
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > _MAX_ANALYSIS_JSON_BYTES
        ):
            raise RehostAnalysisSessionError("analysis artifact is unsafe")
        flags = os.O_RDONLY
        flags |= getattr(os, "O_BINARY", 0)
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        after = os.fstat(descriptor)
        if (
            not stat.S_ISREG(after.st_mode)
            or after.st_nlink != 1
            or after.st_size != before.st_size
            or (
                before.st_ino
                and after.st_ino
                and (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)
            )
        ):
            raise RehostAnalysisSessionError("analysis artifact is unsafe")
        chunks: list[bytes] = []
        remaining = _MAX_ANALYSIS_JSON_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(65536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        if len(raw) != after.st_size or len(raw) > _MAX_ANALYSIS_JSON_BYTES:
            raise RehostAnalysisSessionError("analysis artifact is unsafe")
        document = json.loads(raw.decode("utf-8", errors="strict"))
    except RehostAnalysisSessionError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RehostAnalysisSessionError("analysis evidence is invalid") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if not isinstance(document, dict) or raw != canonical_json(document) + b"\n":
        raise RehostAnalysisSessionError("analysis evidence is not canonical")
    return raw, document


def _create_and_lock(path: Path) -> int:
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
        os.write(descriptor, b"\0")
        os.fsync(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        _lock_descriptor(descriptor)
        return descriptor
    except Exception:
        if "descriptor" in locals():
            os.close(descriptor)
        raise


def _try_lock_existing(path: Path) -> int | None:
    flags = os.O_RDWR
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size != 1
        ):
            raise RehostAnalysisSessionError("analysis lock file is invalid")
        os.lseek(descriptor, 0, os.SEEK_SET)
        try:
            _lock_descriptor(descriptor)
        except RehostAnalysisSessionError:
            os.close(descriptor)
            return None
        return descriptor
    except RehostAnalysisSessionError:
        raise
    except OSError as error:
        if "descriptor" in locals():
            os.close(descriptor)
        raise RehostAnalysisSessionError("analysis lock file is unavailable") from error


def _lock_descriptor(descriptor: int) -> None:
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        raise RehostAnalysisSessionError("analysis session is already active") from error


def _unlock_and_close(descriptor: int) -> None:
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _write_exclusive(path: Path, raw: bytes) -> None:
    try:
        with path.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as error:
        raise RehostAnalysisSessionError("analysis artifact cannot be published") from error


def _atomic_publish(path: Path, raw: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise RehostAnalysisSessionError("analysis artifact already exists")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        _write_exclusive(temporary, raw)
        os.replace(temporary, path)
    except Exception:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def _remove_failed_start(directory: Path) -> None:
    if not directory.exists() or directory.is_symlink():
        return
    for child in sorted(directory.rglob("*"), reverse=True):
        try:
            child.unlink() if child.is_file() else child.rmdir()
        except OSError:
            pass
    try:
        directory.rmdir()
    except OSError:
        pass


def _inventory(directory: Path) -> set[str]:
    try:
        return {path.name for path in directory.iterdir()}
    except OSError as error:
        raise RehostAnalysisSessionError("analysis inventory is unavailable") from error


def _validate_active_inventory(directory: Path) -> None:
    expected = {".session.lock", "analysis.pending.json", "evidence"}
    if _inventory(directory) != expected:
        raise RehostAnalysisSessionError("pending analysis inventory is invalid")
    try:
        lock_metadata = (directory / ".session.lock").lstat()
        pending_metadata = (directory / "analysis.pending.json").lstat()
        evidence_metadata = (directory / "evidence").lstat()
    except OSError as error:
        raise RehostAnalysisSessionError("pending analysis inventory is unavailable") from error
    if (
        not stat.S_ISREG(lock_metadata.st_mode)
        or lock_metadata.st_nlink != 1
        or lock_metadata.st_size != 1
        or not stat.S_ISREG(pending_metadata.st_mode)
        or pending_metadata.st_nlink != 1
        or not stat.S_ISDIR(evidence_metadata.st_mode)
        or evidence_metadata.st_nlink < 1
    ):
        raise RehostAnalysisSessionError("pending analysis inventory is unsafe")


__all__ = [
    "RehostAnalysisSession",
    "RehostAnalysisSessionError",
]
