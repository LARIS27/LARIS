from __future__ import annotations

import hashlib
import hmac
import os
from pathlib import Path
import stat
import subprocess
import time

from laris_onpremise.bundle.rehost_metadata import RehostRuntimeMetadata

from .contracts import (
    HelperReceipt,
    InvestigatorSampler,
    RehostCase,
    RehostRuntimeProfile,
    RehostSubjectKind,
    RehostValidationError,
)
from .wrapper import DerivedStateError, materialize_derived_wrapper


class RehostExecutionError(RuntimeError):
    """Raised when a verified runtime-state payload cannot be rehosted safely."""


_RECEIPT_FIELDS = {
    "mapped_libllama_device",
    "mapped_libllama_inode",
    "restored_bytes",
    "restored_pos_min",
    "restored_pos_max",
    "appended_token_count",
    "logical_batch_count",
    "effective_n_batch",
    "effective_n_ubatch",
    "generated_token_count",
    "prompt_contract",
    "sampler_temperature",
    "sampler_top_k",
    "sampler_top_p",
    "sampler_min_p",
    "sampler_seed",
}
_UNTRUSTED_ENVIRONMENT = {
    "DYLD_INSERT_LIBRARIES",
    "DYLD_LIBRARY_PATH",
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
    "PYTHONHOME",
    "PYTHONPATH",
}


def _metadata(stderr: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in stderr.splitlines():
        if not line.startswith("LARIS_META "):
            continue
        payload = line.removeprefix("LARIS_META ")
        if "=" not in payload:
            raise RehostExecutionError("helper metadata is malformed")
        key, value = payload.split("=", 1)
        if key in values or key not in _RECEIPT_FIELDS or not value:
            raise RehostExecutionError("helper metadata is ambiguous or unsupported")
        values[key] = value
    return values


def _integer_metadata(values: dict[str, str], name: str) -> int:
    value = values.get(name)
    if value is None:
        raise RehostExecutionError(f"helper metadata is missing {name}")
    try:
        parsed = int(value)
    except ValueError as error:
        raise RehostExecutionError(f"helper {name} is invalid") from error
    if parsed < 0:
        raise RehostExecutionError(f"helper {name} is negative")
    return parsed


def _float_metadata(values: dict[str, str], name: str) -> float:
    value = values.get(name)
    if value is None:
        raise RehostExecutionError(f"helper metadata is missing {name}")
    try:
        parsed = float(value)
    except ValueError as error:
        raise RehostExecutionError(f"helper {name} is invalid") from error
    if not parsed > 0 or not parsed < float("inf"):
        raise RehostExecutionError(f"helper {name} is invalid")
    return parsed


def digest_file(path: Path) -> str:
    path = Path(path)
    try:
        before = path.lstat()
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise RehostExecutionError(f"{path.name} is not a regular file")
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
    except RehostExecutionError:
        raise
    except OSError as error:
        raise RehostExecutionError(f"cannot open {path.name}") from error
    digest = hashlib.sha256()
    try:
        observed = os.fstat(descriptor)
        if not stat.S_ISREG(observed.st_mode):
            raise RehostExecutionError(f"{path.name} is not a regular file")
        size = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        size != after.st_size
        or (before.st_dev, before.st_ino) != (observed.st_dev, observed.st_ino)
        or (observed.st_size, observed.st_mtime_ns, observed.st_ctime_ns)
        != (after.st_size, after.st_mtime_ns, after.st_ctime_ns)
    ):
        raise RehostExecutionError(f"{path.name} changed while read")
    return "sha256:" + digest.hexdigest()


def _is_within(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def _is_reparse(metadata: os.stat_result) -> bool:
    flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(getattr(metadata, "st_file_attributes", 0) & flag)


def _verified_run_root(case: RehostCase) -> Path:
    session_root = case.payload_path.parent
    expected_session_prefix = hashlib.sha256(
        case.session_ref.encode("utf-8")
    ).hexdigest()
    portable_sessions_root = session_root.parent
    portable_subject_root = portable_sessions_root.parent
    portable_subjects_root = portable_subject_root.parent
    portable_bundle_root = portable_subjects_root.parent
    portable_run_root = portable_bundle_root.parent
    identity_matches = (
        session_root.name.startswith("s_")
        and expected_session_prefix.startswith(session_root.name[2:])
        and 16 <= len(session_root.name[2:]) <= 64
    )
    portable_user_matches = (
        case.subject_kind is RehostSubjectKind.USER_HMAC
        and portable_sessions_root.name == "sessions"
        and portable_subject_root.name == f"uh_{case.user_hmac}"
        and portable_subjects_root.name == "users"
        and portable_bundle_root.name == "bundle"
        and portable_run_root.name == case.run_id
    )
    portable_endpoint_matches = (
        case.subject_kind is RehostSubjectKind.ENDPOINT_REF
        and portable_sessions_root.name == "sessions"
        and portable_subject_root.name == f"ep_{case.user_hmac}"
        and portable_subjects_root.name == "endpoints"
        and portable_bundle_root.name == "bundle"
        and portable_run_root.name == case.run_id
    )
    if not identity_matches or not (
        portable_user_matches or portable_endpoint_matches
    ):
        raise RehostExecutionError("payload path is not bound to the evidence run")
    try:
        return portable_run_root.resolve(strict=True)
    except OSError as error:
        raise RehostExecutionError("evidence run root is unavailable") from error


def _prepare_analysis_root(case: RehostCase, analysis_root: Path) -> Path:
    analysis_root = Path(analysis_root)
    if not analysis_root.is_absolute():
        raise RehostExecutionError("analysis root must be absolute")
    run_root = _verified_run_root(case)
    try:
        resolved = analysis_root.resolve(strict=False)
    except OSError as error:
        raise RehostExecutionError("analysis root cannot be resolved") from error
    if _is_within(resolved, run_root) or _is_within(run_root, resolved):
        raise RehostExecutionError("analysis root must be disjoint from evidence run")

    probe = analysis_root
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    while True:
        try:
            metadata = probe.lstat()
        except OSError as error:
            raise RehostExecutionError("analysis root ancestry is unavailable") from error
        if stat.S_ISLNK(metadata.st_mode) or _is_reparse(metadata):
            raise RehostExecutionError("analysis root contains a reparse point")
        if probe == probe.parent:
            break
        probe = probe.parent

    try:
        analysis_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        metadata = analysis_root.lstat()
    except OSError as error:
        raise RehostExecutionError("analysis root cannot be created") from error
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or _is_reparse(metadata)
    ):
        raise RehostExecutionError("analysis root is not a private directory")
    if os.name == "posix" and metadata.st_mode & 0o077:
        raise RehostExecutionError("analysis root permissions are not private")
    return analysis_root


def _sanitized_environment(libllama: Path) -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if key.upper() not in _UNTRUSTED_ENVIRONMENT
    }
    environment["LC_ALL"] = "C"
    environment["LD_LIBRARY_PATH"] = str(libllama.parent)
    return environment


def invoke_helper(
    helper: Path,
    model: Path,
    messages_file: Path,
    case: RehostCase,
    *,
    runtime_profile: RehostRuntimeProfile,
    runtime: RehostRuntimeMetadata,
    sampler: InvestigatorSampler,
    libllama: Path,
    expected_helper_digest: str,
    analysis_root: Path,
    timeout_seconds: int,
    n_predict: int = 64,
) -> HelperReceipt:
    """Materialize an analysis-only wrapper and invoke a fresh llama context."""

    if not isinstance(case, RehostCase):
        raise RehostExecutionError("case must be a verified RehostCase")
    if not isinstance(runtime_profile, RehostRuntimeProfile):
        raise RehostExecutionError("runtime profile is not qualified")
    if not isinstance(runtime, RehostRuntimeMetadata):
        raise RehostExecutionError("rehost runtime metadata is unavailable")
    if runtime.source_runtime_profile_digest != case.runtime_profile_digest:
        raise RehostExecutionError("rehost runtime metadata differs from the bundle")
    if not isinstance(sampler, InvestigatorSampler):
        raise RehostExecutionError("investigator sampler is invalid")
    try:
        runtime_profile.verify_case(case)
    except RehostValidationError as error:
        raise RehostExecutionError("runtime profile differs from the bundle") from error
    if timeout_seconds <= 0 or n_predict <= 0:
        raise RehostExecutionError("timeout and predict must be positive")
    if runtime.n_ubatch > runtime.n_batch:
        raise RehostExecutionError("batch topology is invalid")
    helper = Path(helper)
    model = Path(model)
    messages_file = Path(messages_file)
    libllama = Path(libllama)
    for name, path in (
        ("helper", helper),
        ("model", model),
        ("messages", messages_file),
        ("libllama", libllama),
    ):
        try:
            metadata = path.lstat()
        except OSError as error:
            raise RehostExecutionError(f"{name} file is unavailable") from error
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise RehostExecutionError(f"{name} must be a regular file")
    if not hmac.compare_digest(digest_file(model), case.model_digest):
        raise RehostExecutionError("model digest differs from the rehost bundle")
    if not hmac.compare_digest(
        digest_file(helper), expected_helper_digest
    ):
        raise RehostExecutionError("helper digest is not qualified")
    if not hmac.compare_digest(
        digest_file(libllama), runtime_profile.mapped_libllama_digest
    ):
        raise RehostExecutionError("libllama digest is not qualified")
    try:
        qualified_libllama_identity = libllama.lstat()
    except OSError as error:
        raise RehostExecutionError("qualified libllama identity is unavailable") from error

    case_root = _prepare_analysis_root(case, Path(analysis_root)) / case.bundle_id
    try:
        derived = materialize_derived_wrapper(
            payload_path=case.payload_path,
            expected_payload_digest=case.payload_digest,
            destination_seq_id=case.destination_seq_id,
            derived_dir=case_root / "derived",
        )
    except DerivedStateError as error:
        raise RehostExecutionError("derived runtime state could not be created") from error

    command = [
        str(helper),
        "--model", str(model),
        "--wrapped-state", str(derived.path),
        "--expected-libllama", str(libllama),
        "--destination-seq-id", str(case.destination_seq_id),
        "--v-trans", "true" if case.v_trans else "false",
        "--next-position", str(case.next_position),
        "--messages-file", str(messages_file),
        "--ctx", str(case.n_ctx),
        "--parallel", str(case.n_seq_max),
        "--batch", str(runtime.n_batch),
        "--ubatch", str(runtime.n_ubatch),
        "--predict", str(n_predict),
        "--temperature", str(sampler.temperature),
        "--top-k", str(sampler.top_k),
        "--top-p", str(sampler.top_p),
        "--min-p", str(sampler.min_p),
        "--seed", str(sampler.seed),
    ]
    if os.name == "nt" and helper.suffix.casefold() in {".cmd", ".bat"}:
        command = [os.environ.get("COMSPEC", "cmd.exe"), "/d", "/c", *command]
    started = time.monotonic_ns()
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
            timeout=timeout_seconds,
            check=False,
            env=_sanitized_environment(libllama),
        )
    except (OSError, subprocess.TimeoutExpired, UnicodeError) as error:
        raise RehostExecutionError("helper execution failed") from error
    duration_ms = (time.monotonic_ns() - started) // 1_000_000
    metadata = _metadata(completed.stderr)
    if completed.returncode != 0:
        return HelperReceipt(
            exit_code=completed.returncode,
            response=completed.stdout.strip(),
            stderr=completed.stderr,
            mapped_libllama_device=None,
            mapped_libllama_inode=None,
            restored_bytes=None,
            restored_pos_min=None,
            restored_pos_max=None,
            appended_token_count=None,
            logical_batch_count=None,
            effective_n_batch=None,
            effective_n_ubatch=None,
            generated_token_count=None,
            prompt_contract=None,
            sampler=None,
            duration_ms=duration_ms,
        )
    if set(metadata) != _RECEIPT_FIELDS:
        raise RehostExecutionError("successful helper metadata field set is invalid")
    mapped_device = _integer_metadata(metadata, "mapped_libllama_device")
    mapped_inode = _integer_metadata(metadata, "mapped_libllama_inode")
    restored_bytes = _integer_metadata(metadata, "restored_bytes")
    restored_pos_min = _integer_metadata(metadata, "restored_pos_min")
    restored_pos_max = _integer_metadata(metadata, "restored_pos_max")
    appended = _integer_metadata(metadata, "appended_token_count")
    logical_batches = _integer_metadata(metadata, "logical_batch_count")
    effective_n_batch = _integer_metadata(metadata, "effective_n_batch")
    effective_n_ubatch = _integer_metadata(metadata, "effective_n_ubatch")
    generated = _integer_metadata(metadata, "generated_token_count")
    prompt_contract = metadata.get("prompt_contract")
    observed_sampler = InvestigatorSampler(
        temperature=_float_metadata(metadata, "sampler_temperature"),
        top_k=_integer_metadata(metadata, "sampler_top_k"),
        top_p=_float_metadata(metadata, "sampler_top_p"),
        min_p=_float_metadata(metadata, "sampler_min_p"),
        seed=_integer_metadata(metadata, "sampler_seed"),
    )
    if (
        mapped_device != qualified_libllama_identity.st_dev
        or mapped_inode != qualified_libllama_identity.st_ino
    ):
        raise RehostExecutionError(
            "helper mapped libllama differs from the qualified file"
        )
    if not hmac.compare_digest(
        digest_file(libllama), runtime_profile.mapped_libllama_digest
    ):
        raise RehostExecutionError("qualified libllama changed during execution")
    if restored_bytes != derived.size_bytes:
        raise RehostExecutionError("helper restored byte count differs from wrapper")
    if (
        restored_pos_min != case.position_range[0]
        or restored_pos_max != case.position_range[1]
    ):
        raise RehostExecutionError("helper restored position range is invalid")
    if appended <= 0:
        raise RehostExecutionError("helper did not append the investigation question")
    if (
        logical_batches != (appended + runtime.n_batch - 1) // runtime.n_batch
        or effective_n_batch != runtime.n_batch
        or effective_n_ubatch != runtime.n_ubatch
        or prompt_contract != "gguf_default"
        or observed_sampler != sampler
    ):
        raise RehostExecutionError("helper inquiry policy differs from the request")
    response = completed.stdout.strip()
    if generated <= 0 or not response:
        raise RehostExecutionError("helper did not produce a generated response")
    return HelperReceipt(
        exit_code=0,
        response=response,
        stderr=completed.stderr,
        mapped_libllama_device=mapped_device,
        mapped_libllama_inode=mapped_inode,
        restored_bytes=restored_bytes,
        restored_pos_min=restored_pos_min,
        restored_pos_max=restored_pos_max,
        appended_token_count=appended,
        logical_batch_count=logical_batches,
        effective_n_batch=effective_n_batch,
        effective_n_ubatch=effective_n_ubatch,
        generated_token_count=generated,
        prompt_contract=prompt_contract,
        sampler=observed_sampler,
        duration_ms=duration_ms,
    )


__all__ = [
    "RehostExecutionError",
    "digest_file",
    "invoke_helper",
]
