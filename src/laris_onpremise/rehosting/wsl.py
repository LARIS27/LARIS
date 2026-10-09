from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
from typing import Callable, Mapping
import uuid

from laris_onpremise.core.records import canonical_json
from laris_onpremise.rehosting.catalog import (
    PortableBundleCatalog,
    PortableBundleEntry,
)
from laris_onpremise.rehosting.contracts import InvestigatorSampler
from laris_onpremise.rehosting.host_profile import InvestigatorProfile


class WslExecutionError(RuntimeError):
    """Raised when a selected portable bundle cannot be rehosted through WSL."""


_ANALYSIS_ID = re.compile(r"analysis-[A-Za-z0-9_-]{1,64}\Z")
_MAX_RESULT_BYTES = 2 * 1024 * 1024
_MAX_DIAGNOSTIC_BYTES = 64 * 1024
_MAX_MESSAGES_BYTES = 128 * 1024
_MAX_MESSAGES = 128


@dataclass(frozen=True)
class WslRehostRequest:
    bundle_root: Path
    bundle_manifest_digest: str
    entry: PortableBundleEntry
    model: Path
    messages_file: Path
    destination_seq_id: int = 0
    timeout_seconds: int = 120
    predict: int = 64
    sampler: InvestigatorSampler = InvestigatorSampler()


@dataclass(frozen=True)
class WslRehostResult:
    result: Mapping[str, object]


Runner = Callable[..., subprocess.CompletedProcess[bytes]]


class WslExecutor:
    def __init__(
        self,
        *,
        runner: Runner = subprocess.run,
        analysis_id_factory: Callable[[], str] | None = None,
    ) -> None:
        self._runner = runner
        self._analysis_id_factory = analysis_id_factory or (
            lambda: "analysis-" + uuid.uuid4().hex
        )

    def execute(
        self,
        profile: InvestigatorProfile,
        request: WslRehostRequest,
    ) -> WslRehostResult:
        analysis_id = self._analysis_id_factory()
        if _ANALYSIS_ID.fullmatch(analysis_id) is None:
            raise WslExecutionError("analysis identifier is invalid")
        try:
            PortableBundleCatalog.open(
                request.bundle_root,
                expected_manifest_digest=request.bundle_manifest_digest,
            )
            _read_messages(request.messages_file)
            bundle_path = self._translate(profile, request.bundle_root)
            model_path = self._translate(profile, request.model)
            messages_path = self._translate(profile, request.messages_file)
            wsl_analysis = str(PurePosixPath(profile.wsl.work_root) / analysis_id)
        except Exception as error:
            if isinstance(error, WslExecutionError):
                raise
            raise WslExecutionError("WSL rehost execution failed") from error

        execution_error: WslExecutionError | None = None
        result: dict[str, object] | None = None
        try:
            completed = self._run(
                [
                    "wsl.exe",
                    "-d",
                    profile.wsl.distribution,
                    "--exec",
                    profile.wsl.python,
                    "-I",
                    "-B",
                    "-m",
                    "laris_onpremise.rehosting.command",
                    "--bundle-root",
                    bundle_path,
                    "--bundle-manifest-digest",
                    request.bundle_manifest_digest,
                    "--bundle-id",
                    request.entry.bundle_id,
                    "--helper",
                    profile.wsl.helper,
                    "--helper-digest",
                    profile.wsl.helper_digest,
                    "--libllama",
                    profile.wsl.libllama,
                    "--model",
                    model_path,
                    "--model-profile-id",
                    request.entry.model_profile_id,
                    "--runtime-profile",
                    profile.wsl.runtime_profile,
                    "--messages-file",
                    messages_path,
                    "--analysis-root",
                    wsl_analysis,
                    "--destination-seq-id",
                    str(request.destination_seq_id),
                    "--timeout-seconds",
                    str(request.timeout_seconds),
                    "--predict",
                    str(request.predict),
                    "--temperature",
                    str(request.sampler.temperature),
                    "--top-k",
                    str(request.sampler.top_k),
                    "--top-p",
                    str(request.sampler.top_p),
                    "--min-p",
                    str(request.sampler.min_p),
                    "--seed",
                    str(request.sampler.seed),
                ],
                timeout=request.timeout_seconds + 30,
            )
            if completed.returncode != 0:
                raise WslExecutionError(
                    f"WSL rehost worker failed with exit code {completed.returncode}"
                )
            PortableBundleCatalog.open(
                request.bundle_root,
                expected_manifest_digest=request.bundle_manifest_digest,
            )
            result = _parse_result(completed.stdout, request)
        except Exception as error:
            if not isinstance(error, WslExecutionError):
                error = WslExecutionError("WSL rehost execution failed")
            execution_error = error
        try:
            self._cleanup(profile, wsl_analysis)
        except WslExecutionError:
            if execution_error is None:
                raise
        if execution_error is not None:
            raise execution_error
        if result is None:
            raise WslExecutionError("WSL rehost execution produced no result")
        return WslRehostResult(result)

    def _translate(self, profile: InvestigatorProfile, path: Path) -> str:
        if not isinstance(path, Path) or not path.is_absolute():
            raise WslExecutionError("WSL input path must be absolute")
        completed = self._run(
            [
                "wsl.exe",
                "-d",
                profile.wsl.distribution,
                "--exec",
                "/usr/bin/wslpath",
                "-a",
                "-u",
                str(path),
            ],
            timeout=15,
        )
        if completed.returncode != 0:
            raise WslExecutionError("WSL path translation failed")
        try:
            text = completed.stdout.decode("utf-8", errors="strict")
        except UnicodeError as error:
            raise WslExecutionError("WSL path translation is invalid") from error
        lines = text.splitlines()
        if (
            len(lines) != 1
            or not lines[0].startswith("/")
            or "\x00" in lines[0]
            or str(PurePosixPath(lines[0])) != lines[0]
        ):
            raise WslExecutionError("WSL path translation is ambiguous")
        return lines[0]

    def _cleanup(self, profile: InvestigatorProfile, path: str) -> None:
        expected_parent = PurePosixPath(profile.wsl.work_root)
        candidate = PurePosixPath(path)
        if candidate.parent != expected_parent or _ANALYSIS_ID.fullmatch(candidate.name) is None:
            raise WslExecutionError("WSL cleanup target is invalid")
        completed = self._run(
            [
                "wsl.exe",
                "-d",
                profile.wsl.distribution,
                "--exec",
                "/usr/bin/rm",
                "-rf",
                "--",
                str(candidate),
            ],
            timeout=15,
        )
        if completed.returncode != 0:
            raise WslExecutionError("WSL analysis cleanup failed")

    def _run(self, argv: list[str], *, timeout: int) -> subprocess.CompletedProcess[bytes]:
        try:
            completed = self._runner(
                argv,
                capture_output=True,
                text=False,
                check=False,
                timeout=timeout,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise WslExecutionError("WSL command failed") from error
        if len(completed.stdout) > _MAX_RESULT_BYTES or len(completed.stderr) > _MAX_DIAGNOSTIC_BYTES:
            raise WslExecutionError("WSL command output exceeds the bounded limit")
        return completed


def _read_messages(path: Path) -> bytes:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise WslExecutionError("messages file is unavailable") from error
    if not raw or len(raw) > _MAX_MESSAGES_BYTES or b"\x00" in raw:
        raise WslExecutionError("messages file is invalid")

    def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise WslExecutionError("messages file has duplicate fields")
            value[key] = item
        return value

    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=unique,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise WslExecutionError("messages file is invalid") from error
    if (
        not isinstance(value, dict)
        or set(value) != {"messages", "schema_version"}
        or value.get("schema_version") != "1.0"
        or raw != canonical_json(value) + b"\n"
    ):
        raise WslExecutionError("messages file is not canonical")
    items = value.get("messages")
    if not isinstance(items, list) or not 1 <= len(items) <= _MAX_MESSAGES:
        raise WslExecutionError("messages file is invalid")
    for item in items:
        if (
            not isinstance(item, dict)
            or set(item) != {"content", "role"}
            or item.get("role") not in {"user", "assistant"}
            or not isinstance(item.get("content"), str)
            or not item["content"].strip()
            or "\x00" in item["content"]
        ):
            raise WslExecutionError("messages file is invalid")
    return raw


def _parse_result(raw: bytes, request: WslRehostRequest) -> dict[str, object]:
    try:
        value = json.loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise WslExecutionError("WSL result is invalid") from error
    if not isinstance(value, dict) or raw != canonical_json(value) + b"\n":
        raise WslExecutionError("WSL result is not canonical")
    case = value.get("case")
    receipt = value.get("receipt")
    source_runtime = value.get("source_runtime")
    effective_runtime = value.get("effective_runtime")
    inquiry_policy = value.get("inquiry_policy")
    if (
        set(value)
        != {
            "case",
            "effective_runtime",
            "inquiry_policy",
            "model_digest",
            "prompt_contract",
            "receipt",
            "schema_version",
            "source_runtime",
            "status",
        }
        or value.get("schema_version") != "1.0"
        or value.get("status") != "COMPLETE"
        or value.get("prompt_contract") != "gguf_default"
        or value.get("model_digest") != request.entry.model_digest
        or not isinstance(case, dict)
        or not isinstance(receipt, dict)
        or not isinstance(source_runtime, dict)
        or not isinstance(effective_runtime, dict)
        or set(effective_runtime) != {"n_batch", "n_predict", "n_ubatch"}
        or not isinstance(inquiry_policy, dict)
        or set(inquiry_policy) != {"sampler"}
        or not isinstance(inquiry_policy.get("sampler"), dict)
        or case.get("bundle_id") != request.entry.bundle_id
        or case.get("model_digest") != request.entry.model_digest
        or case.get("payload_digest") != request.entry.payload_digest
        or case.get("position_range") != list(request.entry.position_range)
        or receipt.get("exit_code") != 0
        or receipt.get("restored_pos_min") != request.entry.position_range[0]
        or receipt.get("restored_pos_max") != request.entry.position_range[1]
    ):
        raise WslExecutionError("WSL result binding is invalid")
    return value


__all__ = [
    "WslExecutionError",
    "WslExecutor",
    "WslRehostRequest",
    "WslRehostResult",
]
