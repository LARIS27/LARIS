from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import json
import re

from laris_onpremise.attribution.log_linked.contracts import LogLinkedContractError
from laris_onpremise.attribution.log_linked.llama_server_log import (
    LlamaServerLogProfile,
    LlamaServerTraceParser,
)


_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_DEBUG_BODY_MARKERS = (
    b' request: {',
    b' request: [',
    b' response: {',
    b' response: [',
    b'"messages":',
    b'"choices":',
)
_MINIMUM_DISTINCT_RESPONSE_FRAGMENT_BYTES = 16


@dataclass(frozen=True)
class ExecutableIdentity:
    path: str
    digest: str
    version: str

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path.startswith("/") or "\0" in self.path:
            raise LogLinkedContractError("executable path must be absolute")
        if not isinstance(self.digest, str) or _DIGEST.fullmatch(self.digest) is None:
            raise LogLinkedContractError("executable digest is invalid")
        if not isinstance(self.version, str) or not self.version.strip() or "\0" in self.version:
            raise LogLinkedContractError("executable version is invalid")


@dataclass(frozen=True)
class PreflightRequest:
    profile: LlamaServerLogProfile
    launch_arguments: tuple[str, ...]
    expected_identity: ExecutableIdentity
    process_identity_probe: Callable[[], ExecutableIdentity]
    canary_probe: Callable[[str], None]
    log_reader: Callable[[], bytes]
    canary_text: str
    prompt_fragment: str
    response_fragment: str
    authorization_value: str

    def __post_init__(self) -> None:
        if not isinstance(self.profile, LlamaServerLogProfile):
            raise LogLinkedContractError("preflight profile is required")
        if not isinstance(self.launch_arguments, tuple) or not self.launch_arguments:
            raise LogLinkedContractError("launch_arguments must be a non-empty tuple")
        if any(not isinstance(value, str) or not value or "\0" in value for value in self.launch_arguments):
            raise LogLinkedContractError("launch_arguments contain invalid text")
        if not isinstance(self.expected_identity, ExecutableIdentity):
            raise LogLinkedContractError("expected executable identity is required")
        for name in ("process_identity_probe", "canary_probe", "log_reader"):
            if not callable(getattr(self, name)):
                raise LogLinkedContractError(f"{name} must be callable")
        for name in ("canary_text", "prompt_fragment", "response_fragment", "authorization_value"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or "\0" in value:
                raise LogLinkedContractError(f"{name} must be non-empty text")


@dataclass(frozen=True)
class PreflightReceipt:
    adapter_id: str
    passed: bool
    reasons: tuple[str, ...]
    log_digest: str
    event_count: int
    endpoint_event_count: int
    executable_identity: ExecutableIdentity
    launch_arguments_digest: str


class LogPrivacyPreflight:
    def __init__(self, parser: LlamaServerTraceParser) -> None:
        if not isinstance(parser, LlamaServerTraceParser):
            raise LogLinkedContractError("a llama-server trace parser is required")
        self.parser = parser

    def verify(self, request: PreflightRequest) -> PreflightReceipt:
        if request.profile != self.parser.profile:
            raise LogLinkedContractError("preflight and parser profiles differ")

        reasons: list[str] = []
        if not _safe_logging_arguments(request.launch_arguments, request.profile):
            reasons.append("UNSAFE_LOG_ARGUMENTS")

        request.canary_probe(request.canary_text)
        raw_log = request.log_reader()
        if not isinstance(raw_log, bytes):
            raise LogLinkedContractError("log_reader must return bytes")
        events = self.parser.parse(raw_log)
        endpoint_count = sum(
            event.event_type == "ACCESS" and event.endpoint_value is not None
            for event in events
        )
        if endpoint_count == 0:
            reasons.append("ENDPOINT_EVENT_MISSING")
        if _contains_private_content(raw_log, request):
            reasons.append("CONTENT_LEAK_DETECTED")

        identity = request.process_identity_probe()
        if not isinstance(identity, ExecutableIdentity):
            raise LogLinkedContractError(
                "process_identity_probe must return ExecutableIdentity"
            )
        if identity != request.expected_identity:
            reasons.append("EXECUTABLE_IDENTITY_MISMATCH")

        return PreflightReceipt(
            adapter_id=request.profile.adapter_id,
            passed=not reasons,
            reasons=tuple(dict.fromkeys(reasons)),
            log_digest="sha256:" + hashlib.sha256(raw_log).hexdigest(),
            event_count=len(events),
            endpoint_event_count=endpoint_count,
            executable_identity=identity,
            launch_arguments_digest="sha256:"
            + hashlib.sha256(
                json.dumps(
                    list(request.launch_arguments),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
        )


def _safe_logging_arguments(
    arguments: tuple[str, ...], profile: LlamaServerLogProfile
) -> bool:
    forbidden = {"-v", "--verbose", "--log-verbose", "--verbose-prompt", "--log-disable"}
    forbidden.update(profile.forbidden_arguments)
    if any(argument in forbidden for argument in arguments):
        return False
    if any(argument not in arguments for argument in profile.required_arguments):
        return False
    if any(argument == "--log-verbosity=5" for argument in arguments):
        return False

    verbosity_ok = False
    log_file_ok = False
    for index, argument in enumerate(arguments):
        if argument == "--log-verbosity":
            verbosity_ok = index + 1 < len(arguments) and arguments[index + 1] == "4"
        elif argument == "--log-verbosity=4":
            verbosity_ok = True
        elif argument.startswith("--log-verbosity=") and argument != "--log-verbosity=4":
            return False
        if argument == "--log-file":
            log_file_ok = (
                index + 1 < len(arguments)
                and bool(arguments[index + 1])
                and arguments[index + 1].startswith("/")
            )
        elif argument.startswith("--log-file="):
            log_file_ok = argument.removeprefix("--log-file=").startswith("/")
    return verbosity_ok and "--log-timestamps" in arguments and log_file_ok


def _contains_private_content(raw_log: bytes, request: PreflightRequest) -> bool:
    probes = [
        request.canary_text,
        request.prompt_fragment,
        request.authorization_value,
    ]
    response_fragment = request.response_fragment.encode("utf-8")
    if len(response_fragment) >= _MINIMUM_DISTINCT_RESPONSE_FRAGMENT_BYTES:
        probes.append(request.response_fragment)
    if any(value.encode("utf-8") in raw_log for value in probes):
        return True
    lowered = raw_log.lower()
    return any(marker in lowered for marker in _DEBUG_BODY_MARKERS)


__all__ = [
    "ExecutableIdentity",
    "LogPrivacyPreflight",
    "PreflightReceipt",
    "PreflightRequest",
]
