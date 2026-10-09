from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import ipaddress
import re

from laris_onpremise.attribution.log_linked.contracts import (
    LogLinkedContractError,
    canonical_document_bytes,
    parse_application_event,
)


_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,191}\Z")
_ISO_TIMESTAMP = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?Z)\s+"
)
_ELAPSED_TIMESTAMP = re.compile(
    r"^(?P<minutes>\d+)\.(?P<seconds>\d{2})\."
    r"(?P<milliseconds>\d{3})\.(?P<microseconds>\d{3})\s+"
)
_ACCESS = re.compile(
    r"\bdone request:\s+(?P<method>[A-Z]+)\s+(?P<path>\S+)\s+"
    r"(?P<endpoint>\S+)\s+(?P<status>\d{3})\s*$"
)
_SLOT_PREFIX = re.compile(
    r"\bslot\s+[^:]+:\s+id\s+(?P<slot>\d+)\s+\|\s+"
    r"task\s+(?P<task>\d+)\s+\|\s+(?P<message>.+)$"
)
_NEW_PROMPT = re.compile(
    r"\bnew prompt,.*(?:task\.n_tokens|n_prompt_tokens)\s*=\s*(?P<count>\d+)\s*$"
)
_REUSE = re.compile(r"\breusing chunk with size\s+(?P<count>\d+)\s*$")
_PROMPT_EVAL = re.compile(
    r"\bprompt eval time\s*=.*?/\s*(?P<count>\d+)\s+tokens(?:\s+.*)?$"
)
_TERMINAL = re.compile(
    r"\bstop processing:\s+n_tokens\s*=\s*(?P<count>\d+),\s*"
    r"truncated\s*=\s*(?P<truncated>[01])\s*$"
)
_RECOGNIZED_MARKERS = (
    "done request:",
    "new prompt,",
    "reusing chunk with size",
    "prompt eval time",
    "stop processing:",
)


@dataclass(frozen=True)
class LlamaServerLogProfile:
    adapter_id: str
    upstream_commit: str
    maximum_line_bytes: int
    maximum_log_bytes: int
    required_arguments: tuple[str, ...]
    forbidden_arguments: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.adapter_id, str) or _SAFE_ID.fullmatch(self.adapter_id) is None:
            raise LogLinkedContractError("adapter_id is invalid")
        if not isinstance(self.upstream_commit, str) or _COMMIT.fullmatch(self.upstream_commit) is None:
            raise LogLinkedContractError("upstream_commit must be a full lowercase commit")
        for name in ("maximum_line_bytes", "maximum_log_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise LogLinkedContractError(f"{name} must be a positive integer")
        if self.maximum_log_bytes < self.maximum_line_bytes:
            raise LogLinkedContractError("maximum_log_bytes cannot be smaller than one line")
        for name in ("required_arguments", "forbidden_arguments"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or not values:
                raise LogLinkedContractError(f"{name} must be a non-empty tuple")
            if any(not isinstance(value, str) or not value for value in values):
                raise LogLinkedContractError(f"{name} contains invalid text")


@dataclass(frozen=True)
class ApplicationEvent:
    schema_version: str
    adapter_id: str
    source_line_number: int
    source_order: int
    wall_time_utc_ns: int
    event_type: str
    method: str | None
    path_class: str | None
    status_code: int | None
    endpoint_type: str | None
    endpoint_value: str | None
    task_id: int | None
    slot_id: int | None
    prompt_tokens_total: int | None
    prompt_tokens_new: int | None
    prompt_tokens_reused: int | None
    generated_tokens: int | None
    terminal_outcome: str | None

    def __post_init__(self) -> None:
        parse_application_event(canonical_document_bytes(self.to_mapping()))

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "adapter_id": self.adapter_id,
            "source_line_number": self.source_line_number,
            "source_order": self.source_order,
            "wall_time_utc_ns": self.wall_time_utc_ns,
            "event_type": self.event_type,
            "method": self.method,
            "path_class": self.path_class,
            "status_code": self.status_code,
            "endpoint_type": self.endpoint_type,
            "endpoint_value": self.endpoint_value,
            "task_id": self.task_id,
            "slot_id": self.slot_id,
            "prompt_tokens_total": self.prompt_tokens_total,
            "prompt_tokens_new": self.prompt_tokens_new,
            "prompt_tokens_reused": self.prompt_tokens_reused,
            "generated_tokens": self.generated_tokens,
            "terminal_outcome": self.terminal_outcome,
        }


class LlamaServerTraceParser:
    def __init__(
        self,
        profile: LlamaServerLogProfile,
        *,
        elapsed_timestamp_origin_utc_ns: int | None = None,
    ) -> None:
        if not isinstance(profile, LlamaServerLogProfile):
            raise LogLinkedContractError("a llama-server log profile is required")
        if elapsed_timestamp_origin_utc_ns is not None and (
            isinstance(elapsed_timestamp_origin_utc_ns, bool)
            or not isinstance(elapsed_timestamp_origin_utc_ns, int)
            or elapsed_timestamp_origin_utc_ns < 0
        ):
            raise LogLinkedContractError("elapsed timestamp origin is invalid")
        self.profile = profile
        self.elapsed_timestamp_origin_utc_ns = elapsed_timestamp_origin_utc_ns

    def parse(self, raw_log: bytes) -> tuple[ApplicationEvent, ...]:
        if not isinstance(raw_log, bytes):
            raise LogLinkedContractError("raw log bytes are required")
        if len(raw_log) > self.profile.maximum_log_bytes:
            raise LogLinkedContractError("raw log exceeds maximum log size")
        if b"\0" in raw_log:
            raise LogLinkedContractError("raw log contains NUL bytes")
        try:
            text = raw_log.decode("utf-8")
        except UnicodeDecodeError as error:
            raise LogLinkedContractError("raw log is not UTF-8") from error

        events: list[ApplicationEvent] = []
        task_state: dict[tuple[int, int], dict[str, int]] = {}
        for line_number, line in enumerate(text.splitlines(), start=1):
            if len(line.encode("utf-8")) > self.profile.maximum_line_bytes:
                raise LogLinkedContractError("raw log line exceeds maximum line size")
            event = self._parse_line(
                line,
                line_number=line_number,
                source_order=line_number - 1,
                task_state=task_state,
            )
            if event is not None:
                events.append(event)
        return tuple(events)

    def _parse_line(
        self,
        line: str,
        *,
        line_number: int,
        source_order: int,
        task_state: dict[tuple[int, int], dict[str, int]],
    ) -> ApplicationEvent | None:
        iso_timestamp = _ISO_TIMESTAMP.match(line)
        elapsed_timestamp = _ELAPSED_TIMESTAMP.match(line)
        if iso_timestamp is None and elapsed_timestamp is None:
            if any(marker in line for marker in _RECOGNIZED_MARKERS):
                raise LogLinkedContractError(
                    f"recognized log event is malformed at line {line_number}"
                )
            return None
        if iso_timestamp is not None:
            wall_time_utc_ns = _parse_iso_timestamp_ns(
                iso_timestamp.group("timestamp")
            )
        else:
            if self.elapsed_timestamp_origin_utc_ns is None:
                raise LogLinkedContractError(
                    "official elapsed log timestamp requires an explicit timestamp origin"
                )
            wall_time_utc_ns = (
                self.elapsed_timestamp_origin_utc_ns
                + _parse_elapsed_timestamp_ns(elapsed_timestamp)
            )

        access = _ACCESS.search(line)
        if access is not None:
            endpoint = access.group("endpoint")
            endpoint_type = _classify_endpoint(endpoint)
            status = int(access.group("status"))
            return self._event(
                line_number=line_number,
                source_order=source_order,
                wall_time_utc_ns=wall_time_utc_ns,
                event_type="ACCESS",
                method=access.group("method"),
                path_class=_path_class(access.group("path")),
                status_code=status,
                endpoint_type=endpoint_type,
                endpoint_value=endpoint,
                terminal_outcome=_http_outcome(status),
            )

        slot = _SLOT_PREFIX.search(line)
        if slot is not None:
            slot_id = int(slot.group("slot"))
            task_id = int(slot.group("task"))
            message = slot.group("message")
            key = (slot_id, task_id)
            state = task_state.setdefault(key, {})
            match = _NEW_PROMPT.search(message)
            if match is not None:
                count = int(match.group("count"))
                state["prompt_total"] = count
                return self._event(
                    line_number=line_number,
                    source_order=source_order,
                    wall_time_utc_ns=wall_time_utc_ns,
                    event_type="SLOT_PROMPT",
                    task_id=task_id,
                    slot_id=slot_id,
                    prompt_tokens_total=count,
                )
            match = _REUSE.search(message)
            if match is not None:
                count = int(match.group("count"))
                state["reused"] = count
                return self._event(
                    line_number=line_number,
                    source_order=source_order,
                    wall_time_utc_ns=wall_time_utc_ns,
                    event_type="SLOT_REUSE",
                    task_id=task_id,
                    slot_id=slot_id,
                    prompt_tokens_reused=count,
                )
            match = _PROMPT_EVAL.search(message)
            if match is not None:
                count = int(match.group("count"))
                state["prompt_new"] = count
                return self._event(
                    line_number=line_number,
                    source_order=source_order,
                    wall_time_utc_ns=wall_time_utc_ns,
                    event_type="SLOT_PROMPT_EVAL",
                    task_id=task_id,
                    slot_id=slot_id,
                    prompt_tokens_new=count,
                )
            match = _TERMINAL.search(message)
            if match is not None:
                total = int(match.group("count"))
                prompt_total = state.get("prompt_total")
                generated = None if prompt_total is None else max(0, total - prompt_total)
                task_state.pop(key, None)
                return self._event(
                    line_number=line_number,
                    source_order=source_order,
                    wall_time_utc_ns=wall_time_utc_ns,
                    event_type="SLOT_TERMINAL",
                    task_id=task_id,
                    slot_id=slot_id,
                    generated_tokens=generated,
                    terminal_outcome="RELEASED",
                )

        if any(marker in line for marker in _RECOGNIZED_MARKERS):
            raise LogLinkedContractError(
                f"recognized log event is malformed at line {line_number}"
            )
        return None

    def _event(self, **values: object) -> ApplicationEvent:
        defaults: dict[str, object] = {
            "schema_version": "1.0",
            "adapter_id": self.profile.adapter_id,
            "method": None,
            "path_class": None,
            "status_code": None,
            "endpoint_type": None,
            "endpoint_value": None,
            "task_id": None,
            "slot_id": None,
            "prompt_tokens_total": None,
            "prompt_tokens_new": None,
            "prompt_tokens_reused": None,
            "generated_tokens": None,
            "terminal_outcome": None,
        }
        if "line_number" in values:
            values["source_line_number"] = values.pop("line_number")
        defaults.update(values)
        return ApplicationEvent(**defaults)


def _parse_iso_timestamp_ns(value: str) -> int:
    try:
        parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError as error:
        raise LogLinkedContractError("log timestamp is invalid") from error
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    seconds = int(parsed.timestamp())
    fraction = value.split(".", 1)[1].removesuffix("Z") if "." in value else ""
    nanos = int((fraction + "0" * 9)[:9]) if fraction else 0
    return seconds * 1_000_000_000 + nanos


def _parse_elapsed_timestamp_ns(match: re.Match[str]) -> int:
    minutes = int(match.group("minutes"))
    seconds = int(match.group("seconds"))
    milliseconds = int(match.group("milliseconds"))
    microseconds = int(match.group("microseconds"))
    if seconds >= 60:
        raise LogLinkedContractError("official elapsed log timestamp is invalid")
    total_microseconds = (
        ((minutes * 60 + seconds) * 1_000 + milliseconds) * 1_000
        + microseconds
    )
    return total_microseconds * 1_000


def _classify_endpoint(value: str) -> str:
    try:
        address = ipaddress.ip_address(value)
    except ValueError as error:
        raise LogLinkedContractError("recognized log event is malformed: endpoint") from error
    return "IPV4" if address.version == 4 else "IPV6"


def _path_class(value: str) -> str:
    path = value.split("?", 1)[0]
    known = {
        "/v1/chat/completions": "CHAT_COMPLETIONS",
        "/chat/completions": "CHAT_COMPLETIONS",
        "/completion": "COMPLETION",
        "/v1/completions": "COMPLETION",
        "/v1/responses": "RESPONSES",
    }
    return known.get(path, "OTHER")


def _http_outcome(status: int) -> str:
    if status == 499:
        return "CANCELLED"
    return "COMPLETED" if 200 <= status < 400 else "ERROR"


__all__ = [
    "ApplicationEvent",
    "LlamaServerLogProfile",
    "LlamaServerTraceParser",
]
