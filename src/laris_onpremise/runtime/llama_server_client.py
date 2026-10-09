from __future__ import annotations

import http.client
import json
import socket
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from urllib.parse import urlsplit

from laris_onpremise.core.records import JsonValue, RecordValidationError, canonical_json


class UpstreamError(RuntimeError):
    """Base class for pinned llama-server boundary failures."""


class UpstreamContractError(UpstreamError):
    """Raised when an upstream response violates the pinned contract."""


class UpstreamTransportError(UpstreamError):
    """Raised when a request cannot be sent or a response cannot be read."""


class UpstreamTimeoutError(UpstreamTransportError):
    """Raised when a slot does not reach the required state by a deadline."""


class UpstreamHttpError(UpstreamError):
    """Raised for a non-success HTTP status without retaining its body."""

    def __init__(self, status: int) -> None:
        self.status = status
        super().__init__(f"upstream returned HTTP status {status}")


@dataclass(frozen=True)
class HttpSendEvidence:
    evidence_ref: str
    monotonic_time: float
    monotonic_ns: int


@dataclass(frozen=True)
class CompletionResult:
    id_slot: int
    content: str
    terminal: bool
    response_evidence_ref: str
    tokens_evaluated: int | None = None
    tokens_predicted: int | None = None


@dataclass(frozen=True)
class CompletionStreamEvent:
    id_slot: int
    content: str
    terminal: bool
    response_evidence_ref: str


@dataclass(frozen=True)
class SlotStatus:
    slot_id: int
    id_task: int | None
    is_processing: bool


@dataclass(frozen=True)
class SlotReleaseEvidence:
    slot_id: int
    id_task: int | None
    confirmed_monotonic_ns: int
    evidence_ref: str


@dataclass(frozen=True)
class SlotEraseEvidence:
    slot_id: int
    n_erased: int
    confirmed_monotonic_ns: int
    evidence_ref: str


def _parse_json_no_duplicates(raw: bytes) -> object:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise UpstreamContractError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        return json.loads(raw, object_pairs_hook=reject_duplicates)
    except UpstreamContractError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise UpstreamContractError("upstream returned invalid JSON") from error


def _require_object(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise UpstreamContractError(f"{context} must be a JSON object")
    return value


def _require_integer(
    value: object, field: str, *, minimum: int | None = None
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise UpstreamContractError(f"{field} must be an integer")
    if minimum is not None and value < minimum:
        raise UpstreamContractError(f"{field} must be at least {minimum}")
    return value


def _require_string(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise UpstreamContractError(f"{field} must be a string")
    return value


class LlamaServerClient:
    """Strict HTTP client for the pinned non-OAI llama-server endpoints."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 30.0,
        slot_poll_interval_seconds: float = 0.01,
        max_body_bytes: int = 16 * 1024 * 1024,
        slots_body_limit_bytes: int = 256 * 1024,
    ) -> None:
        parsed = urlsplit(base_url)
        if parsed.scheme != "http" or not parsed.hostname:
            raise ValueError("base_url must be an http URL with a host")
        if parsed.path not in ("", "/") or parsed.query or parsed.fragment:
            raise ValueError("base_url must not contain a path, query, or fragment")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if slot_poll_interval_seconds < 0:
            raise ValueError("slot_poll_interval_seconds must be non-negative")
        if max_body_bytes <= 0:
            raise ValueError("max_body_bytes must be positive")
        if slots_body_limit_bytes <= 0:
            raise ValueError("slots_body_limit_bytes must be positive")
        self._host = parsed.hostname
        self._port = parsed.port or 80
        self.timeout_seconds = timeout_seconds
        self.slot_poll_interval_seconds = slot_poll_interval_seconds
        self.max_body_bytes = max_body_bytes
        self.slots_body_limit_bytes = min(
            slots_body_limit_bytes,
            max_body_bytes,
        )

    def _connection(self) -> http.client.HTTPConnection:
        return http.client.HTTPConnection(
            self._host,
            self._port,
            timeout=self.timeout_seconds,
        )

    @staticmethod
    def _send_evidence() -> HttpSendEvidence:
        monotonic_ns = time.monotonic_ns()
        return HttpSendEvidence(
            evidence_ref=f"http-send:{monotonic_ns}",
            monotonic_time=monotonic_ns / 1_000_000_000,
            monotonic_ns=monotonic_ns,
        )

    def _start_request(
        self,
        method: str,
        path: str,
        payload: Mapping[str, JsonValue] | None,
        on_request_sent: Callable[[HttpSendEvidence], None] | None,
    ) -> tuple[http.client.HTTPConnection, http.client.HTTPResponse]:
        connection = self._connection()
        try:
            if payload is None:
                body = None
                headers: dict[str, str] = {"Accept": "application/json"}
            else:
                try:
                    body = canonical_json(payload)
                except RecordValidationError as error:
                    raise UpstreamContractError(
                        "request payload is not canonical JSON"
                    ) from error
                headers = {
                    "Accept": "application/json, text/event-stream",
                    "Content-Type": "application/json",
                }
            evidence = self._send_evidence()
            connection.request(method, path, body=body, headers=headers)
            if on_request_sent is not None:
                on_request_sent(evidence)
            response = connection.getresponse()
            if not 200 <= response.status < 300:
                response.read(self.max_body_bytes + 1)
                raise UpstreamHttpError(response.status)
            return connection, response
        except UpstreamError:
            connection.close()
            raise
        except (OSError, socket.timeout, http.client.HTTPException) as error:
            connection.close()
            raise UpstreamTransportError("upstream HTTP transport failed") from error
        except Exception:
            connection.close()
            raise

    def _read_body(
        self,
        response: http.client.HTTPResponse,
        *,
        body_limit_bytes: int | None = None,
    ) -> bytes:
        limit = self.max_body_bytes if body_limit_bytes is None else body_limit_bytes
        try:
            raw = response.read(limit + 1)
        except (OSError, socket.timeout, http.client.HTTPException) as error:
            raise UpstreamTransportError("failed to read upstream response") from error
        if len(raw) > limit:
            raise UpstreamContractError("upstream response exceeds body limit")
        return raw

    def _request_json(
        self,
        method: str,
        path: str,
        payload: Mapping[str, JsonValue] | None = None,
        *,
        on_request_sent: Callable[[HttpSendEvidence], None] | None = None,
        body_limit_bytes: int | None = None,
    ) -> object:
        connection, response = self._start_request(
            method, path, payload, on_request_sent
        )
        try:
            return _parse_json_no_duplicates(
                self._read_body(
                    response,
                    body_limit_bytes=body_limit_bytes,
                )
            )
        finally:
            connection.close()

    @staticmethod
    def _expected_slot(payload: Mapping[str, JsonValue]) -> int | None:
        value = payload.get("id_slot")
        if value is None:
            return None
        return _require_integer(value, "id_slot", minimum=0)

    @staticmethod
    def _validate_response_slot(
        response: Mapping[str, object], expected_slot: int | None
    ) -> int:
        id_slot = _require_integer(response.get("id_slot"), "id_slot", minimum=0)
        if expected_slot is not None and id_slot != expected_slot:
            raise UpstreamContractError(
                f"expected slot {expected_slot}, received slot {id_slot}"
            )
        return id_slot

    def complete(
        self,
        payload: Mapping[str, JsonValue],
        *,
        on_request_sent: Callable[[HttpSendEvidence], None] | None = None,
    ) -> CompletionResult:
        expected_slot = self._expected_slot(payload)
        response = _require_object(
            self._request_json(
                "POST",
                "/completion",
                payload,
                on_request_sent=on_request_sent,
            ),
            "completion response",
        )
        id_slot = self._validate_response_slot(response, expected_slot)
        content = _require_string(response.get("content"), "content")
        if response.get("stop") is not True:
            raise UpstreamContractError(
                "non-streaming completion response is not terminal"
            )
        tokens_evaluated_value = response.get("tokens_evaluated")
        tokens_evaluated = (
            None
            if tokens_evaluated_value is None
            else _require_integer(
                tokens_evaluated_value, "tokens_evaluated", minimum=0
            )
        )
        tokens_predicted_value = response.get("tokens_predicted")
        tokens_predicted = (
            None
            if tokens_predicted_value is None
            else _require_integer(
                tokens_predicted_value, "tokens_predicted", minimum=0
            )
        )
        response_ns = time.monotonic_ns()
        return CompletionResult(
            id_slot=id_slot,
            content=content,
            terminal=True,
            response_evidence_ref=f"http-response:{response_ns}",
            tokens_evaluated=tokens_evaluated,
            tokens_predicted=tokens_predicted,
        )

    def stream_completion(
        self,
        payload: Mapping[str, JsonValue],
        *,
        on_request_sent: Callable[[HttpSendEvidence], None] | None = None,
    ) -> Iterator[CompletionStreamEvent]:
        expected_slot = self._expected_slot(payload)
        connection, response = self._start_request(
            "POST", "/completion", payload, on_request_sent
        )
        terminal_seen = False
        try:
            while True:
                try:
                    raw_line = response.readline(self.max_body_bytes + 1)
                except (OSError, socket.timeout, http.client.HTTPException) as error:
                    raise UpstreamTransportError(
                        "failed while reading completion stream"
                    ) from error
                if not raw_line:
                    break
                if len(raw_line) > self.max_body_bytes:
                    raise UpstreamContractError("SSE event exceeds body limit")
                line = raw_line.strip()
                if not line or line.startswith(b":"):
                    continue
                if not line.startswith(b"data:"):
                    raise UpstreamContractError("invalid SSE line")
                raw_data = line[5:].lstrip()
                response_object = _require_object(
                    _parse_json_no_duplicates(raw_data), "SSE event"
                )
                content = _require_string(
                    response_object.get("content"), "content"
                )
                stop = response_object.get("stop")
                if not isinstance(stop, bool):
                    raise UpstreamContractError("stop must be a boolean")
                raw_id_slot = _require_integer(
                    response_object.get("id_slot"), "id_slot"
                )
                if not stop and raw_id_slot == -1:
                    id_slot = -1
                else:
                    id_slot = self._validate_response_slot(
                        response_object, expected_slot
                    )
                if terminal_seen:
                    raise UpstreamContractError("received data after terminal event")
                response_ns = time.monotonic_ns()
                terminal_seen = stop
                yield CompletionStreamEvent(
                    id_slot=id_slot,
                    content=content,
                    terminal=stop,
                    response_evidence_ref=f"http-response:{response_ns}",
                )
            if not terminal_seen:
                raise UpstreamContractError(
                    "completion stream ended before terminal event"
                )
        finally:
            connection.close()

    def stream_chat_completion(
        self,
        payload: Mapping[str, JsonValue],
        *,
        on_request_sent: Callable[[HttpSendEvidence], None] | None = None,
    ) -> Iterator[CompletionStreamEvent]:
        expected_slot = self._expected_slot(payload)
        if expected_slot is None:
            raise UpstreamContractError("chat request must bind id_slot")
        connection, response = self._start_request(
            "POST", "/v1/chat/completions", payload, on_request_sent
        )
        terminal_seen = False
        done_seen = False
        try:
            while True:
                try:
                    raw_line = response.readline(self.max_body_bytes + 1)
                except (OSError, socket.timeout, http.client.HTTPException) as error:
                    raise UpstreamTransportError(
                        "failed while reading chat completion stream"
                    ) from error
                if not raw_line:
                    break
                if len(raw_line) > self.max_body_bytes:
                    raise UpstreamContractError("SSE event exceeds body limit")
                line = raw_line.strip()
                if not line or line.startswith(b":"):
                    continue
                if not line.startswith(b"data:"):
                    raise UpstreamContractError("invalid SSE line")
                raw_data = line[5:].lstrip()
                if raw_data == b"[DONE]":
                    if done_seen or not terminal_seen:
                        raise UpstreamContractError("chat stream DONE is out of order")
                    done_seen = True
                    continue
                if done_seen or terminal_seen:
                    raise UpstreamContractError("received data after terminal event")
                response_object = _require_object(
                    _parse_json_no_duplicates(raw_data), "chat SSE event"
                )
                choices = response_object.get("choices")
                if not isinstance(choices, list) or len(choices) != 1:
                    raise UpstreamContractError(
                        "chat SSE event must contain exactly one choice"
                    )
                choice = _require_object(choices[0], "chat SSE choice")
                delta = _require_object(choice.get("delta"), "chat SSE delta")
                raw_content = delta.get("content")
                content = (
                    ""
                    if raw_content is None
                    else _require_string(raw_content, "delta.content")
                )
                finish_reason = choice.get("finish_reason")
                if finish_reason is not None and not isinstance(finish_reason, str):
                    raise UpstreamContractError(
                        "finish_reason must be null or a string"
                    )
                terminal_seen = finish_reason is not None
                response_ns = time.monotonic_ns()
                yield CompletionStreamEvent(
                    id_slot=expected_slot,
                    content=content,
                    terminal=terminal_seen,
                    response_evidence_ref=f"http-response:{response_ns}",
                )
            if not terminal_seen or not done_seen:
                raise UpstreamContractError(
                    "chat completion stream ended before terminal DONE"
                )
        finally:
            connection.close()

    def get_slots(self) -> tuple[SlotStatus, ...]:
        response = self._request_json(
            "GET",
            "/slots",
            body_limit_bytes=self.slots_body_limit_bytes,
        )
        if not isinstance(response, list):
            raise UpstreamContractError("slots response must be a JSON array")
        slots: list[SlotStatus] = []
        seen: set[int] = set()
        for item in response:
            slot = _require_object(item, "slot")
            slot_id = _require_integer(slot.get("id"), "slot.id", minimum=0)
            if slot_id in seen:
                raise UpstreamContractError(f"duplicate slot id: {slot_id}")
            seen.add(slot_id)
            id_task = (
                _require_integer(slot["id_task"], "slot.id_task")
                if "id_task" in slot
                else None
            )
            is_processing = slot.get("is_processing")
            if not isinstance(is_processing, bool):
                raise UpstreamContractError(
                    "slot.is_processing must be a boolean"
                )
            slots.append(
                SlotStatus(
                    slot_id=slot_id,
                    id_task=id_task,
                    is_processing=is_processing,
                )
            )
        return tuple(slots)

    def wait_for_slot_idle(
        self, slot_id: int, *, deadline: float | None
    ) -> SlotReleaseEvidence:
        _require_integer(slot_id, "slot_id", minimum=0)
        while True:
            slots = {slot.slot_id: slot for slot in self.get_slots()}
            if slot_id not in slots:
                raise UpstreamContractError(
                    f"slots response is missing slot {slot_id}"
                )
            status = slots[slot_id]
            if not status.is_processing:
                confirmed_ns = time.monotonic_ns()
                return SlotReleaseEvidence(
                    slot_id=slot_id,
                    id_task=status.id_task,
                    confirmed_monotonic_ns=confirmed_ns,
                    evidence_ref=f"slot-idle:{slot_id}:{confirmed_ns}",
                )
            if deadline is not None and time.monotonic() >= deadline:
                raise UpstreamTimeoutError(
                    f"slot {slot_id} did not become idle before deadline"
                )
            if self.slot_poll_interval_seconds:
                time.sleep(self.slot_poll_interval_seconds)

    def erase_slot(self, slot_id: int) -> SlotEraseEvidence:
        expected_slot = _require_integer(slot_id, "slot_id", minimum=0)
        response = _require_object(
            self._request_json(
                "POST",
                f"/slots/{expected_slot}?action=erase",
            ),
            "slot erase response",
        )
        observed_slot = self._validate_response_slot(response, expected_slot)
        n_erased = _require_integer(
            response.get("n_erased"), "n_erased", minimum=0
        )
        confirmed_ns = time.monotonic_ns()
        return SlotEraseEvidence(
            slot_id=observed_slot,
            n_erased=n_erased,
            confirmed_monotonic_ns=confirmed_ns,
            evidence_ref=f"slot-erased:{observed_slot}:{confirmed_ns}",
        )

    def health(self) -> bool:
        response = _require_object(
            self._request_json("GET", "/health"), "health response"
        )
        return response.get("status") == "ok"
