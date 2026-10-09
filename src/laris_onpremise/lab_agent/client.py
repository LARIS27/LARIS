from __future__ import annotations

import json
import secrets
import ssl
import threading
import time
import urllib.error
import urllib.request
import uuid
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol
from urllib.parse import urlsplit

from laris_onpremise.configuration import ConfigurationError, load_yaml_mapping
from laris_onpremise.lab_agent.runtime_mode import RuntimeMode, parse_runtime_mode


_PROFILE_FIELDS = frozenset(
    {
        "schema_version",
        "agent_id",
        "server_profile",
        "base_url",
        "approved_runtime_profile_id",
        "command_timeout_seconds",
        "tls",
    }
)
_TLS_FIELDS = frozenset(
    {"ca_certificate", "client_certificate", "client_private_key"}
)


class LabAgentClientError(RuntimeError):
    """Raised when the Host cannot safely control the configured Lab Agent."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "LAB_AGENT_CLIENT_ERROR",
        detail: str | None = None,
        outcome_known: bool = True,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.detail = message if detail is None else detail
        self.outcome_known = outcome_known


def _mapping(name: str, value: object, fields: frozenset[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise LabAgentClientError(f"{name} field set is invalid")
    return value


def _text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise LabAgentClientError(f"{name} must be non-empty text")
    return value


def _positive_integer(name: str, value: object, *, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 1
        or value > maximum
    ):
        raise LabAgentClientError(f"{name} must be an integer from 1 to {maximum}")
    return value


def _relative_file(profile_path: Path, name: str, value: object) -> Path:
    raw = _text(name, value)
    candidate = Path(raw)
    if candidate.is_absolute() or candidate.name in {"", "."}:
        raise LabAgentClientError(f"{name} must be a safe relative path")
    if ".." in candidate.parts:
        if (
            len(candidate.parts) < 3
            or candidate.parts[:2] != ("..", "certs")
            or ".." in candidate.parts[2:]
        ):
            raise LabAgentClientError(f"{name} must be a safe relative path")
        return profile_path.parent.parent / Path(*candidate.parts[1:])
    return profile_path.parent / candidate


@dataclass(frozen=True)
class LabAgentProfile:
    agent_id: str
    server_profile: str
    base_url: str
    approved_runtime_profile_id: str
    command_timeout_seconds: int
    ca_certificate: Path
    client_certificate: Path
    client_private_key: Path

    @classmethod
    def load(cls, path: Path) -> "LabAgentProfile":
        try:
            raw = load_yaml_mapping(path)
        except ConfigurationError as error:
            raise LabAgentClientError("Lab Agent profile YAML is invalid") from error
        root = _mapping("profile", raw, _PROFILE_FIELDS)
        if root["schema_version"] != "1.0":
            raise LabAgentClientError("profile schema_version must be 1.0")
        base_url = _text("base_url", root["base_url"])
        parsed = urlsplit(base_url)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise LabAgentClientError("base_url must be an absolute HTTPS origin")
        tls = _mapping("TLS", root["tls"], _TLS_FIELDS)
        return cls(
            agent_id=_text("agent_id", root["agent_id"]),
            server_profile=_text("server_profile", root["server_profile"]),
            base_url=base_url.rstrip("/"),
            approved_runtime_profile_id=_text(
                "approved_runtime_profile_id",
                root["approved_runtime_profile_id"],
            ),
            command_timeout_seconds=_positive_integer(
                "command_timeout_seconds",
                root["command_timeout_seconds"],
                maximum=3600,
            ),
            ca_certificate=_relative_file(
                path,
                "ca_certificate",
                tls["ca_certificate"],
            ),
            client_certificate=_relative_file(
                path,
                "client_certificate",
                tls["client_certificate"],
            ),
            client_private_key=_relative_file(
                path,
                "client_private_key",
                tls["client_private_key"],
            ),
        )


class AgentTransport(Protocol):
    def request(
        self,
        method: str,
        path: str,
        body: dict[str, object] | None,
    ) -> dict[str, object]: ...

    def request_bytes(self, path: str) -> bytes: ...


class _HttpsAgentTransport:
    def __init__(self, profile: LabAgentProfile) -> None:
        self._profile = profile
        self._opener: urllib.request.OpenerDirector | None = None

    def _build_opener(self) -> urllib.request.OpenerDirector:
        context = ssl.create_default_context(cafile=str(self._profile.ca_certificate))
        context.minimum_version = ssl.TLSVersion.TLSv1_3
        context.load_cert_chain(
            certfile=str(self._profile.client_certificate),
            keyfile=str(self._profile.client_private_key),
        )
        return urllib.request.build_opener(urllib.request.HTTPSHandler(context=context))

    def request(
        self,
        method: str,
        path: str,
        body: dict[str, object] | None,
    ) -> dict[str, object]:
        if self._opener is None:
            self._opener = self._build_opener()
        raw = None if body is None else json.dumps(
            body,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        request = urllib.request.Request(
            self._profile.base_url + path,
            data=raw,
            method=method,
            headers={} if raw is None else {"Content-Type": "application/json"},
        )
        try:
            timeout = (
                float(self._profile.command_timeout_seconds)
                if method == "POST"
                else 10.0
            )
            with self._opener.open(request, timeout=timeout) as response:
                payload = response.read(1024 * 1024 + 1)
        except urllib.error.HTTPError as error:
            detail = f"HTTP {error.code}"
            error_code = "LAB_AGENT_HTTP_ERROR"
            error_detail = detail
            try:
                error_payload = error.read(64 * 1024 + 1)
                if len(error_payload) <= 64 * 1024:
                    value = json.loads(error_payload.decode("utf-8"))
                    code = value.get("code") if isinstance(value, dict) else None
                    message = value.get("detail") if isinstance(value, dict) else None
                    if (
                        isinstance(code, str)
                        and re.fullmatch(r"[A-Z][A-Z0-9_]{0,63}", code) is not None
                        and isinstance(message, str)
                        and 0 < len(message) <= 1024
                    ):
                        detail = f"{code}: {message}"
                        error_code = code
                        error_detail = message
            except (OSError, UnicodeError, json.JSONDecodeError):
                pass
            raise LabAgentClientError(
                f"Lab Agent request failed: {detail}",
                code=error_code,
                detail=error_detail,
                outcome_known=True,
            ) from error
        except (OSError, urllib.error.URLError) as error:
            raise LabAgentClientError(
                "Lab Agent request failed",
                code="LAB_AGENT_UNREACHABLE",
                detail="Lab Agent is unreachable",
                outcome_known=False,
            ) from error
        if len(payload) > 1024 * 1024:
            raise LabAgentClientError("Lab Agent response exceeds size limit")
        try:
            value = json.loads(payload.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as error:
            raise LabAgentClientError("Lab Agent response is invalid") from error
        if not isinstance(value, dict):
            raise LabAgentClientError("Lab Agent response must be an object")
        return value

    def request_bytes(self, path: str) -> bytes:
        if self._opener is None:
            self._opener = self._build_opener()
        request = urllib.request.Request(self._profile.base_url + path, method="GET")
        try:
            with self._opener.open(request, timeout=30.0) as response:
                payload = response.read(512 * 1024 * 1024 + 1)
        except (OSError, urllib.error.URLError) as error:
            raise LabAgentClientError("Lab Agent artifact request failed") from error
        if len(payload) > 512 * 1024 * 1024:
            raise LabAgentClientError("Lab Agent artifact exceeds size limit")
        return payload


class LabAgentClient:
    def __init__(
        self,
        profile: LabAgentProfile,
        *,
        transport: AgentTransport | None = None,
    ) -> None:
        self._profile = profile
        self._transport = transport or _HttpsAgentTransport(profile)
        self._counter: int | None = None
        self._pending: tuple[dict[str, object], dict[str, object]] | None = None
        self._lock = threading.RLock()

    @property
    def approved_profile_id(self) -> str:
        return self._profile.approved_runtime_profile_id

    def status(self) -> dict[str, object]:
        response = self._transport.request("GET", "/v1/status", None)
        counter = response.get("accepted_command_counter")
        if isinstance(counter, bool) or not isinstance(counter, int) or counter < 0:
            raise LabAgentClientError("Lab Agent status counter is invalid")
        with self._lock:
            self._counter = counter
            if self._pending is not None:
                _payload, envelope = self._pending
                pending_counter = envelope.get("command_counter")
                if isinstance(pending_counter, int) and counter >= pending_counter:
                    self._pending = None
        return response

    def run_status(self, run_id: str) -> dict[str, object]:
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", run_id) is None:
            raise LabAgentClientError("run_id is invalid")
        return self._transport.request("GET", f"/v1/runs/{run_id}", None)

    def models(self) -> dict[str, object]:
        return self._transport.request("GET", "/v1/models", None)

    def inspect_runtime(self) -> dict[str, object]:
        return self._transport.request("GET", "/v1/runtime-candidate", None)

    def prepare(
        self,
        *,
        runtime_mode: RuntimeMode | str,
        run_id: str,
        scenario_id: str,
        scenario_digest: str,
        expected_request_count: int,
        model_profile_id: str,
        model_digest: str,
        expected_model_name: str,
        expected_slot_count: int,
    ) -> dict[str, object]:
        try:
            parsed_runtime_mode = parse_runtime_mode(runtime_mode)
        except ValueError as error:
            raise LabAgentClientError(str(error)) from error
        payload: dict[str, object] = {
            "runtime_mode": parsed_runtime_mode.value,
            "run_id": run_id,
            "scenario_id": scenario_id,
            "scenario_digest": scenario_digest,
            "approved_profile_id": self.approved_profile_id,
            "expected_request_count": expected_request_count,
            "model_profile_id": model_profile_id,
            "model_digest": model_digest,
            "expected_model_name": expected_model_name,
            "expected_slot_count": expected_slot_count,
        }
        return self._command("/v1/runs/prepare", payload)

    def stop_and_seal(self, run_id: str) -> dict[str, object]:
        return self._command(f"/v1/runs/{run_id}/stop-and-seal", {})

    def release(self, run_id: str) -> dict[str, object]:
        return self._command(f"/v1/runs/{run_id}/release", {})

    def sealed_manifest(self, run_id: str) -> dict[str, object]:
        return self._transport.request(
            "GET",
            f"/v1/runs/{run_id}/sealed-manifest",
            None,
        )

    def artifact(self, run_id: str, artifact_id: str) -> bytes:
        if re.fullmatch(r"art_[A-Za-z0-9_-]{16,96}", artifact_id) is None:
            raise LabAgentClientError("artifact_id is invalid")
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", run_id) is None:
            raise LabAgentClientError("run_id is invalid")
        return self._transport.request_bytes(
            f"/v1/runs/{run_id}/artifacts/{artifact_id}"
        )

    def _command(self, path: str, payload: dict[str, object]) -> dict[str, object]:
        with self._lock:
            if self._counter is None:
                self.status()
            if self._pending is not None:
                pending_payload, envelope = self._pending
                if pending_payload != payload:
                    try:
                        self.status()
                    except LabAgentClientError:
                        pass
                    if self._pending is not None:
                        raise LabAgentClientError(
                            "another Lab Agent command is awaiting a response",
                            code="LAB_COMMAND_OUTCOME_UNKNOWN",
                            detail=(
                                "The previous Lab Agent command outcome is unknown; "
                                "refresh status before retrying"
                            ),
                            outcome_known=False,
                        )
                    assert self._counter is not None
                    envelope = {
                        "schema_version": "1.0",
                        "request_id": str(uuid.uuid4()),
                        "command_counter": self._counter + 1,
                        "issued_at_wall_time_ns": time.time_ns(),
                        "command_nonce": "nonce_" + secrets.token_urlsafe(32),
                        "payload": dict(payload),
                    }
                    self._pending = (dict(payload), envelope)
            else:
                assert self._counter is not None
                envelope = {
                    "schema_version": "1.0",
                    "request_id": str(uuid.uuid4()),
                    "command_counter": self._counter + 1,
                    "issued_at_wall_time_ns": time.time_ns(),
                    "command_nonce": "nonce_" + secrets.token_urlsafe(32),
                    "payload": dict(payload),
                }
                self._pending = (dict(payload), envelope)
            try:
                response = self._transport.request("POST", path, envelope)
            except LabAgentClientError:
                try:
                    self.status()
                except LabAgentClientError:
                    pass
                raise
            self._counter = int(envelope["command_counter"])
            self._pending = None
            return response
