from __future__ import annotations

import json
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from collections.abc import Callable, Mapping
from typing import Any

from laris_onpremise.attribution.proxy.lease_controller import QueueWaitTimeout
from laris_onpremise.attribution.proxy.core import (
    AdmissionClosedError,
    IngressValidationError,
    LarisProxyCore,
)
from laris_onpremise.core.records import canonical_json


class LarisProxyServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        *,
        core: LarisProxyCore,
        max_request_bytes: int = 1024 * 1024,
        health_provider: Callable[[], Mapping[str, object]] | None = None,
    ) -> None:
        if max_request_bytes <= 0:
            raise ValueError("max_request_bytes must be positive")
        self.core = core
        self.max_request_bytes = max_request_bytes
        self.health_provider = health_provider
        super().__init__(server_address, _LarisRequestHandler)


class _LarisRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    @property
    def laris_server(self) -> LarisProxyServer:
        return self.server  # type: ignore[return-value]

    def _send_json(
        self,
        status: int,
        payload: dict[str, Any],
        *,
        request_ref: str | None = None,
        user_hmac: str | None = None,
        identity_key_id: str | None = None,
    ) -> None:
        if (user_hmac is None) != (identity_key_id is None):
            raise ValueError(
                "user_hmac and identity_key_id must be emitted together"
            )
        body = canonical_json(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if request_ref is not None:
            self.send_header("X-LARIS-Request-Ref", request_ref)
        if user_hmac is not None and identity_key_id is not None:
            self.send_header("X-LARIS-User-HMAC", user_hmac)
            self.send_header("X-LARIS-Identity-Key-Id", identity_key_id)
        self.end_headers()
        self.wfile.write(body)

    def _identity_headers(self) -> tuple[str, str, str, str | None]:
        actor = self.headers.get("X-LARIS-Test-Actor")
        session_ref = self.headers.get("X-LARIS-Test-Session-Ref")
        auth_receipt = self.headers.get("X-LARIS-Test-Auth-Receipt")
        retry_of = self.headers.get("X-LARIS-Test-Retry-Of")
        if not actor or not session_ref or not auth_receipt:
            raise IngressValidationError(
                "synthetic test identity headers are required"
            )
        return actor, session_ref, auth_receipt, retry_of

    def _read_body(self) -> bytes:
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise IngressValidationError("Content-Length is required")
        try:
            length = int(raw_length)
        except ValueError as error:
            raise IngressValidationError("invalid Content-Length") from error
        if length < 0 or length > self.laris_server.max_request_bytes:
            raise IngressValidationError("request body exceeds size limit")
        return self.rfile.read(length)

    def do_GET(self) -> None:
        if self.path == "/health":
            if self.laris_server.health_provider is not None:
                try:
                    health = dict(self.laris_server.health_provider())
                except Exception:
                    self._send_json(503, {"error_code": "health_unavailable"})
                    return
                self._send_json(200, health)
                return
            self._send_json(
                200,
                {"status": "ok"},
            )
            return
        self._send_json(404, {"error_code": "not_found"})

    def do_POST(self) -> None:
        if self.path not in {"/completion", "/v1/chat/completions"}:
            self._send_json(404, {"error_code": "not_found"})
            return
        try:
            raw_body = self._read_body()
            actor, session_ref, auth_receipt, retry_of = self._identity_headers()
            parsed = self.laris_server.core.parse_ingress_body(raw_body)
            connection_ref = f"conn_{uuid.uuid4().hex}"
            if self.path == "/v1/chat/completions":
                if parsed.get("stream") is not True:
                    raise IngressValidationError("chat requests must stream")
                self._handle_stream(
                    request_kind="chat",
                    actor=actor,
                    session_ref=session_ref,
                    auth_receipt=auth_receipt,
                    retry_of=retry_of,
                    connection_ref=connection_ref,
                    raw_body=raw_body,
                )
            elif parsed.get("stream") is True:
                self._handle_stream(
                    request_kind="completion",
                    actor=actor,
                    session_ref=session_ref,
                    auth_receipt=auth_receipt,
                    retry_of=retry_of,
                    connection_ref=connection_ref,
                    raw_body=raw_body,
                )
            else:
                result = self.laris_server.core.handle_completion(
                    asserted_subject=actor,
                    auth_receipt_id=auth_receipt,
                    session_ref=session_ref,
                    connection_ref=connection_ref,
                    retry_of=retry_of,
                    raw_body=raw_body,
                )
                response_document: dict[str, Any] = {
                    "content": result.content,
                    "id_slot": result.slot_id,
                    "stop": True,
                }
                if result.tokens_evaluated is not None:
                    response_document["tokens_evaluated"] = result.tokens_evaluated
                if result.tokens_predicted is not None:
                    response_document["tokens_predicted"] = result.tokens_predicted
                self._send_json(
                    200,
                    response_document,
                    request_ref=result.request_ref,
                    user_hmac=result.user_hmac,
                    identity_key_id=result.identity_key_id,
                )
        except IngressValidationError:
            self._send_json(400, {"error_code": "invalid_ingress"})
        except QueueWaitTimeout:
            self._send_json(503, {"error_code": "queue_timeout"})
        except AdmissionClosedError:
            self._send_json(503, {"error_code": "admission_closed"})
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception:
            self._send_json(502, {"error_code": "pipeline_failure"})

    def _handle_stream(
        self,
        *,
        request_kind: str,
        actor: str,
        session_ref: str,
        auth_receipt: str,
        retry_of: str | None,
        connection_ref: str,
        raw_body: bytes,
    ) -> None:
        start_stream = (
            self.laris_server.core.start_stream_chat
            if request_kind == "chat"
            else self.laris_server.core.start_stream_completion
        )
        session = start_stream(
            asserted_subject=actor,
            auth_receipt_id=auth_receipt,
            session_ref=session_ref,
            connection_ref=connection_ref,
            retry_of=retry_of,
            raw_body=raw_body,
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("X-LARIS-Request-Ref", session.request_ref)
        self.send_header("X-LARIS-Lease-Id", session.lease_id)
        self.send_header("X-LARIS-Slot-Id-Plus-One", str(session.slot_id + 1))
        self.send_header("X-LARIS-Slot-Epoch", str(session.slot_epoch))
        self.send_header("X-LARIS-Seq-Id-Plus-One", str(session.seq_id + 1))
        self.send_header("X-LARIS-User-HMAC", session.user_hmac)
        self.send_header("X-LARIS-Identity-Key-Id", session.identity_key_id)
        self.end_headers()
        try:
            for event in session.events:
                data = canonical_json(
                    {
                        "content": event.content,
                        "id_slot": event.slot_id,
                        "stop": event.terminal,
                    }
                )
                self.wfile.write(b"data: " + data + b"\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            session.events.close()

    def log_message(self, format: str, *args: object) -> None:
        return
