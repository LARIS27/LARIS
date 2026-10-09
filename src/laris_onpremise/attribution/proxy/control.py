from __future__ import annotations

import json
import os
import socket
import stat
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from laris_onpremise.attribution.proxy.core import LarisProxyCore
from laris_onpremise.core.records import canonical_json


class ProxyControlError(RuntimeError):
    """Raised when a local proxy control request violates the contract."""


MAX_PROXY_DRAIN_TIMEOUT_MS = 600_000


@dataclass(frozen=True)
class ProxyControlResult:
    response: dict[str, object]
    shutdown_requested: bool


def _write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _strict_json_object(payload: bytes) -> dict[str, object]:
    def reject_duplicates(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise ProxyControlError(f"duplicate control key: {key}")
            value[key] = item
        return value

    try:
        value = json.loads(payload, object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProxyControlError("control request is not strict JSON") from error
    if not isinstance(value, dict):
        raise ProxyControlError("control request must be a JSON object")
    return value


class ProxyControlProtocol:
    def __init__(
        self,
        *,
        core: LarisProxyCore,
        run_nonce: str,
        evidence_dir: Path,
    ) -> None:
        if len(run_nonce) < 16:
            raise ValueError("run_nonce must contain at least 16 characters")
        self.core = core
        self.run_nonce = run_nonce
        self.evidence_dir = evidence_dir

    def _validate_common(self, request: Mapping[str, object]) -> str:
        if request.get("schema_version") != "1.0":
            raise ProxyControlError("unsupported proxy control schema")
        if request.get("run_nonce") != self.run_nonce:
            raise ProxyControlError("proxy control run nonce mismatch")
        command = request.get("command")
        if not isinstance(command, str):
            raise ProxyControlError("proxy control command is missing")
        return command

    def handle(self, request: Mapping[str, object]) -> ProxyControlResult:
        command = self._validate_common(request)
        if command == "CLOSE_ADMISSION":
            if set(request) != {
                "schema_version",
                "command",
                "run_nonce",
                "timeout_ms",
            }:
                raise ProxyControlError("CLOSE_ADMISSION fields are not exact")
            timeout_ms = request["timeout_ms"]
            if (
                isinstance(timeout_ms, bool)
                or not isinstance(timeout_ms, int)
                or not 1 <= timeout_ms <= MAX_PROXY_DRAIN_TIMEOUT_MS
            ):
                raise ProxyControlError("timeout_ms is invalid")
            evidence = self.core.close_admission(
                deadline=time.monotonic() + timeout_ms / 1000
            )
            response: dict[str, object] = {
                "schema_version": "1.0",
                "state": "PROXY_ADMISSION_CLOSED",
                "run_nonce": self.run_nonce,
                "closed_at_monotonic_ns": evidence.closed_at_monotonic_ns,
                "drained_at_monotonic_ns": evidence.drained_at_monotonic_ns,
                "active_request_count": evidence.active_request_count,
            }
            _write_exclusive(
                self.evidence_dir / "PROXY_ADMISSION_CLOSED.json",
                canonical_json(response) + b"\n",
            )
            return ProxyControlResult(response, shutdown_requested=False)
        if command == "SHUTDOWN":
            if set(request) != {"schema_version", "command", "run_nonce"}:
                raise ProxyControlError("SHUTDOWN fields are not exact")
            if not self.core.shutdown_ready:
                raise ProxyControlError("proxy admission is not drained")
            return ProxyControlResult(
                {
                    "schema_version": "1.0",
                    "state": "PROXY_SHUTDOWN_READY",
                    "run_nonce": self.run_nonce,
                },
                shutdown_requested=True,
            )
        raise ProxyControlError("unsupported proxy control command")


def send_proxy_control(
    socket_path: Path,
    request: Mapping[str, object],
    *,
    timeout_seconds: float,
    expected_server_pid: int | None = None,
) -> dict[str, object]:
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as channel:
        channel.settimeout(timeout_seconds)
        channel.connect(str(socket_path))
        if expected_server_pid is not None:
            credentials = channel.getsockopt(
                socket.SOL_SOCKET,
                socket.SO_PEERCRED,
                struct.calcsize("3i"),
            )
            server_pid, _, _ = struct.unpack("3i", credentials)
            if server_pid != expected_server_pid:
                raise ProxyControlError("proxy control peer PID mismatch")
        channel.sendall(canonical_json(dict(request)) + b"\n")
        payload = bytearray()
        while b"\n" not in payload:
            chunk = channel.recv(4096)
            if not chunk:
                raise ProxyControlError("proxy control response ended early")
            payload.extend(chunk)
            if len(payload) > 16_384:
                raise ProxyControlError("proxy control response is too large")
        line, remainder = bytes(payload).split(b"\n", 1)
        if remainder:
            raise ProxyControlError("proxy control response has trailing data")
        response = _strict_json_object(line)
        if response.get("schema_version") != "1.0":
            raise ProxyControlError("unsupported proxy control response")
        return response


class UnixProxyControlServer:
    def __init__(
        self,
        *,
        socket_path: Path,
        protocol: ProxyControlProtocol,
        expected_uid: int | None = None,
    ) -> None:
        self.socket_path = socket_path
        self.protocol = protocol
        self.expected_uid = os.getuid() if expected_uid is None else expected_uid

    def _handle_connection(self, channel: socket.socket) -> bool:
        credentials = channel.getsockopt(
            socket.SOL_SOCKET,
            socket.SO_PEERCRED,
            struct.calcsize("3i"),
        )
        _, peer_uid, _ = struct.unpack("3i", credentials)
        if peer_uid != self.expected_uid:
            raise ProxyControlError("proxy control peer UID mismatch")
        payload = bytearray()
        while b"\n" not in payload:
            chunk = channel.recv(4096)
            if not chunk:
                raise ProxyControlError("proxy control request ended early")
            payload.extend(chunk)
            if len(payload) > 16_384:
                raise ProxyControlError("proxy control request is too large")
        line, remainder = bytes(payload).split(b"\n", 1)
        if remainder:
            raise ProxyControlError("proxy control request has trailing data")
        result = self.protocol.handle(_strict_json_object(line))
        channel.sendall(canonical_json(result.response) + b"\n")
        return result.shutdown_requested

    def serve_until_shutdown(self) -> None:
        if not hasattr(socket, "SO_PEERCRED"):
            raise ProxyControlError("SO_PEERCRED is required")
        self.socket_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.socket_path.exists() or self.socket_path.is_symlink():
            raise FileExistsError(self.socket_path)
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        created = False
        try:
            listener.bind(str(self.socket_path))
            created = True
            os.chmod(self.socket_path, 0o600)
            if stat.S_IMODE(self.socket_path.stat().st_mode) != 0o600:
                raise ProxyControlError("proxy control socket mode is not 0600")
            listener.listen(4)
            while True:
                channel, _ = listener.accept()
                with channel:
                    try:
                        if self._handle_connection(channel):
                            return
                    except ProxyControlError as error:
                        response = {
                            "schema_version": "1.0",
                            "state": "PROXY_CONTROL_ERROR",
                            "error": str(error),
                        }
                        channel.sendall(canonical_json(response) + b"\n")
        finally:
            listener.close()
            if created:
                try:
                    mode = self.socket_path.lstat().st_mode
                except FileNotFoundError:
                    pass
                else:
                    if stat.S_ISSOCK(mode):
                        self.socket_path.unlink()
