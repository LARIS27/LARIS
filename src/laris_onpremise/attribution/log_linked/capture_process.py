from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import select
import stat
import subprocess
from typing import Callable

from laris_onpremise.kv_state.observation.runtime_state_profile import (
    RuntimeStateObserverProfile,
    load_runtime_state_observer_profile,
)


_ATTACHED = b"ATTACHED runtime-state-service\n"
_READY = b"READY runtime-state-service\n"
_TOPOLOGY_PREFIX = b"TOPOLOGY runtime-state-service "
_TOPOLOGY_FIELDS = frozenset(
    {
        "n_ctx",
        "n_layer",
        "n_pos_per_embd",
        "n_seq_max",
        "n_stream",
        "profile_digest",
        "v_trans",
    }
)
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")


class LogLinkedCaptureError(RuntimeError):
    """Raised when the standalone observer child cannot be trusted."""


@dataclass(frozen=True)
class LogLinkedCaptureConfiguration:
    python_executable: Path
    socket_path: Path
    frame_dir: Path
    profile_path: Path
    model_registration_receipt: Path
    runtime_instance_id: str
    runtime_pid: int
    runtime_start_time_ticks: int
    native_observer: Path
    runtime_binary: Path
    bpf_object: Path
    ready_timeout_seconds: float
    control_timeout_seconds: float
    service_log_path: Path

    def __post_init__(self) -> None:
        for field in (
            "python_executable",
            "socket_path",
            "frame_dir",
            "profile_path",
            "model_registration_receipt",
            "native_observer",
            "runtime_binary",
            "bpf_object",
            "service_log_path",
        ):
            value = getattr(self, field)
            if not isinstance(value, Path) or not value.is_absolute():
                raise ValueError(f"{field} must be an absolute pathlib.Path")
        if (
            not isinstance(self.runtime_instance_id, str)
            or not self.runtime_instance_id
        ):
            raise ValueError("runtime_instance_id must be non-empty text")
        for field in ("runtime_pid", "runtime_start_time_ticks"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field} must be a positive integer")
        for field in ("ready_timeout_seconds", "control_timeout_seconds"):
            value = getattr(self, field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or value <= 0
            ):
                raise ValueError(f"{field} must be positive")


def build_log_linked_service_command(
    configuration: LogLinkedCaptureConfiguration,
    profile: RuntimeStateObserverProfile,
) -> tuple[str, ...]:
    if not isinstance(configuration, LogLinkedCaptureConfiguration):
        raise TypeError("configuration must be LogLinkedCaptureConfiguration")
    if not isinstance(profile, RuntimeStateObserverProfile):
        raise TypeError("profile must be RuntimeStateObserverProfile")
    targets = {target.probe_id: target for target in profile.probe_targets}
    required_targets = {
        "kv_constructor",
        "n_pos_per_embd",
        "view_2d",
        "buffer_base",
        "buffer_size",
        "apply_ubatch",
    }
    if not required_targets.issubset(targets):
        raise LogLinkedCaptureError(
            "observer profile omits a required writer-boundary probe"
        )
    return (
        str(configuration.python_executable),
        "-m",
        "laris_onpremise.observer.runtime_state_service",
        "--socket-path",
        str(configuration.socket_path),
        "--frame-dir",
        str(configuration.frame_dir),
        "--profile",
        str(configuration.profile_path),
        "--model-registration-receipt",
        str(configuration.model_registration_receipt),
        "--runtime-instance-id",
        configuration.runtime_instance_id,
        "--runtime-pid",
        str(configuration.runtime_pid),
        "--runtime-start-time-ticks",
        str(configuration.runtime_start_time_ticks),
        "--native-observer",
        str(configuration.native_observer),
        "--runtime-binary",
        str(configuration.runtime_binary),
        "--attachment-mode",
        "live",
        "--capture-mode",
        "writer-boundary",
        "--session-mode",
        "log-linked",
        "--bpf-object",
        str(configuration.bpf_object),
        "--constructor-offset",
        str(targets["kv_constructor"].resolved_file_offset),
        "--n-pos-per-embd-offset",
        str(targets["n_pos_per_embd"].resolved_file_offset),
        "--view-offset",
        str(targets["view_2d"].resolved_file_offset),
        "--buffer-base-offset",
        str(targets["buffer_base"].resolved_file_offset),
        "--buffer-size-offset",
        str(targets["buffer_size"].resolved_file_offset),
        "--apply-offset",
        str(targets["apply_ubatch"].resolved_file_offset),
        "--tensor-data-offset",
        str(profile.offsets.tensor_data),
        "--tensor-type-offset",
        str(profile.offsets.tensor_type),
        "--ready-timeout-seconds",
        str(configuration.ready_timeout_seconds),
        "--control-timeout-seconds",
        str(configuration.control_timeout_seconds),
    )


def _parse_topology_marker(line: bytes) -> dict[str, object]:
    if not isinstance(line, bytes) or not line.startswith(_TOPOLOGY_PREFIX):
        raise LogLinkedCaptureError(
            "runtime-state topology marker prefix is invalid"
        )
    payload = line[len(_TOPOLOGY_PREFIX) :]
    if not payload.endswith(b"\n"):
        raise LogLinkedCaptureError(
            "runtime-state topology marker is not terminated"
        )

    def reject_duplicates(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate topology key")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload[:-1].decode("ascii"),
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise LogLinkedCaptureError(
            "runtime-state topology marker is invalid"
        ) from error
    if not isinstance(value, dict) or set(value) != _TOPOLOGY_FIELDS:
        raise LogLinkedCaptureError(
            "runtime-state topology field set is invalid"
        )
    for field in ("n_ctx", "n_layer", "n_pos_per_embd", "n_seq_max", "n_stream"):
        item = value[field]
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise LogLinkedCaptureError(
                f"runtime-state topology {field} must be a positive integer"
            )
    if not isinstance(value["v_trans"], bool):
        raise LogLinkedCaptureError(
            "runtime-state topology v_trans must be boolean"
        )
    digest = value["profile_digest"]
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise LogLinkedCaptureError(
            "runtime-state topology profile digest is invalid"
        )
    canonical = _TOPOLOGY_PREFIX + json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii") + b"\n"
    if line != canonical:
        raise LogLinkedCaptureError(
            "runtime-state topology marker is not canonical"
        )
    return value


def _verify_private_socket(socket_path: Path) -> None:
    try:
        metadata = socket_path.lstat()
    except OSError as error:
        raise LogLinkedCaptureError(
            "runtime-state socket is missing"
        ) from error
    if (
        not stat.S_ISSOCK(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or (
            hasattr(os, "geteuid")
            and hasattr(metadata, "st_uid")
            and metadata.st_uid != os.geteuid()
        )
    ):
        raise LogLinkedCaptureError(
            "runtime-state socket is not private"
        )


class LogLinkedCaptureProcess:
    def __init__(
        self,
        *,
        configuration: LogLinkedCaptureConfiguration,
        process_factory: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
        socket_probe: Callable[[Path], None] = _verify_private_socket,
    ) -> None:
        self.configuration = configuration
        self.process_factory = process_factory
        self.socket_probe = socket_probe
        self.profile = load_runtime_state_observer_profile(
            configuration.profile_path
        )
        self._process: subprocess.Popen[bytes] | None = None
        self._service_log = None
        self._topology: dict[str, object] | None = None

    @property
    def topology(self) -> dict[str, object]:
        if self._topology is None:
            raise LogLinkedCaptureError(
                "runtime-state topology is unavailable"
            )
        return dict(self._topology)

    def start(
        self,
        topology_bootstrap: Callable[[], None],
    ) -> dict[str, object]:
        if self._process is not None:
            raise LogLinkedCaptureError(
                "Log-linked capture process was already started"
            )
        if not callable(topology_bootstrap):
            raise TypeError("topology_bootstrap must be callable")
        cfg = self.configuration
        cfg.socket_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if cfg.socket_path.exists() or cfg.socket_path.is_symlink():
            raise LogLinkedCaptureError(
                "runtime-state socket path already exists"
            )
        cfg.frame_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
        cfg.service_log_path.parent.mkdir(
            parents=True,
            exist_ok=True,
            mode=0o700,
        )
        self._service_log = cfg.service_log_path.open("xb")
        command = build_log_linked_service_command(cfg, self.profile)
        try:
            self._process = self.process_factory(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=self._service_log,
                bufsize=0,
            )
            attached = self._readline(cfg.ready_timeout_seconds)
            if attached != _ATTACHED:
                raise LogLinkedCaptureError(
                    "runtime-state service ATTACHED marker is invalid"
                )
            topology_bootstrap()
            topology = _parse_topology_marker(
                self._readline(cfg.ready_timeout_seconds)
            )
            self._validate_topology(topology)
            ready = self._readline(cfg.ready_timeout_seconds)
            if ready != _READY:
                raise LogLinkedCaptureError(
                    "runtime-state service READY marker is invalid"
                )
            self.socket_probe(cfg.socket_path)
            self._topology = topology
            return dict(topology)
        except BaseException:
            self.abort()
            raise

    def wait(self) -> None:
        process = self._process
        if process is None:
            raise LogLinkedCaptureError(
                "Log-linked capture process was not started"
            )
        try:
            return_code = process.wait(
                timeout=self.configuration.control_timeout_seconds
            )
        except subprocess.TimeoutExpired as error:
            raise LogLinkedCaptureError(
                "runtime-state service did not stop"
            ) from error
        finally:
            self._close_log()
        if return_code != 0:
            raise LogLinkedCaptureError(
                "runtime-state service stopped with an error"
            )

    def abort(self) -> None:
        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(
                    timeout=self.configuration.control_timeout_seconds
                )
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(
                    timeout=self.configuration.control_timeout_seconds
                )
        self._close_log()

    def _readline(self, timeout_seconds: float) -> bytes:
        process = self._process
        if process is None or process.stdout is None:
            raise LogLinkedCaptureError(
                "runtime-state service stdout is unavailable"
            )
        try:
            process.stdout.fileno()
        except (AttributeError, OSError):
            line = process.stdout.readline()
        else:
            readable, _, _ = select.select(
                (process.stdout,),
                (),
                (),
                timeout_seconds,
            )
            if not readable:
                raise LogLinkedCaptureError(
                    "runtime-state service marker timed out"
                )
            line = process.stdout.readline()
        if line:
            return line
        return_code = process.poll()
        if return_code is not None:
            raise LogLinkedCaptureError(
                "runtime-state service stopped before publishing its marker"
            )
        raise LogLinkedCaptureError(
            "runtime-state service marker is unavailable"
        )

    def _validate_topology(self, topology: dict[str, object]) -> None:
        expected = {
            "n_seq_max": self.profile.n_seq_max,
            "n_stream": self.profile.n_stream,
            "profile_digest": self.profile.profile_digest,
        }
        if not self.profile.has_dynamic_topology:
            expected.update(
                {
                    "n_layer": self.profile.n_layer,
                    "n_pos_per_embd": self.profile.n_pos_per_embd,
                }
            )
        if any(topology[field] != value for field, value in expected.items()):
            raise LogLinkedCaptureError(
                "runtime-state topology differs from the observer profile"
            )
        if self.profile.has_dynamic_topology and (
            topology["n_layer"] > self.profile.max_n_layer
            or topology["n_pos_per_embd"]
            > self.profile.max_n_pos_per_embd
        ):
            raise LogLinkedCaptureError(
                "runtime-state topology exceeds the observer capability"
            )
        if (
            self.profile.v_trans is not None
            and topology["v_trans"] != self.profile.v_trans
        ):
            raise LogLinkedCaptureError(
                "runtime-state topology value layout differs from the profile"
            )

    def _close_log(self) -> None:
        if self._service_log is not None:
            self._service_log.close()
            self._service_log = None


__all__ = [
    "LogLinkedCaptureConfiguration",
    "LogLinkedCaptureError",
    "LogLinkedCaptureProcess",
    "build_log_linked_service_command",
]
