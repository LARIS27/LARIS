from __future__ import annotations

import os
import json
import stat
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from laris_onpremise.runtime.elf import ProcessInspectionError, read_process_identity


class _ProxyCore(Protocol):
    @property
    def admission_open(self) -> bool: ...


class _Upstream(Protocol):
    def health(self) -> bool: ...

    def get_slots(self) -> tuple[object, ...]: ...


def _positive_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _nonempty_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be non-empty text")
    return value


@dataclass(frozen=True)
class LiveObserverHealthState:
    status: str
    observer_pid: int
    observer_start_time_ticks: int
    target_pid: int
    target_start_time_ticks: int
    probe_symbol: str
    lost_event_count: int
    counter_gap_count: int

    def __post_init__(self) -> None:
        if self.status not in {"ATTACHED", "STOPPING", "FAILED"}:
            raise ValueError("unsupported observer health status")
        for name in ("observer_pid", "target_pid"):
            _positive_integer(name, getattr(self, name))
        for name in (
            "observer_start_time_ticks",
            "target_start_time_ticks",
        ):
            _positive_integer(name, getattr(self, name))
        _nonempty_text("probe_symbol", self.probe_symbol)
        _nonnegative_integer("lost_event_count", self.lost_event_count)
        _nonnegative_integer("counter_gap_count", self.counter_gap_count)

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": "1.0",
            "status": self.status,
            "observer_pid": self.observer_pid,
            "observer_start_time_ticks": self.observer_start_time_ticks,
            "target_pid": self.target_pid,
            "target_start_time_ticks": self.target_start_time_ticks,
            "probe_symbol": self.probe_symbol,
            "lost_event_count": self.lost_event_count,
            "counter_gap_count": self.counter_gap_count,
        }


_OBSERVER_HEALTH_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "observer_pid",
        "observer_start_time_ticks",
        "target_pid",
        "target_start_time_ticks",
        "probe_symbol",
        "lost_event_count",
        "counter_gap_count",
    }
)

_RUNTIME_TOPOLOGY_FIELDS = frozenset(
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


def _validated_runtime_topology(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != _RUNTIME_TOPOLOGY_FIELDS:
        raise ValueError("runtime topology field set is invalid")
    for field in ("n_ctx", "n_layer", "n_pos_per_embd", "n_seq_max", "n_stream"):
        item = value[field]
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise ValueError(f"runtime topology {field} must be positive")
    if not isinstance(value["v_trans"], bool):
        raise ValueError("runtime topology v_trans must be boolean")
    digest = value["profile_digest"]
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise ValueError("runtime topology profile digest is invalid")
    return dict(value)


class RuntimeTopologyFileProvider:
    def __init__(self, path: Path, *, maximum_bytes: int = 4096) -> None:
        if maximum_bytes < 1:
            raise ValueError("maximum_bytes must be positive")
        self._path = Path(path)
        self._maximum_bytes = maximum_bytes

    def __call__(self) -> dict[str, object]:
        try:
            metadata = self._path.lstat()
            if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
                raise ValueError("runtime topology must be a regular file")
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(self._path, flags)
            try:
                opened = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(opened.st_mode)
                    or opened.st_dev != metadata.st_dev
                    or opened.st_ino != metadata.st_ino
                ):
                    raise ValueError("runtime topology file identity changed")
                raw = os.read(descriptor, self._maximum_bytes + 1)
                if len(raw) > self._maximum_bytes:
                    raise ValueError("runtime topology file is too large")
            finally:
                os.close(descriptor)
        except ValueError:
            raise
        except OSError as error:
            raise ValueError("runtime topology file cannot be read") from error

        def reject_duplicates(
            pairs: list[tuple[str, object]],
        ) -> dict[str, object]:
            result: dict[str, object] = {}
            for key, item in pairs:
                if key in result:
                    raise ValueError("runtime topology contains a duplicate key")
                result[key] = item
            return result

        try:
            value = json.loads(
                raw.decode("ascii"), object_pairs_hook=reject_duplicates
            )
        except ValueError:
            raise
        except (UnicodeError, json.JSONDecodeError) as error:
            raise ValueError("runtime topology is not strict JSON") from error
        topology = _validated_runtime_topology(value)
        canonical = (
            json.dumps(
                topology,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("ascii")
        if raw != canonical:
            raise ValueError("runtime topology file is not canonical")
        return topology


class FileObserverStatusProvider:
    def __init__(self, path: Path, *, maximum_bytes: int = 16 * 1024) -> None:
        if maximum_bytes < 1:
            raise ValueError("maximum_bytes must be positive")
        self._path = Path(path)
        self._maximum_bytes = maximum_bytes

    def __call__(self) -> LiveObserverHealthState:
        try:
            metadata = self._path.lstat()
            if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
                raise ValueError("observer health must be a regular file")
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(self._path, flags)
            try:
                opened = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(opened.st_mode)
                    or opened.st_dev != metadata.st_dev
                    or opened.st_ino != metadata.st_ino
                ):
                    raise ValueError("observer health file identity changed")
                raw = os.read(descriptor, self._maximum_bytes + 1)
                if len(raw) > self._maximum_bytes:
                    raise ValueError("observer health file is too large")
            finally:
                os.close(descriptor)
        except ValueError:
            raise
        except OSError as error:
            raise ValueError("observer health file cannot be read") from error

        def reject_duplicates(
            pairs: list[tuple[str, object]],
        ) -> dict[str, object]:
            result: dict[str, object] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("observer health contains a duplicate key")
                result[key] = value
            return result

        try:
            value = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicates)
        except ValueError:
            raise
        except (UnicodeError, json.JSONDecodeError) as error:
            raise ValueError("observer health is not strict JSON") from error
        if not isinstance(value, dict) or set(value) != _OBSERVER_HEALTH_FIELDS:
            raise ValueError("observer health field set is invalid")
        if value["schema_version"] != "1.0":
            raise ValueError("observer health schema_version must be 1.0")
        return LiveObserverHealthState(
            status=value["status"],
            observer_pid=value["observer_pid"],
            observer_start_time_ticks=value["observer_start_time_ticks"],
            target_pid=value["target_pid"],
            target_start_time_ticks=value["target_start_time_ticks"],
            probe_symbol=value["probe_symbol"],
            lost_event_count=value["lost_event_count"],
            counter_gap_count=value["counter_gap_count"],
        )


def process_matches(pid: int, start_time_ticks: int) -> bool:
    try:
        identity = read_process_identity(pid)
    except (OSError, ValueError, ProcessInspectionError):
        return False
    return identity.pid == pid and identity.start_time_ticks == start_time_ticks


def path_is_writable(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except OSError:
        return False
    if stat.S_ISLNK(metadata.st_mode):
        return False
    expected_type = stat.S_ISDIR if path.is_dir() else stat.S_ISREG
    return expected_type(metadata.st_mode) and os.access(path, os.W_OK)


def unix_socket_identity(path: Path) -> tuple[int, int]:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISSOCK(metadata.st_mode):
        raise ValueError("runtime-state socket is not a trusted Unix socket")
    if os.name == "posix" and stat.S_IMODE(metadata.st_mode) != 0o600:
        raise ValueError("runtime-state socket mode is not 0600")
    return metadata.st_dev, metadata.st_ino


class ProxyHealthProvider:
    """Project verified live process state through the proxy health endpoint."""

    def __init__(
        self,
        *,
        server_id: str,
        core: _ProxyCore,
        upstream: _Upstream,
        hosting_application: str,
        model_name: str,
        slot_count: int,
        runtime_library: str,
        probe_symbol: str,
        runtime_pid: int,
        runtime_start_time_ticks: int,
        observer_status: Callable[[], LiveObserverHealthState],
        runtime_state_service_pid: int,
        runtime_state_service_start_time_ticks: int,
        runtime_state_socket: Path,
        ledger_path: Path,
        runtime_state_payload_root: Path,
        runtime_topology: Mapping[str, object],
        process_matches: Callable[[int, int], bool] = process_matches,
        socket_identity: Callable[[Path], tuple[int, int]] = unix_socket_identity,
        path_writable: Callable[[Path], bool] = path_is_writable,
    ) -> None:
        self._server_id = _nonempty_text("server_id", server_id)
        self._core = core
        self._upstream = upstream
        self._hosting_application = _nonempty_text(
            "hosting_application", hosting_application
        )
        self._model_name = _nonempty_text("model_name", model_name)
        self._slot_count = _positive_integer("slot_count", slot_count)
        self._runtime_library = _nonempty_text("runtime_library", runtime_library)
        self._probe_symbol = _nonempty_text("probe_symbol", probe_symbol)
        self._runtime_pid = _positive_integer("runtime_pid", runtime_pid)
        self._runtime_start_time_ticks = _positive_integer(
            "runtime_start_time_ticks", runtime_start_time_ticks
        )
        self._observer_status = observer_status
        self._runtime_state_service_pid = _positive_integer(
            "runtime_state_service_pid", runtime_state_service_pid
        )
        self._runtime_state_service_start_time_ticks = _positive_integer(
            "runtime_state_service_start_time_ticks",
            runtime_state_service_start_time_ticks,
        )
        self._runtime_state_socket = Path(runtime_state_socket)
        if not self._runtime_state_socket.is_absolute():
            raise ValueError("runtime_state_socket must be absolute")
        self._socket_identity = socket_identity
        self._runtime_state_socket_identity = socket_identity(
            self._runtime_state_socket
        )
        for name, value in zip(
            ("runtime_state_socket_device", "runtime_state_socket_inode"),
            self._runtime_state_socket_identity,
        ):
            _nonnegative_integer(name, value)
        self._ledger_path = Path(ledger_path)
        self._runtime_state_payload_root = Path(runtime_state_payload_root)
        self._runtime_topology = _validated_runtime_topology(
            dict(runtime_topology)
        )
        self._process_matches = process_matches
        self._path_writable = path_writable

    def _upstream_ready(self) -> bool:
        try:
            if not self._upstream.health():
                return False
            slots = self._upstream.get_slots()
            slot_ids = tuple(sorted(getattr(slot, "slot_id", None) for slot in slots))
        except Exception:
            return False
        return slot_ids == tuple(range(self._slot_count))

    def __call__(self) -> Mapping[str, object]:
        runtime_alive = self._process_matches(
            self._runtime_pid, self._runtime_start_time_ticks
        )
        upstream_ready = runtime_alive and self._upstream_ready()
        try:
            observer = self._observer_status()
        except Exception:
            observer = LiveObserverHealthState(
                status="FAILED",
                observer_pid=self._runtime_pid,
                observer_start_time_ticks=self._runtime_start_time_ticks,
                target_pid=self._runtime_pid,
                target_start_time_ticks=self._runtime_start_time_ticks,
                probe_symbol=self._probe_symbol,
                lost_event_count=1,
                counter_gap_count=1,
            )
        observer_ready = (
            observer.status == "ATTACHED"
            and observer.probe_symbol == self._probe_symbol
            and observer.target_pid == self._runtime_pid
            and observer.target_start_time_ticks == self._runtime_start_time_ticks
            and self._process_matches(
                observer.observer_pid, observer.observer_start_time_ticks
            )
            and observer.lost_event_count == 0
            and observer.counter_gap_count == 0
            and runtime_alive
        )
        runtime_state_service_alive = self._process_matches(
            self._runtime_state_service_pid,
            self._runtime_state_service_start_time_ticks,
        )
        try:
            current_socket_identity = self._socket_identity(
                self._runtime_state_socket
            )
        except (OSError, ValueError):
            current_socket_identity = None
        runtime_state_observer_ready = (
            runtime_state_service_alive
            and current_socket_identity == self._runtime_state_socket_identity
            and runtime_alive
        )
        ledger_writable = self._path_writable(self._ledger_path)
        runtime_state_payload_writable = self._path_writable(
            self._runtime_state_payload_root
        )
        evidence_ready = ledger_writable and runtime_state_payload_writable
        admission_open = self._core.admission_open
        proxy_ready = admission_open and upstream_ready
        all_ready = (
            proxy_ready
            and observer_ready
            and runtime_state_observer_ready
            and evidence_ready
        )
        status = "READY" if all_ready else "NOT_READY"
        return {
            "schema_version": "1.0",
            "status": status,
            "server_id": self._server_id,
            "proxy": {
                "status": "READY" if proxy_ready else "NOT_READY",
                "admission_open": admission_open,
            },
            "upstream": {
                "status": "READY" if upstream_ready else "NOT_READY",
                "hosting_application": self._hosting_application,
                "model_name": self._model_name,
                "slot_count": self._slot_count,
            },
            "runtime": {
                "status": "READY" if runtime_alive else "NOT_READY",
                "runtime_library": self._runtime_library,
                "runtime_pid": self._runtime_pid,
                "topology": dict(self._runtime_topology),
            },
            "observer": {
                "status": "READY" if observer_ready else "NOT_READY",
                "attached": observer_ready,
                "target_pid": observer.target_pid,
                "probe_symbol": self._probe_symbol,
                "lost_event_count": observer.lost_event_count,
                "counter_gap_count": observer.counter_gap_count,
            },
            "runtime_state_observer": {
                "status": (
                    "READY" if runtime_state_observer_ready else "NOT_READY"
                ),
                "service_pid": self._runtime_state_service_pid,
                "service_start_time_ticks": (
                    self._runtime_state_service_start_time_ticks
                ),
                "socket_device": self._runtime_state_socket_identity[0],
                "socket_inode": self._runtime_state_socket_identity[1],
            },
            "evidence": {
                "status": "READY" if evidence_ready else "NOT_READY",
                "ledger_writable": ledger_writable,
                "runtime_state_payload_writable": runtime_state_payload_writable,
            },
        }
