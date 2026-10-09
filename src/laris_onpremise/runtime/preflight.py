from __future__ import annotations

import hashlib
import gzip
import json
import os
import platform
import re
import subprocess
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from laris_onpremise.core.records import canonical_json


HOST_PREFLIGHT_SCHEMA_VERSION: Final[str] = "1.0"
SUPPORTED_PROFILE_ID: Final[str] = (
    "ubuntu-24.04-linux6.17-x86_64-clang18-libbpf1.3"
)
VM24_LINUX_7_PROFILE_ID: Final[str] = (
    "ubuntu-24.04-linux7.0-x86_64-clang18-libbpf1.3"
)
REQUIRED_KERNEL_CONFIGS: Final[tuple[str, ...]] = (
    "CONFIG_BPF",
    "CONFIG_BPF_EVENTS",
    "CONFIG_BPF_SYSCALL",
    "CONFIG_DEBUG_INFO_BTF",
    "CONFIG_PERF_EVENTS",
    "CONFIG_UPROBES",
    "CONFIG_UPROBE_EVENTS",
)
REQUIRED_BPF_PROGRAM_TYPES: Final[tuple[str, ...]] = (
    "kprobe",
    "perf_event",
    "tracing",
)
REQUIRED_BPF_MAP_TYPES: Final[tuple[str, ...]] = ("percpu_array", "ringbuf")
REQUIRED_BPF_HELPERS: Final[tuple[str, ...]] = (
    "bpf_ringbuf_output",
    "bpf_ringbuf_reserve",
    "bpf_ringbuf_submit",
)


class HostQualificationError(ValueError):
    """Raised when measured host facts do not meet the pinned profile."""


def _require_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_nonnegative_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _require_bool(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _require_canonical_tuple(name: str, value: object) -> tuple[str, ...]:
    if not isinstance(value, tuple) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError(f"{name} must be a tuple of non-empty strings")
    if tuple(sorted(set(value))) != value:
        raise ValueError(f"{name} must be sorted and unique")
    return value


@dataclass(frozen=True)
class HostProfile:
    profile_id: str
    os_id: str
    os_version_id: str
    architecture: str
    kernel_major_minor: str
    python_major_minor: str
    clang_major: str
    gcc_major: str
    libbpf_major_minor: str
    bpftool_major_minor: str
    required_kernel_configs: tuple[str, ...]
    required_bpf_program_types: tuple[str, ...]
    required_bpf_map_types: tuple[str, ...]
    required_bpf_helpers: tuple[str, ...]

    @classmethod
    def supported(cls) -> "HostProfile":
        """Return the only host profile accepted by this phase."""
        return cls(
            profile_id=SUPPORTED_PROFILE_ID,
            os_id="ubuntu",
            os_version_id="24.04",
            architecture="x86_64",
            kernel_major_minor="6.17",
            python_major_minor="3.12",
            clang_major="18",
            gcc_major="13",
            libbpf_major_minor="1.3",
            bpftool_major_minor="7.7",
            required_kernel_configs=REQUIRED_KERNEL_CONFIGS,
            required_bpf_program_types=REQUIRED_BPF_PROGRAM_TYPES,
            required_bpf_map_types=REQUIRED_BPF_MAP_TYPES,
            required_bpf_helpers=REQUIRED_BPF_HELPERS,
        )

    @classmethod
    def vm24_linux_7_0(cls) -> "HostProfile":
        return cls(
            **{
                **cls.supported().__dict__,
                "profile_id": VM24_LINUX_7_PROFILE_ID,
                "kernel_major_minor": "7.0",
            }
        )

    @classmethod
    def known_profiles(cls) -> tuple["HostProfile", ...]:
        return (cls.supported(), cls.vm24_linux_7_0())


@dataclass(frozen=True)
class HostPreflight:
    schema_version: str
    profile_id: str
    collected_at_wall_time_ns: int
    collected_at_monotonic_ns: int
    host_boot_id: str
    os_id: str
    os_version_id: str
    architecture: str
    kernel_release: str
    python_version: str
    clang_version: str
    gcc_version: str
    libbpf_version: str
    bpftool_version: str
    effective_uid: int
    procfs_available: bool
    tracefs_mounted: bool
    btf_vmlinux_readable: bool
    kernel_configs: tuple[str, ...]
    bpf_program_types: tuple[str, ...]
    bpf_map_types: tuple[str, ...]
    bpf_helpers: tuple[str, ...]
    bpf_feature_report_digest: str
    privileged_bpf_probe_succeeded: bool
    pidfd_open_available: bool
    pidfd_send_signal_available: bool
    rename_noreplace_available: bool

    def __post_init__(self) -> None:
        if self.schema_version != HOST_PREFLIGHT_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {HOST_PREFLIGHT_SCHEMA_VERSION}"
            )
        for name in (
            "profile_id",
            "host_boot_id",
            "os_id",
            "os_version_id",
            "architecture",
            "kernel_release",
            "python_version",
            "clang_version",
            "gcc_version",
            "libbpf_version",
            "bpftool_version",
            "bpf_feature_report_digest",
        ):
            _require_text(name, getattr(self, name))
        for name in ("collected_at_wall_time_ns", "collected_at_monotonic_ns", "effective_uid"):
            _require_nonnegative_integer(name, getattr(self, name))
        for name in (
            "procfs_available",
            "tracefs_mounted",
            "btf_vmlinux_readable",
            "privileged_bpf_probe_succeeded",
            "pidfd_open_available",
            "pidfd_send_signal_available",
            "rename_noreplace_available",
        ):
            _require_bool(name, getattr(self, name))
        for name in (
            "kernel_configs",
            "bpf_program_types",
            "bpf_map_types",
            "bpf_helpers",
        ):
            _require_canonical_tuple(name, getattr(self, name))
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", self.bpf_feature_report_digest):
            raise ValueError("bpf_feature_report_digest must be a sha256 digest")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "collected_at_wall_time_ns": self.collected_at_wall_time_ns,
            "collected_at_monotonic_ns": self.collected_at_monotonic_ns,
            "host_boot_id": self.host_boot_id,
            "os_id": self.os_id,
            "os_version_id": self.os_version_id,
            "architecture": self.architecture,
            "kernel_release": self.kernel_release,
            "python_version": self.python_version,
            "clang_version": self.clang_version,
            "gcc_version": self.gcc_version,
            "libbpf_version": self.libbpf_version,
            "bpftool_version": self.bpftool_version,
            "effective_uid": self.effective_uid,
            "procfs_available": self.procfs_available,
            "tracefs_mounted": self.tracefs_mounted,
            "btf_vmlinux_readable": self.btf_vmlinux_readable,
            "kernel_configs": list(self.kernel_configs),
            "bpf_program_types": list(self.bpf_program_types),
            "bpf_map_types": list(self.bpf_map_types),
            "bpf_helpers": list(self.bpf_helpers),
            "bpf_feature_report_digest": self.bpf_feature_report_digest,
            "privileged_bpf_probe_succeeded": self.privileged_bpf_probe_succeeded,
            "pidfd_open_available": self.pidfd_open_available,
            "pidfd_send_signal_available": self.pidfd_send_signal_available,
            "rename_noreplace_available": self.rename_noreplace_available,
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json(self.to_mapping())

    def canonical_digest(self) -> str:
        return "sha256:" + hashlib.sha256(self.canonical_bytes()).hexdigest()


def _version_prefix(value: str, count: int) -> str | None:
    match = re.search(r"\d+(?:\.\d+){0,2}", value)
    if match is None:
        return None
    parts = match.group(0).split(".")
    if len(parts) < count:
        return None
    return ".".join(parts[:count])


def _read_os_release() -> dict[str, str]:
    path = Path("/etc/os-release")
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in line or line.startswith("#"):
            continue
        key, value = line.split("=", maxsplit=1)
        values[key] = value.strip().strip('"')
    return values


def _command_output(command: tuple[str, ...]) -> tuple[int, str]:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except OSError:
        return 127, ""
    return result.returncode, result.stdout + result.stderr


def _read_kernel_configs(
    kernel_release: str,
    *,
    candidates: tuple[Path, ...] | None = None,
) -> tuple[str, ...]:
    if candidates is None:
        candidates = (
            Path(f"/boot/config-{kernel_release}"),
            Path("/proc/config.gz"),
        )
    for path in candidates:
        if not path.is_file():
            continue
        if path.suffix == ".gz":
            try:
                text = gzip.open(path, "rt", encoding="utf-8", errors="replace").read()
            except OSError:
                continue
        else:
            text = path.read_text(encoding="utf-8", errors="replace")
        configured = {
            line.split("=", maxsplit=1)[0]
            for line in text.splitlines()
            if line.endswith("=y") or line.endswith("=m")
        }
        return tuple(sorted(configured & set(REQUIRED_KERNEL_CONFIGS)))
    return ()


def _kernel_configs_from_report(report: object) -> tuple[str, ...]:
    if not isinstance(report, Mapping):
        return ()
    configured: set[str] = set()

    def visit(value: object) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                if key in REQUIRED_KERNEL_CONFIGS and child in {True, "y", "m"}:
                    configured.add(key)
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(report)
    return tuple(sorted(configured))


_SECTION_NAMES: Final[dict[str, str]] = {
    "program": "program_types",
    "map": "map_types",
    "helper": "helpers",
}
_BPFTOOL_77_TOP_LEVEL_SECTIONS: Final[frozenset[str]] = frozenset(
    {
        "system_config",
        "syscall_config",
        "program_types",
        "map_types",
        "helpers",
        "misc",
    }
)
_REQUIRED_HELPER_SECTION_KEYS: Final[tuple[str, ...]] = (
    "kprobe_available_helpers",
    "perf_event_available_helpers",
    "tracing_available_helpers",
)
_PROGRAM_TYPE_KEY_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^have_[a-z0-9_]+_prog_type$"
)
_MAP_TYPE_KEY_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^have_[a-z0-9_]+_map_type$"
)
_HELPER_KEY_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^[a-z0-9_]+_available_helpers$"
)


def _section(report: object, kind: str) -> object | None:
    if not isinstance(report, Mapping):
        return None
    name = _SECTION_NAMES.get(kind)
    if name is None:
        return None
    return report.get(name)


def _feature_key_candidates(name: str, kind: str) -> frozenset[str]:
    if kind == "program":
        return frozenset({f"have_{name}_prog_type"})
    if kind == "map":
        return frozenset({f"have_{name}_map_type"})
    return frozenset()


def _parse_boolean_feature_section(
    section: object,
    required: tuple[str, ...],
    kind: str,
) -> tuple[str, ...] | None:
    if not isinstance(section, Mapping) or any(
        not isinstance(key, str) or not isinstance(value, bool)
        for key, value in section.items()
    ):
        return None
    return tuple(
        name
        for name in required
        if any(
            section.get(key) is True
            for key in _feature_key_candidates(name, kind)
        )
    )


def _parse_helper_section(
    section: object,
    required: tuple[str, ...],
) -> tuple[str, ...] | None:
    if not isinstance(section, Mapping):
        return None
    if any(key not in section for key in _REQUIRED_HELPER_SECTION_KEYS):
        return None
    for key, value in section.items():
        if not isinstance(key, str) or not key.endswith("_available_helpers"):
            return None
        if not isinstance(value, list) or any(
            not isinstance(item, str) for item in value
        ):
            return None
    # Uprobe observation needs the ringbuf helpers available to kprobe programs.
    kprobe_helpers = section["kprobe_available_helpers"]
    return tuple(name for name in required if name in kprobe_helpers)


def _feature_names(
    report: object,
    required: tuple[str, ...],
    kind: str,
) -> tuple[str, ...]:
    """Return required features from the exact bpftool JSON section only."""
    section = _section(report, kind)
    if kind in {"program", "map"}:
        parsed = _parse_boolean_feature_section(section, required, kind)
    elif kind == "helper":
        parsed = _parse_helper_section(section, required)
    else:
        parsed = None
    return parsed if parsed is not None else ()


def _complete_feature_report(report: object) -> bool:
    if (
        not isinstance(report, Mapping)
        or set(report) != _BPFTOOL_77_TOP_LEVEL_SECTIONS
        or any(not isinstance(value, Mapping) for value in report.values())
    ):
        return False
    program_section = _section(report, "program")
    map_section = _section(report, "map")
    if not isinstance(program_section, Mapping) or not isinstance(
        map_section, Mapping
    ):
        return False
    helper_section = _section(report, "helper")
    if not isinstance(helper_section, Mapping):
        return False
    if any(
        not isinstance(key, str)
        or _PROGRAM_TYPE_KEY_PATTERN.fullmatch(key) is None
        or not isinstance(value, bool)
        for key, value in program_section.items()
    ) or any(
        not isinstance(key, str)
        or _MAP_TYPE_KEY_PATTERN.fullmatch(key) is None
        or not isinstance(value, bool)
        for key, value in map_section.items()
    ) or any(
        not isinstance(key, str)
        or _HELPER_KEY_PATTERN.fullmatch(key) is None
        or not isinstance(value, list)
        or any(not isinstance(item, str) for item in value)
        or f"have_{key.removesuffix('_available_helpers')}_prog_type"
        not in program_section
        for key, value in helper_section.items()
    ):
        return False
    if any(
        not any(
            key in program_section
            for key in _feature_key_candidates(name, "program")
        )
        for name in REQUIRED_BPF_PROGRAM_TYPES
    ) or any(
        not any(key in map_section for key in _feature_key_candidates(name, "map"))
        for name in REQUIRED_BPF_MAP_TYPES
    ):
        return False
    return (
        _parse_boolean_feature_section(
            program_section,
            REQUIRED_BPF_PROGRAM_TYPES,
            "program",
        )
        is not None
        and _parse_boolean_feature_section(
            map_section,
            REQUIRED_BPF_MAP_TYPES,
            "map",
        )
        is not None
        and _parse_helper_section(
            helper_section,
            REQUIRED_BPF_HELPERS,
        )
        is not None
    )


def _unescape_mountinfo_path(value: str) -> str | None:
    def replacement(match: re.Match[str]) -> str:
        return chr(int(match.group(1), 8))

    if re.search(r"\\(?![0-7]{3})", value):
        return None
    try:
        return re.sub(r"\\([0-7]{3})", replacement, value)
    except ValueError:
        return None


def _tracefs_mounted_from_mountinfo(
    mountinfo: str,
    *,
    mountpoints: tuple[str, ...] = (
        "/sys/kernel/tracing",
        "/sys/kernel/debug/tracing",
    ),
) -> bool:
    """Check exact tracefs mounts from Linux mountinfo, failing closed."""
    if not isinstance(mountinfo, str) or not mountinfo:
        return False
    tracefs_found = False
    for line in mountinfo.splitlines():
        fields = line.split()
        try:
            separator = fields.index("-")
        except ValueError:
            return False
        if separator < 6 or len(fields) < separator + 4:
            return False
        mountpoint = _unescape_mountinfo_path(fields[4])
        if mountpoint is None:
            return False
        filesystem_type = fields[separator + 1]
        if mountpoint in mountpoints and filesystem_type == "tracefs":
            tracefs_found = True
    return tracefs_found


def _read_tracefs_mounted(
    mountinfo_path: Path = Path("/proc/self/mountinfo"),
) -> bool:
    try:
        return _tracefs_mounted_from_mountinfo(
            mountinfo_path.read_text(encoding="utf-8", errors="strict")
        )
    except OSError:
        return False


def _renameat2_symbol_available() -> bool:
    if platform.system() != "Linux":
        return False
    try:
        import ctypes

        return hasattr(ctypes.CDLL(None), "renameat2")
    except OSError:
        return False


def _collect_host_preflight(
    runner: Callable[[tuple[str, ...]], tuple[int, str]] = _command_output,
    *,
    profile: HostProfile | None = None,
) -> HostPreflight:
    """Measure host facts without loading or attaching any observer program."""
    selected_profile = profile or HostProfile.supported()
    if selected_profile not in HostProfile.known_profiles():
        raise HostQualificationError("profile must be a known supported profile")
    os_release = _read_os_release()
    kernel_release = platform.release()
    probe_status, probe_output = runner(
        ("bpftool", "feature", "probe", "kernel", "full", "-j")
    ) if platform.system() == "Linux" else (127, "")
    try:
        report: object = json.loads(probe_output) if probe_status == 0 else {}
    except json.JSONDecodeError:
        report = {}
    clang_status, clang_output = runner(("clang", "--version"))
    gcc_status, gcc_output = runner(("gcc", "--version"))
    libbpf_status, libbpf_output = runner(("pkg-config", "--modversion", "libbpf"))
    bpftool_status, bpftool_output = runner(("bpftool", "version"))
    _ = (clang_status, gcc_status, libbpf_status, bpftool_status)
    procfs = Path("/proc").is_dir()
    boot_id_path = Path("/proc/sys/kernel/random/boot_id")
    boot_id = (
        boot_id_path.read_text(encoding="ascii", errors="replace").strip()
        if boot_id_path.is_file()
        else "unavailable"
    )
    tracefs = _read_tracefs_mounted()
    btf_readable = Path("/sys/kernel/btf/vmlinux").is_file() and os.access(
        "/sys/kernel/btf/vmlinux", os.R_OK
    )
    return HostPreflight(
        schema_version=HOST_PREFLIGHT_SCHEMA_VERSION,
        profile_id=selected_profile.profile_id,
        collected_at_wall_time_ns=time.time_ns(),
        collected_at_monotonic_ns=time.monotonic_ns(),
        host_boot_id=boot_id or "unavailable",
        os_id=os_release.get("ID", platform.system().lower() or "unknown"),
        os_version_id=os_release.get("VERSION_ID", "unavailable"),
        architecture=platform.machine() or "unknown",
        kernel_release=kernel_release or "unavailable",
        python_version=platform.python_version() or "unavailable",
        clang_version=clang_output.strip() or "unavailable",
        gcc_version=gcc_output.strip() or "unavailable",
        libbpf_version=libbpf_output.strip() or "unavailable",
        bpftool_version=bpftool_output.strip() or "unavailable",
        effective_uid=os.geteuid() if hasattr(os, "geteuid") else 0,
        procfs_available=procfs,
        tracefs_mounted=tracefs,
        btf_vmlinux_readable=btf_readable,
        kernel_configs=tuple(
            sorted(
                set(_read_kernel_configs(kernel_release))
                | set(_kernel_configs_from_report(report))
            )
        ),
        bpf_program_types=_feature_names(report, REQUIRED_BPF_PROGRAM_TYPES, "program"),
        bpf_map_types=_feature_names(report, REQUIRED_BPF_MAP_TYPES, "map"),
        bpf_helpers=_feature_names(report, REQUIRED_BPF_HELPERS, "helper"),
        bpf_feature_report_digest="sha256:" + hashlib.sha256(probe_output.encode("utf-8")).hexdigest(),
        privileged_bpf_probe_succeeded=(
            probe_status == 0 and _complete_feature_report(report)
        ),
        pidfd_open_available=hasattr(os, "pidfd_open"),
        pidfd_send_signal_available=hasattr(__import__("signal"), "pidfd_send_signal"),
        rename_noreplace_available=_renameat2_symbol_available(),
    )


def collect_host_preflight(
    profile: HostProfile | None = None,
) -> HostPreflight:
    """Collect a non-invasive host qualification record."""
    return _collect_host_preflight(profile=profile)


def require_qualified_host(profile: HostProfile, host: HostPreflight) -> None:
    """Fail closed unless every pinned host requirement is satisfied."""
    if not isinstance(profile, HostProfile):
        raise TypeError("profile must be a HostProfile")
    if not isinstance(host, HostPreflight):
        raise TypeError("host must be a HostPreflight")
    if profile not in HostProfile.known_profiles():
        raise HostQualificationError("profile must be a known supported profile")
    reasons: list[str] = []
    exact = {
        "profile_id": (host.profile_id, profile.profile_id),
        "os_id": (host.os_id, profile.os_id),
        "os_version_id": (host.os_version_id, profile.os_version_id),
        "architecture": (host.architecture, profile.architecture),
    }
    for field, (actual, expected) in exact.items():
        if actual != expected:
            reasons.append(f"{field} must be {expected!r}, observed {actual!r}")
    version_fields = (
        ("kernel_release", host.kernel_release, profile.kernel_major_minor, 2),
        ("python_version", host.python_version, profile.python_major_minor, 2),
        ("clang_version", host.clang_version, profile.clang_major, 1),
        ("gcc_version", host.gcc_version, profile.gcc_major, 1),
        ("libbpf_version", host.libbpf_version, profile.libbpf_major_minor, 2),
        ("bpftool_version", host.bpftool_version, profile.bpftool_major_minor, 2),
    )
    for field, actual, expected, count in version_fields:
        if _version_prefix(actual, count) != expected:
            reasons.append(f"{field} must have version prefix {expected!r}")
    required_features = (
        ("kernel_configs", set(profile.required_kernel_configs), set(host.kernel_configs)),
        ("bpf_program_types", set(profile.required_bpf_program_types), set(host.bpf_program_types)),
        ("bpf_map_types", set(profile.required_bpf_map_types), set(host.bpf_map_types)),
        ("bpf_helpers", set(profile.required_bpf_helpers), set(host.bpf_helpers)),
    )
    for field, required, observed in required_features:
        missing = sorted(required - observed)
        if missing:
            reasons.append(f"{field} missing {missing}")
    for field in (
        "procfs_available",
        "tracefs_mounted",
        "btf_vmlinux_readable",
        "privileged_bpf_probe_succeeded",
        "pidfd_open_available",
        "pidfd_send_signal_available",
        "rename_noreplace_available",
    ):
        if not getattr(host, field):
            reasons.append(f"{field} must be true")
    if reasons:
        raise HostQualificationError("host qualification failed: " + "; ".join(reasons))
