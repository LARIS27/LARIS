"""Compatibility exports for :mod:`laris_onpremise.runtime.preflight`."""

from laris_onpremise.runtime.preflight import (
    HOST_PREFLIGHT_SCHEMA_VERSION,
    REQUIRED_BPF_HELPERS,
    REQUIRED_BPF_MAP_TYPES,
    REQUIRED_BPF_PROGRAM_TYPES,
    REQUIRED_KERNEL_CONFIGS,
    SUPPORTED_PROFILE_ID,
    VM24_LINUX_7_PROFILE_ID,
    HostPreflight,
    HostProfile,
    HostQualificationError,
    _collect_host_preflight,
    _feature_names,
    _kernel_configs_from_report,
    _read_kernel_configs,
    _renameat2_symbol_available,
    _tracefs_mounted_from_mountinfo,
    collect_host_preflight,
    require_qualified_host,
)

__all__ = [
    "HOST_PREFLIGHT_SCHEMA_VERSION",
    "REQUIRED_BPF_HELPERS",
    "REQUIRED_BPF_MAP_TYPES",
    "REQUIRED_BPF_PROGRAM_TYPES",
    "REQUIRED_KERNEL_CONFIGS",
    "SUPPORTED_PROFILE_ID",
    "VM24_LINUX_7_PROFILE_ID",
    "HostPreflight",
    "HostProfile",
    "HostQualificationError",
    "collect_host_preflight",
    "require_qualified_host",
]
