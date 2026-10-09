from __future__ import annotations

from enum import StrEnum


class RuntimeMode(StrEnum):
    ATTACHED = "ATTACHED"
    MANAGED_LEGACY = "MANAGED_LEGACY"


def parse_runtime_mode(value: object) -> RuntimeMode:
    if not isinstance(value, str):
        raise ValueError("runtime_mode must be text")
    try:
        return RuntimeMode(value)
    except ValueError as error:
        raise ValueError("runtime_mode is unsupported") from error
