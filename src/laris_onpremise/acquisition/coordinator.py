"""Compatibility import for relocated Proxy acquisition coordination."""

from laris_onpremise.attribution.proxy.coordinator import (
    AcquisitionCoordinationError,
    AcquisitionCoordinator,
)

__all__ = ["AcquisitionCoordinationError", "AcquisitionCoordinator"]
