"""Compatibility import for the relocated runtime-state acquirer."""

from laris_onpremise.kv_state.acquisition.acquirer import *
from laris_onpremise.kv_state.acquisition.acquirer import (
    _validate_backing_ranges,
    _validate_target_marker,
)
