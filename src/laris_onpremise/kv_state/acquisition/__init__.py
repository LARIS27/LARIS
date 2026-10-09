"""Common KV-state acquisition contracts and implementations."""
"""Common selective runtime-state acquisition contracts and services."""

from .contracts import (
    SIMPLE_KV_MEMORY_FAMILY,
    SIMPLE_KV_PAYLOAD_FORMAT,
    RuntimeStateAcquirer,
    RuntimeStateBindingError,
    RuntimeStateCaptureRequest,
    RuntimeStateContractError,
    RuntimeStatePayloadCommit,
    validate_payload_binding,
)
from .acquirer import (
    AcquisitionBindingError,
    AcquisitionError,
    AcquisitionObserverIntegrityError,
    AcquisitionStabilityError,
    AcquisitionTargetConflict,
    ObservedKvBackingRange,
    RuntimeStateAcquisitionSource,
    RuntimeStateAcquisitionView,
    RuntimeStateTargetMarker,
    SelectiveRuntimeStateAcquirer,
)
from .layout import (
    LayoutValidationError,
    ReadSpan,
    SelectedCell,
    SimpleKvLayout,
    TensorRowLayout,
    plan_selected_rows,
)
from .payload import (
    PayloadValidationError,
    SimpleKvPayloadSummary,
    encode_simple_kv_payload,
    inspect_simple_kv_payload,
)
from .reader import (
    AddressReader,
    MemoryReadError,
    ProcessIdentityError,
    ProcessMemoryReader,
)
from laris_onpremise.kv_state.evidence.store import (
    PayloadStoreError,
    PayloadStoreRootIntegrityError,
    RuntimeStatePayloadStore,
)

__all__ = [
    "SIMPLE_KV_MEMORY_FAMILY",
    "SIMPLE_KV_PAYLOAD_FORMAT",
    "RuntimeStateAcquirer",
    "RuntimeStateBindingError",
    "RuntimeStateCaptureRequest",
    "RuntimeStateContractError",
    "RuntimeStatePayloadCommit",
    "validate_payload_binding",
    "AcquisitionBindingError",
    "AcquisitionError",
    "AcquisitionObserverIntegrityError",
    "AcquisitionStabilityError",
    "AcquisitionTargetConflict",
    "ObservedKvBackingRange",
    "RuntimeStateAcquisitionSource",
    "RuntimeStateAcquisitionView",
    "RuntimeStateTargetMarker",
    "SelectiveRuntimeStateAcquirer",
    "LayoutValidationError",
    "ReadSpan",
    "SelectedCell",
    "SimpleKvLayout",
    "TensorRowLayout",
    "plan_selected_rows",
    "PayloadValidationError",
    "SimpleKvPayloadSummary",
    "encode_simple_kv_payload",
    "inspect_simple_kv_payload",
    "AddressReader",
    "MemoryReadError",
    "ProcessIdentityError",
    "ProcessMemoryReader",
    "PayloadStoreError",
    "PayloadStoreRootIntegrityError",
    "RuntimeStatePayloadStore",
]
