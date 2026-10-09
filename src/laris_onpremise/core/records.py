from __future__ import annotations

import base64
import hashlib
import hmac
import json
import math
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import TypeAlias


JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]

SCHEMA_VERSION = "1.0"


class RecordValidationError(ValueError):
    """Raised when a record violates the LARIS data contract."""


class IdentityKeyNotFoundError(LookupError):
    """Raised when an identity key ID is not present in the deployment registry."""


class LeaseState(str, Enum):
    RESERVED = "RESERVED"
    SUBMITTED = "SUBMITTED"
    ACTIVE = "ACTIVE"
    QUIESCING = "QUIESCING"
    CHECKPOINTING = "CHECKPOINTING"
    CHECKPOINTED = "CHECKPOINTED"
    CAPTURE_SKIPPED = "CAPTURE_SKIPPED"
    RELEASED = "RELEASED"
    UNKNOWN = "UNKNOWN"
    QUARANTINED = "QUARANTINED"


LEASE_EVENT_TYPES = frozenset(
    {
        "request_received",
        "request_queued",
        "request_queue_cancelled",
        "lease_reserved",
        "dispatch_authorized",
        "upstream_request_sent",
        "slot_assignment_confirmed",
        "terminal_response_received",
        "slot_idle_confirmed",
        "runtime_state_capture_started",
        "runtime_state_acquisition_requested",
        "runtime_state_acquisition_copied",
        "runtime_state_acquisition_completed",
        "runtime_state_payload_committed",
        "runtime_state_post_copy_failed",
        "capture_skipped",
        "lease_released",
        "lease_unknown",
        "slot_quarantined",
        "controller_recovered",
        "runtime_started",
        "audit_violation",
    }
)


def _require_nonempty_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RecordValidationError(f"{name} must be a non-empty string")
    return value


def _require_no_nul(name: str, value: str) -> str:
    if "\0" in value:
        raise RecordValidationError(f"{name} must not contain NUL")
    return value


def _require_positive_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RecordValidationError(f"{name} must be a positive integer")
    return value


def _require_nonnegative_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RecordValidationError(f"{name} must be a non-negative integer")
    return value


def _normalize_json(value: object, path: str = "$") -> JsonValue:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RecordValidationError(f"{path} contains a non-finite number")
        return value
    if isinstance(value, list):
        return [_normalize_json(item, f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, Mapping):
        normalized: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise RecordValidationError(f"{path} contains a non-string key")
            normalized[key] = _normalize_json(item, f"{path}.{key}")
        return normalized
    raise RecordValidationError(f"{path} contains an unsupported JSON value")


@dataclass(frozen=True)
class RequestIdentity:
    user_hmac: str
    identity_key_id: str
    auth_receipt_id: str
    session_ref: str
    request_ref: str
    connection_ref: str
    retry_of: str | None = None

    def __post_init__(self) -> None:
        _require_nonempty_text("user_hmac", self.user_hmac)
        _require_nonempty_text("identity_key_id", self.identity_key_id)
        _require_nonempty_text("auth_receipt_id", self.auth_receipt_id)
        _require_nonempty_text("session_ref", self.session_ref)
        _require_nonempty_text("request_ref", self.request_ref)
        _require_nonempty_text("connection_ref", self.connection_ref)
        if self.retry_of is not None:
            _require_nonempty_text("retry_of", self.retry_of)
            if self.retry_of == self.request_ref:
                raise RecordValidationError("retry_of must differ from request_ref")


@dataclass(frozen=True)
class IdentityKey:
    identity_key_id: str
    secret: bytes = field(repr=False)

    def __post_init__(self) -> None:
        _require_nonempty_text("identity_key_id", self.identity_key_id)
        if not isinstance(self.secret, bytes) or len(self.secret) < 16:
            raise RecordValidationError(
                "identity key secret must contain at least 16 bytes"
            )
        object.__setattr__(self, "secret", bytes(self.secret))


class IdentityKeyRegistry:
    """Deployment-scoped resolver that binds a key ID to exactly one secret."""

    def __init__(self, keys: Iterable[IdentityKey]) -> None:
        resolved: dict[str, IdentityKey] = {}
        for key in keys:
            if not isinstance(key, IdentityKey):
                raise RecordValidationError(
                    "identity key registry entries must be IdentityKey objects"
                )
            if key.identity_key_id in resolved:
                raise RecordValidationError(
                    f"duplicate identity_key_id: {key.identity_key_id}"
                )
            resolved[key.identity_key_id] = key
        if not resolved:
            raise RecordValidationError(
                "identity key registry must contain at least one key"
            )
        self._keys = resolved

    def resolve(self, identity_key_id: str) -> IdentityKey:
        _require_nonempty_text("identity_key_id", identity_key_id)
        try:
            return self._keys[identity_key_id]
        except KeyError as error:
            raise IdentityKeyNotFoundError(
                f"identity key ID is unknown: {identity_key_id}"
            ) from error


@dataclass(frozen=True)
class LeaseRecord:
    lease_id: str
    request: RequestIdentity
    runtime_instance_id: str
    context_ref: str
    context_ref_source: str
    expected_slot_id: int
    expected_seq_id: int
    lease_generation: int
    response_confirmed_slot_id: int | None
    observed_seq_id: int | None
    slot_epoch: int | None
    state: LeaseState
    holds_sequence: bool
    backend_released: bool = False
    capture_id: str | None = None
    capture_skipped_reason: str | None = None

    def __post_init__(self) -> None:
        _require_nonempty_text("lease_id", self.lease_id)
        if not isinstance(self.request, RequestIdentity):
            raise RecordValidationError("request must be a RequestIdentity")
        _require_nonempty_text("runtime_instance_id", self.runtime_instance_id)
        _require_nonempty_text("context_ref", self.context_ref)
        _require_nonempty_text("context_ref_source", self.context_ref_source)
        _require_nonnegative_integer("expected_slot_id", self.expected_slot_id)
        _require_nonnegative_integer("expected_seq_id", self.expected_seq_id)
        _require_positive_integer("lease_generation", self.lease_generation)
        if self.response_confirmed_slot_id is not None:
            _require_nonnegative_integer(
                "response_confirmed_slot_id", self.response_confirmed_slot_id
            )
        if self.observed_seq_id is not None:
            _require_nonnegative_integer("observed_seq_id", self.observed_seq_id)
        if self.slot_epoch is not None:
            _require_positive_integer("slot_epoch", self.slot_epoch)
        if not isinstance(self.state, LeaseState):
            raise RecordValidationError("state must be a LeaseState")
        if not isinstance(self.holds_sequence, bool):
            raise RecordValidationError("holds_sequence must be a boolean")
        if not isinstance(self.backend_released, bool):
            raise RecordValidationError("backend_released must be a boolean")
        if self.capture_id is not None:
            _require_nonempty_text("capture_id", self.capture_id)
        if self.capture_skipped_reason is not None:
            _require_nonempty_text(
                "capture_skipped_reason", self.capture_skipped_reason
            )

    @classmethod
    def reserved(
        cls,
        *,
        lease_id: str,
        request: RequestIdentity,
        runtime_instance_id: str,
        context_ref: str,
        context_ref_source: str,
        expected_slot_id: int,
        expected_seq_id: int,
        lease_generation: int,
        slot_count: int,
    ) -> "LeaseRecord":
        _require_positive_integer("slot_count", slot_count)
        _require_nonnegative_integer("expected_slot_id", expected_slot_id)
        if expected_slot_id >= slot_count:
            raise RecordValidationError(
                f"expected_slot_id must be less than slot_count ({slot_count})"
            )
        if expected_seq_id != expected_slot_id:
            raise RecordValidationError(
                "expected_seq_id must equal expected_slot_id in the pinned P1 profile"
            )
        return cls(
            lease_id=lease_id,
            request=request,
            runtime_instance_id=runtime_instance_id,
            context_ref=context_ref,
            context_ref_source=context_ref_source,
            expected_slot_id=expected_slot_id,
            expected_seq_id=expected_seq_id,
            lease_generation=lease_generation,
            response_confirmed_slot_id=None,
            observed_seq_id=None,
            slot_epoch=None,
            state=LeaseState.RESERVED,
            holds_sequence=True,
        )


@dataclass(frozen=True)
class LeaseEvent:
    schema_version: str
    event_id: str
    event_type: str
    lease_id: str | None
    request_ref: str | None
    runtime_instance_id: str | None
    wall_time_ns: int
    monotonic_ns: int
    details: Mapping[str, JsonValue]

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise RecordValidationError(
                f"schema_version must be {SCHEMA_VERSION}"
            )
        _require_nonempty_text("event_id", self.event_id)
        if self.event_type not in LEASE_EVENT_TYPES:
            raise RecordValidationError(f"unsupported event_type: {self.event_type}")
        for name, value in (
            ("lease_id", self.lease_id),
            ("request_ref", self.request_ref),
            ("runtime_instance_id", self.runtime_instance_id),
        ):
            if value is not None:
                _require_nonempty_text(name, value)
        _require_nonnegative_integer("wall_time_ns", self.wall_time_ns)
        _require_nonnegative_integer("monotonic_ns", self.monotonic_ns)
        normalized = _normalize_json(self.details, "$.details")
        if not isinstance(normalized, dict):
            raise RecordValidationError("details must be a JSON object")
        object.__setattr__(self, "details", normalized)


def canonical_json(value: object) -> bytes:
    normalized = _normalize_json(value)
    return json.dumps(
        normalized,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_length_prefixed(*parts: str) -> bytes:
    encoded = bytearray()
    for index, part in enumerate(parts):
        if not isinstance(part, str):
            raise RecordValidationError(f"parts[{index}] must be a string")
        raw = part.encode("utf-8")
        if len(raw) > 0xFFFFFFFF:
            raise RecordValidationError(f"parts[{index}] is too large")
        encoded.extend(len(raw).to_bytes(4, "big"))
        encoded.extend(raw)
    return bytes(encoded)


def derive_user_hmac(
    registry: IdentityKeyRegistry,
    identity_key_id: str,
    tenant_id: str,
    normalized_authenticated_subject_id: str,
) -> str:
    if not isinstance(registry, IdentityKeyRegistry):
        raise RecordValidationError(
            "registry must be an IdentityKeyRegistry"
        )
    identity_key_id = _require_nonempty_text("identity_key_id", identity_key_id)
    identity_key = registry.resolve(identity_key_id)
    tenant_id = _require_no_nul(
        "tenant_id", _require_nonempty_text("tenant_id", tenant_id)
    )
    normalized_authenticated_subject_id = _require_no_nul(
        "normalized_authenticated_subject_id",
        _require_nonempty_text(
            "normalized_authenticated_subject_id",
            normalized_authenticated_subject_id,
        ),
    )
    message = (
        tenant_id.encode("utf-8")
        + b"\0"
        + normalized_authenticated_subject_id.encode("utf-8")
    )
    digest = hmac.new(identity_key.secret, message, hashlib.sha256).digest()
    encoded = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return encoded


def new_request_ref() -> str:
    return f"req_{uuid.uuid4().hex}"
