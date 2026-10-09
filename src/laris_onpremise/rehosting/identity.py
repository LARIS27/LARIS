from __future__ import annotations

import hmac

from laris_onpremise.core.records import (
    IdentityKeyNotFoundError,
    IdentityKeyRegistry,
    RecordValidationError,
    derive_user_hmac,
)
from laris_onpremise.rehosting.catalog import (
    PortableBundleCatalog,
    PortableBundleEntry,
)


class RehostIdentityError(ValueError):
    """Raised when an investigator subject cannot select one bundle identity."""


def resolve_subject_sessions(
    catalog: PortableBundleCatalog,
    *,
    tenant_id: str,
    registry: IdentityKeyRegistry,
    raw_subject_id: str,
) -> tuple[PortableBundleEntry, ...]:
    if not isinstance(catalog, PortableBundleCatalog):
        raise TypeError("catalog must be a PortableBundleCatalog")
    subject = _validated_subject(raw_subject_id)
    candidates: dict[str, str] = {}
    for identity_key_id in sorted({item.identity_key_id for item in catalog.entries}):
        try:
            candidates[identity_key_id] = derive_user_hmac(
                registry,
                identity_key_id,
                tenant_id,
                subject,
            )
        except IdentityKeyNotFoundError as error:
            raise RehostIdentityError(
                f"required identity key is unavailable: {identity_key_id}"
            ) from error
        except (RecordValidationError, UnicodeError) as error:
            raise RehostIdentityError("investigator identity configuration is invalid") from error

    matches = tuple(
        item
        for item in catalog.entries
        if hmac.compare_digest(candidates[item.identity_key_id], item.user_hmac)
    )
    if not matches:
        raise RehostIdentityError("no bundle matches the supplied subject identifier")
    if len({item.identity_key_id for item in matches}) != 1:
        raise RehostIdentityError("bundle identity selection is ambiguous")
    return tuple(sorted(matches, key=lambda item: (item.last_turn_index, item.session_ref)))


def _validated_subject(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise RehostIdentityError("subject identifier is invalid")
    if any(ord(character) < 0x20 or ord(character) == 0x7F for character in value):
        raise RehostIdentityError("subject identifier is invalid")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeError as error:
        raise RehostIdentityError("subject identifier is invalid") from error
    return value
