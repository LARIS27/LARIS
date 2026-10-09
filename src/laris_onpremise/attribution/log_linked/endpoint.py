from __future__ import annotations

import base64
import hmac
from ipaddress import IPv4Address, IPv6Address
import re


class EndpointDerivationError(ValueError):
    """Raised when an endpoint cannot enter the pseudonymous contract."""


_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,191}\Z")


def _parse_port(value: str) -> None:
    if not value.isascii() or not value.isdecimal():
        raise EndpointDerivationError("transport port is invalid")
    port = int(value, 10)
    if not 1 <= port <= 65_535:
        raise EndpointDerivationError("transport port is invalid")


def normalize_endpoint(endpoint_type: str, value: str) -> str:
    """Return one canonical IP address for in-memory correlation only."""

    if endpoint_type not in {"IPV4", "IPV6"}:
        raise EndpointDerivationError("endpoint_type is unsupported")
    if not isinstance(value, str) or not value or value != value.strip():
        raise EndpointDerivationError("endpoint value is invalid")
    if "%" in value:
        raise EndpointDerivationError("scoped IPv6 endpoints are unsupported")

    address_text = value
    if endpoint_type == "IPV4":
        if value.count(":") == 1:
            address_text, port_text = value.rsplit(":", 1)
            _parse_port(port_text)
        elif ":" in value:
            raise EndpointDerivationError("IPV4 endpoint is invalid")
        try:
            return str(IPv4Address(address_text))
        except ValueError as error:
            raise EndpointDerivationError("IPV4 endpoint is invalid") from error

    if value.startswith("["):
        closing = value.find("]")
        if closing < 0:
            raise EndpointDerivationError("bracketed IPV6 endpoint is invalid")
        address_text = value[1:closing]
        suffix = value[closing + 1 :]
        if suffix:
            if not suffix.startswith(":") or len(suffix) == 1:
                raise EndpointDerivationError("bracketed IPV6 endpoint is invalid")
            _parse_port(suffix[1:])
    elif "[" in value or "]" in value:
        raise EndpointDerivationError("bracketed IPV6 endpoint is invalid")

    try:
        address = IPv6Address(address_text)
    except ValueError as error:
        raise EndpointDerivationError("IPV6 endpoint is invalid") from error
    if address.ipv4_mapped is not None:
        mapped = int(address.ipv4_mapped)
        return f"::ffff:{mapped >> 16:x}:{mapped & 0xffff:x}"
    return str(address)


def validate_correlation_key_id(value: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise EndpointDerivationError("correlation_key_id is unsafe")
    return value


def derive_endpoint_ref(
    *, correlation_key: bytes, endpoint_type: str, endpoint_value: str
) -> str:
    """Derive the portable pseudonymous reference for one endpoint."""

    if type(correlation_key) is not bytes or len(correlation_key) != 32:
        raise EndpointDerivationError("correlation_key must be exactly 32-byte bytes")
    normalized_value = normalize_endpoint(endpoint_type, endpoint_value)
    message = f"{endpoint_type}:{normalized_value}".encode("utf-8")
    return (
        base64.urlsafe_b64encode(hmac.digest(correlation_key, message, "sha256"))
        .rstrip(b"=")
        .decode("ascii")
    )


__all__ = [
    "EndpointDerivationError",
    "derive_endpoint_ref",
    "normalize_endpoint",
    "validate_correlation_key_id",
]
