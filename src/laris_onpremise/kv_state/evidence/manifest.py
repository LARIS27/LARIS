from __future__ import annotations

from collections.abc import Mapping
import json
import re


class AcquisitionManifestError(ValueError):
    """Raised when an acquisition manifest violates the sealed wire contract."""


ACQUISITION_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "capture_id",
        "lease_id",
        "request_ref",
        "runtime_instance_id",
        "runtime_pid",
        "runtime_start_time_ticks",
        "slot_id",
        "slot_epoch",
        "observed_seq_id",
        "observed_stream_index",
        "snapshot_id",
        "snapshot_cut_monotonic_ns",
        "target_marker_before_digest",
        "target_marker_after_digest",
        "layout_profile_digest",
        "kv_backing_provenance_digest",
        "kv_backing_range_count",
        "memory_family",
        "payload_format",
        "payload_file",
        "payload_size_bytes",
        "payload_digest",
        "position_range",
        "kv_cell_ranges",
        "next_position",
        "planned_read_span_count",
        "executed_read_count",
        "read_rounds",
        "read_byte_count",
        "read_size_range",
        "read_trace_digest",
        "acquisition_evidence_ref",
        "contains_prompt_text",
        "contains_response_text",
        "contains_explicit_token_history",
    }
)

_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")


def _text(document: Mapping[str, object], name: str) -> str:
    value = document[name]
    if not isinstance(value, str) or not value.strip() or "\0" in value:
        raise AcquisitionManifestError(f"{name} must be non-empty text")
    return value


def _integer(
    document: Mapping[str, object],
    name: str,
    *,
    minimum: int,
) -> int:
    value = document[name]
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise AcquisitionManifestError(
            f"{name} must be an integer of at least {minimum}"
        )
    return value


def _digest(document: Mapping[str, object], name: str) -> str:
    value = document[name]
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise AcquisitionManifestError(
            f"{name} must be lowercase sha256:<64 hex>"
        )
    return value


def _range(document: Mapping[str, object], name: str) -> tuple[int, int]:
    value = document[name]
    if not isinstance(value, list) or len(value) != 2:
        raise AcquisitionManifestError(f"{name} must be a two-item array")
    start, end = value
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, int)
        or not isinstance(end, int)
        or start < 0
        or end < start
    ):
        raise AcquisitionManifestError(f"{name} is not an inclusive range")
    return start, end


def validate_acquisition_manifest(document: Mapping[str, object]) -> None:
    """Validate the exact manifest contract without relying on optional packages."""

    if not isinstance(document, Mapping):
        raise AcquisitionManifestError("manifest must be an object")
    if document.get("schema_version") != "1.1":
        raise AcquisitionManifestError("schema_version must be 1.1")
    actual = set(document)
    if actual != ACQUISITION_MANIFEST_FIELDS:
        missing = sorted(ACQUISITION_MANIFEST_FIELDS - actual)
        extra = sorted(actual - ACQUISITION_MANIFEST_FIELDS)
        detail = missing[0] if missing else extra[0]
        raise AcquisitionManifestError(
            f"manifest field set is invalid: {detail}"
        )

    for name in (
        "capture_id",
        "lease_id",
        "request_ref",
        "runtime_instance_id",
        "snapshot_id",
        "acquisition_evidence_ref",
    ):
        _text(document, name)
    for name in (
        "runtime_pid",
        "runtime_start_time_ticks",
        "slot_epoch",
        "kv_backing_range_count",
        "payload_size_bytes",
        "next_position",
        "planned_read_span_count",
        "executed_read_count",
        "read_byte_count",
    ):
        _integer(document, name, minimum=1)
    for name in (
        "slot_id",
        "observed_seq_id",
        "observed_stream_index",
        "snapshot_cut_monotonic_ns",
    ):
        _integer(document, name, minimum=0)
    for name in (
        "target_marker_before_digest",
        "target_marker_after_digest",
        "layout_profile_digest",
        "kv_backing_provenance_digest",
        "payload_digest",
        "read_trace_digest",
    ):
        _digest(document, name)
    if (
        document["target_marker_before_digest"]
        != document["target_marker_after_digest"]
    ):
        raise AcquisitionManifestError(
            "target marker digests must match for a single-read capture"
        )

    constants = {
        "memory_family": "simple_kv",
        "payload_format": "llama-simple-kv-seq-payload-v1",
        "payload_file": "runtime_state_payload.bin",
        "read_rounds": 1,
        "contains_prompt_text": False,
        "contains_response_text": False,
        "contains_explicit_token_history": False,
    }
    for name, expected in constants.items():
        if type(document[name]) is not type(expected) or document[name] != expected:
            raise AcquisitionManifestError(f"{name} must equal {expected!r}")

    position_start, position_end = _range(document, "position_range")
    if position_start > position_end:
        raise AcquisitionManifestError("position_range is invalid")
    if document["next_position"] != position_end + 1:
        raise AcquisitionManifestError(
            "next_position must immediately follow position_range"
        )

    cell_ranges = document["kv_cell_ranges"]
    if not isinstance(cell_ranges, list) or not cell_ranges:
        raise AcquisitionManifestError("kv_cell_ranges must be a non-empty array")
    previous_end = -1
    for index, value in enumerate(cell_ranges):
        nested = {"value": value}
        start, end = _range(nested, "value")
        if start <= previous_end:
            raise AcquisitionManifestError(
                f"kv_cell_ranges[{index}] overlaps or is unsorted"
            )
        previous_end = end

    read_min, read_max = _range(document, "read_size_range")
    if read_min <= 0 or read_max <= 0:
        raise AcquisitionManifestError("read_size_range must be positive")
    if document["executed_read_count"] != document["planned_read_span_count"]:
        raise AcquisitionManifestError(
            "executed_read_count must equal the single-read plan"
        )
    minimum_total = document["executed_read_count"] * read_min
    maximum_total = document["executed_read_count"] * read_max
    if not minimum_total <= document["read_byte_count"] <= maximum_total:
        raise AcquisitionManifestError("read_byte_count is inconsistent")


def canonical_acquisition_manifest_bytes(
    document: Mapping[str, object],
) -> bytes:
    validate_acquisition_manifest(document)
    try:
        return (
            json.dumps(
                dict(document),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise AcquisitionManifestError("manifest is not canonical JSON") from error


def parse_acquisition_manifest_bytes(payload: bytes) -> dict[str, object]:
    if not isinstance(payload, bytes):
        raise AcquisitionManifestError("manifest bytes are required")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise AcquisitionManifestError(
                    f"duplicate manifest field: {key}"
                )
            result[key] = value
        return result

    try:
        document = json.loads(
            payload.decode("utf-8"), object_pairs_hook=reject_duplicates
        )
    except AcquisitionManifestError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AcquisitionManifestError("manifest JSON is invalid") from error
    if not isinstance(document, dict):
        raise AcquisitionManifestError("manifest must be an object")
    canonical = canonical_acquisition_manifest_bytes(document)
    if canonical != payload:
        raise AcquisitionManifestError("manifest bytes are not canonical")
    return document
