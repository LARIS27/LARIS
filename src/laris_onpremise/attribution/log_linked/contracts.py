from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
import json
from pathlib import PurePosixPath
import re


class LogLinkedContractError(ValueError):
    """Raised when Log-linked evidence violates its public contract."""


class AttributionDecision(StrEnum):
    CERTIFIED = "CERTIFIED"
    RANKED = "RANKED"
    AMBIGUOUS = "AMBIGUOUS"
    INCOMPLETE = "INCOMPLETE"


class StateUnitKind(StrEnum):
    BASE = "BASE"
    DELTA = "DELTA"


APPLICATION_EVENT_FIELDS = frozenset(
    {
        "schema_version",
        "adapter_id",
        "source_line_number",
        "source_order",
        "wall_time_utc_ns",
        "event_type",
        "method",
        "path_class",
        "status_code",
        "endpoint_type",
        "endpoint_value",
        "task_id",
        "slot_id",
        "prompt_tokens_total",
        "prompt_tokens_new",
        "prompt_tokens_reused",
        "generated_tokens",
        "terminal_outcome",
    }
)
STATE_UNIT_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "unit_id",
        "unit_kind",
        "runtime_instance_id",
        "runtime_pid",
        "runtime_start_time_ticks",
        "seq_id",
        "sequence_generation",
        "apply_call_id",
        "apply_monotonic_ns",
        "previous_unit_digest",
        "base_unit_digest",
        "position_ranges",
        "kv_cell_ranges",
        "stream_indexes",
        "mutation_kinds",
        "layout_profile_digest",
        "layout_topology_digest",
        "kv_backing_provenance_digest",
        "observer_health",
        "acquisition_version",
        "payload_file",
        "payload_size_bytes",
        "payload_digest",
        "contains_prompt_text",
        "contains_response_text",
        "contains_explicit_token_history",
        "state",
    }
)
ATTRIBUTION_RESULT_FIELDS = frozenset(
    {
        "schema_version",
        "analysis_version",
        "adapter_id",
        "runtime_instance_id",
        "request_ref",
        "task_id",
        "slot_id",
        "slot_epoch",
        "endpoint_type",
        "endpoint_ref",
        "correlation_key_id",
        "state_binding",
        "endpoint_binding",
        "final_decision",
        "cutoff_apply_call_id",
        "cutoff_monotonic_ns",
        "state_unit_chain_digests",
        "source_application_log_digest",
        "source_observer_terminal_seal_digest",
        "clock_normalization_receipt_digest",
    }
)
EVIDENCE_PACKAGE_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "package_id",
        "adapter_id",
        "runtime_instance_id",
        "created_at_utc_ns",
        "application_log_file",
        "application_log_size_bytes",
        "application_log_digest",
        "log_provenance_file",
        "log_provenance_digest",
        "observer_terminal_seal_file",
        "observer_terminal_seal_digest",
        "writer_capture_receipt_file",
        "writer_capture_receipt_digest",
        "state_units_root",
        "state_unit_manifest_digests",
        "file_inventory",
        "normalized_application_events_file",
        "normalized_application_events_digest",
        "attribution_results_file",
        "attribution_results_digest",
        "endpoint_derivation_receipt_file",
        "endpoint_derivation_receipt_digest",
        "evaluation_metrics_file",
        "evaluation_metrics_digest",
    }
)
WRITER_CAPTURE_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "capture_mode",
        "runtime_instance_id",
        "runtime_pid",
        "runtime_start_time_ticks",
        "runtime_executable_digest",
        "observer_profile_digest",
        "apply_count",
        "unit_count",
        "fragment_count",
        "copied_byte_count",
        "native_ring_frame_count",
        "native_ring_high_water_mark",
        "native_ring_high_water_unit",
        "persistence_queue_high_water_mark",
        "observer_cpu_time_ns",
        "safe_failure_counts",
        "contains_prompt_text",
        "contains_response_text",
        "contains_token_ids",
        "contains_endpoint_values",
        "contains_request_refs",
        "contains_private_truth",
    }
)
WRITER_CAPTURE_SAFE_FAILURE_FIELDS = frozenset(
    {
        "ring_reserve_failure_count",
        "user_memory_read_failure_count",
        "unmatched_call_count",
        "unsupported_mutation_count",
        "mapping_mutation_count",
        "final_inflight_count",
        "incomplete_unit_count",
        "persistence_failure_count",
        "terminal_receipt_failure_count",
    }
)

_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,191}\Z")
_RUNTIME_INSTANCE_ID = re.compile(
    r"(?:[A-Za-z0-9][A-Za-z0-9._-]{0,191}|sha256:[0-9a-f]{64})\Z"
)
_UNIT_ID = re.compile(r"su_[0-9a-f]{64}\Z")
_PACKAGE_ID = re.compile(r"lep_[0-9a-f]{64}\Z")
_ENDPOINT_REF = re.compile(r"[A-Za-z0-9_-]{43}\Z")
_FORBIDDEN_CONTENT_KEYS = frozenset(
    {
        "authorization",
        "body_digest",
        "content",
        "cookie",
        "expanded_chat_template",
        "messages",
        "prompt",
        "prompt_digest",
        "prompt_text",
        "request_body",
        "response",
        "response_body",
        "response_digest",
        "response_text",
        "system_prompt",
        "token_ids",
    }
)


def _require_nonempty_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LogLinkedContractError(f"{name} must be a non-empty string")
    return value


def _require_nonnegative_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise LogLinkedContractError(f"{name} must be a non-negative integer")
    return value


def _require_positive_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise LogLinkedContractError(f"{name} must be a positive integer")
    return value


def _require_exact_fields(
    name: str, document: Mapping[str, object], expected: frozenset[str]
) -> None:
    actual = set(document)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        detail = missing[0] if missing else extra[0]
        raise LogLinkedContractError(f"{name} field set is invalid: {detail}")


def _require_safe_id(name: str, value: object) -> str:
    text = _require_nonempty_text(name, value)
    if _SAFE_ID.fullmatch(text) is None:
        raise LogLinkedContractError(f"{name} is not a safe identifier")
    return text


def _require_runtime_instance_id(value: object) -> str:
    text = _require_nonempty_text("runtime_instance_id", value)
    if _RUNTIME_INSTANCE_ID.fullmatch(text) is None:
        raise LogLinkedContractError("runtime_instance_id is invalid")
    return text


def _require_pattern(name: str, value: object, pattern: re.Pattern[str]) -> str:
    text = _require_nonempty_text(name, value)
    if pattern.fullmatch(text) is None:
        raise LogLinkedContractError(f"{name} has an invalid format")
    return text


def _require_digest(name: str, value: object, *, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise LogLinkedContractError(f"{name} must be a lowercase sha256 digest")
    return value


def _require_optional_nonnegative_integer(name: str, value: object) -> int | None:
    if value is None:
        return None
    return _require_nonnegative_integer(name, value)


def _require_optional_positive_integer(name: str, value: object) -> int | None:
    if value is None:
        return None
    return _require_positive_integer(name, value)


def _require_false(name: str, value: object) -> None:
    if value is not False:
        raise LogLinkedContractError(f"{name} must be false")


def _require_enum(name: str, value: object, allowed: frozenset[str]) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise LogLinkedContractError(f"{name} is invalid")
    return value


def _require_safe_relative_path(name: str, value: object) -> str:
    text = _require_nonempty_text(name, value)
    if "\\" in text or "\0" in text or ":" in text:
        raise LogLinkedContractError(f"{name} must be a safe relative path")
    if any(part in {"", ".", ".."} for part in text.split("/")):
        raise LogLinkedContractError(f"{name} must be a safe relative path")
    path = PurePosixPath(text)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise LogLinkedContractError(f"{name} must be a safe relative path")
    return text


def _require_range_list(name: str, value: object, *, allow_empty: bool) -> None:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise LogLinkedContractError(f"{name} must be an ordered range array")
    previous_end = -1
    for index, item in enumerate(value):
        if not isinstance(item, list) or len(item) != 2:
            raise LogLinkedContractError(f"{name}[{index}] must be a two-item range")
        start, end = item
        _require_nonnegative_integer(f"{name}[{index}][0]", start)
        _require_nonnegative_integer(f"{name}[{index}][1]", end)
        if end < start or start <= previous_end:
            raise LogLinkedContractError(f"{name} must be sorted and non-overlapping")
        previous_end = end


def _require_sorted_unique_integers(name: str, value: object, *, allow_empty: bool) -> None:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise LogLinkedContractError(f"{name} must be an integer array")
    for index, item in enumerate(value):
        _require_nonnegative_integer(f"{name}[{index}]", item)
    if value != sorted(set(value)):
        raise LogLinkedContractError(f"{name} must be sorted and unique")


def _require_string_list(
    name: str,
    value: object,
    *,
    allow_empty: bool,
    digest: bool = False,
) -> None:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise LogLinkedContractError(f"{name} must be a string array")
    for index, item in enumerate(value):
        if digest:
            _require_digest(f"{name}[{index}]", item)
        else:
            _require_safe_id(f"{name}[{index}]", item)
    if value != sorted(set(value)):
        raise LogLinkedContractError(f"{name} must be sorted and unique")


def _reject_forbidden_content_fields(value: object) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if key in _FORBIDDEN_CONTENT_KEYS:
                raise LogLinkedContractError(f"content-bearing field is forbidden: {key}")
            _reject_forbidden_content_fields(nested)
    elif isinstance(value, list):
        for nested in value:
            _reject_forbidden_content_fields(nested)


def canonical_document_bytes(document: Mapping[str, object]) -> bytes:
    if not isinstance(document, Mapping):
        raise LogLinkedContractError("document must be an object")
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
        raise LogLinkedContractError("document is not canonical JSON") from error


def _parse_document(payload: bytes, *, name: str) -> dict[str, object]:
    if not isinstance(payload, bytes):
        raise LogLinkedContractError(f"{name} bytes are required")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise LogLinkedContractError(f"duplicate {name} field: {key}")
            result[key] = value
        return result

    try:
        document = json.loads(payload.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except LogLinkedContractError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LogLinkedContractError(f"{name} JSON is invalid") from error
    if not isinstance(document, dict):
        raise LogLinkedContractError(f"{name} must be an object")
    if canonical_document_bytes(document) != payload:
        raise LogLinkedContractError(f"{name} bytes are not canonical")
    return document


def _require_schema_version(document: Mapping[str, object]) -> None:
    if document.get("schema_version") != "1.0":
        raise LogLinkedContractError("schema_version must be 1.0")


def parse_application_event(payload: bytes) -> dict[str, object]:
    document = _parse_document(payload, name="application event")
    _require_exact_fields("application event", document, APPLICATION_EVENT_FIELDS)
    _require_schema_version(document)
    _reject_forbidden_content_fields(document)
    _require_safe_id("adapter_id", document["adapter_id"])
    _require_positive_integer("source_line_number", document["source_line_number"])
    _require_nonnegative_integer("source_order", document["source_order"])
    _require_nonnegative_integer("wall_time_utc_ns", document["wall_time_utc_ns"])
    event_type = _require_enum(
        "event_type",
        document["event_type"],
        frozenset({"ACCESS", "SLOT_PROMPT", "SLOT_REUSE", "SLOT_PROMPT_EVAL", "SLOT_TERMINAL"}),
    )
    if document["method"] is not None:
        _require_enum("method", document["method"], frozenset({"GET", "POST", "PUT", "DELETE", "PATCH"}))
    if document["path_class"] is not None:
        _require_safe_id("path_class", document["path_class"])
    status = _require_optional_nonnegative_integer("status_code", document["status_code"])
    if status is not None and not 100 <= status <= 599:
        raise LogLinkedContractError("status_code must be an HTTP status")
    endpoint_type = document["endpoint_type"]
    endpoint_value = document["endpoint_value"]
    if (endpoint_type is None) != (endpoint_value is None):
        raise LogLinkedContractError("endpoint type and value must be present together")
    if endpoint_type is not None:
        _require_enum("endpoint_type", endpoint_type, frozenset({"IPV4", "IPV6"}))
        _require_nonempty_text("endpoint_value", endpoint_value)
    for field in ("task_id", "slot_id", "prompt_tokens_total", "prompt_tokens_new", "prompt_tokens_reused", "generated_tokens"):
        _require_optional_nonnegative_integer(field, document[field])
    if document["terminal_outcome"] is not None:
        _require_enum(
            "terminal_outcome",
            document["terminal_outcome"],
            frozenset({"COMPLETED", "CANCELLED", "ERROR", "RELEASED"}),
        )
    if event_type == "ACCESS":
        for field in ("method", "path_class", "status_code", "endpoint_type", "endpoint_value"):
            if document[field] is None:
                raise LogLinkedContractError(f"ACCESS requires {field}")
    else:
        for field in ("task_id", "slot_id"):
            if document[field] is None:
                raise LogLinkedContractError(f"{event_type} requires {field}")
    return document


def parse_state_unit_manifest(payload: bytes) -> dict[str, object]:
    document = _parse_document(payload, name="state-unit manifest")
    _require_exact_fields("state-unit manifest", document, STATE_UNIT_MANIFEST_FIELDS)
    _require_schema_version(document)
    _reject_forbidden_content_fields(document)
    _require_pattern("unit_id", document["unit_id"], _UNIT_ID)
    kind = _require_enum("unit_kind", document["unit_kind"], frozenset(member.value for member in StateUnitKind))
    _require_runtime_instance_id(document["runtime_instance_id"])
    _require_positive_integer("runtime_pid", document["runtime_pid"])
    _require_positive_integer("runtime_start_time_ticks", document["runtime_start_time_ticks"])
    _require_nonnegative_integer("seq_id", document["seq_id"])
    _require_positive_integer("sequence_generation", document["sequence_generation"])
    _require_nonnegative_integer("apply_call_id", document["apply_call_id"])
    _require_nonnegative_integer("apply_monotonic_ns", document["apply_monotonic_ns"])
    _require_digest("previous_unit_digest", document["previous_unit_digest"], nullable=True)
    _require_digest("base_unit_digest", document["base_unit_digest"], nullable=True)
    state = _require_enum("state", document["state"], frozenset({"COMPLETE", "INCOMPLETE"}))
    allow_empty = state == "INCOMPLETE"
    _require_range_list("position_ranges", document["position_ranges"], allow_empty=allow_empty)
    _require_range_list("kv_cell_ranges", document["kv_cell_ranges"], allow_empty=allow_empty)
    _require_sorted_unique_integers("stream_indexes", document["stream_indexes"], allow_empty=allow_empty)
    _require_string_list("mutation_kinds", document["mutation_kinds"], allow_empty=True)
    for field in ("layout_profile_digest", "layout_topology_digest", "kv_backing_provenance_digest", "payload_digest"):
        _require_digest(field, document[field])
    health = document["observer_health"]
    if not isinstance(health, Mapping):
        raise LogLinkedContractError("observer_health must be an object")
    health_fields = frozenset({"observer_loss_count", "parse_failure_count", "unmatched_call_count", "unsupported_mutation_count"})
    _require_exact_fields("observer_health", health, health_fields)
    for field in health_fields:
        _require_nonnegative_integer(field, health[field])
    version = document["acquisition_version"]
    if not isinstance(version, Mapping):
        raise LogLinkedContractError(
            "acquisition_version must be an object"
        )
    version_fields = frozenset(
        {
            "stream_index",
            "stream_apply_call_id",
            "mutation_generation",
            "backing_generation",
            "layout_generation",
            "target_cell_map_digest",
            "state",
        }
    )
    _require_exact_fields(
        "acquisition_version", version, version_fields
    )
    _require_nonnegative_integer("stream_index", version["stream_index"])
    _require_positive_integer(
        "stream_apply_call_id", version["stream_apply_call_id"]
    )
    for field in (
        "mutation_generation",
        "backing_generation",
        "layout_generation",
    ):
        _require_nonnegative_integer(field, version[field])
    _require_digest(
        "target_cell_map_digest", version["target_cell_map_digest"]
    )
    if version["state"] != "READY":
        raise LogLinkedContractError(
            "acquisition_version state must equal READY"
        )
    if version["stream_index"] not in document["stream_indexes"]:
        raise LogLinkedContractError(
            "acquisition_version stream is outside the state unit"
        )
    if version["stream_apply_call_id"] != document["apply_call_id"]:
        raise LogLinkedContractError(
            "acquisition_version apply differs from the state unit"
        )
    if document["payload_file"] != "runtime_state_payload.bin":
        raise LogLinkedContractError("payload_file must equal runtime_state_payload.bin")
    size = _require_nonnegative_integer("payload_size_bytes", document["payload_size_bytes"])
    for field in ("contains_prompt_text", "contains_response_text", "contains_explicit_token_history"):
        _require_false(field, document[field])
    if state == "COMPLETE" and size == 0:
        raise LogLinkedContractError("a COMPLETE state unit requires a payload")
    if kind == StateUnitKind.BASE:
        if document["previous_unit_digest"] is not None or document["base_unit_digest"] is not None:
            raise LogLinkedContractError("a BASE unit cannot reference a predecessor")
    elif state == "COMPLETE":
        for field in ("previous_unit_digest", "base_unit_digest"):
            if document[field] is None:
                raise LogLinkedContractError(f"a COMPLETE DELTA requires {field}")
    return document


def _parse_binding_assessment(
    name: str, value: object
) -> tuple[AttributionDecision, str | None]:
    if not isinstance(value, Mapping):
        raise LogLinkedContractError(f"{name} must be an object")
    _require_exact_fields(
        name,
        value,
        frozenset(
            {
                "decision",
                "reason_code",
                "candidate_ids",
                "candidate_count",
                "selected_candidate_id",
                "optimal_candidate_count",
                "optimal_assignment_count",
                "ranking_stage",
            }
        ),
    )
    decision = AttributionDecision(
        _require_enum(
            f"{name}.decision",
            value["decision"],
            frozenset(item.value for item in AttributionDecision),
        )
    )
    _require_safe_id(f"{name}.reason_code", value["reason_code"])
    _require_string_list(
        f"{name}.candidate_ids", value["candidate_ids"], allow_empty=True
    )
    candidate_count = _require_nonnegative_integer(
        f"{name}.candidate_count", value["candidate_count"]
    )
    if candidate_count != len(value["candidate_ids"]):
        raise LogLinkedContractError(
            f"{name}.candidate_count must match candidate_ids"
        )
    selected = value["selected_candidate_id"]
    if selected is not None:
        _require_safe_id(f"{name}.selected_candidate_id", selected)
        if selected not in value["candidate_ids"]:
            raise LogLinkedContractError(
                f"{name}.selected_candidate_id is not a candidate"
            )
    optimal_candidate_count = _require_optional_nonnegative_integer(
        f"{name}.optimal_candidate_count", value["optimal_candidate_count"]
    )
    _require_optional_nonnegative_integer(
        f"{name}.optimal_assignment_count", value["optimal_assignment_count"]
    )
    ranking_stage = value["ranking_stage"]
    if ranking_stage is not None:
        _require_safe_id(f"{name}.ranking_stage", ranking_stage)
    if decision in {AttributionDecision.CERTIFIED, AttributionDecision.RANKED}:
        if selected is None or optimal_candidate_count != 1:
            raise LogLinkedContractError(
                f"{name} resolved decision requires one selected candidate"
            )
    elif selected is not None:
        raise LogLinkedContractError(
            f"{name} unresolved decision cannot select a candidate"
        )
    if decision is AttributionDecision.RANKED and ranking_stage is None:
        raise LogLinkedContractError(f"{name}.ranking_stage is required for RANKED")
    if decision is not AttributionDecision.RANKED and ranking_stage is not None:
        raise LogLinkedContractError(
            f"{name}.ranking_stage is allowed only for RANKED"
        )
    return decision, ranking_stage


def parse_attribution_result(payload: bytes) -> dict[str, object]:
    document = _parse_document(payload, name="attribution result")
    _require_exact_fields(
        "attribution result", document, ATTRIBUTION_RESULT_FIELDS
    )
    _require_schema_version(document)
    _reject_forbidden_content_fields(document)
    if document["analysis_version"] != "global-structural-ranked-v2":
        raise LogLinkedContractError("analysis_version is unsupported")
    for field in ("adapter_id", "request_ref", "correlation_key_id"):
        _require_safe_id(field, document[field])
    _require_runtime_instance_id(document["runtime_instance_id"])
    _require_nonnegative_integer("task_id", document["task_id"])
    _require_nonnegative_integer("slot_id", document["slot_id"])
    _require_positive_integer("slot_epoch", document["slot_epoch"])
    state_decision, _ = _parse_binding_assessment(
        "state_binding", document["state_binding"]
    )
    endpoint_decision, _ = _parse_binding_assessment(
        "endpoint_binding", document["endpoint_binding"]
    )
    final_decision = AttributionDecision(
        _require_enum(
            "final_decision",
            document["final_decision"],
            frozenset(item.value for item in AttributionDecision),
        )
    )
    decisions = {state_decision, endpoint_decision}
    expected_final = (
        AttributionDecision.INCOMPLETE
        if AttributionDecision.INCOMPLETE in decisions
        else AttributionDecision.AMBIGUOUS
        if AttributionDecision.AMBIGUOUS in decisions
        else AttributionDecision.CERTIFIED
        if decisions == {AttributionDecision.CERTIFIED}
        else AttributionDecision.RANKED
    )
    if final_decision is not expected_final:
        raise LogLinkedContractError("final_decision differs from its bindings")
    if (document["endpoint_type"] is None) != (document["endpoint_ref"] is None):
        raise LogLinkedContractError("endpoint type and reference must be present together")
    if document["endpoint_type"] is not None:
        _require_enum("endpoint_type", document["endpoint_type"], frozenset({"IPV4", "IPV6"}))
        _require_pattern("endpoint_ref", document["endpoint_ref"], _ENDPOINT_REF)
    cutoff_apply = _require_optional_nonnegative_integer("cutoff_apply_call_id", document["cutoff_apply_call_id"])
    cutoff_time = _require_optional_nonnegative_integer("cutoff_monotonic_ns", document["cutoff_monotonic_ns"])
    chain = document["state_unit_chain_digests"]
    if not isinstance(chain, list):
        raise LogLinkedContractError(
            "state_unit_chain_digests must be a digest array"
        )
    for index, digest_value in enumerate(chain):
        _require_digest(
            f"state_unit_chain_digests[{index}]", digest_value
        )
    if len(chain) != len(set(chain)):
        raise LogLinkedContractError(
            "state_unit_chain_digests must not contain duplicates"
        )
    for field in ("source_application_log_digest", "source_observer_terminal_seal_digest", "clock_normalization_receipt_digest"):
        _require_digest(field, document[field])
    if endpoint_decision in {
        AttributionDecision.CERTIFIED,
        AttributionDecision.RANKED,
    }:
        if document["endpoint_ref"] is None:
            raise LogLinkedContractError(
                "resolved endpoint binding requires an endpoint reference"
            )
    elif document["endpoint_ref"] is not None:
        raise LogLinkedContractError(
            "unresolved endpoint binding cannot publish an endpoint reference"
        )
    if state_decision in {
        AttributionDecision.CERTIFIED,
        AttributionDecision.RANKED,
    }:
        if cutoff_apply is None or cutoff_time is None or not chain:
            raise LogLinkedContractError(
                "resolved state binding requires cutoff and state-unit chain"
            )
    elif cutoff_apply is not None or cutoff_time is not None or chain:
        raise LogLinkedContractError(
            "unresolved state binding cannot publish a cutoff or state-unit chain"
        )
    return document


def parse_evidence_package_manifest(payload: bytes) -> dict[str, object]:
    document = _parse_document(payload, name="evidence package manifest")
    _require_exact_fields("evidence package manifest", document, EVIDENCE_PACKAGE_MANIFEST_FIELDS)
    _require_schema_version(document)
    _reject_forbidden_content_fields(document)
    _require_pattern("package_id", document["package_id"], _PACKAGE_ID)
    _require_safe_id("adapter_id", document["adapter_id"])
    _require_runtime_instance_id(document["runtime_instance_id"])
    _require_nonnegative_integer("created_at_utc_ns", document["created_at_utc_ns"])
    _require_positive_integer("application_log_size_bytes", document["application_log_size_bytes"])
    for field in (
        "application_log_file",
        "log_provenance_file",
        "observer_terminal_seal_file",
        "writer_capture_receipt_file",
        "state_units_root",
        "normalized_application_events_file",
        "attribution_results_file",
        "endpoint_derivation_receipt_file",
    ):
        _require_safe_relative_path(field, document[field])
    for field in (
        "application_log_digest",
        "log_provenance_digest",
        "observer_terminal_seal_digest",
        "writer_capture_receipt_digest",
        "normalized_application_events_digest",
        "attribution_results_digest",
        "endpoint_derivation_receipt_digest",
    ):
        _require_digest(field, document[field])
    _require_string_list("state_unit_manifest_digests", document["state_unit_manifest_digests"], allow_empty=False, digest=True)
    inventory = document["file_inventory"]
    if not isinstance(inventory, list) or not inventory:
        raise LogLinkedContractError("file_inventory must be a non-empty array")
    inventory_paths: list[str] = []
    for index, item in enumerate(inventory):
        if not isinstance(item, Mapping):
            raise LogLinkedContractError(
                f"file_inventory[{index}] must be an object"
            )
        _require_exact_fields(
            f"file_inventory[{index}]",
            item,
            frozenset({"path", "size_bytes", "sha256"}),
        )
        path = _require_safe_relative_path(
            f"file_inventory[{index}].path", item["path"]
        )
        _require_nonnegative_integer(
            f"file_inventory[{index}].size_bytes", item["size_bytes"]
        )
        _require_digest(f"file_inventory[{index}].sha256", item["sha256"])
        inventory_paths.append(path)
    if inventory_paths != sorted(set(inventory_paths)):
        raise LogLinkedContractError(
            "file_inventory must be sorted by path and contain no duplicates"
        )
    metrics_file = document["evaluation_metrics_file"]
    metrics_digest = document["evaluation_metrics_digest"]
    if (metrics_file is None) != (metrics_digest is None):
        raise LogLinkedContractError("evaluation path and digest must be present together")
    if metrics_file is not None:
        _require_safe_relative_path("evaluation_metrics_file", metrics_file)
        _require_digest("evaluation_metrics_digest", metrics_digest)
    return document


def parse_writer_capture_receipt(payload: bytes) -> dict[str, object]:
    document = _parse_document(payload, name="writer capture receipt")
    _require_exact_fields(
        "writer capture receipt", document, WRITER_CAPTURE_RECEIPT_FIELDS
    )
    _require_schema_version(document)
    _reject_forbidden_content_fields(document)
    _require_enum(
        "capture_mode", document["capture_mode"], frozenset({"writer-boundary"})
    )
    _require_runtime_instance_id(document["runtime_instance_id"])
    _require_positive_integer("runtime_pid", document["runtime_pid"])
    _require_positive_integer(
        "runtime_start_time_ticks", document["runtime_start_time_ticks"]
    )
    _require_digest(
        "runtime_executable_digest", document["runtime_executable_digest"]
    )
    _require_digest("observer_profile_digest", document["observer_profile_digest"])
    for field in (
        "apply_count",
        "unit_count",
        "fragment_count",
        "copied_byte_count",
        "native_ring_frame_count",
        "native_ring_high_water_mark",
        "persistence_queue_high_water_mark",
        "observer_cpu_time_ns",
    ):
        _require_nonnegative_integer(field, document[field])
    _require_enum(
        "native_ring_high_water_unit",
        document["native_ring_high_water_unit"],
        frozenset({"FRAMES_PER_SEALED_UNIT"}),
    )
    failures = document["safe_failure_counts"]
    if not isinstance(failures, Mapping):
        raise LogLinkedContractError("safe_failure_counts must be an object")
    _require_exact_fields(
        "safe_failure_counts", failures, WRITER_CAPTURE_SAFE_FAILURE_FIELDS
    )
    for field in WRITER_CAPTURE_SAFE_FAILURE_FIELDS:
        _require_nonnegative_integer(
            f"safe_failure_counts.{field}", failures[field]
        )
    for field in (
        "contains_prompt_text",
        "contains_response_text",
        "contains_token_ids",
        "contains_endpoint_values",
        "contains_request_refs",
        "contains_private_truth",
    ):
        _require_false(field, document[field])
    if document["unit_count"] > document["fragment_count"]:
        raise LogLinkedContractError(
            "unit_count cannot exceed fragment_count"
        )
    if document["apply_count"] > document["unit_count"]:
        raise LogLinkedContractError("apply_count cannot exceed unit_count")
    return document


@dataclass(frozen=True)
class SequenceGenerationRef:
    runtime_instance_id: str
    seq_id: int
    sequence_generation: int

    def __post_init__(self) -> None:
        _require_nonempty_text("runtime_instance_id", self.runtime_instance_id)
        _require_nonnegative_integer("seq_id", self.seq_id)
        _require_positive_integer(
            "sequence_generation", self.sequence_generation
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "runtime_instance_id": self.runtime_instance_id,
            "seq_id": self.seq_id,
            "sequence_generation": self.sequence_generation,
        }


@dataclass(frozen=True)
class ApplyBoundary:
    apply_call_id: int
    monotonic_ns: int
    sequence_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        _require_nonnegative_integer("apply_call_id", self.apply_call_id)
        _require_nonnegative_integer("monotonic_ns", self.monotonic_ns)
        if not isinstance(self.sequence_ids, tuple) or not self.sequence_ids:
            raise LogLinkedContractError(
                "sequence_ids must be a non-empty tuple"
            )
        for index, sequence_id in enumerate(self.sequence_ids):
            _require_nonnegative_integer(
                f"sequence_ids[{index}]", sequence_id
            )
        if tuple(sorted(set(self.sequence_ids))) != self.sequence_ids:
            raise LogLinkedContractError(
                "sequence_ids must be sorted and unique"
            )

    def to_mapping(self) -> dict[str, object]:
        return {
            "apply_call_id": self.apply_call_id,
            "monotonic_ns": self.monotonic_ns,
            "sequence_ids": list(self.sequence_ids),
        }


__all__ = [
    "APPLICATION_EVENT_FIELDS",
    "ApplyBoundary",
    "ATTRIBUTION_RESULT_FIELDS",
    "AttributionDecision",
    "EVIDENCE_PACKAGE_MANIFEST_FIELDS",
    "LogLinkedContractError",
    "SequenceGenerationRef",
    "STATE_UNIT_MANIFEST_FIELDS",
    "WRITER_CAPTURE_RECEIPT_FIELDS",
    "WRITER_CAPTURE_SAFE_FAILURE_FIELDS",
    "StateUnitKind",
    "canonical_document_bytes",
    "parse_application_event",
    "parse_attribution_result",
    "parse_evidence_package_manifest",
    "parse_state_unit_manifest",
    "parse_writer_capture_receipt",
]
