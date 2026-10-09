from __future__ import annotations

from dataclasses import dataclass
import ipaddress
import os
from pathlib import Path
import re
import stat
from typing import Protocol

from laris_onpremise.core.records import canonical_json

from .contracts import AttributionDecision
from .correlation import AttributionResult, CorrelationInputs
from .endpoint import derive_endpoint_ref, validate_correlation_key_id


class EvaluationError(RuntimeError):
    """Raised when blind-correlation evaluation inputs are inconsistent."""


_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,191}\Z")
_RUNTIME_INSTANCE_ID = re.compile(
    r"(?:[A-Za-z0-9][A-Za-z0-9._-]{0,191}|sha256:[0-9a-f]{64})\Z"
)


@dataclass(frozen=True)
class EvaluationOracleEntry:
    """Experiment-only truth. Never pass this object to the correlator."""

    request_ref: str
    endpoint_type: str
    endpoint_value: str
    runtime_instance_id: str
    seq_id: int
    sequence_generation: int
    slot_id: int
    state_unit_chain_digests: tuple[str, ...]
    request_started_monotonic_ns: int
    request_finished_monotonic_ns: int

    def __post_init__(self) -> None:
        _safe_id("request_ref", self.request_ref)
        _runtime_instance_id(self.runtime_instance_id)
        if self.endpoint_type not in {"IPV4", "IPV6"}:
            raise EvaluationError("oracle endpoint_type is invalid")
        try:
            address = ipaddress.ip_address(self.endpoint_value)
        except ValueError as error:
            raise EvaluationError("oracle endpoint_value is invalid") from error
        expected_type = "IPV4" if address.version == 4 else "IPV6"
        if self.endpoint_type != expected_type:
            raise EvaluationError("oracle endpoint type and value differ")
        _nonnegative("seq_id", self.seq_id)
        _positive("sequence_generation", self.sequence_generation)
        _nonnegative("slot_id", self.slot_id)
        if (
            not isinstance(self.state_unit_chain_digests, tuple)
            or not self.state_unit_chain_digests
            or len(set(self.state_unit_chain_digests))
            != len(self.state_unit_chain_digests)
            or any(
                not isinstance(value, str) or _DIGEST.fullmatch(value) is None
                for value in self.state_unit_chain_digests
            )
        ):
            raise EvaluationError("oracle state-unit chain is invalid")
        _nonnegative(
            "request_started_monotonic_ns", self.request_started_monotonic_ns
        )
        _nonnegative(
            "request_finished_monotonic_ns", self.request_finished_monotonic_ns
        )
        if self.request_finished_monotonic_ns < self.request_started_monotonic_ns:
            raise EvaluationError("oracle request time range is invalid")


@dataclass(frozen=True)
class BlindRunTelemetry:
    expected_application_event_count: int
    observed_application_event_count: int
    observer_event_loss_count: int
    captured_bytes: int
    maximum_capture_backlog: int
    observation_started_monotonic_ns: int
    observation_finished_monotonic_ns: int

    def __post_init__(self) -> None:
        for name in (
            "expected_application_event_count",
            "observed_application_event_count",
            "observer_event_loss_count",
            "captured_bytes",
            "maximum_capture_backlog",
            "observation_started_monotonic_ns",
            "observation_finished_monotonic_ns",
        ):
            _nonnegative(name, getattr(self, name))
        if self.observed_application_event_count > self.expected_application_event_count:
            raise EvaluationError("observed application event count exceeds oracle")
        if (
            self.observation_finished_monotonic_ns
            <= self.observation_started_monotonic_ns
        ):
            raise EvaluationError("observation duration must be positive")


@dataclass(frozen=True)
class AttributionEvaluationMetrics:
    total_request_count: int
    certified_count: int
    ranked_count: int
    resolved_count: int
    ambiguous_count: int
    incomplete_count: int
    correct_attribution_count: int
    false_attribution_count: int
    missed_attribution_count: int
    event_loss_count: int
    captured_bytes: int
    maximum_capture_backlog: int
    throughput_requests_per_second: float
    mean_request_latency_ns: int
    p95_request_latency_ns: int

    def to_mapping(self) -> dict[str, object]:
        total = self.total_request_count
        return {
            "schema_version": "1.0",
            "status": "COMPLETE",
            "total_request_count": total,
            "certified_count": self.certified_count,
            "ranked_count": self.ranked_count,
            "resolved_count": self.resolved_count,
            "ambiguous_count": self.ambiguous_count,
            "incomplete_count": self.incomplete_count,
            "correct_attribution_count": self.correct_attribution_count,
            "false_attribution_count": self.false_attribution_count,
            "missed_attribution_count": self.missed_attribution_count,
            "event_loss_count": self.event_loss_count,
            "captured_bytes": self.captured_bytes,
            "maximum_capture_backlog": self.maximum_capture_backlog,
            "throughput_requests_per_second": self.throughput_requests_per_second,
            "mean_request_latency_ns": self.mean_request_latency_ns,
            "p95_request_latency_ns": self.p95_request_latency_ns,
            "certified_rate": _rate(self.certified_count, total),
            "ranked_rate": _rate(self.ranked_count, total),
            "resolved_rate": _rate(self.resolved_count, total),
            "ambiguous_rate": _rate(self.ambiguous_count, total),
            "incomplete_rate": _rate(self.incomplete_count, total),
            "correct_attribution_rate": _rate(
                self.correct_attribution_count, total
            ),
            "false_attribution_rate": _rate(
                self.false_attribution_count, total
            ),
            "missed_attribution_rate": _rate(
                self.missed_attribution_count, total
            ),
        }


class _BlindAttributionAnalyzer(Protocol):
    def analyze(
        self,
        inputs: object,
        *,
        correlation_key: bytes,
        correlation_key_id: str,
    ) -> tuple[AttributionResult, ...]: ...


class BlindAttributionRunner:
    """A deliberately oracle-free adapter around the production analyzer."""

    def __init__(self, analyzer: _BlindAttributionAnalyzer) -> None:
        if not callable(getattr(analyzer, "analyze", None)):
            raise EvaluationError("attribution analyzer is invalid")
        self.analyzer = analyzer

    def run(
        self,
        inputs: CorrelationInputs | object,
        *,
        correlation_key: bytes,
        correlation_key_id: str,
    ) -> tuple[AttributionResult, ...]:
        if type(correlation_key) is not bytes or len(correlation_key) != 32:
            raise EvaluationError("correlation_key must be exactly 32 bytes")
        try:
            validate_correlation_key_id(correlation_key_id)
        except ValueError as error:
            raise EvaluationError("correlation_key_id is invalid") from error
        results = self.analyzer.analyze(
            inputs,
            correlation_key=correlation_key,
            correlation_key_id=correlation_key_id,
        )
        if not isinstance(results, tuple) or any(
            not isinstance(result, AttributionResult) for result in results
        ):
            raise EvaluationError("attribution analyzer returned invalid results")
        return results


class AttributionEvaluation:
    """Compare already-produced blind results with isolated controlled truth."""

    def evaluate(
        self,
        results: tuple[AttributionResult, ...],
        oracle: tuple[EvaluationOracleEntry, ...],
        *,
        correlation_key: bytes,
        telemetry: BlindRunTelemetry,
    ) -> AttributionEvaluationMetrics:
        if not isinstance(results, tuple) or any(
            not isinstance(result, AttributionResult) for result in results
        ):
            raise EvaluationError("attribution results are invalid")
        if not isinstance(oracle, tuple) or not oracle or any(
            not isinstance(entry, EvaluationOracleEntry) for entry in oracle
        ):
            raise EvaluationError("evaluation oracle is invalid")
        if type(correlation_key) is not bytes or len(correlation_key) != 32:
            raise EvaluationError("correlation_key must be exactly 32 bytes")
        if not isinstance(telemetry, BlindRunTelemetry):
            raise EvaluationError("blind telemetry is invalid")
        oracle_by_request = {entry.request_ref: entry for entry in oracle}
        if len(oracle_by_request) != len(oracle):
            raise EvaluationError("evaluation oracle contains a duplicate request")
        result_by_request = {str(result.request_ref): result for result in results}
        if len(result_by_request) != len(results):
            raise EvaluationError("attribution results contain a duplicate request")
        unknown = set(result_by_request) - set(oracle_by_request)
        if unknown:
            raise EvaluationError("attribution result is absent from the oracle")

        correct_attribution = 0
        false_attribution = 0
        certified = 0
        ranked = 0
        ambiguous = 0
        incomplete = 0
        for entry in oracle:
            result = result_by_request.get(entry.request_ref)
            if result is None:
                incomplete += 1
                continue
            if result.final_decision is AttributionDecision.AMBIGUOUS:
                ambiguous += 1
                continue
            if result.final_decision is AttributionDecision.INCOMPLETE:
                incomplete += 1
                continue
            if result.final_decision is AttributionDecision.CERTIFIED:
                certified += 1
            else:
                ranked += 1
            expected_endpoint_ref = derive_endpoint_ref(
                correlation_key=correlation_key,
                endpoint_type=entry.endpoint_type,
                endpoint_value=entry.endpoint_value,
            )
            if (
                result.endpoint_ref == expected_endpoint_ref
                and result.runtime_instance_id == entry.runtime_instance_id
                and result.slot_id == entry.slot_id
                and result.state_unit_chain_digests
                == entry.state_unit_chain_digests
            ):
                correct_attribution += 1
            else:
                false_attribution += 1

        total = len(oracle)
        latencies = sorted(
            entry.request_finished_monotonic_ns
            - entry.request_started_monotonic_ns
            for entry in oracle
        )
        duration_ns = (
            telemetry.observation_finished_monotonic_ns
            - telemetry.observation_started_monotonic_ns
        )
        missing_application_events = (
            telemetry.expected_application_event_count
            - telemetry.observed_application_event_count
        )
        return AttributionEvaluationMetrics(
            total_request_count=total,
            certified_count=certified,
            ranked_count=ranked,
            resolved_count=certified + ranked,
            ambiguous_count=ambiguous,
            incomplete_count=incomplete,
            correct_attribution_count=correct_attribution,
            false_attribution_count=false_attribution,
            missed_attribution_count=total - correct_attribution,
            event_loss_count=(
                missing_application_events + telemetry.observer_event_loss_count
            ),
            captured_bytes=telemetry.captured_bytes,
            maximum_capture_backlog=telemetry.maximum_capture_backlog,
            throughput_requests_per_second=round(
                total * 1_000_000_000 / duration_ns, 6
            ),
            mean_request_latency_ns=sum(latencies) // len(latencies),
            p95_request_latency_ns=latencies[
                max(0, (95 * len(latencies) + 99) // 100 - 1)
            ],
        )


def write_evaluation_report_new(
    path: Path | str, metrics: AttributionEvaluationMetrics
) -> Path:
    destination = Path(path)
    if not isinstance(metrics, AttributionEvaluationMetrics):
        raise EvaluationError("evaluation metrics are invalid")
    if destination.exists() or _is_link(destination):
        raise EvaluationError("evaluation report already exists")
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    payload = canonical_json(metrics.to_mapping()) + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(destination, flags, 0o600)
    except OSError as error:
        raise EvaluationError("evaluation report cannot be created") from error
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise EvaluationError("evaluation report write made no progress")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if os.name == "posix":
        os.chmod(destination, 0o600, follow_symlinks=False)
    return destination


def _rate(value: int, total: int) -> float:
    return round(value / total, 6) if total else 0.0


def _safe_id(name: str, value: object) -> None:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise EvaluationError(f"{name} is invalid")


def _runtime_instance_id(value: object) -> None:
    if not isinstance(value, str) or _RUNTIME_INSTANCE_ID.fullmatch(value) is None:
        raise EvaluationError("runtime_instance_id is invalid")


def _nonnegative(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EvaluationError(f"{name} must be non-negative")


def _positive(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise EvaluationError(f"{name} must be positive")


def _is_link(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    return stat.S_ISLNK(metadata.st_mode) or bool(
        getattr(metadata, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    )


__all__ = [
    "AttributionEvaluation",
    "AttributionEvaluationMetrics",
    "BlindAttributionRunner",
    "BlindRunTelemetry",
    "EvaluationError",
    "EvaluationOracleEntry",
    "write_evaluation_report_new",
]
