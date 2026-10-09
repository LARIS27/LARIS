from __future__ import annotations

import hashlib
import hmac
import re
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence

from laris_onpremise.core.records import canonical_json


_HMAC_SHA256 = re.compile(r"^hmac-sha256:[0-9a-f]{64}$")


class ContinuityValidationError(ValueError):
    """Raised when session continuity evidence is invalid."""


class ApplicationContextMode(str, Enum):
    CUMULATIVE_CONTEXT = "cumulative_context"
    TURN_ONLY = "turn_only"


class ContinuityMechanism(str, Enum):
    FULL_CONTEXT_RECOMPUTE = "full_context_recompute"
    ROLLING_CONTEXT_RECOMPUTE = "rolling_context_recompute"
    NONE = "none"


def _require_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContinuityValidationError(f"{name} must be a non-empty string")
    return value


def _require_commitment(name: str, value: object) -> str:
    if not isinstance(value, str) or _HMAC_SHA256.fullmatch(value) is None:
        raise ContinuityValidationError(
            f"{name} must be an hmac-sha256 commitment"
        )
    return value


def _normalize_messages(
    messages: Sequence[tuple[str, str]],
) -> list[dict[str, str]]:
    if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence):
        raise ContinuityValidationError("messages must be a sequence")
    normalized: list[dict[str, str]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, tuple) or len(message) != 2:
            raise ContinuityValidationError(
                f"messages[{index}] must be a role-content tuple"
            )
        role, content = message
        if role not in {"system", "user", "assistant", "tool"}:
            raise ContinuityValidationError(
                f"messages[{index}] contains an unsupported role"
            )
        if not isinstance(content, str):
            raise ContinuityValidationError(
                f"messages[{index}] content must be a string"
            )
        normalized.append({"role": role, "content": content})
    if not normalized:
        raise ContinuityValidationError("messages must not be empty")
    return normalized


def derive_context_commitment(
    key: bytes,
    scope: str,
    messages: Sequence[tuple[str, str]],
) -> str:
    if not isinstance(key, bytes) or len(key) < 16:
        raise ContinuityValidationError("key must contain at least 16 bytes")
    _require_text("scope", scope)
    normalized = _normalize_messages(messages)
    payload = {
        "domain": "LARIS-CONTEXT-v1",
        "messages": normalized,
        "scope": scope,
    }
    return "hmac-sha256:" + hmac.new(
        key,
        canonical_json(payload),
        hashlib.sha256,
    ).hexdigest()


def _require_complete_dialogue_turns(
    messages: Sequence[tuple[str, str]],
) -> None:
    if not messages or len(messages) % 2 != 0:
        raise ContinuityValidationError(
            "rolling context must drop complete turns"
        )
    for index in range(0, len(messages), 2):
        if messages[index][0] != "user" or messages[index + 1][0] != "assistant":
            raise ContinuityValidationError(
                "rolling context must drop complete turns"
            )


def _derive_rolling_transition_commitment(
    *,
    key: bytes,
    actor_ref: str,
    session_ref: str,
    predecessor_turn_ref: str,
    turn_ref: str,
    predecessor_transcript_commitment: str,
    submitted_prefix_commitment: str,
    predecessor_dropped_message_count: int,
) -> str:
    if not isinstance(key, bytes) or len(key) < 16:
        raise ContinuityValidationError("key must contain at least 16 bytes")
    payload = {
        "actor_ref": actor_ref,
        "domain": "LARIS-ROLLING-CONTEXT-TRANSITION-v1",
        "predecessor_dropped_message_count": (
            predecessor_dropped_message_count
        ),
        "predecessor_transcript_commitment": (
            predecessor_transcript_commitment
        ),
        "predecessor_turn_ref": predecessor_turn_ref,
        "session_ref": session_ref,
        "submitted_prefix_commitment": submitted_prefix_commitment,
        "turn_ref": turn_ref,
    }
    return "hmac-sha256:" + hmac.new(
        key,
        canonical_json(payload),
        hashlib.sha256,
    ).hexdigest()


@dataclass(frozen=True)
class ContextEvidence:
    schema_version: str
    actor_ref: str
    session_ref: str
    turn_ref: str
    turn_index: int
    predecessor_turn_ref: str | None
    predecessor_transcript_commitment: str | None
    application_context_mode: ApplicationContextMode
    continuity_mechanism: ContinuityMechanism
    submitted_prefix_commitment: str | None
    submitted_context_commitment: str
    resulting_transcript_commitment: str
    submitted_message_count: int
    submitted_utf8_bytes: int
    predecessor_dropped_message_count: int | None = None
    rolling_transition_commitment: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version not in {"1.0", "1.1"}:
            raise ContinuityValidationError("schema_version must be 1.0 or 1.1")
        for name in ("actor_ref", "session_ref", "turn_ref"):
            _require_text(name, getattr(self, name))
        if isinstance(self.turn_index, bool) or self.turn_index < 1:
            raise ContinuityValidationError("turn_index must be a positive integer")
        if self.predecessor_turn_ref is not None:
            _require_text("predecessor_turn_ref", self.predecessor_turn_ref)
        if self.predecessor_transcript_commitment is not None:
            _require_commitment(
                "predecessor_transcript_commitment",
                self.predecessor_transcript_commitment,
            )
        if self.submitted_prefix_commitment is not None:
            _require_commitment(
                "submitted_prefix_commitment",
                self.submitted_prefix_commitment,
            )
        _require_commitment(
            "submitted_context_commitment", self.submitted_context_commitment
        )
        _require_commitment(
            "resulting_transcript_commitment",
            self.resulting_transcript_commitment,
        )
        if (
            isinstance(self.submitted_message_count, bool)
            or self.submitted_message_count < 1
        ):
            raise ContinuityValidationError(
                "submitted_message_count must be a positive integer"
            )
        if (
            isinstance(self.submitted_utf8_bytes, bool)
            or self.submitted_utf8_bytes < 0
        ):
            raise ContinuityValidationError(
                "submitted_utf8_bytes must be a non-negative integer"
            )
        dropped = self.predecessor_dropped_message_count
        if dropped is not None and (
            isinstance(dropped, bool)
            or not isinstance(dropped, int)
            or dropped <= 0
        ):
            raise ContinuityValidationError(
                "predecessor_dropped_message_count must be a positive integer"
            )
        if self.rolling_transition_commitment is not None:
            _require_commitment(
                "rolling_transition_commitment",
                self.rolling_transition_commitment,
            )
        if self.schema_version == "1.0":
            if (
                self.continuity_mechanism
                is ContinuityMechanism.ROLLING_CONTEXT_RECOMPUTE
            ):
                raise ContinuityValidationError(
                    "schema 1.0 cannot claim rolling context"
                )
            if (
                dropped is not None
                or self.rolling_transition_commitment is not None
            ):
                raise ContinuityValidationError(
                    "schema 1.0 cannot contain a rolling transition"
                )
        elif (
            self.continuity_mechanism
            is ContinuityMechanism.ROLLING_CONTEXT_RECOMPUTE
        ):
            if dropped is None or self.rolling_transition_commitment is None:
                raise ContinuityValidationError(
                    "rolling context boundary is incomplete"
                )
        elif (
            dropped is not None
            or self.rolling_transition_commitment is not None
        ):
                raise ContinuityValidationError(
                    "only rolling context can contain a rolling transition"
                )

    def to_mapping(self) -> dict[str, object]:
        mapping = {
            "schema_version": self.schema_version,
            "actor_ref": self.actor_ref,
            "session_ref": self.session_ref,
            "turn_ref": self.turn_ref,
            "turn_index": self.turn_index,
            "predecessor_turn_ref": self.predecessor_turn_ref,
            "predecessor_transcript_commitment": (
                self.predecessor_transcript_commitment
            ),
            "application_context_mode": self.application_context_mode.value,
            "continuity_mechanism": self.continuity_mechanism.value,
            "submitted_prefix_commitment": self.submitted_prefix_commitment,
            "submitted_context_commitment": self.submitted_context_commitment,
            "resulting_transcript_commitment": (
                self.resulting_transcript_commitment
            ),
            "submitted_message_count": self.submitted_message_count,
            "submitted_utf8_bytes": self.submitted_utf8_bytes,
        }
        if self.schema_version == "1.1":
            mapping.update(
                {
                    "predecessor_dropped_message_count": (
                        self.predecessor_dropped_message_count
                    ),
                    "rolling_transition_commitment": (
                        self.rolling_transition_commitment
                    ),
                }
            )
        return mapping

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ContextEvidence":
        legacy_fields = {
            "schema_version",
            "actor_ref",
            "session_ref",
            "turn_ref",
            "turn_index",
            "predecessor_turn_ref",
            "predecessor_transcript_commitment",
            "application_context_mode",
            "continuity_mechanism",
            "submitted_prefix_commitment",
            "submitted_context_commitment",
            "resulting_transcript_commitment",
            "submitted_message_count",
            "submitted_utf8_bytes",
        }
        current_fields = legacy_fields | {
            "predecessor_dropped_message_count",
            "rolling_transition_commitment",
        }
        if not isinstance(value, Mapping):
            raise ContinuityValidationError(
                "context evidence field set is invalid"
            )
        data = dict(value)
        schema_version = data.get("schema_version")
        expected_fields = legacy_fields if schema_version == "1.0" else current_fields
        if set(data) != expected_fields:
            raise ContinuityValidationError(
                "context evidence field set is invalid"
            )
        try:
            data["application_context_mode"] = ApplicationContextMode(
                data["application_context_mode"]
            )
            data["continuity_mechanism"] = ContinuityMechanism(
                data["continuity_mechanism"]
            )
            evidence = cls(**data)  # type: ignore[arg-type]
        except (TypeError, ValueError) as error:
            raise ContinuityValidationError(
                "context evidence values are invalid"
            ) from error
        allowed = {
            ApplicationContextMode.CUMULATIVE_CONTEXT: {
                ContinuityMechanism.FULL_CONTEXT_RECOMPUTE,
                ContinuityMechanism.ROLLING_CONTEXT_RECOMPUTE,
            },
            ApplicationContextMode.TURN_ONLY: {ContinuityMechanism.NONE},
        }[evidence.application_context_mode]
        if evidence.continuity_mechanism not in allowed:
            raise ContinuityValidationError("mode and mechanism are inconsistent")
        return evidence


def validate_context_transition(
    predecessor: ContextEvidence,
    current: ContextEvidence,
    *,
    key: bytes,
) -> None:
    if not isinstance(predecessor, ContextEvidence) or not isinstance(
        current, ContextEvidence
    ):
        raise ContinuityValidationError(
            "context transition requires ContextEvidence records"
        )
    if (
        current.actor_ref != predecessor.actor_ref
        or current.session_ref != predecessor.session_ref
        or current.predecessor_turn_ref != predecessor.turn_ref
        or current.turn_index != predecessor.turn_index + 1
        or current.predecessor_transcript_commitment
        != predecessor.resulting_transcript_commitment
    ):
        raise ContinuityValidationError(
            "context transition predecessor binding is invalid"
        )
    if (
        current.application_context_mode
        is not ApplicationContextMode.CUMULATIVE_CONTEXT
    ):
        raise ContinuityValidationError(
            "only cumulative context can claim a context transition"
        )
    if (
        current.continuity_mechanism
        is ContinuityMechanism.FULL_CONTEXT_RECOMPUTE
    ):
        if current.submitted_prefix_commitment != (
            predecessor.resulting_transcript_commitment
        ):
            raise ContinuityValidationError(
                "full context predecessor transcript is invalid"
            )
        return
    if (
        current.continuity_mechanism
        is not ContinuityMechanism.ROLLING_CONTEXT_RECOMPUTE
        or current.schema_version != "1.1"
    ):
        raise ContinuityValidationError(
            "rolling predecessor suffix is invalid"
        )
    dropped = current.predecessor_dropped_message_count
    prefix_commitment = current.submitted_prefix_commitment
    if (
        dropped is None
        or dropped % 2 != 0
        or prefix_commitment is None
        or current.rolling_transition_commitment is None
    ):
        raise ContinuityValidationError(
            "rolling predecessor suffix is invalid"
        )
    expected = _derive_rolling_transition_commitment(
        key=key,
        actor_ref=current.actor_ref,
        session_ref=current.session_ref,
        predecessor_turn_ref=predecessor.turn_ref,
        turn_ref=current.turn_ref,
        predecessor_transcript_commitment=(
            predecessor.resulting_transcript_commitment
        ),
        submitted_prefix_commitment=prefix_commitment,
        predecessor_dropped_message_count=dropped,
    )
    if not hmac.compare_digest(expected, current.rolling_transition_commitment):
        raise ContinuityValidationError(
            "rolling predecessor suffix is invalid"
        )


def build_context_evidence(
    *,
    key: bytes,
    actor_ref: str,
    session_ref: str,
    turn_ref: str,
    turn_index: int,
    predecessor_turn_ref: str | None,
    predecessor_transcript_commitment: str | None,
    application_context_mode: ApplicationContextMode,
    continuity_mechanism: ContinuityMechanism,
    submitted_messages: Sequence[tuple[str, str]],
    submitted_prefix_message_count: int,
    assistant_response: str,
    predecessor_messages: Sequence[tuple[str, str]] | None = None,
) -> ContextEvidence:
    if not isinstance(application_context_mode, ApplicationContextMode):
        raise ContinuityValidationError(
            "application_context_mode must be an ApplicationContextMode"
        )
    if not isinstance(continuity_mechanism, ContinuityMechanism):
        raise ContinuityValidationError(
            "continuity_mechanism must be a ContinuityMechanism"
        )
    allowed_mechanisms = {
        ApplicationContextMode.CUMULATIVE_CONTEXT: {
            ContinuityMechanism.FULL_CONTEXT_RECOMPUTE,
            ContinuityMechanism.ROLLING_CONTEXT_RECOMPUTE,
        },
        ApplicationContextMode.TURN_ONLY: {ContinuityMechanism.NONE},
    }[application_context_mode]
    if continuity_mechanism not in allowed_mechanisms:
        raise ContinuityValidationError("mode and mechanism are inconsistent")
    if isinstance(turn_index, bool) or not isinstance(turn_index, int) or turn_index < 1:
        raise ContinuityValidationError("turn_index must be a positive integer")
    if (
        isinstance(submitted_prefix_message_count, bool)
        or not isinstance(submitted_prefix_message_count, int)
        or submitted_prefix_message_count < 0
    ):
        raise ContinuityValidationError(
            "submitted_prefix_message_count must be non-negative"
        )
    normalized = _normalize_messages(submitted_messages)
    if submitted_prefix_message_count > len(normalized):
        raise ContinuityValidationError(
            "submitted_prefix_message_count exceeds submitted messages"
        )
    if not isinstance(assistant_response, str):
        raise ContinuityValidationError("assistant_response must be text")

    if turn_index == 1:
        if predecessor_turn_ref is not None:
            raise ContinuityValidationError("the first turn cannot have a predecessor")
    elif predecessor_turn_ref is None:
        raise ContinuityValidationError("later turns require a predecessor_turn_ref")

    prefix_commitment: str | None = None
    predecessor_dropped_message_count: int | None = None
    rolling_transition_commitment: str | None = None
    if application_context_mode is ApplicationContextMode.CUMULATIVE_CONTEXT:
        if turn_index == 1:
            if predecessor_transcript_commitment is not None:
                raise ContinuityValidationError(
                    "the first turn cannot bind a predecessor transcript"
                )
            if submitted_prefix_message_count != 0:
                raise ContinuityValidationError(
                    "the first cumulative turn cannot have a submitted prefix"
                )
        else:
            if predecessor_transcript_commitment is None:
                raise ContinuityValidationError(
                    "later cumulative turns require a predecessor transcript"
                )
            if submitted_prefix_message_count == 0:
                raise ContinuityValidationError(
                    "later cumulative turns require a submitted prefix"
                )
            prefix_messages = [
                (item["role"], item["content"])
                for item in normalized[:submitted_prefix_message_count]
            ]
            if (
                continuity_mechanism
                is ContinuityMechanism.FULL_CONTEXT_RECOMPUTE
            ):
                prefix_commitment = derive_context_commitment(
                    key,
                    "resulting_transcript",
                    prefix_messages,
                )
                if not hmac.compare_digest(
                    prefix_commitment,
                    _require_commitment(
                        "predecessor_transcript_commitment",
                        predecessor_transcript_commitment,
                    ),
                ):
                    raise ContinuityValidationError(
                        "submitted prefix does not match predecessor transcript"
                    )
            else:
                if predecessor_messages is None:
                    raise ContinuityValidationError(
                        "rolling context requires predecessor messages"
                    )
                predecessor_normalized = _normalize_messages(
                    predecessor_messages
                )
                predecessor_tuples = [
                    (item["role"], item["content"])
                    for item in predecessor_normalized
                ]
                observed_predecessor = derive_context_commitment(
                    key,
                    "resulting_transcript",
                    predecessor_tuples,
                )
                if not hmac.compare_digest(
                    observed_predecessor,
                    predecessor_transcript_commitment,
                ):
                    raise ContinuityValidationError(
                        "rolling predecessor transcript is inconsistent"
                    )
                _require_complete_dialogue_turns(prefix_messages)
                dropped = len(predecessor_tuples) - len(prefix_messages)
                if dropped <= 0 or dropped % 2 != 0:
                    raise ContinuityValidationError(
                        "rolling context must drop complete turns"
                    )
                _require_complete_dialogue_turns(predecessor_tuples[:dropped])
                if predecessor_tuples[dropped:] != prefix_messages:
                    raise ContinuityValidationError(
                        "submitted prefix is not an exact predecessor suffix"
                    )
                predecessor_dropped_message_count = dropped
                prefix_commitment = derive_context_commitment(
                    key,
                    "rolling_predecessor_suffix",
                    prefix_messages,
                )
                rolling_transition_commitment = (
                    _derive_rolling_transition_commitment(
                        key=key,
                        actor_ref=actor_ref,
                        session_ref=session_ref,
                        predecessor_turn_ref=predecessor_turn_ref,
                        turn_ref=turn_ref,
                        predecessor_transcript_commitment=(
                            predecessor_transcript_commitment
                        ),
                        submitted_prefix_commitment=prefix_commitment,
                        predecessor_dropped_message_count=dropped,
                    )
                )
    else:
        if predecessor_transcript_commitment is not None:
            raise ContinuityValidationError(
                "turn-only/none cannot claim a predecessor transcript"
            )
        if submitted_prefix_message_count != 0:
            raise ContinuityValidationError(
                "turn-only/none cannot contain a submitted prefix"
            )
        if predecessor_messages is not None:
            raise ContinuityValidationError(
                "turn-only/none cannot contain predecessor messages"
            )

    normalized_tuples = [
        (item["role"], item["content"]) for item in normalized
    ]
    resulting_messages = normalized_tuples + [("assistant", assistant_response)]
    return ContextEvidence(
        schema_version="1.1",
        actor_ref=actor_ref,
        session_ref=session_ref,
        turn_ref=turn_ref,
        turn_index=turn_index,
        predecessor_turn_ref=predecessor_turn_ref,
        predecessor_transcript_commitment=predecessor_transcript_commitment,
        application_context_mode=application_context_mode,
        continuity_mechanism=continuity_mechanism,
        submitted_prefix_commitment=prefix_commitment,
        submitted_context_commitment=derive_context_commitment(
            key,
            "submitted_context",
            normalized_tuples,
        ),
        resulting_transcript_commitment=derive_context_commitment(
            key,
            "resulting_transcript",
            resulting_messages,
        ),
        submitted_message_count=len(normalized),
        submitted_utf8_bytes=sum(
            len(item["content"].encode("utf-8")) for item in normalized
        ),
        predecessor_dropped_message_count=predecessor_dropped_message_count,
        rolling_transition_commitment=rolling_transition_commitment,
    )
