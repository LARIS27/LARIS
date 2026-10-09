from __future__ import annotations

import base64
import binascii
import hashlib
import json
from pathlib import Path
import socket
from typing import Final

from .runtime_state_contract import (
    RuntimeStateFrameError,
    RuntimeStateSnapshot,
    runtime_state_snapshot_from_frame,
)
from .runtime_state_service import (
    CompletedApplyPage,
    CompletedNativeUnitPage,
    LogLinkedRuntimeCut,
    NativeCapturedUnitDescriptor,
    NativeRuntimeStateWireError,
    ObservedApplyBoundary,
    ObservedApplyTokenCell,
    encode_completed_applies_request,
    encode_completed_native_units_request,
    encode_log_linked_cut_request,
    encode_read_native_unit_request,
    encode_snapshot_request,
    native_stream_version_from_document,
)


_MAX_RESPONSE_BYTES: Final[int] = 64 * 1024 * 1024


class RuntimeStateClientError(RuntimeError):
    """Raised when the local snapshot service response cannot be trusted."""


class RuntimeStateClient:
    def __init__(self, socket_path: Path, *, timeout_seconds: float = 5.0) -> None:
        if not isinstance(socket_path, Path) or not socket_path.is_absolute():
            raise RuntimeStateClientError(
                "socket_path must be an absolute pathlib.Path"
            )
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or timeout_seconds <= 0
        ):
            raise RuntimeStateClientError("timeout_seconds must be positive")
        self.socket_path = socket_path
        self.timeout_seconds = float(timeout_seconds)

    def snapshot(
        self,
        *,
        not_before_ns: int,
        not_after_ns: int,
        observed_seq_id: int,
    ) -> RuntimeStateSnapshot:
        document = self._request_document(
            encode_snapshot_request(
                not_before_ns,
                not_after_ns,
                observed_seq_id,
            )
        )
        try:
            return runtime_state_snapshot_from_frame(document)
        except RuntimeStateFrameError as error:
            raise RuntimeStateClientError(
                "snapshot response is not a canonical complete frame"
            ) from error

    def log_linked_cut(
        self,
        *,
        not_before_ns: int,
        not_after_ns: int,
        observed_seq_id: int,
    ) -> LogLinkedRuntimeCut:
        document = self._request_document(
            encode_log_linked_cut_request(
                not_before_ns,
                not_after_ns,
                observed_seq_id,
            )
        )
        if not isinstance(document, dict) or set(document) != {
            "snapshot",
            "stream_version",
        }:
            raise RuntimeStateClientError(
                "Log-linked cut response fields are invalid"
            )
        try:
            return LogLinkedRuntimeCut(
                snapshot=runtime_state_snapshot_from_frame(
                    document["snapshot"]
                ),
                stream_version=native_stream_version_from_document(
                    document["stream_version"]
                ),
            )
        except (
            RuntimeStateFrameError,
            NativeRuntimeStateWireError,
        ) as error:
            raise RuntimeStateClientError(
                "Log-linked cut response is invalid"
            ) from error

    def stop(self) -> None:
        document = self._request_document(b"STOP {}\n")
        if document != {"state": "sealed"}:
            raise RuntimeStateClientError(
                "snapshot service stop response is invalid"
            )

    def arm(self) -> None:
        document = self._request_document(b"ARM {}\n")
        if document != {"state": "armed"}:
            raise RuntimeStateClientError(
                "snapshot service arm response is invalid"
            )

    def drain(self) -> None:
        document = self._request_document(b"DRAIN {}\n")
        if document != {"state": "drained"}:
            raise RuntimeStateClientError(
                "runtime-state drain response is invalid"
            )

    def completed_applies(self, *, after_cursor: int) -> CompletedApplyPage:
        document = self._request_document(
            encode_completed_applies_request(after_cursor)
        )
        return _completed_apply_page_from_document(
            document, after_cursor=after_cursor
        )

    def completed_native_units(
        self, *, after_cursor: int
    ) -> CompletedNativeUnitPage:
        document = self._request_document(
            encode_completed_native_units_request(after_cursor)
        )
        return _completed_native_unit_page_from_document(
            document, after_cursor=after_cursor
        )

    def read_native_unit(
        self,
        *,
        unit_id: str,
        offset: int,
        size: int,
    ) -> bytes:
        document = self._request_document(
            encode_read_native_unit_request(
                unit_id=unit_id,
                offset=offset,
                size=size,
            )
        )
        if not isinstance(document, dict) or set(document) != {
            "byte_count",
            "data_base64",
            "offset",
            "payload_sha256",
            "schema_version",
            "total_size",
            "unit_id",
        }:
            raise RuntimeStateClientError("native-unit read fields are invalid")
        if document["schema_version"] != "1.0":
            raise RuntimeStateClientError("native-unit read schema is invalid")
        if document["unit_id"] != unit_id or document["offset"] != offset:
            raise RuntimeStateClientError("native-unit read identity differs")
        byte_count = _positive_integer(document["byte_count"], "byte_count")
        total_size = _positive_integer(document["total_size"], "total_size")
        if byte_count != size or offset > total_size - byte_count:
            raise RuntimeStateClientError("native-unit read range differs")
        digest = document["payload_sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise RuntimeStateClientError("native-unit read digest is invalid")
        encoded = document["data_base64"]
        if not isinstance(encoded, str):
            raise RuntimeStateClientError("native-unit read payload is invalid")
        try:
            payload = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as error:
            raise RuntimeStateClientError(
                "native-unit read payload is invalid"
            ) from error
        if (
            len(payload) != byte_count
            or base64.b64encode(payload).decode("ascii") != encoded
        ):
            raise RuntimeStateClientError("native-unit read payload differs")
        return payload

    def read_complete_native_unit(
        self, descriptor: NativeCapturedUnitDescriptor
    ) -> bytes:
        if not isinstance(descriptor, NativeCapturedUnitDescriptor):
            raise RuntimeStateClientError("native-unit descriptor is invalid")
        result = bytearray()
        while len(result) < descriptor.payload_size:
            size = min(1024 * 1024, descriptor.payload_size - len(result))
            result.extend(
                self.read_native_unit(
                    unit_id=descriptor.unit_id,
                    offset=len(result),
                    size=size,
                )
            )
        payload = bytes(result)
        if hashlib.sha256(payload).hexdigest() != descriptor.payload_sha256:
            raise RuntimeStateClientError("native-unit payload digest mismatch")
        return payload

    def _request_document(self, request: bytes) -> object:
        if self.socket_path.is_symlink():
            raise RuntimeStateClientError("snapshot socket path must not be a symlink")
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.settimeout(self.timeout_seconds)
        try:
            client.connect(str(self.socket_path))
            client.sendall(request)
            client.shutdown(socket.SHUT_WR)
            response = bytearray()
            while len(response) <= _MAX_RESPONSE_BYTES:
                chunk = client.recv(min(65536, _MAX_RESPONSE_BYTES + 1 - len(response)))
                if not chunk:
                    break
                response.extend(chunk)
        except (OSError, TimeoutError) as error:
            raise RuntimeStateClientError("snapshot service communication failed") from error
        finally:
            client.close()
        if len(response) > _MAX_RESPONSE_BYTES:
            raise RuntimeStateClientError("snapshot response exceeds the size limit")
        if not response.endswith(b"\n") or response.count(b"\n") != 1:
            raise RuntimeStateClientError("snapshot response framing is invalid")

        def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
            result: dict[str, object] = {}
            for key, value in pairs:
                if key in result:
                    raise RuntimeStateClientError(
                        "snapshot response contains a duplicate field"
                    )
                result[key] = value
            return result

        try:
            document = json.loads(
                response[:-1], object_pairs_hook=reject_duplicates
            )
        except RuntimeStateClientError:
            raise
        except (UnicodeError, json.JSONDecodeError) as error:
            raise RuntimeStateClientError("snapshot response JSON is invalid") from error
        if isinstance(document, dict) and document.get("state") == "error":
            message = document.get("message")
            if not isinstance(message, str) or not message:
                message = "snapshot service rejected the request"
            raise RuntimeStateClientError(message)
        return document


def _completed_apply_page_from_document(
    value: object, *, after_cursor: int
) -> CompletedApplyPage:
    if not isinstance(value, dict) or set(value) != {
        "boundaries",
        "next_cursor",
        "truncated",
    }:
        raise RuntimeStateClientError("completed-apply page fields are invalid")
    next_cursor = _nonnegative_integer(value["next_cursor"], "next_cursor")
    if not isinstance(value["truncated"], bool):
        raise RuntimeStateClientError("completed-apply truncated flag is invalid")
    raw_boundaries = value["boundaries"]
    if not isinstance(raw_boundaries, list):
        raise RuntimeStateClientError("completed-apply boundaries are invalid")
    boundaries: list[ObservedApplyBoundary] = []
    expected_cursor = after_cursor + 1
    for raw_boundary in raw_boundaries:
        if not isinstance(raw_boundary, dict) or set(raw_boundary) != {
            "apply_call_id",
            "cursor",
            "monotonic_ns",
            "mutation_kinds",
            "observer_loss_count",
            "parse_failure_count",
            "runtime_instance_id",
            "runtime_pid",
            "runtime_start_time_ticks",
            "sequence_ids",
            "token_cells",
            "unmatched_call_count",
            "unsupported_mutation_count",
            "watermark_monotonic_ns",
        }:
            raise RuntimeStateClientError(
                "completed-apply boundary fields are invalid"
            )
        cursor = _positive_integer(raw_boundary["cursor"], "cursor")
        if cursor != expected_cursor:
            raise RuntimeStateClientError(
                "completed-apply boundary cursor is not contiguous"
            )
        expected_cursor += 1
        runtime_instance_id = raw_boundary["runtime_instance_id"]
        if not isinstance(runtime_instance_id, str) or not runtime_instance_id:
            raise RuntimeStateClientError("runtime_instance_id is invalid")
        sequence_ids = _sorted_unique_integers(
            raw_boundary["sequence_ids"], "sequence_ids", allow_empty=True
        )
        mutation_kinds_value = raw_boundary["mutation_kinds"]
        if not isinstance(mutation_kinds_value, list) or any(
            not isinstance(item, str) or not item
            for item in mutation_kinds_value
        ):
            raise RuntimeStateClientError("mutation_kinds are invalid")
        mutation_kinds = tuple(mutation_kinds_value)
        if mutation_kinds != tuple(sorted(set(mutation_kinds))):
            raise RuntimeStateClientError("mutation_kinds are not canonical")
        raw_cells = raw_boundary["token_cells"]
        if not isinstance(raw_cells, list):
            raise RuntimeStateClientError("completed-apply token_cells are invalid")
        cells: list[ObservedApplyTokenCell] = []
        for raw_cell in raw_cells:
            if not isinstance(raw_cell, dict) or set(raw_cell) != {
                "cell_index",
                "ownership_generation",
                "seq_id",
                "stream_index",
                "token_position",
            }:
                raise RuntimeStateClientError(
                    "completed-apply token-cell fields are invalid"
                )
            cell = ObservedApplyTokenCell(
                seq_id=_nonnegative_integer(raw_cell["seq_id"], "seq_id"),
                stream_index=_nonnegative_integer(
                    raw_cell["stream_index"], "stream_index"
                ),
                token_position=_nonnegative_integer(
                    raw_cell["token_position"], "token_position"
                ),
                cell_index=_nonnegative_integer(
                    raw_cell["cell_index"], "cell_index"
                ),
                ownership_generation=_positive_integer(
                    raw_cell["ownership_generation"], "ownership_generation"
                ),
            )
            if cell.seq_id not in sequence_ids:
                raise RuntimeStateClientError(
                    "completed-apply token-cell sequence is undeclared"
                )
            cells.append(cell)
        apply_call_id = _positive_integer(
            raw_boundary["apply_call_id"], "apply_call_id"
        )
        if any(cell.ownership_generation != apply_call_id for cell in cells):
            raise RuntimeStateClientError(
                "completed-apply ownership generation differs"
            )
        boundaries.append(
            ObservedApplyBoundary(
                cursor=cursor,
                runtime_instance_id=runtime_instance_id,
                runtime_pid=_positive_integer(
                    raw_boundary["runtime_pid"], "runtime_pid"
                ),
                runtime_start_time_ticks=_positive_integer(
                    raw_boundary["runtime_start_time_ticks"],
                    "runtime_start_time_ticks",
                ),
                apply_call_id=apply_call_id,
                monotonic_ns=_nonnegative_integer(
                    raw_boundary["monotonic_ns"], "monotonic_ns"
                ),
                sequence_ids=sequence_ids,
                token_cells=tuple(cells),
                mutation_kinds=mutation_kinds,
                observer_loss_count=_nonnegative_integer(
                    raw_boundary["observer_loss_count"],
                    "observer_loss_count",
                ),
                parse_failure_count=_nonnegative_integer(
                    raw_boundary["parse_failure_count"],
                    "parse_failure_count",
                ),
                unmatched_call_count=_nonnegative_integer(
                    raw_boundary["unmatched_call_count"],
                    "unmatched_call_count",
                ),
                unsupported_mutation_count=_nonnegative_integer(
                    raw_boundary["unsupported_mutation_count"],
                    "unsupported_mutation_count",
                ),
                watermark_monotonic_ns=_nonnegative_integer(
                    raw_boundary["watermark_monotonic_ns"],
                    "watermark_monotonic_ns",
                ),
            )
        )
    if next_cursor != after_cursor + len(boundaries):
        raise RuntimeStateClientError(
            "completed-apply next_cursor differs from the page"
        )
    return CompletedApplyPage(
        next_cursor=next_cursor,
        truncated=value["truncated"],
        boundaries=tuple(boundaries),
    )


def _completed_native_unit_page_from_document(
    value: object, *, after_cursor: int
) -> CompletedNativeUnitPage:
    if not isinstance(value, dict) or set(value) != {
        "next_cursor",
        "truncated",
        "units",
    }:
        raise RuntimeStateClientError("completed native-unit page fields are invalid")
    next_cursor = _nonnegative_integer(value["next_cursor"], "next_cursor")
    if not isinstance(value["truncated"], bool):
        raise RuntimeStateClientError("completed native-unit truncated flag is invalid")
    raw_units = value["units"]
    if not isinstance(raw_units, list):
        raise RuntimeStateClientError("completed native-unit list is invalid")
    units: list[NativeCapturedUnitDescriptor] = []
    expected_cursor = after_cursor + 1
    expected_fields = {
        "apply_call_id",
        "cursor",
        "fragment_count",
        "payload_sha256",
        "payload_size",
        "runtime_instance_id",
        "seq_id",
        "stream_index",
        "token_cells",
        "unit_id",
        "writer_monotonic_ns",
    }
    for raw_unit in raw_units:
        if not isinstance(raw_unit, dict) or set(raw_unit) != expected_fields:
            raise RuntimeStateClientError("native-unit descriptor fields are invalid")
        cursor = _positive_integer(raw_unit["cursor"], "cursor")
        if cursor != expected_cursor:
            raise RuntimeStateClientError(
                "native-unit descriptor cursor is not contiguous"
            )
        expected_cursor += 1
        raw_cells = raw_unit["token_cells"]
        if not isinstance(raw_cells, list):
            raise RuntimeStateClientError("native-unit token_cells are invalid")
        token_cells: list[tuple[int, int]] = []
        for raw_cell in raw_cells:
            if not isinstance(raw_cell, list) or len(raw_cell) != 2:
                raise RuntimeStateClientError("native-unit token cell is invalid")
            token_cells.append(
                (
                    _nonnegative_integer(raw_cell[0], "token_position"),
                    _nonnegative_integer(raw_cell[1], "cell_index"),
                )
            )
        try:
            units.append(
                NativeCapturedUnitDescriptor(
                    cursor=cursor,
                    unit_id=raw_unit["unit_id"],
                    runtime_instance_id=raw_unit["runtime_instance_id"],
                    apply_call_id=_positive_integer(
                        raw_unit["apply_call_id"], "apply_call_id"
                    ),
                    seq_id=_nonnegative_integer(raw_unit["seq_id"], "seq_id"),
                    stream_index=_nonnegative_integer(
                        raw_unit["stream_index"], "stream_index"
                    ),
                    writer_monotonic_ns=_nonnegative_integer(
                        raw_unit["writer_monotonic_ns"], "writer_monotonic_ns"
                    ),
                    token_cells=tuple(token_cells),
                    fragment_count=_positive_integer(
                        raw_unit["fragment_count"], "fragment_count"
                    ),
                    payload_size=_positive_integer(
                        raw_unit["payload_size"], "payload_size"
                    ),
                    payload_sha256=raw_unit["payload_sha256"],
                )
            )
        except Exception as error:
            if isinstance(error, RuntimeStateClientError):
                raise
            raise RuntimeStateClientError(
                "native-unit descriptor is invalid"
            ) from error
    if next_cursor != after_cursor + len(units):
        raise RuntimeStateClientError("native-unit next_cursor differs from the page")
    return CompletedNativeUnitPage(
        next_cursor=next_cursor,
        truncated=value["truncated"],
        units=tuple(units),
    )


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeStateClientError(f"{name} is invalid")
    return value


def _positive_integer(value: object, name: str) -> int:
    result = _nonnegative_integer(value, name)
    if result == 0:
        raise RuntimeStateClientError(f"{name} is invalid")
    return result


def _sorted_unique_integers(
    value: object, name: str, *, allow_empty: bool
) -> tuple[int, ...]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise RuntimeStateClientError(f"{name} is invalid")
    result = tuple(_nonnegative_integer(item, name) for item in value)
    if result != tuple(sorted(set(result))):
        raise RuntimeStateClientError(f"{name} is not canonical")
    return result


__all__ = [
    "CompletedApplyPage",
    "CompletedNativeUnitPage",
    "NativeCapturedUnitDescriptor",
    "ObservedApplyBoundary",
    "ObservedApplyTokenCell",
    "LogLinkedRuntimeCut",
    "RuntimeStateClient",
    "RuntimeStateClientError",
]
