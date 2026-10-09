from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import sqlite3
import threading
import time
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Final

from laris_onpremise.core.records import (
    SCHEMA_VERSION,
    JsonValue,
    LeaseEvent,
    LeaseRecord,
    LeaseState,
    RequestIdentity,
    canonical_json,
)


GENESIS_DIGEST: Final[str] = "sha256:" + ("0" * 64)
BINDING_LEDGER_SCHEMA_VERSION: Final[str] = "4.0"


class LedgerError(RuntimeError):
    """Base class for durable ledger failures."""


class LedgerIntegrityError(LedgerError):
    """Raised when the append-only event chain fails verification."""


class LedgerPolicyError(LedgerError):
    """Raised when an event tries to store a non-schema field."""


class LedgerSchemaError(LedgerError):
    """Raised when a ledger belongs to an unsupported historical schema."""


class LedgerWriterConflictError(LedgerError):
    """Raised when another Controller already owns the writer lock."""


class LeaseConflictError(LedgerError):
    """Raised when a durable Lease uniqueness invariant is violated."""


@dataclass(frozen=True)
class StoredEvent:
    event_counter: int
    event: LeaseEvent
    previous_digest: str
    event_digest: str
    event_mac: str


@dataclass(frozen=True)
class _ReplayedLease:
    lease: LeaseRecord
    updated_event_counter: int
    dispatch_authorized: bool = False
    acquisition_requested: bool = False
    acquisition_copied: bool = False
    acquisition_completion: _AcquisitionCompletion | None = None


@dataclass(frozen=True)
class _AcquisitionCompletion:
    capture_id: str
    payload_digest: str
    payload_size_bytes: int
    evidence_ref: str


_EVENT_DETAIL_FIELDS: Final[dict[str, frozenset[str]]] = {
    "request_received": frozenset(
        {
            "user_hmac",
            "identity_key_id",
            "auth_receipt_id",
            "session_ref",
            "connection_ref",
            "retry_of",
        }
    ),
    "request_queued": frozenset({"reason"}),
    "request_queue_cancelled": frozenset({"reason"}),
    "lease_reserved": frozenset(
        {
            "user_hmac",
            "identity_key_id",
            "auth_receipt_id",
            "session_ref",
            "connection_ref",
            "retry_of",
            "context_ref",
            "context_ref_source",
            "expected_slot_id",
            "expected_seq_id",
            "lease_generation",
        }
    ),
    "dispatch_authorized": frozenset({"evidence_ref"}),
    "upstream_request_sent": frozenset({"evidence_ref"}),
    "slot_assignment_confirmed": frozenset(
        {
            "response_confirmed_slot_id",
            "slot_epoch",
            "evidence_ref",
        }
    ),
    "terminal_response_received": frozenset({"evidence_ref", "reason"}),
    "slot_idle_confirmed": frozenset({"evidence_ref"}),
    "runtime_state_capture_started": frozenset({"capture_id"}),
    "runtime_state_acquisition_requested": frozenset(
        {"capture_id", "observed_seq_id", "evidence_ref"}
    ),
    "runtime_state_acquisition_copied": frozenset(
        {"capture_id", "evidence_ref"}
    ),
    "runtime_state_acquisition_completed": frozenset(
        {
            "capture_id",
            "request_ref",
            "runtime_instance_id",
            "slot_id",
            "slot_epoch",
            "observed_seq_id",
            "payload_digest",
            "payload_size_bytes",
            "evidence_ref",
        }
    ),
    "runtime_state_payload_committed": frozenset(
        {
            "capture_id",
            "payload_digest",
            "payload_size_bytes",
            "evidence_ref",
            "context_ref",
            "expected_slot_id",
            "response_confirmed_slot_id",
            "slot_epoch",
        }
    ),
    "runtime_state_post_copy_failed": frozenset(
        {"reason", "evidence_ref"}
    ),
    "capture_skipped": frozenset({"reason"}),
    "lease_released": frozenset(),
    "lease_unknown": frozenset({"reason", "controller_instance_id"}),
    "slot_quarantined": frozenset({"reason", "evidence_ref"}),
    "controller_recovered": frozenset(
        {"controller_instance_id", "reason"}
    ),
    "runtime_started": frozenset(
        {"controller_instance_id", "evidence_ref", "reason"}
    ),
    "audit_violation": frozenset(
        {
            "reason",
            "evidence_ref",
            "expected_slot_id",
            "response_confirmed_slot_id",
            "expected_seq_id",
            "observed_seq_id",
        }
    ),
}


_BINDING_SCHEMA_SQL: Final[str] = """
CREATE TABLE IF NOT EXISTS events (
    event_counter INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL UNIQUE,
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    previous_digest TEXT NOT NULL,
    event_digest TEXT NOT NULL,
    event_mac TEXT NOT NULL,
    wall_time_ns INTEGER NOT NULL,
    monotonic_ns INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS leases (
    lease_id TEXT PRIMARY KEY,
    user_hmac TEXT NOT NULL,
    identity_key_id TEXT NOT NULL,
    auth_receipt_id TEXT NOT NULL,
    session_ref TEXT NOT NULL,
    request_ref TEXT NOT NULL UNIQUE,
    connection_ref TEXT NOT NULL,
    retry_of TEXT,
    runtime_instance_id TEXT NOT NULL,
    context_ref TEXT NOT NULL,
    context_ref_source TEXT NOT NULL,
    expected_slot_id INTEGER NOT NULL,
    expected_seq_id INTEGER NOT NULL,
    lease_generation INTEGER NOT NULL,
    response_confirmed_slot_id INTEGER,
    observed_seq_id INTEGER,
    slot_epoch INTEGER,
    state TEXT NOT NULL,
    holds_sequence INTEGER NOT NULL CHECK (holds_sequence IN (0, 1)),
    backend_released INTEGER NOT NULL CHECK (backend_released IN (0, 1)),
    capture_id TEXT,
    capture_skipped_reason TEXT,
    updated_event_counter INTEGER NOT NULL,
    UNIQUE(
        runtime_instance_id,
        context_ref,
        expected_slot_id,
        lease_generation
    ),
    FOREIGN KEY(updated_event_counter) REFERENCES events(event_counter)
);

CREATE UNIQUE INDEX IF NOT EXISTS one_holder_per_sequence
ON leases(runtime_instance_id, context_ref, expected_seq_id)
WHERE holds_sequence = 1;

CREATE TABLE IF NOT EXISTS sequence_counters (
    runtime_instance_id TEXT NOT NULL,
    context_ref TEXT NOT NULL,
    expected_slot_id INTEGER NOT NULL,
    last_lease_generation INTEGER NOT NULL,
    last_slot_epoch INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY(runtime_instance_id, context_ref, expected_slot_id)
);

CREATE TABLE IF NOT EXISTS ledger_metadata (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    ledger_id TEXT NOT NULL UNIQUE,
    schema_version TEXT NOT NULL,
    schema_fingerprint TEXT NOT NULL
);
"""


def _normalized_schema_sql(value: object) -> str | None:
    if value is None:
        return None
    return " ".join(str(value).split())


def _schema_fingerprint(connection: sqlite3.Connection) -> str:
    objects = connection.execute(
        "SELECT type, name, tbl_name, sql FROM sqlite_master "
        "WHERE type IN ('table', 'index', 'trigger', 'view') "
        "AND name NOT LIKE 'sqlite_%' "
        "ORDER BY type, name"
    ).fetchall()
    descriptor: dict[str, JsonValue] = {"objects": [], "tables": {}}
    object_items = descriptor["objects"]
    table_items = descriptor["tables"]
    assert isinstance(object_items, list)
    assert isinstance(table_items, dict)
    for row in objects:
        object_items.append(
            {
                "type": str(row["type"]),
                "name": str(row["name"]),
                "table": str(row["tbl_name"]),
                "sql": _normalized_schema_sql(row["sql"]),
            }
        )
        if row["type"] != "table":
            continue
        table_name = str(row["name"])
        columns = [
            {
                "cid": int(item["cid"]),
                "name": str(item["name"]),
                "type": str(item["type"]),
                "notnull": int(item["notnull"]),
                "default": item["dflt_value"],
                "pk": int(item["pk"]),
                "hidden": int(item["hidden"]),
            }
            for item in connection.execute(
                "SELECT * FROM pragma_table_xinfo(?) ORDER BY cid", (table_name,)
            ).fetchall()
        ]
        foreign_keys = [
            {
                "id": int(item["id"]),
                "seq": int(item["seq"]),
                "table": str(item["table"]),
                "from": str(item["from"]),
                "to": str(item["to"]),
                "on_update": str(item["on_update"]),
                "on_delete": str(item["on_delete"]),
                "match": str(item["match"]),
            }
            for item in connection.execute(
                "SELECT * FROM pragma_foreign_key_list(?) ORDER BY id, seq",
                (table_name,),
            ).fetchall()
        ]
        indexes: list[JsonValue] = []
        for index in connection.execute(
            "SELECT * FROM pragma_index_list(?) ORDER BY name", (table_name,)
        ).fetchall():
            index_name = str(index["name"])
            columns_for_index = [
                {
                    "seqno": int(item["seqno"]),
                    "cid": int(item["cid"]),
                    "name": None if item["name"] is None else str(item["name"]),
                    "desc": int(item["desc"]),
                    "coll": None if item["coll"] is None else str(item["coll"]),
                    "key": int(item["key"]),
                }
                for item in connection.execute(
                    "SELECT * FROM pragma_index_xinfo(?) ORDER BY seqno",
                    (index_name,),
                ).fetchall()
            ]
            indexes.append(
                {
                    "name": index_name,
                    "unique": int(index["unique"]),
                    "origin": str(index["origin"]),
                    "partial": int(index["partial"]),
                    "columns": columns_for_index,
                }
            )
        table_items[table_name] = {
            "columns": columns,
            "foreign_keys": foreign_keys,
            "indexes": indexes,
        }
    return "sha256:" + hashlib.sha256(canonical_json(descriptor)).hexdigest()


@lru_cache(maxsize=1)
def _expected_schema_fingerprint() -> str:
    reference = sqlite3.connect(":memory:")
    reference.row_factory = sqlite3.Row
    try:
        reference.execute("PRAGMA foreign_keys=ON")
        reference.executescript(_BINDING_SCHEMA_SQL)
        return _schema_fingerprint(reference)
    finally:
        reference.close()


class _ProcessWriterLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._stream = path.open("a+b")
        self._locked = False
        try:
            self._stream.seek(0, os.SEEK_END)
            if self._stream.tell() == 0:
                self._stream.write(b"\0")
                self._stream.flush()
            self._stream.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self._stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(
                    self._stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
                )
            self._locked = True
        except OSError as error:
            self._stream.close()
            raise LedgerWriterConflictError(
                f"binding ledger writer is already active: {path}"
            ) from error

    def close(self) -> None:
        if not self._locked:
            return
        try:
            self._stream.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self._stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._stream.fileno(), fcntl.LOCK_UN)
        finally:
            self._locked = False
            self._stream.close()


class _Unset:
    pass


UNSET = _Unset()


class BindingLedger:
    """SQLite-backed event chain with a mutable Lease materialized view."""

    def __init__(self, path: Path | str, *, mac_key: bytes) -> None:
        self.path = Path(path)
        self.anchor_path = self.path.with_suffix(
            self.path.suffix + ".anchor.json"
        )
        self._connection: sqlite3.Connection | None = None
        self._writer_lock: _ProcessWriterLock | None = None
        self._mutex = threading.RLock()
        self._mac_key = b""

        if not isinstance(mac_key, bytes) or len(mac_key) < 16:
            raise LedgerPolicyError("mac_key must contain at least 16 bytes")
        self._mac_key = bytes(mac_key)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer_lock = _ProcessWriterLock(
            self.path.with_suffix(self.path.suffix + ".writer.lock")
        )
        try:
            self._connection = sqlite3.connect(
                self.path,
                timeout=5.0,
                isolation_level=None,
                check_same_thread=False,
            )
            self._connection.row_factory = sqlite3.Row
            self._initialize()
            self._initialize_anchor()
        except Exception:
            self.close()
            raise

    def _initialize(self) -> None:
        connection = self._require_connection()
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA foreign_keys=ON")
        self._validate_existing_schema(connection)
        connection.executescript(_BINDING_SCHEMA_SQL)
        schema_fingerprint = _schema_fingerprint(connection)
        if schema_fingerprint != _expected_schema_fingerprint():
            raise LedgerSchemaError(
                "binding ledger schema fingerprint does not match the current contract"
            )
        metadata = connection.execute(
            "SELECT ledger_id, schema_version, schema_fingerprint "
            "FROM ledger_metadata WHERE singleton = 1"
        ).fetchone()
        if metadata is None:
            connection.execute(
                "INSERT INTO ledger_metadata "
                "(singleton, ledger_id, schema_version, schema_fingerprint) "
                "VALUES (1, ?, ?, ?)",
                (
                    f"ledger_{uuid.uuid4().hex}",
                    BINDING_LEDGER_SCHEMA_VERSION,
                    schema_fingerprint,
                ),
            )
        elif (
            metadata["schema_version"] != BINDING_LEDGER_SCHEMA_VERSION
            or metadata["schema_fingerprint"] != schema_fingerprint
        ):
            raise LedgerSchemaError(
                "binding ledger metadata schema fingerprint is invalid"
            )

    @staticmethod
    def _validate_existing_schema(connection: sqlite3.Connection) -> None:
        tables = {
            str(row["name"])
            for row in connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        }
        if not tables:
            return

        expected_tables = {
            "events",
            "leases",
            "sequence_counters",
            "ledger_metadata",
        }
        if tables != expected_tables:
            raise LedgerSchemaError(
                "historical binding ledger schema is unsupported by the current runtime"
            )
        actual = _schema_fingerprint(connection)
        expected = _expected_schema_fingerprint()
        if actual != expected:
            raise LedgerSchemaError(
                "binding ledger schema fingerprint does not match the current contract"
            )
        metadata = connection.execute(
            "SELECT schema_version, schema_fingerprint FROM ledger_metadata "
            "WHERE singleton = 1"
        ).fetchone()
        if (
            metadata is None
            or metadata["schema_version"] != BINDING_LEDGER_SCHEMA_VERSION
            or metadata["schema_fingerprint"] != actual
        ):
            raise LedgerSchemaError(
                "binding ledger metadata schema fingerprint is invalid"
            )

    def _anchor_core(self) -> dict[str, JsonValue]:
        connection = self._require_connection()
        metadata = connection.execute(
            "SELECT ledger_id FROM ledger_metadata WHERE singleton = 1"
        ).fetchone()
        if metadata is None:
            raise LedgerIntegrityError("ledger identity is missing")
        head = connection.execute(
            """
            SELECT event_counter, event_digest
            FROM events
            ORDER BY event_counter DESC
            LIMIT 1
            """
        ).fetchone()
        return {
            "schema_version": SCHEMA_VERSION,
            "ledger_id": metadata["ledger_id"],
            "event_count": 0 if head is None else int(head["event_counter"]),
            "chain_head_digest": (
                GENESIS_DIGEST if head is None else head["event_digest"]
            ),
        }

    def _write_anchor(self) -> None:
        core = self._anchor_core()
        anchor_mac = "hmac-sha256:" + hmac.new(
            self._mac_key, canonical_json(core), hashlib.sha256
        ).hexdigest()
        document = dict(core)
        document["anchor_mac"] = anchor_mac
        payload = canonical_json(document) + b"\n"
        temporary = self.anchor_path.with_name(
            f".{self.anchor_path.name}.{uuid.uuid4().hex}.tmp"
        )
        try:
            with temporary.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.anchor_path)
            if os.name == "posix":
                directory_fd = os.open(self.anchor_path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        finally:
            temporary.unlink(missing_ok=True)

    def _initialize_anchor(self) -> None:
        event_count = int(
            self._require_connection()
            .execute("SELECT COUNT(*) AS count FROM events")
            .fetchone()["count"]
        )
        if self.anchor_path.exists():
            self.verify_chain()
        elif event_count == 0:
            self._write_anchor()
        else:
            raise LedgerIntegrityError("existing ledger anchor is missing")

    def verify_anchor(self) -> None:
        try:
            raw = self.anchor_path.read_text(encoding="utf-8")
            document = json.loads(raw)
        except (OSError, json.JSONDecodeError) as error:
            raise LedgerIntegrityError("ledger anchor cannot be read") from error
        expected_keys = {
            "schema_version",
            "ledger_id",
            "event_count",
            "chain_head_digest",
            "anchor_mac",
        }
        if not isinstance(document, dict) or set(document) != expected_keys:
            raise LedgerIntegrityError("ledger anchor schema is invalid")
        core = {key: document[key] for key in expected_keys - {"anchor_mac"}}
        expected_mac = "hmac-sha256:" + hmac.new(
            self._mac_key, canonical_json(core), hashlib.sha256
        ).hexdigest()
        if not isinstance(document["anchor_mac"], str) or not hmac.compare_digest(
            expected_mac, document["anchor_mac"]
        ):
            raise LedgerIntegrityError("ledger anchor MAC mismatch")
        if core != self._anchor_core():
            raise LedgerIntegrityError("ledger anchor does not match chain head")

    def _require_connection(self) -> sqlite3.Connection:
        if self._connection is None:
            raise LedgerError("binding ledger is closed")
        return self._connection

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._mutex:
            connection = self._require_connection()
            self.verify_chain()
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
                verified = self._verify_event_chain(connection)
                self._verify_materialized_state(connection, verified)
            except Exception:
                connection.rollback()
                raise
            else:
                connection.commit()
                self._write_anchor()

    @staticmethod
    def _event_payload(event: LeaseEvent) -> dict[str, JsonValue]:
        return {
            "schema_version": event.schema_version,
            "event_id": event.event_id,
            "event_type": event.event_type,
            "lease_id": event.lease_id,
            "request_ref": event.request_ref,
            "runtime_instance_id": event.runtime_instance_id,
            "wall_time_ns": event.wall_time_ns,
            "monotonic_ns": event.monotonic_ns,
            "details": dict(event.details),
        }

    @staticmethod
    def _event_from_payload(payload: Mapping[str, object]) -> LeaseEvent:
        details = payload.get("details")
        if not isinstance(details, Mapping):
            raise LedgerIntegrityError("event details are not an object")
        return LeaseEvent(
            schema_version=payload.get("schema_version"),
            event_id=payload.get("event_id"),
            event_type=payload.get("event_type"),
            lease_id=payload.get("lease_id"),
            request_ref=payload.get("request_ref"),
            runtime_instance_id=payload.get("runtime_instance_id"),
            wall_time_ns=payload.get("wall_time_ns"),
            monotonic_ns=payload.get("monotonic_ns"),
            details=details,
        )

    @staticmethod
    def _validate_event_policy(event: LeaseEvent) -> None:
        required = _EVENT_DETAIL_FIELDS[event.event_type]
        actual = set(event.details)
        rejected = sorted(actual - required)
        if rejected:
            raise LedgerPolicyError(
                f"event {event.event_type} contains disallowed field: {rejected[0]}"
            )
        missing = sorted(required - actual)
        if missing:
            raise LedgerPolicyError(
                f"event {event.event_type} is missing required field: {missing[0]}"
            )

        def require_text(field: str) -> str:
            value = event.details[field]
            if not isinstance(value, str) or not value.strip():
                raise LedgerPolicyError(
                    f"event {event.event_type} field {field} must be a non-empty string"
                )
            return value

        def require_nonnegative_int(field: str, *, nullable: bool = False) -> int | None:
            value = event.details[field]
            if nullable and value is None:
                return None
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise LedgerPolicyError(
                    f"event {event.event_type} field {field} must be a non-negative integer"
                )
            return value

        def require_positive_int(field: str, *, nullable: bool = False) -> int | None:
            value = event.details[field]
            if nullable and value is None:
                return None
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise LedgerPolicyError(
                    f"event {event.event_type} field {field} must be a positive integer"
                )
            return value

        text_fields = {
            "user_hmac",
            "identity_key_id",
            "auth_receipt_id",
            "session_ref",
            "connection_ref",
            "request_ref",
            "runtime_instance_id",
            "controller_instance_id",
            "context_ref",
            "context_ref_source",
            "capture_id",
            "evidence_ref",
            "reason",
        }
        for field in sorted(actual & text_fields):
            require_text(field)
        if "retry_of" in actual:
            retry_of = event.details["retry_of"]
            if retry_of is not None and (
                not isinstance(retry_of, str) or not retry_of.strip()
            ):
                raise LedgerPolicyError(
                    f"event {event.event_type} field retry_of must be null or a non-empty string"
                )
            if retry_of is not None and retry_of == event.request_ref:
                raise LedgerPolicyError(
                    f"event {event.event_type} field retry_of must differ from request_ref"
                )

        for field in sorted(
            actual
            & {
                "expected_slot_id",
                "expected_seq_id",
                "response_confirmed_slot_id",
                "observed_seq_id",
                "slot_id",
            }
        ):
            require_nonnegative_int(
                field, nullable=event.event_type == "audit_violation"
            )
        for field in sorted(
            actual & {"lease_generation", "slot_epoch", "payload_size_bytes"}
        ):
            require_positive_int(
                field, nullable=event.event_type == "audit_violation"
            )
        if "payload_digest" in actual:
            digest = event.details["payload_digest"]
            if (
                not isinstance(digest, str)
                or re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None
            ):
                raise LedgerPolicyError(
                    f"event {event.event_type} field payload_digest is invalid"
                )
        if event.event_type == "request_received" and event.request_ref is None:
            raise LedgerPolicyError(
                "event request_received requires a request_ref"
            )
        if event.event_type == "lease_reserved":
            if (
                event.lease_id is None
                or event.request_ref is None
                or event.runtime_instance_id is None
            ):
                raise LedgerPolicyError(
                    "event lease_reserved requires lease, request, and runtime identifiers"
                )
            if event.details["expected_seq_id"] != event.details["expected_slot_id"]:
                raise LedgerPolicyError(
                    "event lease_reserved expected_seq_id must equal expected_slot_id"
                )
        if event.event_type == "runtime_state_acquisition_completed":
            if event.details["request_ref"] != event.request_ref:
                raise LedgerPolicyError(
                    "event runtime_state_acquisition_completed field request_ref does not match the event"
                )
            if event.details["runtime_instance_id"] != event.runtime_instance_id:
                raise LedgerPolicyError(
                    "event runtime_state_acquisition_completed field runtime_instance_id does not match the event"
                )

    def _append_in_transaction(
        self, connection: sqlite3.Connection, event: LeaseEvent
    ) -> StoredEvent:
        self._validate_event_policy(event)
        previous_row = connection.execute(
            "SELECT event_digest FROM events ORDER BY event_counter DESC LIMIT 1"
        ).fetchone()
        previous_digest = (
            GENESIS_DIGEST if previous_row is None else previous_row["event_digest"]
        )
        payload_bytes = canonical_json(self._event_payload(event))
        event_digest = "sha256:" + hashlib.sha256(
            previous_digest.encode("ascii") + payload_bytes
        ).hexdigest()
        event_mac = "hmac-sha256:" + hmac.new(
            self._mac_key, event_digest.encode("ascii"), hashlib.sha256
        ).hexdigest()
        cursor = connection.execute(
            """
            INSERT INTO events (
                event_id,
                event_type,
                payload_json,
                previous_digest,
                event_digest,
                event_mac,
                wall_time_ns,
                monotonic_ns
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.event_id,
                event.event_type,
                payload_bytes.decode("utf-8"),
                previous_digest,
                event_digest,
                event_mac,
                event.wall_time_ns,
                event.monotonic_ns,
            ),
        )
        return StoredEvent(
            event_counter=int(cursor.lastrowid),
            event=event,
            previous_digest=previous_digest,
            event_digest=event_digest,
            event_mac=event_mac,
        )

    def append(self, event: LeaseEvent) -> StoredEvent:
        with self._transaction() as connection:
            return self._append_in_transaction(connection, event)

    def read_events(self) -> tuple[StoredEvent, ...]:
        with self._mutex:
            rows = self._require_connection().execute(
                """
                SELECT event_counter, payload_json, previous_digest,
                       event_digest, event_mac
                FROM events
                ORDER BY event_counter
                """
            ).fetchall()
        result: list[StoredEvent] = []
        for row in rows:
            try:
                payload = json.loads(row["payload_json"])
                event = self._event_from_payload(payload)
            except Exception as error:
                raise LedgerIntegrityError(
                    f"event {row['event_counter']} cannot be decoded"
                ) from error
            result.append(
                StoredEvent(
                    event_counter=row["event_counter"],
                    event=event,
                    previous_digest=row["previous_digest"],
                    event_digest=row["event_digest"],
                    event_mac=row["event_mac"],
                )
            )
        return tuple(result)

    def _verify_event_chain(
        self, connection: sqlite3.Connection
    ) -> tuple[StoredEvent, ...]:
        self._validate_existing_schema(connection)
        rows = connection.execute(
            """
            SELECT event_counter, payload_json, previous_digest,
                   event_digest, event_mac
            FROM events
            ORDER BY event_counter
            """
        ).fetchall()
        expected_previous = GENESIS_DIGEST
        verified: list[StoredEvent] = []
        for expected_counter, row in enumerate(rows, start=1):
            if row["event_counter"] != expected_counter:
                raise LedgerIntegrityError(
                    f"event counter discontinuity at {row['event_counter']}"
                )
            if row["previous_digest"] != expected_previous:
                raise LedgerIntegrityError(
                    f"event {expected_counter} previous digest mismatch"
                )
            payload_bytes = row["payload_json"].encode("utf-8")
            expected_digest = "sha256:" + hashlib.sha256(
                expected_previous.encode("ascii") + payload_bytes
            ).hexdigest()
            if not hmac.compare_digest(expected_digest, row["event_digest"]):
                raise LedgerIntegrityError(
                    f"event {expected_counter} digest mismatch"
                )
            expected_mac = "hmac-sha256:" + hmac.new(
                self._mac_key, expected_digest.encode("ascii"), hashlib.sha256
            ).hexdigest()
            if not hmac.compare_digest(expected_mac, row["event_mac"]):
                raise LedgerIntegrityError(
                    f"event {expected_counter} MAC mismatch"
                )
            try:
                event = self._event_from_payload(json.loads(row["payload_json"]))
            except Exception as error:
                raise LedgerIntegrityError(
                    f"event {expected_counter} payload is invalid"
                ) from error
            try:
                self._validate_event_policy(event)
            except LedgerPolicyError as error:
                raise LedgerIntegrityError(
                    f"event {expected_counter} violates event detail policy"
                ) from error
            verified.append(
                StoredEvent(
                    event_counter=expected_counter,
                    event=event,
                    previous_digest=expected_previous,
                    event_digest=expected_digest,
                    event_mac=row["event_mac"],
                )
            )
            expected_previous = expected_digest
        return tuple(verified)

    def verify_chain(self) -> None:
        with self._mutex:
            connection = self._require_connection()
            verified = self._verify_event_chain(connection)
            self._verify_materialized_state(connection, verified)
            self.verify_anchor()

    @classmethod
    def _verify_materialized_state(
        cls,
        connection: sqlite3.Connection,
        events: tuple[StoredEvent, ...],
    ) -> None:
        requests: dict[str, RequestIdentity] = {}
        replayed: dict[str, _ReplayedLease] = {}
        counters: dict[tuple[str, str, int], tuple[int, int]] = {}

        for stored in events:
            event = stored.event
            details = event.details
            if event.event_type == "request_received":
                assert event.request_ref is not None
                request = RequestIdentity(
                    user_hmac=str(details["user_hmac"]),
                    identity_key_id=str(details["identity_key_id"]),
                    auth_receipt_id=str(details["auth_receipt_id"]),
                    session_ref=str(details["session_ref"]),
                    request_ref=event.request_ref,
                    connection_ref=str(details["connection_ref"]),
                    retry_of=details["retry_of"],
                )
                previous_request = requests.get(event.request_ref)
                if previous_request is not None and previous_request != request:
                    raise LedgerIntegrityError(
                        "request identity changed within the event chain"
                    )
                requests[event.request_ref] = request
                continue

            if event.event_type == "lease_reserved":
                if (
                    event.lease_id is None
                    or event.request_ref is None
                    or event.runtime_instance_id is None
                ):
                    raise LedgerIntegrityError(
                        "lease_reserved event has an incomplete binding"
                    )
                if event.lease_id in replayed:
                    raise LedgerIntegrityError(
                        "Lease was reserved more than once in the event chain"
                    )
                request = RequestIdentity(
                    user_hmac=str(details["user_hmac"]),
                    identity_key_id=str(details["identity_key_id"]),
                    auth_receipt_id=str(details["auth_receipt_id"]),
                    session_ref=str(details["session_ref"]),
                    request_ref=event.request_ref,
                    connection_ref=str(details["connection_ref"]),
                    retry_of=details["retry_of"],
                )
                observed_request = requests.get(event.request_ref)
                if observed_request is not None and observed_request != request:
                    raise LedgerIntegrityError(
                        "lease_reserved identity differs from request_received"
                    )
                lease = LeaseRecord(
                    lease_id=event.lease_id,
                    request=request,
                    runtime_instance_id=event.runtime_instance_id,
                    context_ref=str(details["context_ref"]),
                    context_ref_source=str(details["context_ref_source"]),
                    expected_slot_id=int(details["expected_slot_id"]),
                    expected_seq_id=int(details["expected_seq_id"]),
                    lease_generation=int(details["lease_generation"]),
                    response_confirmed_slot_id=None,
                    observed_seq_id=None,
                    slot_epoch=None,
                    state=LeaseState.RESERVED,
                    holds_sequence=True,
                    backend_released=False,
                )
                key = (
                    lease.runtime_instance_id,
                    lease.context_ref,
                    lease.expected_slot_id,
                )
                previous_generation, previous_epoch = counters.get(key, (0, 0))
                if lease.lease_generation != previous_generation + 1:
                    raise LedgerIntegrityError(
                        "lease generation is not reproducible from the event chain"
                    )
                counters[key] = (lease.lease_generation, previous_epoch)
                replayed[event.lease_id] = _ReplayedLease(
                    lease=lease, updated_event_counter=stored.event_counter
                )
                continue

            materializing_types = {
                "dispatch_authorized",
                "upstream_request_sent",
                "slot_assignment_confirmed",
                "terminal_response_received",
                "slot_idle_confirmed",
                "runtime_state_capture_started",
                "runtime_state_acquisition_requested",
                "runtime_state_acquisition_copied",
                "runtime_state_payload_committed",
                "runtime_state_post_copy_failed",
                "capture_skipped",
                "lease_released",
                "lease_unknown",
                "slot_quarantined",
            }
            lease_lifecycle_types = materializing_types | {
                "runtime_state_acquisition_completed"
            }
            if event.event_type not in lease_lifecycle_types:
                continue
            if event.lease_id is None or event.lease_id not in replayed:
                raise LedgerIntegrityError(
                    f"event {event.event_type} references an unknown Lease"
                )
            replay_state = replayed[event.lease_id]
            current = replay_state.lease
            if event.request_ref != current.request.request_ref:
                raise LedgerIntegrityError(
                    "Lease event binding request_ref differs from the reservation"
                )
            if event.runtime_instance_id != current.runtime_instance_id:
                raise LedgerIntegrityError(
                    "Lease event binding runtime_instance_id differs from the reservation"
                )

            changes: dict[str, object] = {}
            dispatch_authorized = replay_state.dispatch_authorized
            acquisition_requested = replay_state.acquisition_requested
            acquisition_copied = replay_state.acquisition_copied
            acquisition_completion = replay_state.acquisition_completion
            materializes = event.event_type in materializing_types

            def require_state(*expected: LeaseState) -> None:
                if current.state not in expected:
                    expected_names = ", ".join(item.value for item in expected)
                    raise LedgerIntegrityError(
                        f"event {event.event_type} requires Lease state "
                        f"{expected_names}; current state is {current.state.value}"
                    )

            if event.event_type == "dispatch_authorized":
                require_state(LeaseState.RESERVED)
                if dispatch_authorized:
                    raise LedgerIntegrityError(
                        "event dispatch_authorized may occur only once per Lease"
                    )
                dispatch_authorized = True
                changes["state"] = LeaseState.RESERVED
            elif event.event_type == "upstream_request_sent":
                require_state(LeaseState.RESERVED)
                if not dispatch_authorized:
                    raise LedgerIntegrityError(
                        "event upstream_request_sent requires prior dispatch_authorized"
                    )
                changes["state"] = LeaseState.SUBMITTED
            elif event.event_type == "slot_assignment_confirmed":
                require_state(LeaseState.SUBMITTED)
                confirmed_slot = int(details["response_confirmed_slot_id"])
                slot_epoch = int(details["slot_epoch"])
                if confirmed_slot != current.expected_slot_id:
                    raise LedgerIntegrityError(
                        "confirmed slot differs from the reserved slot"
                    )
                key = (
                    current.runtime_instance_id,
                    current.context_ref,
                    current.expected_slot_id,
                )
                generation, previous_epoch = counters[key]
                if slot_epoch != previous_epoch + 1:
                    raise LedgerIntegrityError(
                        "slot epoch is not reproducible from the event chain"
                    )
                counters[key] = (generation, slot_epoch)
                changes.update(
                    state=LeaseState.ACTIVE,
                    response_confirmed_slot_id=confirmed_slot,
                    slot_epoch=slot_epoch,
                )
            elif event.event_type == "terminal_response_received":
                require_state(LeaseState.ACTIVE)
                changes["state"] = LeaseState.QUIESCING
            elif event.event_type == "slot_idle_confirmed":
                require_state(LeaseState.QUIESCING)
                if current.backend_released:
                    raise LedgerIntegrityError(
                        "event slot_idle_confirmed may occur only once per Lease"
                    )
                changes.update(
                    state=LeaseState.QUIESCING, backend_released=True
                )
            elif event.event_type == "runtime_state_capture_started":
                require_state(LeaseState.QUIESCING)
                if not current.backend_released:
                    raise LedgerIntegrityError(
                        "event runtime_state_capture_started requires an idle backend"
                    )
                if current.capture_id is not None:
                    raise LedgerIntegrityError(
                        "event runtime_state_capture_started may occur only once per Lease"
                    )
                changes.update(
                    state=LeaseState.CHECKPOINTING,
                    capture_id=str(details["capture_id"]),
                )
            elif event.event_type == "runtime_state_acquisition_requested":
                require_state(LeaseState.CHECKPOINTING)
                if details["capture_id"] != current.capture_id:
                    raise LedgerIntegrityError(
                        "acquisition request capture_id differs from the Lease"
                    )
                if acquisition_requested:
                    raise LedgerIntegrityError(
                        "exactly one acquisition request is allowed per capture"
                    )
                if (
                    current.response_confirmed_slot_id
                    != current.expected_slot_id
                    or current.slot_epoch is None
                    or details["observed_seq_id"] != current.expected_seq_id
                ):
                    raise LedgerIntegrityError(
                        "acquisition request does not match the confirmed Lease binding"
                    )
                acquisition_requested = True
                changes.update(
                    state=LeaseState.CHECKPOINTING,
                    observed_seq_id=int(details["observed_seq_id"]),
                )
            elif event.event_type == "runtime_state_acquisition_copied":
                require_state(LeaseState.CHECKPOINTING)
                if not acquisition_requested:
                    raise LedgerIntegrityError(
                        "verified copy requires an acquisition request"
                    )
                if acquisition_copied:
                    raise LedgerIntegrityError(
                        "verified copy may occur only once per capture"
                    )
                if details["capture_id"] != current.capture_id:
                    raise LedgerIntegrityError(
                        "verified copy capture_id differs from the Lease"
                    )
                if not current.holds_sequence:
                    raise LedgerIntegrityError(
                        "verified copy requires physical slot ownership"
                    )
                acquisition_copied = True
                changes.update(
                    state=LeaseState.CHECKPOINTING,
                    holds_sequence=False,
                )
            elif event.event_type == "runtime_state_acquisition_completed":
                require_state(LeaseState.CHECKPOINTING)
                if not acquisition_requested:
                    raise LedgerIntegrityError(
                        "acquisition completion requires exactly one acquisition request"
                    )
                if acquisition_completion is not None:
                    raise LedgerIntegrityError(
                        "exactly one acquisition completion is allowed per capture"
                    )
                if (
                    event.request_ref != current.request.request_ref
                    or event.runtime_instance_id != current.runtime_instance_id
                    or details["capture_id"] != current.capture_id
                    or details["request_ref"] != current.request.request_ref
                    or details["runtime_instance_id"]
                    != current.runtime_instance_id
                    or details["slot_id"] != current.expected_slot_id
                    or details["slot_id"]
                    != current.response_confirmed_slot_id
                    or details["slot_epoch"] != current.slot_epoch
                    or details["observed_seq_id"] != current.expected_seq_id
                    or details["observed_seq_id"] != current.observed_seq_id
                ):
                    raise LedgerIntegrityError(
                        "acquisition completion does not match the current Lease binding"
                    )
                acquisition_completion = _AcquisitionCompletion(
                    capture_id=str(details["capture_id"]),
                    payload_digest=str(details["payload_digest"]),
                    payload_size_bytes=int(details["payload_size_bytes"]),
                    evidence_ref=str(details["evidence_ref"]),
                )
                materializes = False
            elif event.event_type == "runtime_state_payload_committed":
                require_state(LeaseState.CHECKPOINTING)
                if not acquisition_requested or acquisition_completion is None:
                    raise LedgerIntegrityError(
                        "payload commit requires exactly one acquisition completion"
                    )
                if (
                    details["capture_id"] != current.capture_id
                    or details["capture_id"] != acquisition_completion.capture_id
                    or details["payload_digest"]
                    != acquisition_completion.payload_digest
                    or details["payload_size_bytes"]
                    != acquisition_completion.payload_size_bytes
                    or details["context_ref"] != current.context_ref
                    or details["expected_slot_id"] != current.expected_slot_id
                    or details["response_confirmed_slot_id"]
                    != current.response_confirmed_slot_id
                    or details["slot_epoch"] != current.slot_epoch
                ):
                    raise LedgerIntegrityError(
                        "payload commit does not match acquisition completion and Lease binding"
                    )
                changes["state"] = LeaseState.CHECKPOINTED
            elif event.event_type == "runtime_state_post_copy_failed":
                require_state(LeaseState.CHECKPOINTING, LeaseState.CHECKPOINTED)
                if not acquisition_copied or current.holds_sequence:
                    raise LedgerIntegrityError(
                        "post-copy failure requires a non-holding verified copy"
                    )
                changes.update(
                    state=LeaseState.QUARANTINED,
                    holds_sequence=False,
                )
            elif event.event_type == "capture_skipped":
                require_state(LeaseState.QUIESCING)
                if not current.backend_released:
                    raise LedgerIntegrityError(
                        "event capture_skipped requires an idle backend"
                    )
                changes.update(
                    state=LeaseState.CAPTURE_SKIPPED,
                    capture_skipped_reason=str(details["reason"]),
                )
            elif event.event_type == "lease_released":
                require_state(LeaseState.CHECKPOINTED, LeaseState.CAPTURE_SKIPPED)
                if not current.backend_released:
                    raise LedgerIntegrityError(
                        "event lease_released requires an idle backend"
                    )
                changes.update(state=LeaseState.RELEASED, holds_sequence=False)
            elif event.event_type == "lease_unknown":
                if current.state in {
                    LeaseState.RELEASED,
                    LeaseState.UNKNOWN,
                    LeaseState.QUARANTINED,
                }:
                    raise LedgerIntegrityError(
                        f"event lease_unknown is illegal from {current.state.value}"
                    )
                changes["state"] = LeaseState.UNKNOWN
            elif event.event_type == "slot_quarantined":
                if current.state in {
                    LeaseState.RELEASED,
                    LeaseState.UNKNOWN,
                    LeaseState.QUARANTINED,
                }:
                    raise LedgerIntegrityError(
                        f"event slot_quarantined is illegal from {current.state.value}"
                    )
                changes.update(state=LeaseState.QUARANTINED, holds_sequence=True)
            updated = replace(current, **changes) if materializes else current
            replayed[event.lease_id] = _ReplayedLease(
                lease=updated,
                updated_event_counter=(
                    stored.event_counter
                    if materializes
                    else replay_state.updated_event_counter
                ),
                dispatch_authorized=dispatch_authorized,
                acquisition_requested=acquisition_requested,
                acquisition_copied=acquisition_copied,
                acquisition_completion=acquisition_completion,
            )

        for replay_state in replayed.values():
            request_ref = replay_state.lease.request.request_ref
            observed_request = requests.get(request_ref)
            if (
                observed_request is not None
                and observed_request != replay_state.lease.request
            ):
                raise LedgerIntegrityError(
                    "request identity differs from the reserved Lease"
                )

        lease_rows = connection.execute(
            "SELECT * FROM leases ORDER BY lease_id"
        ).fetchall()
        if {str(row["lease_id"]) for row in lease_rows} != set(replayed):
            raise LedgerIntegrityError(
                "materialized Lease set does not match the event chain"
            )
        for row in lease_rows:
            lease_id = str(row["lease_id"])
            expected = replayed[lease_id]
            try:
                actual = cls._lease_from_row(row)
            except Exception as error:
                raise LedgerIntegrityError(
                    f"materialized Lease {lease_id} is invalid"
                ) from error
            if (
                actual != expected.lease
                or row["updated_event_counter"] != expected.updated_event_counter
            ):
                raise LedgerIntegrityError(
                    f"materialized Lease {lease_id} does not match the event chain"
                )

        counter_rows = connection.execute(
            "SELECT * FROM sequence_counters "
            "ORDER BY runtime_instance_id, context_ref, expected_slot_id"
        ).fetchall()
        actual_counters = {
            (
                str(row["runtime_instance_id"]),
                str(row["context_ref"]),
                int(row["expected_slot_id"]),
            ): (
                int(row["last_lease_generation"]),
                int(row["last_slot_epoch"]),
            )
            for row in counter_rows
        }
        if actual_counters != counters:
            raise LedgerIntegrityError(
                "materialized sequence counter state does not match the event chain"
            )

    @staticmethod
    def _new_event(
        event_type: str,
        *,
        lease: LeaseRecord | None = None,
        runtime_instance_id: str | None = None,
        details: Mapping[str, JsonValue] | None = None,
    ) -> LeaseEvent:
        return LeaseEvent(
            schema_version=SCHEMA_VERSION,
            event_id=f"evt_{uuid.uuid4().hex}",
            event_type=event_type,
            lease_id=None if lease is None else lease.lease_id,
            request_ref=None if lease is None else lease.request.request_ref,
            runtime_instance_id=(
                runtime_instance_id
                if lease is None
                else lease.runtime_instance_id
            ),
            wall_time_ns=time.time_ns(),
            monotonic_ns=time.monotonic_ns(),
            details={} if details is None else details,
        )

    @staticmethod
    def _next_lease_generation(
        connection: sqlite3.Connection,
        *,
        runtime_instance_id: str,
        context_ref: str,
        expected_slot_id: int,
    ) -> tuple[int, sqlite3.Row | None]:
        counter = connection.execute(
            """
            SELECT last_lease_generation
            FROM sequence_counters
            WHERE runtime_instance_id = ?
              AND context_ref = ?
              AND expected_slot_id = ?
            """,
            (runtime_instance_id, context_ref, expected_slot_id),
        ).fetchone()
        generation = 1 if counter is None else counter["last_lease_generation"] + 1
        return generation, counter

    def _create_lease_in_transaction(
        self, connection: sqlite3.Connection, lease: LeaseRecord
    ) -> None:
        if lease.state is not LeaseState.RESERVED:
            raise LeaseConflictError("new Lease must be RESERVED")
        expected_generation, counter = self._next_lease_generation(
            connection,
            runtime_instance_id=lease.runtime_instance_id,
            context_ref=lease.context_ref,
            expected_slot_id=lease.expected_slot_id,
        )
        if lease.lease_generation != expected_generation:
            raise LeaseConflictError(
                "lease_generation must be the next durable generation"
            )

        event = self._new_event(
            "lease_reserved",
            lease=lease,
            details={
                "user_hmac": lease.request.user_hmac,
                "identity_key_id": lease.request.identity_key_id,
                "auth_receipt_id": lease.request.auth_receipt_id,
                "session_ref": lease.request.session_ref,
                "connection_ref": lease.request.connection_ref,
                "retry_of": lease.request.retry_of,
                "context_ref": lease.context_ref,
                "context_ref_source": lease.context_ref_source,
                "expected_slot_id": lease.expected_slot_id,
                "expected_seq_id": lease.expected_seq_id,
                "lease_generation": lease.lease_generation,
            },
        )
        stored = self._append_in_transaction(connection, event)
        try:
            connection.execute(
                """
                INSERT INTO leases (
                    lease_id, user_hmac, identity_key_id,
                    auth_receipt_id, session_ref, request_ref,
                    connection_ref, retry_of, runtime_instance_id,
                    context_ref, context_ref_source, expected_slot_id,
                    expected_seq_id, lease_generation,
                    response_confirmed_slot_id, observed_seq_id,
                    slot_epoch, state, holds_sequence, backend_released,
                    capture_id, capture_skipped_reason,
                    updated_event_counter
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                self._lease_values(lease, stored.event_counter),
            )
            if counter is None:
                connection.execute(
                    """
                    INSERT INTO sequence_counters (
                        runtime_instance_id, context_ref, expected_slot_id,
                        last_lease_generation, last_slot_epoch
                    ) VALUES (?, ?, ?, ?, 0)
                    """,
                    (
                        lease.runtime_instance_id,
                        lease.context_ref,
                        lease.expected_slot_id,
                        lease.lease_generation,
                    ),
                )
            else:
                connection.execute(
                    """
                    UPDATE sequence_counters
                    SET last_lease_generation = ?
                    WHERE runtime_instance_id = ?
                      AND context_ref = ?
                      AND expected_slot_id = ?
                    """,
                    (
                        lease.lease_generation,
                        lease.runtime_instance_id,
                        lease.context_ref,
                        lease.expected_slot_id,
                    ),
                )
        except sqlite3.IntegrityError as error:
            raise LeaseConflictError(
                f"Lease conflicts with durable binding: {lease.lease_id}"
            ) from error

    def create_lease(self, lease: LeaseRecord) -> LeaseRecord:
        with self._transaction() as connection:
            self._create_lease_in_transaction(connection, lease)
        return lease

    def reserve_lease(
        self,
        *,
        lease_id: str,
        request: RequestIdentity,
        runtime_instance_id: str,
        context_ref: str,
        context_ref_source: str,
        expected_slot_id: int,
        expected_seq_id: int,
        slot_count: int,
    ) -> LeaseRecord:
        with self._transaction() as connection:
            lease_generation, _ = self._next_lease_generation(
                connection,
                runtime_instance_id=runtime_instance_id,
                context_ref=context_ref,
                expected_slot_id=expected_slot_id,
            )
            lease = LeaseRecord.reserved(
                lease_id=lease_id,
                request=request,
                runtime_instance_id=runtime_instance_id,
                context_ref=context_ref,
                context_ref_source=context_ref_source,
                expected_slot_id=expected_slot_id,
                expected_seq_id=expected_seq_id,
                lease_generation=lease_generation,
                slot_count=slot_count,
            )
            self._create_lease_in_transaction(connection, lease)
        return lease

    @staticmethod
    def _lease_values(lease: LeaseRecord, event_counter: int) -> tuple[object, ...]:
        request = lease.request
        return (
            lease.lease_id,
            request.user_hmac,
            request.identity_key_id,
            request.auth_receipt_id,
            request.session_ref,
            request.request_ref,
            request.connection_ref,
            request.retry_of,
            lease.runtime_instance_id,
            lease.context_ref,
            lease.context_ref_source,
            lease.expected_slot_id,
            lease.expected_seq_id,
            lease.lease_generation,
            lease.response_confirmed_slot_id,
            lease.observed_seq_id,
            lease.slot_epoch,
            lease.state.value,
            int(lease.holds_sequence),
            int(lease.backend_released),
            lease.capture_id,
            lease.capture_skipped_reason,
            event_counter,
        )

    @staticmethod
    def _lease_from_row(row: sqlite3.Row) -> LeaseRecord:
        request = RequestIdentity(
            user_hmac=row["user_hmac"],
            identity_key_id=row["identity_key_id"],
            auth_receipt_id=row["auth_receipt_id"],
            session_ref=row["session_ref"],
            request_ref=row["request_ref"],
            connection_ref=row["connection_ref"],
            retry_of=row["retry_of"],
        )
        return LeaseRecord(
            lease_id=row["lease_id"],
            request=request,
            runtime_instance_id=row["runtime_instance_id"],
            context_ref=row["context_ref"],
            context_ref_source=row["context_ref_source"],
            expected_slot_id=row["expected_slot_id"],
            expected_seq_id=row["expected_seq_id"],
            lease_generation=row["lease_generation"],
            response_confirmed_slot_id=row["response_confirmed_slot_id"],
            observed_seq_id=row["observed_seq_id"],
            slot_epoch=row["slot_epoch"],
            state=LeaseState(row["state"]),
            holds_sequence=bool(row["holds_sequence"]),
            backend_released=bool(row["backend_released"]),
            capture_id=row["capture_id"],
            capture_skipped_reason=row["capture_skipped_reason"],
        )

    def get_lease(self, lease_id: str) -> LeaseRecord:
        self.verify_chain()
        with self._mutex:
            row = self._require_connection().execute(
                "SELECT * FROM leases WHERE lease_id = ?", (lease_id,)
            ).fetchone()
        if row is None:
            raise KeyError(lease_id)
        return self._lease_from_row(row)

    def confirm_slot_assignment(
        self,
        lease_id: str,
        *,
        response_confirmed_slot_id: int,
        evidence_ref: str,
    ) -> LeaseRecord:
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM leases WHERE lease_id = ?", (lease_id,)
            ).fetchone()
            if row is None:
                raise KeyError(lease_id)
            current = self._lease_from_row(row)
            if response_confirmed_slot_id != current.expected_slot_id:
                raise LeaseConflictError("response slot does not match expected slot")
            if current.slot_epoch is not None:
                raise LeaseConflictError("slot_epoch is already assigned")
            counter = connection.execute(
                """
                SELECT last_slot_epoch
                FROM sequence_counters
                WHERE runtime_instance_id = ?
                  AND context_ref = ?
                  AND expected_slot_id = ?
                """,
                (
                    current.runtime_instance_id,
                    current.context_ref,
                    current.expected_slot_id,
                ),
            ).fetchone()
            if counter is None:
                raise LedgerIntegrityError("sequence counter is missing")
            slot_epoch = counter["last_slot_epoch"] + 1
            updated = replace(
                current,
                response_confirmed_slot_id=response_confirmed_slot_id,
                slot_epoch=slot_epoch,
                state=LeaseState.ACTIVE,
            )
            event = self._new_event(
                "slot_assignment_confirmed",
                lease=current,
                details={
                    "response_confirmed_slot_id": response_confirmed_slot_id,
                    "slot_epoch": slot_epoch,
                    "evidence_ref": evidence_ref,
                },
            )
            stored = self._append_in_transaction(connection, event)
            connection.execute(
                """
                UPDATE sequence_counters
                SET last_slot_epoch = ?
                WHERE runtime_instance_id = ?
                  AND context_ref = ?
                  AND expected_slot_id = ?
                """,
                (
                    slot_epoch,
                    current.runtime_instance_id,
                    current.context_ref,
                    current.expected_slot_id,
                ),
            )
            connection.execute(
                """
                UPDATE leases SET
                    response_confirmed_slot_id = ?,
                    slot_epoch = ?,
                    state = ?,
                    updated_event_counter = ?
                WHERE lease_id = ?
                """,
                (
                    updated.response_confirmed_slot_id,
                    updated.slot_epoch,
                    updated.state.value,
                    stored.event_counter,
                    updated.lease_id,
                ),
            )
        return updated

    def transition_lease(
        self,
        lease_id: str,
        *,
        state: LeaseState,
        event: LeaseEvent,
        holds_sequence: bool | None = None,
        response_confirmed_slot_id: int | None | _Unset = UNSET,
        observed_seq_id: int | None | _Unset = UNSET,
        slot_epoch: int | None | _Unset = UNSET,
        backend_released: bool | None = None,
        capture_id: str | None | _Unset = UNSET,
        capture_skipped_reason: str | None | _Unset = UNSET,
    ) -> LeaseRecord:
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM leases WHERE lease_id = ?", (lease_id,)
            ).fetchone()
            if row is None:
                raise KeyError(lease_id)
            current = self._lease_from_row(row)
            if event.lease_id != current.lease_id:
                raise LeaseConflictError("event lease_id does not match Lease")
            if event.request_ref not in (None, current.request.request_ref):
                raise LeaseConflictError("event request_ref does not match Lease")
            if event.runtime_instance_id not in (
                None,
                current.runtime_instance_id,
            ):
                raise LeaseConflictError(
                    "event runtime_instance_id does not match Lease"
                )

            changes: dict[str, object] = {"state": state}
            if holds_sequence is not None:
                changes["holds_sequence"] = holds_sequence
            if not isinstance(response_confirmed_slot_id, _Unset):
                changes["response_confirmed_slot_id"] = response_confirmed_slot_id
            if not isinstance(observed_seq_id, _Unset):
                changes["observed_seq_id"] = observed_seq_id
            if not isinstance(slot_epoch, _Unset):
                changes["slot_epoch"] = slot_epoch
            if backend_released is not None:
                changes["backend_released"] = backend_released
            if not isinstance(capture_id, _Unset):
                changes["capture_id"] = capture_id
            if not isinstance(capture_skipped_reason, _Unset):
                changes["capture_skipped_reason"] = capture_skipped_reason
            updated = replace(current, **changes)
            stored = self._append_in_transaction(connection, event)
            try:
                connection.execute(
                    """
                    UPDATE leases SET
                        response_confirmed_slot_id = ?,
                        observed_seq_id = ?,
                        slot_epoch = ?,
                        state = ?,
                        holds_sequence = ?,
                        backend_released = ?,
                        capture_id = ?,
                        capture_skipped_reason = ?,
                        updated_event_counter = ?
                    WHERE lease_id = ?
                    """,
                    (
                        updated.response_confirmed_slot_id,
                        updated.observed_seq_id,
                        updated.slot_epoch,
                        updated.state.value,
                        int(updated.holds_sequence),
                        int(updated.backend_released),
                        updated.capture_id,
                        updated.capture_skipped_reason,
                        stored.event_counter,
                        updated.lease_id,
                    ),
                )
            except sqlite3.IntegrityError as error:
                raise LeaseConflictError(
                    f"Lease transition conflicts with durable binding: {lease_id}"
                ) from error
        return updated

    def recover_controller_state(
        self, *, controller_instance_id: str, runtime_instance_id: str
    ) -> tuple[str, ...]:
        with self._transaction() as connection:
            rows = connection.execute(
                """
                SELECT * FROM leases
                WHERE runtime_instance_id = ? AND holds_sequence = 1
                ORDER BY lease_id
                """,
                (runtime_instance_id,),
            ).fetchall()
            recovered: list[str] = []
            for row in rows:
                current = self._lease_from_row(row)
                updated = replace(current, state=LeaseState.UNKNOWN)
                event = self._new_event(
                    "lease_unknown",
                    lease=current,
                    details={
                        "reason": "controller_restart_requires_reconciliation",
                        "controller_instance_id": controller_instance_id,
                    },
                )
                stored = self._append_in_transaction(connection, event)
                connection.execute(
                    """
                    UPDATE leases
                    SET state = ?, updated_event_counter = ?
                    WHERE lease_id = ?
                    """,
                    (
                        updated.state.value,
                        stored.event_counter,
                        updated.lease_id,
                    ),
                )
                recovered.append(updated.lease_id)

            post_copy_rows = connection.execute(
                """
                SELECT * FROM leases
                WHERE runtime_instance_id = ?
                  AND holds_sequence = 0
                  AND state IN (?, ?)
                ORDER BY lease_id
                """,
                (
                    runtime_instance_id,
                    LeaseState.CHECKPOINTING.value,
                    LeaseState.CHECKPOINTED.value,
                ),
            ).fetchall()
            for row in post_copy_rows:
                current = self._lease_from_row(row)
                event = self._new_event(
                    "runtime_state_post_copy_failed",
                    lease=current,
                    details={
                        "reason": "controller_restart_after_verified_copy",
                        "evidence_ref": (
                            f"controller-recovery:{controller_instance_id}"
                        ),
                    },
                )
                stored = self._append_in_transaction(connection, event)
                connection.execute(
                    """
                    UPDATE leases
                    SET state = ?, holds_sequence = 0, updated_event_counter = ?
                    WHERE lease_id = ?
                    """,
                    (
                        LeaseState.QUARANTINED.value,
                        stored.event_counter,
                        current.lease_id,
                    ),
                )

            recovery_event = self._new_event(
                "controller_recovered",
                runtime_instance_id=runtime_instance_id,
                details={
                    "controller_instance_id": controller_instance_id,
                    "reason": "held_leases_marked_unknown",
                },
            )
            self._append_in_transaction(connection, recovery_event)
        return tuple(recovered)

    def handle_runtime_restart(
        self,
        *,
        controller_instance_id: str,
        old_runtime_instance_id: str,
        new_runtime_instance_id: str,
        evidence_ref: str,
    ) -> tuple[str, ...]:
        recovered = self.recover_controller_state(
            controller_instance_id=controller_instance_id,
            runtime_instance_id=old_runtime_instance_id,
        )
        self.append(
            self._new_event(
                "runtime_started",
                runtime_instance_id=new_runtime_instance_id,
                details={
                    "controller_instance_id": controller_instance_id,
                    "evidence_ref": evidence_ref,
                    "reason": "verified_backend_process_restart",
                },
            )
        )
        return recovered

    def durability_profile(self) -> dict[str, object]:
        with self._mutex:
            connection = self._require_connection()
            journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
            synchronous = connection.execute("PRAGMA synchronous").fetchone()[0]
        return {
            "journal_mode": str(journal_mode).lower(),
            "synchronous": int(synchronous),
        }

    def close(self) -> None:
        with self._mutex:
            if self._connection is not None:
                self._connection.close()
                self._connection = None
            if self._writer_lock is not None:
                self._writer_lock.close()
                self._writer_lock = None
            self._mac_key = b""

    def __enter__(self) -> "BindingLedger":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
