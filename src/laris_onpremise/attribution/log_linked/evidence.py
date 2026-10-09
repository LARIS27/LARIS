from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile

from .contracts import (
    canonical_document_bytes,
    parse_application_event,
    parse_attribution_result,
    parse_evidence_package_manifest,
    parse_state_unit_manifest,
    parse_writer_capture_receipt,
)
from .correlation import AttributionResult
from .llama_server_log import ApplicationEvent


class LogLinkedEvidenceError(RuntimeError):
    """Raised when a Log-linked evidence package is unsafe or inconsistent."""


_FULL_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_PROVENANCE_FIELDS = frozenset(
    {
        "schema_version",
        "adapter_id",
        "upstream_commit",
        "server_executable_sha256",
        "logging_arguments",
        "log_started_at_utc_ns",
        "log_ended_at_utc_ns",
        "parser_profile_id",
        "clock_sources",
    }
)
_FORBIDDEN_LOG_ARGUMENTS = frozenset(
    {"--verbose", "--log-verbose", "--verbose-prompt"}
)


@dataclass(frozen=True)
class EvidenceInputs:
    adapter_id: str
    runtime_instance_id: str
    created_at_utc_ns: int
    raw_application_log: Path
    log_provenance: Mapping[str, object]
    observer_terminal_seal: Path
    writer_capture_receipt: Path
    state_units_root: Path
    normalized_application_events: tuple[ApplicationEvent, ...]
    attribution_results: tuple[AttributionResult, ...]
    endpoint_derivation_receipt: Mapping[str, object]
    evaluation_metrics: Mapping[str, object] | None


@dataclass(frozen=True)
class LogLinkedEvidencePackage:
    root: Path
    manifest_path: Path
    manifest: Mapping[str, object]

    @classmethod
    def create(
        cls, root: Path | str, inputs: EvidenceInputs
    ) -> LogLinkedEvidencePackage:
        destination = Path(root)
        if destination.exists() or _is_link_like(destination):
            raise LogLinkedEvidenceError("evidence package already exists")
        if not isinstance(inputs, EvidenceInputs):
            raise LogLinkedEvidenceError("EvidenceInputs are required")
        parent = destination.parent
        _ensure_private_directory(parent)

        raw_log = _read_regular_file(inputs.raw_application_log)
        if not raw_log:
            raise LogLinkedEvidenceError("application log must not be empty")
        terminal_seal = _read_regular_file(inputs.observer_terminal_seal)
        if not terminal_seal:
            raise LogLinkedEvidenceError("observer terminal seal must not be empty")
        writer_receipt = _read_regular_file(inputs.writer_capture_receipt)
        try:
            parse_writer_capture_receipt(writer_receipt)
        except ValueError as error:
            raise LogLinkedEvidenceError(
                f"writer capture receipt is invalid: {error}"
            ) from error
        provenance = _validate_log_provenance(inputs)
        endpoint_receipt = _canonical_mapping(
            "endpoint derivation receipt", inputs.endpoint_derivation_receipt
        )
        metrics = (
            None
            if inputs.evaluation_metrics is None
            else _canonical_mapping("evaluation metrics", inputs.evaluation_metrics)
        )
        normalized = _serialize_application_events(
            inputs.normalized_application_events,
            adapter_id=inputs.adapter_id,
        )
        attributions = _serialize_attribution_results(
            inputs.attribution_results,
            adapter_id=inputs.adapter_id,
            runtime_instance_id=inputs.runtime_instance_id,
        )
        raw_endpoints = {
            str(event.endpoint_value)
            for event in inputs.normalized_application_events
            if event.endpoint_value is not None
        }
        _reject_raw_endpoints(endpoint_receipt, raw_endpoints, "endpoint receipt")
        _reject_raw_endpoints(attributions, raw_endpoints, "attribution results")
        _reject_raw_endpoints(writer_receipt, raw_endpoints, "writer capture receipt")

        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name}.staging-", dir=str(parent)
            )
        )
        _make_private(staging, directory=True)
        try:
            for relative in ("application", "observer", "derived"):
                _ensure_private_directory(staging / relative)
            _write_new(staging / "application" / "raw_llama_server.log", raw_log)
            _write_new(
                staging / "application" / "log_provenance.json", provenance
            )
            _write_new(
                staging / "observer" / "runtime_state_terminal_seal.json",
                terminal_seal,
            )
            _write_new(
                staging / "observer" / "writer_capture_receipt.json",
                writer_receipt,
            )
            _copy_tree_checked(
                inputs.state_units_root, staging / "observer" / "state-units"
            )
            _write_new(
                staging / "derived" / "normalized_application_events.jsonl",
                normalized,
            )
            _write_new(
                staging / "derived" / "attribution_results.jsonl", attributions
            )
            _write_new(
                staging / "derived" / "endpoint_derivation_receipt.json",
                endpoint_receipt,
            )
            if metrics is not None:
                _ensure_private_directory(staging / "evaluation")
                _write_new(staging / "evaluation" / "metrics.json", metrics)

            inventory = _inventory(staging)
            state_manifest_digests = sorted(
                item["sha256"]
                for item in inventory
                if str(item["path"]).endswith("/state_unit_manifest.json")
            )
            if not state_manifest_digests:
                raise LogLinkedEvidenceError(
                    "state-unit inventory contains no manifests"
                )
            manifest_without_id: dict[str, object] = {
                "schema_version": "1.0",
                "adapter_id": inputs.adapter_id,
                "runtime_instance_id": inputs.runtime_instance_id,
                "created_at_utc_ns": inputs.created_at_utc_ns,
                "application_log_file": "application/raw_llama_server.log",
                "application_log_size_bytes": len(raw_log),
                "application_log_digest": _digest(raw_log),
                "log_provenance_file": "application/log_provenance.json",
                "log_provenance_digest": _digest(provenance),
                "observer_terminal_seal_file": (
                    "observer/runtime_state_terminal_seal.json"
                ),
                "observer_terminal_seal_digest": _digest(terminal_seal),
                "writer_capture_receipt_file": (
                    "observer/writer_capture_receipt.json"
                ),
                "writer_capture_receipt_digest": _digest(writer_receipt),
                "state_units_root": "observer/state-units",
                "state_unit_manifest_digests": state_manifest_digests,
                "file_inventory": inventory,
                "normalized_application_events_file": (
                    "derived/normalized_application_events.jsonl"
                ),
                "normalized_application_events_digest": _digest(normalized),
                "attribution_results_file": "derived/attribution_results.jsonl",
                "attribution_results_digest": _digest(attributions),
                "endpoint_derivation_receipt_file": (
                    "derived/endpoint_derivation_receipt.json"
                ),
                "endpoint_derivation_receipt_digest": _digest(endpoint_receipt),
                "evaluation_metrics_file": (
                    None if metrics is None else "evaluation/metrics.json"
                ),
                "evaluation_metrics_digest": (
                    None if metrics is None else _digest(metrics)
                ),
            }
            package_id = "lep_" + hashlib.sha256(
                canonical_document_bytes(manifest_without_id)
            ).hexdigest()
            manifest = {"package_id": package_id, **manifest_without_id}
            manifest_bytes = canonical_document_bytes(manifest)
            parsed = parse_evidence_package_manifest(manifest_bytes)
            _reject_raw_endpoints(manifest_bytes, raw_endpoints, "package manifest")
            _write_new(
                staging / "evidence_package_manifest.json", manifest_bytes
            )
            _fsync_tree(staging)
            if destination.exists() or _is_link_like(destination):
                raise LogLinkedEvidenceError("evidence package already exists")
            try:
                os.rename(staging, destination)
            except OSError as error:
                if destination.exists() or _is_link_like(destination):
                    raise LogLinkedEvidenceError(
                        "evidence package already exists"
                    ) from error
                raise LogLinkedEvidenceError(
                    "evidence package publication failed"
                ) from error
            _make_private(destination, directory=True)
            _fsync_directory(parent)
        except Exception:
            if staging.exists() or _is_link_like(staging):
                _remove_owned_staging(staging, parent)
            raise
        return cls(destination, destination / "evidence_package_manifest.json", parsed)

    @classmethod
    def verify(cls, root: Path | str) -> LogLinkedEvidencePackage:
        package_root = Path(root)
        _verify_directory(package_root, "evidence package root")
        manifest_path = package_root / "evidence_package_manifest.json"
        manifest_bytes = _read_regular_file(manifest_path)
        try:
            manifest = parse_evidence_package_manifest(manifest_bytes)
        except ValueError as error:
            raise LogLinkedEvidenceError(
                f"evidence package manifest is invalid: {error}"
            ) from error

        expected_inventory = manifest["file_inventory"]
        actual_inventory = _inventory(package_root, omit_manifest=True)
        if actual_inventory != expected_inventory:
            raise LogLinkedEvidenceError(
                "evidence package contains an unexpected or tampered file"
            )
        _verify_directory_inventory(package_root, actual_inventory, manifest)
        by_path = {str(item["path"]): item for item in actual_inventory}
        _verify_claim(
            by_path,
            str(manifest["application_log_file"]),
            int(manifest["application_log_size_bytes"]),
            str(manifest["application_log_digest"]),
        )
        for path_field, digest_field in (
            ("log_provenance_file", "log_provenance_digest"),
            ("observer_terminal_seal_file", "observer_terminal_seal_digest"),
            ("writer_capture_receipt_file", "writer_capture_receipt_digest"),
            (
                "normalized_application_events_file",
                "normalized_application_events_digest",
            ),
            ("attribution_results_file", "attribution_results_digest"),
            (
                "endpoint_derivation_receipt_file",
                "endpoint_derivation_receipt_digest",
            ),
        ):
            path = str(manifest[path_field])
            _verify_claim(by_path, path, None, str(manifest[digest_field]))
        if manifest["evaluation_metrics_file"] is not None:
            _verify_claim(
                by_path,
                str(manifest["evaluation_metrics_file"]),
                None,
                str(manifest["evaluation_metrics_digest"]),
            )

        state_manifest_digests = sorted(
            str(item["sha256"])
            for item in actual_inventory
            if str(item["path"]).endswith("/state_unit_manifest.json")
        )
        if state_manifest_digests != manifest["state_unit_manifest_digests"]:
            raise LogLinkedEvidenceError("state-unit manifest inventory differs")
        _verify_state_units(package_root, actual_inventory)

        provenance = _read_regular_file(
            package_root / str(manifest["log_provenance_file"])
        )
        _parse_canonical_mapping(provenance, "log provenance")
        normalized_path = package_root / str(
            manifest["normalized_application_events_file"]
        )
        events = _parse_json_lines(
            _read_regular_file(normalized_path),
            parser=parse_application_event,
            label="normalized application events",
        )
        attribution_bytes = _read_regular_file(
            package_root / str(manifest["attribution_results_file"])
        )
        _parse_json_lines(
            attribution_bytes,
            parser=parse_attribution_result,
            label="attribution results",
        )
        endpoint_receipt = _read_regular_file(
            package_root / str(manifest["endpoint_derivation_receipt_file"])
        )
        _parse_canonical_mapping(endpoint_receipt, "endpoint derivation receipt")
        writer_receipt = _read_regular_file(
            package_root / str(manifest["writer_capture_receipt_file"])
        )
        try:
            parse_writer_capture_receipt(writer_receipt)
        except ValueError as error:
            raise LogLinkedEvidenceError(
                f"writer capture receipt is invalid: {error}"
            ) from error
        if manifest["evaluation_metrics_file"] is not None:
            _parse_canonical_mapping(
                _read_regular_file(
                    package_root / str(manifest["evaluation_metrics_file"])
                ),
                "evaluation metrics",
            )

        raw_endpoints = {
            str(event["endpoint_value"])
            for event in events
            if event["endpoint_value"] is not None
        }
        _reject_raw_endpoints(
            attribution_bytes, raw_endpoints, "attribution results"
        )
        _reject_raw_endpoints(
            endpoint_receipt, raw_endpoints, "endpoint derivation receipt"
        )
        _reject_raw_endpoints(
            writer_receipt, raw_endpoints, "writer capture receipt"
        )
        _reject_raw_endpoints(manifest_bytes, raw_endpoints, "package manifest")
        expected_id = "lep_" + hashlib.sha256(
            canonical_document_bytes(
                {key: value for key, value in manifest.items() if key != "package_id"}
            )
        ).hexdigest()
        if not hmac.compare_digest(str(manifest["package_id"]), expected_id):
            raise LogLinkedEvidenceError("evidence package identifier differs")
        return cls(package_root, manifest_path, manifest)


def _validate_log_provenance(inputs: EvidenceInputs) -> bytes:
    document = dict(inputs.log_provenance)
    if set(document) != _PROVENANCE_FIELDS:
        raise LogLinkedEvidenceError("log provenance field set is invalid")
    if document["schema_version"] != "1.0":
        raise LogLinkedEvidenceError("log provenance schema is invalid")
    if (
        document["adapter_id"] != inputs.adapter_id
        or document["parser_profile_id"] != inputs.adapter_id
    ):
        raise LogLinkedEvidenceError("log provenance adapter differs")
    if not isinstance(document["upstream_commit"], str) or _FULL_COMMIT.fullmatch(
        str(document["upstream_commit"])
    ) is None:
        raise LogLinkedEvidenceError("log provenance commit is invalid")
    if not isinstance(document["server_executable_sha256"], str) or _DIGEST.fullmatch(
        str(document["server_executable_sha256"])
    ) is None:
        raise LogLinkedEvidenceError("log provenance executable digest is invalid")
    arguments = document["logging_arguments"]
    if not isinstance(arguments, list) or any(
        not isinstance(item, str) or not item for item in arguments
    ):
        raise LogLinkedEvidenceError("logging arguments are invalid")
    if _FORBIDDEN_LOG_ARGUMENTS.intersection(arguments):
        raise LogLinkedEvidenceError("content-bearing logging argument is forbidden")
    required = {"--log-file", "--log-timestamps", "--log-verbosity", "4"}
    if not required.issubset(arguments):
        raise LogLinkedEvidenceError("required content-free logging arguments are absent")
    start = document["log_started_at_utc_ns"]
    end = document["log_ended_at_utc_ns"]
    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or start < 0
        or isinstance(end, bool)
        or not isinstance(end, int)
        or end < start
    ):
        raise LogLinkedEvidenceError("log provenance time range is invalid")
    clocks = document["clock_sources"]
    if not isinstance(clocks, list) or len(clocks) != len(set(clocks)) or any(
        not isinstance(item, str) or not item for item in clocks
    ):
        raise LogLinkedEvidenceError("clock sources must be unique")
    return _canonical_mapping("log provenance", document)


def _serialize_application_events(
    events: Sequence[ApplicationEvent], *, adapter_id: str
) -> bytes:
    if not isinstance(events, tuple) or not events:
        raise LogLinkedEvidenceError("normalized application events are required")
    payloads: list[bytes] = []
    for event in events:
        if not isinstance(event, ApplicationEvent) or event.adapter_id != adapter_id:
            raise LogLinkedEvidenceError("normalized application event is invalid")
        payload = canonical_document_bytes(event.to_mapping())
        parse_application_event(payload)
        payloads.append(payload)
    return b"".join(payloads)


def _serialize_attribution_results(
    results: Sequence[AttributionResult], *, adapter_id: str, runtime_instance_id: str
) -> bytes:
    if not isinstance(results, tuple) or not results:
        raise LogLinkedEvidenceError("attribution results are required")
    payloads: list[bytes] = []
    for result in results:
        if (
            not isinstance(result, AttributionResult)
            or result.adapter_id != adapter_id
            or result.runtime_instance_id != runtime_instance_id
        ):
            raise LogLinkedEvidenceError("attribution result is invalid")
        payloads.append(result.to_bytes())
    return b"".join(payloads)


def _canonical_mapping(name: str, value: Mapping[str, object]) -> bytes:
    if not isinstance(value, Mapping):
        raise LogLinkedEvidenceError(f"{name} must be an object")
    try:
        return canonical_document_bytes(value)
    except ValueError as error:
        raise LogLinkedEvidenceError(f"{name} is invalid: {error}") from error


def _parse_canonical_mapping(payload: bytes, name: str) -> dict[str, object]:
    try:
        document = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LogLinkedEvidenceError(f"{name} is not JSON") from error
    if not isinstance(document, dict) or canonical_document_bytes(document) != payload:
        raise LogLinkedEvidenceError(f"{name} is not canonical JSON")
    return document


def _parse_json_lines(payload: bytes, *, parser, label: str) -> list[dict[str, object]]:
    if not payload:
        raise LogLinkedEvidenceError(f"{label} are empty")
    documents: list[dict[str, object]] = []
    for line in payload.splitlines(keepends=True):
        try:
            documents.append(parser(line))
        except ValueError as error:
            raise LogLinkedEvidenceError(f"{label} are invalid: {error}") from error
    return documents


def _verify_state_units(root: Path, inventory: list[dict[str, object]]) -> None:
    by_path = {str(item["path"]): item for item in inventory}
    for path, item in by_path.items():
        if not path.endswith("/state_unit_manifest.json"):
            continue
        payload = _read_regular_file(root / path)
        try:
            manifest = parse_state_unit_manifest(payload)
        except ValueError as error:
            raise LogLinkedEvidenceError("state-unit manifest is invalid") from error
        payload_path = (root / path).parent / str(manifest["payload_file"])
        relative_payload = payload_path.relative_to(root).as_posix()
        payload_item = by_path.get(relative_payload)
        if payload_item is None:
            raise LogLinkedEvidenceError("state-unit payload is unavailable")
        if (
            payload_item["size_bytes"] != manifest["payload_size_bytes"]
            or not hmac.compare_digest(
                str(payload_item["sha256"]), str(manifest["payload_digest"])
            )
        ):
            raise LogLinkedEvidenceError("state-unit payload claim differs")


def _verify_claim(
    by_path: Mapping[str, Mapping[str, object]],
    path: str,
    size: int | None,
    digest: str,
) -> None:
    item = by_path.get(path)
    if item is None:
        raise LogLinkedEvidenceError("manifest references an unavailable file")
    if size is not None and item["size_bytes"] != size:
        raise LogLinkedEvidenceError("manifest file size differs")
    if not hmac.compare_digest(str(item["sha256"]), digest):
        raise LogLinkedEvidenceError("manifest file digest differs")


def _inventory(root: Path, *, omit_manifest: bool = False) -> list[dict[str, object]]:
    _verify_directory(root, "evidence package root")
    items: list[dict[str, object]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if _is_link_like(path):
            raise LogLinkedEvidenceError("evidence package contains a symbolic link")
        metadata = os.stat(path, follow_symlinks=False)
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise LogLinkedEvidenceError("evidence package contains an unexpected artifact")
        relative = path.relative_to(root).as_posix()
        if omit_manifest and relative == "evidence_package_manifest.json":
            continue
        payload = _read_regular_file(path)
        items.append(
            {"path": relative, "size_bytes": len(payload), "sha256": _digest(payload)}
        )
    return sorted(items, key=lambda item: str(item["path"]))


def _verify_directory_inventory(
    root: Path,
    inventory: list[dict[str, object]],
    manifest: Mapping[str, object],
) -> None:
    expected_directories = {"application", "observer", "derived"}
    if manifest["evaluation_metrics_file"] is not None:
        expected_directories.add("evaluation")
    for item in inventory:
        parts = Path(str(item["path"])).parts[:-1]
        for length in range(1, len(parts) + 1):
            expected_directories.add(Path(*parts[:length]).as_posix())
    actual_directories: set[str] = set()
    for path in root.rglob("*"):
        if _is_link_like(path):
            raise LogLinkedEvidenceError("evidence package contains a symbolic link")
        if path.is_dir():
            actual_directories.add(path.relative_to(root).as_posix())
    if actual_directories != expected_directories:
        raise LogLinkedEvidenceError(
            "evidence package contains an unexpected directory"
        )


def _copy_tree_checked(source: Path, destination: Path) -> None:
    _verify_directory(source, "state-unit root")
    _ensure_private_directory(destination)

    def copy_directory(source_root: Path, destination_root: Path) -> None:
        for item in sorted(source_root.iterdir(), key=lambda path: path.name):
            if _is_link_like(item):
                raise LogLinkedEvidenceError("state-unit root contains a symbolic link")
            metadata = os.stat(item, follow_symlinks=False)
            target = destination_root / item.name
            if stat.S_ISDIR(metadata.st_mode):
                _ensure_private_directory(target)
                copy_directory(item, target)
            elif stat.S_ISREG(metadata.st_mode):
                _write_new(target, _read_regular_file(item))
            else:
                raise LogLinkedEvidenceError(
                    "state-unit root contains an unexpected artifact"
                )

    copy_directory(source, destination)


def _read_regular_file(path: Path) -> bytes:
    if _is_link_like(path):
        raise LogLinkedEvidenceError("evidence input is a symbolic link")
    try:
        before = os.stat(path, follow_symlinks=False)
    except OSError as error:
        raise LogLinkedEvidenceError("required evidence input is unavailable") from error
    if not stat.S_ISREG(before.st_mode):
        raise LogLinkedEvidenceError("required evidence input is not a regular file")
    payload = path.read_bytes()
    after = os.stat(path, follow_symlinks=False)
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise LogLinkedEvidenceError("evidence input changed while reading")
    return payload


def _write_new(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as error:
        raise LogLinkedEvidenceError("evidence file cannot be created") from error
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise LogLinkedEvidenceError("evidence write made no progress")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _make_private(path, directory=False)


def _ensure_private_directory(path: Path) -> None:
    if path.exists() and (_is_link_like(path) or not path.is_dir()):
        raise LogLinkedEvidenceError("evidence directory is unsafe")
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    _verify_directory(path, "evidence directory")
    _make_private(path, directory=True)


def _verify_directory(path: Path, label: str) -> None:
    if _is_link_like(path):
        raise LogLinkedEvidenceError(f"{label} is a symbolic link")
    try:
        metadata = os.stat(path, follow_symlinks=False)
    except OSError as error:
        raise LogLinkedEvidenceError(f"{label} is unavailable") from error
    if not stat.S_ISDIR(metadata.st_mode):
        raise LogLinkedEvidenceError(f"{label} is not a directory")


def _is_link_like(path: Path) -> bool:
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(metadata.st_mode):
        return True
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(reparse and getattr(metadata, "st_file_attributes", 0) & reparse)


def _make_private(path: Path, *, directory: bool) -> None:
    if os.name == "posix":
        os.chmod(path, 0o700 if directory else 0o600, follow_symlinks=False)


def _digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _reject_raw_endpoints(payload: bytes, endpoints: set[str], label: str) -> None:
    for endpoint in endpoints:
        if endpoint.encode("utf-8") in payload:
            raise LogLinkedEvidenceError(f"raw endpoint leaked into {label}")


def _fsync_tree(root: Path) -> None:
    if os.name != "posix":
        return
    directories = [root]
    directories.extend(path for path in root.rglob("*") if path.is_dir())
    for directory in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        _fsync_directory(directory)


def _fsync_directory(path: Path) -> None:
    if os.name != "posix":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _remove_owned_staging(staging: Path, parent: Path) -> None:
    try:
        if staging.parent.resolve() != parent.resolve():
            raise LogLinkedEvidenceError("evidence staging cleanup target is unsafe")
        if not staging.name.startswith(".") or ".staging-" not in staging.name:
            raise LogLinkedEvidenceError("evidence staging cleanup target is unsafe")
        if _is_link_like(staging):
            staging.unlink()
        else:
            shutil.rmtree(staging)
    except FileNotFoundError:
        return


__all__ = [
    "EvidenceInputs",
    "LogLinkedEvidenceError",
    "LogLinkedEvidencePackage",
]
