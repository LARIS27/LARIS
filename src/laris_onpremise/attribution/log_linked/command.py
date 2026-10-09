from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
import json
from pathlib import Path
import re
import sys
from typing import Protocol


EXIT_SUCCESS = 0
EXIT_INVALID_INPUT = 2
EXIT_PREFLIGHT_UNSUPPORTED = 3
EXIT_OBSERVATION_INCOMPLETE = 4
EXIT_ATTRIBUTION_UNRESOLVED = 5
EXIT_VERIFICATION_FAILED = 6

_ENDPOINT_REF = re.compile(r"[A-Za-z0-9_-]{43}\Z")
_REQUEST_REF = re.compile(r"rr_[0-9a-f]{64}\Z")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,191}\Z")


class LogLinkedCommandError(RuntimeError):
    """Raised when operator input cannot start a Log-linked stage."""


class LogLinkedVerificationError(RuntimeError):
    """Raised when sealed evidence or export verification fails."""


class _Workflow(Protocol):
    def preflight(self, profile: Path) -> Mapping[str, object]: ...

    def observe(
        self, profile: Path, evidence_root: Path
    ) -> Mapping[str, object]: ...

    def interrupt_observation(self) -> Mapping[str, object]: ...

    def analyze(
        self,
        evidence_root: Path,
        correlation_key_file: Path,
        correlation_key_id: str,
    ) -> Mapping[str, object]: ...

    def export(
        self,
        evidence_root: Path,
        endpoint_ref: str,
        bundle_root: Path,
        session_request_ref: str | None = None,
    ) -> Mapping[str, object]: ...


def add_parser(subcommands: object) -> argparse.ArgumentParser:
    add = getattr(subcommands, "add_parser", None)
    if not callable(add):
        raise TypeError("subcommands must support add_parser")
    root = add(
        "log-linked",
        help="Capture and attribute endpoint-unattributed llama.cpp state",
    )
    commands = root.add_subparsers(dest="log_linked_command", required=True)

    preflight = commands.add_parser(
        "preflight",
        help="Verify a content-free official llama-server log profile",
    )
    preflight.add_argument("--profile", type=Path, required=True)

    observe = commands.add_parser(
        "observe",
        help="Record Observer-driven sequence state units",
    )
    observe.add_argument("--profile", type=Path, required=True)
    observe.add_argument("--evidence-root", type=Path, required=True)

    correlate = commands.add_parser(
        "correlate",
        help="Analyze a sealed observation with its content-free log",
    )
    correlate.add_argument("--evidence-root", type=Path, required=True)
    correlate.add_argument("--correlation-key-file", type=Path, required=True)
    correlate.add_argument("--correlation-key-id", required=True)

    exported = commands.add_parser(
        "export",
        help="Export one resolved endpoint attribution as a rehosting bundle",
    )
    exported.add_argument("--evidence-root", type=Path, required=True)
    exported.add_argument("--endpoint-ref", required=True)
    exported.add_argument("--session-request-ref")
    exported.add_argument("--bundle-root", type=Path, required=True)
    return root


def _json(document: Mapping[str, object]) -> None:
    print(
        json.dumps(
            dict(document),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )


def _absolute(path: Path, label: str) -> Path:
    if not isinstance(path, Path) or not path.is_absolute():
        raise LogLinkedCommandError(f"{label} must be an absolute path")
    return path


def _existing_file(path: Path, label: str) -> Path:
    value = _absolute(path, label)
    if value.is_symlink() or not value.is_file():
        raise LogLinkedCommandError(f"{label} is unavailable")
    return value


def _existing_directory(path: Path, label: str) -> Path:
    value = _absolute(path, label)
    if value.is_symlink() or not value.is_dir():
        raise LogLinkedCommandError(f"{label} is unavailable")
    return value


class LogLinkedWorkflow:
    """Deployment workflow; the hosting application is always externally owned."""

    def __init__(self) -> None:
        self._active_observation: object | None = None

    def preflight(self, profile: Path) -> Mapping[str, object]:
        from .workflow import run_preflight

        return run_preflight(profile)

    def observe(
        self, profile: Path, evidence_root: Path
    ) -> Mapping[str, object]:
        from .workflow import ObservationSession

        session = ObservationSession(profile=profile, evidence_root=evidence_root)
        self._active_observation = session
        try:
            return session.run()
        finally:
            if session.finished:
                self._active_observation = None

    def interrupt_observation(self) -> Mapping[str, object]:
        session = self._active_observation
        if session is None or not callable(getattr(session, "interrupt", None)):
            return {"schema_version": "1.0", "status": "INCOMPLETE"}
        result = session.interrupt()
        self._active_observation = None
        return result

    def analyze(
        self,
        evidence_root: Path,
        correlation_key_file: Path,
        correlation_key_id: str,
    ) -> Mapping[str, object]:
        from .workflow import correlate_observation

        return correlate_observation(
            evidence_root=evidence_root,
            correlation_key_file=correlation_key_file,
            correlation_key_id=correlation_key_id,
        )

    def export(
        self,
        evidence_root: Path,
        endpoint_ref: str,
        bundle_root: Path,
        session_request_ref: str | None = None,
    ) -> Mapping[str, object]:
        from .workflow import export_endpoint_bundle

        return export_endpoint_bundle(
            evidence_root=evidence_root,
            endpoint_ref=endpoint_ref,
            bundle_root=bundle_root,
            session_request_ref=session_request_ref,
        )


def run(
    arguments: argparse.Namespace,
    *,
    workflow_factory: Callable[[], _Workflow] = LogLinkedWorkflow,
) -> int:
    workflow = workflow_factory()
    try:
        command = arguments.log_linked_command
        if command == "preflight":
            profile = _existing_file(arguments.profile, "profile")
            result = workflow.preflight(profile)
            _json(result)
            return (
                EXIT_SUCCESS
                if result.get("status") == "SUPPORTED"
                else EXIT_PREFLIGHT_UNSUPPORTED
            )
        if command == "observe":
            profile = _existing_file(arguments.profile, "profile")
            evidence_root = _absolute(arguments.evidence_root, "evidence_root")
            if evidence_root.exists() or evidence_root.is_symlink():
                raise LogLinkedCommandError("evidence_root already exists")
            try:
                result = workflow.observe(profile, evidence_root)
            except KeyboardInterrupt:
                result = workflow.interrupt_observation()
                _json(result)
                return EXIT_OBSERVATION_INCOMPLETE
            _json(result)
            return (
                EXIT_SUCCESS
                if result.get("status") == "COMPLETE"
                else EXIT_OBSERVATION_INCOMPLETE
            )
        if command == "correlate":
            evidence_root = _existing_directory(
                arguments.evidence_root, "evidence_root"
            )
            key_file = _existing_file(
                arguments.correlation_key_file, "correlation_key_file"
            )
            key_id = arguments.correlation_key_id
            if not isinstance(key_id, str) or _SAFE_ID.fullmatch(key_id) is None:
                raise LogLinkedCommandError("correlation_key_id is invalid")
            result = workflow.analyze(evidence_root, key_file, key_id)
            _json(result)
            return (
                EXIT_SUCCESS
                if result.get("status") == "COMPLETE"
                else EXIT_ATTRIBUTION_UNRESOLVED
            )
        if command == "export":
            evidence_root = _existing_directory(
                arguments.evidence_root, "evidence_root"
            )
            endpoint_ref = arguments.endpoint_ref
            if (
                not isinstance(endpoint_ref, str)
                or _ENDPOINT_REF.fullmatch(endpoint_ref) is None
            ):
                raise LogLinkedCommandError("endpoint_ref is invalid")
            session_request_ref = arguments.session_request_ref
            if session_request_ref is not None and (
                not isinstance(session_request_ref, str)
                or _REQUEST_REF.fullmatch(session_request_ref) is None
            ):
                raise LogLinkedCommandError("session_request_ref is invalid")
            bundle_root = _absolute(arguments.bundle_root, "bundle_root")
            if bundle_root.exists() or bundle_root.is_symlink():
                raise LogLinkedCommandError("bundle_root already exists")
            result = workflow.export(
                evidence_root,
                endpoint_ref,
                bundle_root,
                session_request_ref,
            )
            _json(result)
            return EXIT_SUCCESS
        raise LogLinkedCommandError("log-linked command is unsupported")
    except LogLinkedVerificationError as error:
        print(f"laris log-linked: {error}", file=sys.stderr)
        return EXIT_VERIFICATION_FAILED
    except (LogLinkedCommandError, OSError, ValueError) as error:
        print(f"laris log-linked: {error}", file=sys.stderr)
        return EXIT_INVALID_INPUT
    except RuntimeError as error:
        print(f"laris log-linked: {error}", file=sys.stderr)
        return EXIT_VERIFICATION_FAILED


__all__ = [
    "EXIT_ATTRIBUTION_UNRESOLVED",
    "EXIT_INVALID_INPUT",
    "EXIT_OBSERVATION_INCOMPLETE",
    "EXIT_PREFLIGHT_UNSUPPORTED",
    "EXIT_SUCCESS",
    "EXIT_VERIFICATION_FAILED",
    "LogLinkedCommandError",
    "LogLinkedVerificationError",
    "LogLinkedWorkflow",
    "add_parser",
    "run",
]
