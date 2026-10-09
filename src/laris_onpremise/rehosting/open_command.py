from __future__ import annotations

from argparse import Namespace
from contextlib import nullcontext
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Callable, ContextManager, TextIO

from laris_onpremise.core.records import canonical_json
from laris_onpremise.rehosting.analysis_session import RehostAnalysisSession
from laris_onpremise.rehosting.catalog import PortableBundleCatalog
from laris_onpremise.rehosting.contracts import (
    InvestigatorSampler,
    RehostSubjectKind,
)
from laris_onpremise.rehosting.host_profile import (
    InvestigatorProfile,
    InvestigatorProfileError,
)
from laris_onpremise.rehosting.identity import resolve_subject_sessions
from laris_onpremise.rehosting.profile_command import default_profile_path
from laris_onpremise.rehosting.runtime_catalog import resolve_model
from laris_onpremise.rehosting.terminal import TerminalPresenter
from laris_onpremise.rehosting.wsl import WslExecutor, WslRehostRequest


def open_main(
    arguments: Namespace,
    *,
    input_stream: TextIO = sys.stdin,
    output_stream: TextIO = sys.stdout,
    error_stream: TextIO = sys.stderr,
    executor_factory: Callable[[], WslExecutor] = WslExecutor,
) -> int:
    analysis_session: RehostAnalysisSession | None = None
    try:
        endpoint_ref_argument = getattr(arguments, "endpoint_ref", None)
        if arguments.json and (
            arguments.bundle_root is None
            or arguments.question_file is None
            or (arguments.subject_id_file is None) == (endpoint_ref_argument is None)
        ):
            raise ValueError(
                "--json requires --bundle-root, --question-file, and exactly one "
                "of --subject-id-file or --endpoint-ref"
            )
        presenter = _presenter(arguments, output_stream)
        if presenter is not None:
            presenter.banner()
            presenter.startup_scan()
        profile_path = _absolute(Path(arguments.profile or default_profile_path()))
        try:
            with _activity(presenter, "Loading investigator profile"):
                profile = InvestigatorProfile.load(profile_path)
        except InvestigatorProfileError as error:
            try:
                profile_path.lstat()
            except FileNotFoundError:
                bundle_hint = (
                    str(_absolute(Path(arguments.bundle_root)))
                    if arguments.bundle_root is not None
                    else "<bundle-path>"
                )
                raise ValueError(
                    "investigator profile is unavailable; run: "
                    f'laris rehost profile create --bundle-root "{bundle_hint}"'
                ) from error
            raise
        _stage(presenter, "PROFILE", "VERIFIED", "local investigator profile")
        analysis_root = (
            _absolute(Path(arguments.analysis_root))
            if arguments.analysis_root is not None
            else profile.analysis_root
        )
        RehostAnalysisSession.recover_stale(analysis_root)
        bundle_root = (
            _absolute(Path(arguments.bundle_root))
            if arguments.bundle_root is not None
            else _absolute(Path(_prompt(input_stream, output_stream, "Bundle path: ").strip()))
        )
        with _activity(presenter, "Verifying portable bundle"):
            catalog = PortableBundleCatalog.open(bundle_root)
        _stage(presenter, "BUNDLE", "VERIFIED", catalog.source_run_id)
        if catalog.subject_kind is RehostSubjectKind.ENDPOINT_REF:
            if arguments.subject_id_file is not None:
                raise ValueError(
                    "endpoint bundles require --endpoint-ref, not --subject-id-file"
                )
            endpoint_ref = (
                endpoint_ref_argument
                if endpoint_ref_argument is not None
                else _prompt(input_stream, output_stream, "Endpoint ref: ").strip()
            )
            matches = tuple(
                entry for entry in catalog.entries
                if entry.endpoint_ref == endpoint_ref
            )
            if not matches:
                raise ValueError("endpoint reference does not match the bundle")
            _stage(presenter, "ENDPOINT", "MATCHED", f"{len(matches)} session(s)")
        else:
            if endpoint_ref_argument is not None:
                raise ValueError(
                    "user-HMAC bundles require --subject-id-file, not --endpoint-ref"
                )
            raw_subject = (
                _read_text_file(Path(arguments.subject_id_file), "subject ID")
                if arguments.subject_id_file is not None
                else _prompt(input_stream, output_stream, "Subject ID: ")
            )
            with _activity(presenter, "Matching subject inside selected bundle"):
                matches = resolve_subject_sessions(
                    catalog,
                    tenant_id=profile.tenant_id,
                    registry=profile.identity_registry,
                    raw_subject_id=raw_subject,
                )
            _stage(presenter, "SUBJECT", "MATCHED", f"{len(matches)} session(s)")
        entry = _select_session(
            matches,
            arguments.session_ref,
            input_stream=input_stream,
            output_stream=output_stream,
        )
        _stage(
            presenter,
            "SESSION",
            "SELECTED",
            f"{entry.model_profile_id} / {entry.last_turn_index} captured turns",
        )
        with _activity(presenter, "Resolving exact local model"):
            model = resolve_model(profile, entry)
        _stage(presenter, "MODEL", "READY", entry.model_profile_id)
        executor = executor_factory()
        sampler = _sampler(arguments)
        question_file = (
            _absolute(Path(arguments.question_file))
            if arguments.question_file is not None
            else None
        )
        if question_file is not None:
            messages = _investigator_messages(
                _read_text_file(question_file, "question"),
                [],
                session=False,
            )
            analysis_session = RehostAnalysisSession.start(
                analysis_root,
                catalog=catalog,
                entry=entry,
                mode="simple",
                sampler=sampler,
                predict=arguments.predict,
            )
            temporary_messages = _temporary_messages(
                profile.analysis_root,
                messages,
            )
            turn_started_at = datetime.now(timezone.utc)
            try:
                try:
                    result = _execute_question(
                        executor,
                        profile,
                        catalog,
                        entry,
                        model,
                        temporary_messages,
                        arguments,
                        presenter,
                        sampler=sampler,
                    )
                except Exception as error:
                    analysis_session.record_failure(
                        messages=messages,
                        reason=str(error),
                        started_at=turn_started_at,
                    )
                    analysis_session.finalize(
                        status="INCOMPLETE",
                        reason="inquiry_failed",
                    )
                    analysis_session = None
                    raise
                analysis_session.record_success(
                    messages=messages,
                    result=result.result,
                    started_at=turn_started_at,
                )
                analysis_session.finalize(status="COMPLETE")
                analysis_session = None
            finally:
                try:
                    temporary_messages.unlink()
                except OSError:
                    pass
            _render_result(result, output_stream, as_json=arguments.json)
            return 0

        mode = _select_mode(
            getattr(arguments, "mode", None),
            input_stream=input_stream,
            output_stream=output_stream,
        )
        output_stream.write(f"Selected mode: {mode.upper()}\n")
        output_stream.write("Type /exit to quit.\n\n")
        output_stream.flush()
        analysis_session = RehostAnalysisSession.start(
            analysis_root,
            catalog=catalog,
            entry=entry,
            mode=mode,
            sampler=sampler,
            predict=arguments.predict,
        )
        history: list[tuple[str, str]] = []
        while True:
            question = _prompt(input_stream, output_stream, "laris> ")
            if question.strip().casefold() in {"/exit", "/quit"}:
                completed = analysis_session.finalize(status="COMPLETE")
                analysis_session = None
                output_stream.write(f"analysis> {completed.parent}\n")
                output_stream.flush()
                return 0
            if not question.strip():
                continue
            messages = _investigator_messages(
                question,
                history,
                session=mode == "session",
            )
            temporary_messages = _temporary_messages(
                profile.analysis_root,
                messages,
            )
            turn_started_at = datetime.now(timezone.utc)
            try:
                try:
                    result = _execute_question(
                        executor,
                        profile,
                        catalog,
                        entry,
                        model,
                        temporary_messages,
                        arguments,
                        presenter,
                        sampler=sampler,
                    )
                except Exception as error:
                    analysis_session.record_failure(
                        messages=messages,
                        reason=str(error),
                        started_at=turn_started_at,
                    )
                    _stage(presenter, "INQUIRY", "FAILED", tone="red")
                    error_stream.write(f"laris rehost: turn failed: {error}\n")
                    error_stream.flush()
                    continue
            finally:
                try:
                    temporary_messages.unlink()
                except OSError:
                    pass
            analysis_session.record_success(
                messages=messages,
                result=result.result,
                started_at=turn_started_at,
            )
            _render_result(
                result,
                output_stream,
                as_json=False,
                analysis_directory=analysis_session.directory,
            )
            response = _response(result)
            if mode == "session" and response is not None:
                history.append((question, response))
    except EOFError:
        if analysis_session is not None:
            _finalize_incomplete(analysis_session, "input_closed", error_stream)
        error_stream.write("laris rehost: input closed\n")
        return 130
    except KeyboardInterrupt:
        if analysis_session is not None:
            _finalize_incomplete(
                analysis_session,
                "investigator_cancelled",
                error_stream,
            )
        error_stream.write("laris rehost: cancelled\n")
        return 130
    except Exception as error:
        if analysis_session is not None:
            _finalize_incomplete(analysis_session, "analysis_failed", error_stream)
        error_stream.write(f"laris rehost: {error}\n")
        return 2


def _presenter(arguments: Namespace, stream: TextIO) -> TerminalPresenter | None:
    if arguments.json:
        return None
    tty = hasattr(stream, "isatty") and stream.isatty()
    styled = (
        not arguments.plain
        and tty
        and "NO_COLOR" not in os.environ
        and os.environ.get("TERM") != "dumb"
    )
    return TerminalPresenter(
        stream,
        color=styled,
        motion=styled and not arguments.no_motion,
    )


def _activity(
    presenter: TerminalPresenter | None,
    label: str,
) -> ContextManager[None]:
    if presenter is None:
        return nullcontext()
    return presenter.activity(label)


def _stage(
    presenter: TerminalPresenter | None,
    name: str,
    status: str,
    detail: str | None = None,
    *,
    tone: str = "green",
) -> None:
    if presenter is not None:
        presenter.stage(name, status, detail, tone=tone)


def _select_mode(
    configured: str | None,
    *,
    input_stream: TextIO,
    output_stream: TextIO,
) -> str:
    if configured in {"simple", "session"}:
        return configured
    output_stream.write("Select rehost mode:\n")
    output_stream.write("  A - SIMPLE: each question starts from the preserved state\n")
    output_stream.write("  B - SESSION: prior investigator turns are accumulated\n")
    while True:
        selected = _prompt(input_stream, output_stream, "Choose mode [A/B]: ").strip().upper()
        if selected == "A":
            return "simple"
        if selected == "B":
            return "session"
        output_stream.write("Please enter A or B.\n")


def _temporary_messages(
    root: Path,
    messages: tuple[dict[str, str], ...],
) -> Path:
    if not messages:
        raise ValueError("messages are invalid")
    root.mkdir(parents=True, exist_ok=True)
    descriptor, raw_path = tempfile.mkstemp(
        prefix=".laris-messages-",
        suffix=".json",
        dir=root,
    )
    os.close(descriptor)
    path = Path(raw_path)
    path.write_bytes(
        canonical_json(
            {
                "messages": list(messages),
                "schema_version": "1.0",
            }
        )
        + b"\n"
    )
    return path


def _investigator_messages(
    question: str,
    history: list[tuple[str, str]],
    *,
    session: bool,
) -> tuple[dict[str, str], ...]:
    if not question.strip() or "\x00" in question:
        raise ValueError("question is invalid")
    messages: list[dict[str, str]] = []
    if session:
        for prior_question, prior_answer in history:
            messages.append({"role": "user", "content": prior_question})
            messages.append({"role": "assistant", "content": prior_answer})
    messages.append({"role": "user", "content": question})
    return tuple(messages)


def _execute_question(
    executor: WslExecutor,
    profile: InvestigatorProfile,
    catalog: PortableBundleCatalog,
    entry: object,
    model: Path,
    messages_file: Path,
    arguments: Namespace,
    presenter: TerminalPresenter | None,
    *,
    sampler: InvestigatorSampler,
):
    request = WslRehostRequest(
        bundle_root=catalog.bundle_root,
        bundle_manifest_digest=catalog.manifest_digest,
        entry=entry,
        model=model,
        messages_file=messages_file,
        timeout_seconds=arguments.timeout_seconds,
        predict=arguments.predict,
        sampler=sampler,
    )
    with _activity(presenter, "Restoring state and running inquiry"):
        return executor.execute(profile, request)


def _render_result(
    result: object,
    output_stream: TextIO,
    *,
    as_json: bool,
    analysis_directory: Path | None = None,
) -> None:
    if as_json:
        output_stream.write(
            json.dumps(
                result.result,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
        return
    response = _response(result)
    output_stream.write("assistant> ")
    output_stream.write((response if isinstance(response, str) else "(no response)") + "\n")
    if analysis_directory is not None:
        output_stream.write(f"analysis> {analysis_directory}\n")
    output_stream.write("\n")
    output_stream.flush()


def _sampler(arguments: Namespace) -> InvestigatorSampler:
    return InvestigatorSampler(
        temperature=getattr(arguments, "temperature", 0.8),
        top_k=getattr(arguments, "top_k", 40),
        top_p=getattr(arguments, "top_p", 0.95),
        min_p=getattr(arguments, "min_p", 0.05),
        seed=getattr(arguments, "seed", 0),
    )


def _finalize_incomplete(
    session: RehostAnalysisSession,
    reason: str,
    error_stream: TextIO,
) -> None:
    try:
        session.finalize(status="INCOMPLETE", reason=reason)
    except Exception as error:
        error_stream.write(f"laris rehost: analysis finalization failed: {error}\n")


def _response(result: object) -> str | None:
    receipt = result.result.get("receipt")
    response = receipt.get("response") if isinstance(receipt, dict) else None
    return response if isinstance(response, str) else None


def _prompt(input_stream: TextIO, output_stream: TextIO, prompt: str) -> str:
    output_stream.write(prompt)
    output_stream.flush()
    value = input_stream.readline()
    if value == "":
        raise EOFError
    if value.endswith("\n"):
        value = value[:-1]
    if value.endswith("\r"):
        value = value[:-1]
    return value


def _select_session(
    matches: tuple[object, ...],
    session_ref: str | None,
    *,
    input_stream: TextIO,
    output_stream: TextIO,
):
    if session_ref is not None:
        selected = [item for item in matches if item.session_ref == session_ref]
        if len(selected) != 1:
            raise ValueError("session selection does not match the bundle")
        return selected[0]
    if len(matches) == 1:
        return matches[0]
    output_stream.write(f"Matched sessions: {len(matches)}\n")
    for index, item in enumerate(matches, 1):
        output_stream.write(
            f"  {index}. {item.model_profile_id}  {item.last_turn_index} turns  "
            f"{item.session_ref[:15]}...\n"
        )
    while True:
        raw = _prompt(input_stream, output_stream, "Select session: ")
        try:
            index = int(raw)
        except ValueError:
            continue
        if 1 <= index <= len(matches):
            return matches[index - 1]


def _read_text_file(path: Path, description: str) -> str:
    try:
        value = _absolute(path).read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ValueError(f"{description} file is unavailable") from error
    if value.endswith(("\n", "\r")):
        raise ValueError(f"{description} file must not contain a line ending")
    return value


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


__all__ = ["default_profile_path", "open_main"]
