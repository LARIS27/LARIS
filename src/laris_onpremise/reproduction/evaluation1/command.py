"""Reviewer-facing Evaluation1 reproduction commands."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from argparse import Namespace
from pathlib import Path
from typing import Callable, Sequence, TextIO

from laris_onpremise.rehosting.open_command import open_main as rehost_open_main

from .contracts import CaptureMode
from .resolver import Evaluation1Release


def _add_selector(parser: argparse.ArgumentParser, *, tuple_required: bool) -> None:
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--mode", choices=tuple(mode.value for mode in CaptureMode), required=True)
    if tuple_required:
        parser.add_argument("--checkpoint-id", choices=tuple(f"C{i}" for i in range(1, 7)), required=True)
        parser.add_argument("--user-ref", choices=tuple(f"U{i:02d}" for i in range(1, 11)), required=True)


def add_parser(subcommands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    reproduce = subcommands.add_parser("reproduce", help="Reproduce published paper evaluations")
    evaluations = reproduce.add_subparsers(dest="reproduce_evaluation", required=True)
    evaluation1 = evaluations.add_parser("evaluation1", help="Inspect and rehost Evaluation1 bundles")
    commands = evaluation1.add_subparsers(dest="evaluation1_command", required=True)

    listing = commands.add_parser("list", help="List the 60 public tuples for one model and mode")
    _add_selector(listing, tuple_required=False)

    show = commands.add_parser("show", help="Verify and show one public tuple")
    _add_selector(show, tuple_required=True)

    rehost = commands.add_parser("rehost", help="Materialize and query one verified public bundle")
    _add_selector(rehost, tuple_required=True)
    question = rehost.add_mutually_exclusive_group(required=True)
    question.add_argument("--question-file", type=Path)
    question.add_argument("--record-id")
    rehost.add_argument("--profile", type=Path)
    rehost.add_argument("--analysis-root", type=Path)
    rehost.add_argument("--timeout-seconds", type=int, default=120)
    rehost.add_argument("--predict", type=int, default=128)
    rehost.add_argument("--temperature", type=float, default=0.0)
    rehost.add_argument("--top-k", type=int, default=1)
    rehost.add_argument("--top-p", type=float, default=1.0)
    rehost.add_argument("--min-p", type=float, default=0.0)
    rehost.add_argument("--seed", type=int, default=0)
    rehost.add_argument("--analysis-mode", choices=("simple", "session"), default="simple")
    rehost.add_argument("--json", action="store_true")

    recall = commands.add_parser("verify-recall", help="Compare a response with the published oracle")
    _add_selector(recall, tuple_required=True)
    recall.add_argument("--record-id", required=True)
    recall.add_argument("--response-file", type=Path, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="laris reproduce evaluation1")
    commands = parser.add_subparsers(dest="evaluation1_command", required=True)
    listing = commands.add_parser("list")
    _add_selector(listing, tuple_required=False)
    show = commands.add_parser("show")
    _add_selector(show, tuple_required=True)
    rehost = commands.add_parser("rehost")
    _add_selector(rehost, tuple_required=True)
    group = rehost.add_mutually_exclusive_group(required=True)
    group.add_argument("--question-file", type=Path)
    group.add_argument("--record-id")
    rehost.add_argument("--profile", type=Path)
    rehost.add_argument("--analysis-root", type=Path)
    rehost.add_argument("--timeout-seconds", type=int, default=120)
    rehost.add_argument("--predict", type=int, default=128)
    rehost.add_argument("--temperature", type=float, default=0.0)
    rehost.add_argument("--top-k", type=int, default=1)
    rehost.add_argument("--top-p", type=float, default=1.0)
    rehost.add_argument("--min-p", type=float, default=0.0)
    rehost.add_argument("--seed", type=int, default=0)
    rehost.add_argument("--analysis-mode", choices=("simple", "session"), default="simple")
    rehost.add_argument("--json", action="store_true")
    recall = commands.add_parser("verify-recall")
    _add_selector(recall, tuple_required=True)
    recall.add_argument("--record-id", required=True)
    recall.add_argument("--response-file", type=Path, required=True)
    return parser


def _emit(value: object, output_stream: TextIO) -> None:
    output_stream.write(
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    )


def run(
    arguments: Namespace,
    *,
    output_stream: TextIO = sys.stdout,
    rehost_runner: Callable[[Namespace], int] = rehost_open_main,
) -> int:
    release = Evaluation1Release.open(arguments.release_root)
    command = arguments.evaluation1_command
    if command == "list":
        entries = release.list_entries(arguments.model_id, arguments.mode)
        _emit(
            {
                "schema_version": "1.0",
                "model_id": arguments.model_id,
                "mode": arguments.mode,
                "count": len(entries),
                "entries": [release.entry_document(entry) for entry in entries],
            },
            output_stream,
        )
        return 0
    entry = release.resolve(
        arguments.model_id,
        arguments.mode,
        arguments.checkpoint_id,
        arguments.user_ref,
    )
    identifier = release.verify_identifier(entry)
    if command == "show":
        result = release.entry_document(entry)
        result["identifier_verification"] = identifier["status"]
        _emit(result, output_stream)
        return 0
    if command == "verify-recall":
        release.validate_record(arguments.checkpoint_id, arguments.record_id)
        expected = release.expected_answer(
            arguments.checkpoint_id,
            arguments.user_ref,
            arguments.record_id,
        )
        response = arguments.response_file.read_text(encoding="utf-8").strip()
        matched = expected.casefold() in response.casefold()
        _emit(
            {
                "schema_version": "1.0",
                "checkpoint_id": arguments.checkpoint_id,
                "record_id": arguments.record_id,
                "match": matched,
                "expected_sha256": "sha256:" + hashlib.sha256(expected.encode()).hexdigest(),
                "response_sha256": "sha256:" + hashlib.sha256(response.encode()).hexdigest(),
            },
            output_stream,
        )
        return 0 if matched else 1
    if command != "rehost":
        raise ValueError("unsupported Evaluation1 reproduction command")

    with tempfile.TemporaryDirectory(prefix="laris-evaluation1-") as temporary:
        temporary_root = Path(temporary)
        bundle_root = temporary_root / "bundle"
        release.materialize(
            arguments.model_id,
            arguments.mode,
            arguments.checkpoint_id,
            arguments.user_ref,
            bundle_root,
        )
        if arguments.question_file is None:
            release.validate_record(arguments.checkpoint_id, arguments.record_id)
            question_file = temporary_root / "question.txt"
            question_file.write_text(
                release.question(arguments.checkpoint_id, arguments.record_id),
                encoding="utf-8",
            )
        else:
            question_file = arguments.question_file
        subject_file: Path | None = None
        endpoint_ref: str | None = None
        if entry.mode is CaptureMode.PROXY:
            subject_file = temporary_root / "subject-id.txt"
            subject_file.write_text(
                release.subject_id(entry.public_user_ref),
                encoding="utf-8",
            )
        else:
            endpoint_ref = entry.subject_ref
        forwarded = Namespace(
            bundle_root=bundle_root,
            profile=arguments.profile,
            subject_id_file=subject_file,
            endpoint_ref=endpoint_ref,
            session_ref=entry.session_ref,
            question_file=question_file,
            analysis_root=arguments.analysis_root,
            timeout_seconds=arguments.timeout_seconds,
            predict=arguments.predict,
            temperature=arguments.temperature,
            top_k=arguments.top_k,
            top_p=arguments.top_p,
            min_p=arguments.min_p,
            seed=arguments.seed,
            plain=True,
            no_motion=True,
            mode=arguments.analysis_mode,
            json=arguments.json,
        )
        return rehost_runner(forwarded)


def main(
    argv: Sequence[str] | None = None,
    *,
    output_stream: TextIO = sys.stdout,
    rehost_runner: Callable[[Namespace], int] = rehost_open_main,
) -> int:
    arguments = build_parser().parse_args(argv)
    return run(
        arguments,
        output_stream=output_stream,
        rehost_runner=rehost_runner,
    )
