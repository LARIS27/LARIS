"""Public command-line entry point for current LARIS runtime workflows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from laris_onpremise.attribution.log_linked.command import (
    add_parser as add_log_linked_parser,
    run as run_log_linked_command,
)
from laris_onpremise.attribution.proxy.command import (
    add_arguments as add_proxy_arguments,
    run as run_proxy_command,
)
from laris_onpremise.rehosting.command import main as rehost_main
from laris_onpremise.rehosting.open_command import open_main as rehost_open_main
from laris_onpremise.rehosting.profile_command import (
    add_identity_main as profile_add_identity_main,
    create_main as profile_create_main,
)
from laris_onpremise.reproduction.evaluation1.command import (
    add_parser as add_reproduction_parser,
    run as run_reproduction_command,
)
from laris_onpremise.runtime_control.command import (
    add_parser as add_runtime_parser,
    run as run_runtime_command,
)


def _add_rehost_parser(
    subcommands: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    rehost = subcommands.add_parser(
        "rehost", help="Rehost verified runtime-state payloads"
    )
    commands = rehost.add_subparsers(dest="rehost_command", required=True)

    cumulative = commands.add_parser(
        "cumulative", help="Query one cumulative-context payload"
    )
    cumulative.add_argument("--run-root", required=True)
    cumulative.add_argument("--user-hmac")
    cumulative.add_argument("--session-ref")
    cumulative.add_argument("--bundle-id")
    cumulative.add_argument("--destination-seq-id", type=int, default=0)
    cumulative.add_argument("--helper", required=True)
    cumulative.add_argument("--helper-digest", required=True)
    cumulative.add_argument("--libllama", required=True)
    cumulative.add_argument("--model", required=True)
    cumulative.add_argument("--model-profile-id", required=True)
    cumulative.add_argument("--runtime-profile", required=True)
    cumulative.add_argument("--index-digest", required=True)
    cumulative.add_argument("--messages-file", required=True)
    cumulative.add_argument("--analysis-root", required=True)
    cumulative.add_argument("--timeout-seconds", type=int, default=120)
    cumulative.add_argument("--predict", type=int, default=64)

    opened = commands.add_parser(
        "open", help="Open one portable bundle and query a restored session"
    )
    opened.add_argument("--bundle-root", type=Path)
    opened.add_argument("--profile", type=Path)
    opened.add_argument("--subject-id-file", type=Path)
    opened.add_argument("--endpoint-ref")
    opened.add_argument("--session-ref")
    opened.add_argument("--question-file", type=Path)
    opened.add_argument("--analysis-root", type=Path)
    opened.add_argument("--timeout-seconds", type=int, default=120)
    opened.add_argument("--predict", type=int, default=64)
    opened.add_argument("--temperature", type=float, default=0.8)
    opened.add_argument("--top-k", type=int, default=40)
    opened.add_argument("--top-p", type=float, default=0.95)
    opened.add_argument("--min-p", type=float, default=0.05)
    opened.add_argument("--seed", type=int, default=0)
    opened.add_argument("--plain", action="store_true")
    opened.add_argument("--no-motion", action="store_true")
    opened.add_argument("--mode", choices=("simple", "session"))
    opened.add_argument("--json", action="store_true")

    profile = commands.add_parser(
        "profile", help="Create or update investigator configuration"
    )
    profile_commands = profile.add_subparsers(
        dest="rehost_profile_command", required=True
    )
    create = profile_commands.add_parser("create")
    create.add_argument("--bundle-root", type=Path, required=True)
    create.add_argument("--profile", type=Path)
    create.add_argument("--tenant-id")
    create.add_argument("--identity-key", action="append")
    create.add_argument("--model-root", type=Path, action="append")
    create.add_argument("--analysis-root", type=Path)
    create.add_argument("--wsl-distribution")
    create.add_argument("--wsl-python")
    create.add_argument("--wsl-runtime-profile")
    create.add_argument("--wsl-helper")
    create.add_argument("--wsl-helper-digest")
    create.add_argument("--wsl-libllama")
    create.add_argument("--wsl-work-root")
    create.add_argument("--preserve-existing-identity-keys", action="store_true")
    create.add_argument("--replace", action="store_true")
    create.add_argument("--json", action="store_true")
    add_identity = profile_commands.add_parser("add-identity")
    add_identity.add_argument("--profile", type=Path)
    add_identity.add_argument("--identity-key", required=True)
    add_identity.add_argument("--json", action="store_true")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="laris",
        description="LARIS on-premise capture, attribution, and rehosting.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    add_proxy_arguments(
        subcommands.add_parser("proxy", help="Run the Sequence Lease Proxy")
    )
    add_runtime_parser(subcommands)
    add_log_linked_parser(subcommands)
    add_reproduction_parser(subcommands)
    _add_rehost_parser(subcommands)
    return parser


def _cumulative_arguments(arguments: argparse.Namespace) -> list[str]:
    values = [
        "--run-root", arguments.run_root,
        "--helper", arguments.helper,
        "--helper-digest", arguments.helper_digest,
        "--libllama", arguments.libllama,
        "--model", arguments.model,
        "--model-profile-id", arguments.model_profile_id,
        "--runtime-profile", arguments.runtime_profile,
        "--index-digest", arguments.index_digest,
        "--messages-file", arguments.messages_file,
        "--analysis-root", arguments.analysis_root,
        "--destination-seq-id", str(arguments.destination_seq_id),
        "--timeout-seconds", str(arguments.timeout_seconds),
        "--predict", str(arguments.predict),
    ]
    for option, value in (
        ("--user-hmac", arguments.user_hmac),
        ("--session-ref", arguments.session_ref),
        ("--bundle-id", arguments.bundle_id),
    ):
        if value is not None:
            values.extend((option, value))
    return values


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "proxy":
            return run_proxy_command(arguments)
        if arguments.command == "runtime":
            return run_runtime_command(arguments)
        if arguments.command == "log-linked":
            return run_log_linked_command(arguments)
        if arguments.command == "reproduce":
            return run_reproduction_command(arguments)
        if arguments.command == "rehost" and arguments.rehost_command == "cumulative":
            return rehost_main(_cumulative_arguments(arguments))
        if arguments.command == "rehost" and arguments.rehost_command == "open":
            return rehost_open_main(arguments)
        if (
            arguments.command == "rehost"
            and arguments.rehost_command == "profile"
            and arguments.rehost_profile_command == "create"
        ):
            return profile_create_main(arguments)
        if (
            arguments.command == "rehost"
            and arguments.rehost_command == "profile"
            and arguments.rehost_profile_command == "add-identity"
        ):
            return profile_add_identity_main(arguments)
    except (OSError, ValueError) as error:
        print(f"laris: {error}", file=sys.stderr)
        return 2
    parser.error("unsupported command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
