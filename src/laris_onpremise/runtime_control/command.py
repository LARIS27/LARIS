from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

from laris_onpremise.configuration import LiveScenario
from laris_onpremise.lab_agent.client import (
    LabAgentClient,
    LabAgentClientError,
    LabAgentProfile,
)
from laris_onpremise.lab_agent.runtime_mode import RuntimeMode


def add_parser(subcommands) -> argparse.ArgumentParser:
    runtime = subcommands.add_parser(
        "runtime",
        help="Inspect and control an existing llama.cpp runtime through the Lab Agent",
    )
    commands = runtime.add_subparsers(dest="runtime_command", required=True)
    for name, help_text in (
        ("inspect", "Inspect one compatible existing runtime"),
        ("status", "Read the persistent Lab Agent state"),
    ):
        parser = commands.add_parser(name, help=help_text)
        parser.add_argument("--agent-profile", type=Path, required=True)
    attach = commands.add_parser(
        "attach", help="Attach LARIS Proxy and observers to an existing runtime"
    )
    attach.add_argument("--agent-profile", type=Path, required=True)
    attach.add_argument("--scenario", type=Path, required=True)
    for name, help_text in (
        ("stop-and-seal", "Close admission and seal the attached run"),
        ("detach", "Release LARIS while retaining the external runtime"),
    ):
        parser = commands.add_parser(name, help=help_text)
        parser.add_argument("--agent-profile", type=Path, required=True)
        parser.add_argument("--run-id", required=True)
    return runtime


def _client(path: Path) -> LabAgentClient:
    return LabAgentClient(LabAgentProfile.load(path))


def _write_result(value: object) -> None:
    print(json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")))


def _run_id(scenario: LiveScenario) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"run-{timestamp}_{scenario.run_name}_{secrets.token_hex(4)}"


def _attach(client: LabAgentClient, scenario_path: Path) -> dict[str, object]:
    raw = scenario_path.read_bytes()
    scenario = LiveScenario.load(scenario_path)
    inspected = client.inspect_runtime()
    candidate = inspected.get("candidate")
    if inspected.get("attach_ready") is not True or not isinstance(candidate, dict):
        raise LabAgentClientError(
            "Attached runtime is not ready",
            code="ATTACHED_RUNTIME_NOT_READY",
            detail="Attached runtime is not ready",
        )
    slot_ids = candidate.get("slot_ids")
    if not isinstance(slot_ids, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in slot_ids
    ):
        raise LabAgentClientError("Attached runtime slot inventory is invalid")
    expected_slots = scenario.runtime_slots
    if expected_slots is None or len(slot_ids) != expected_slots:
        raise LabAgentClientError(
            "Attached runtime slot count does not match the scenario",
            code="ATTACHED_RUNTIME_MISMATCH",
            detail="Attached runtime slot count does not match the scenario",
        )
    required = {
        "model_profile_id": candidate.get("model_profile_id"),
        "model_digest": candidate.get("model_digest"),
        "model_name": candidate.get("model_name"),
    }
    if any(not isinstance(value, str) or not value for value in required.values()):
        raise LabAgentClientError("Attached runtime model identity is invalid")
    return client.prepare(
        runtime_mode=RuntimeMode.ATTACHED,
        run_id=_run_id(scenario),
        scenario_id=scenario.scenario_id,
        scenario_digest="sha256:" + hashlib.sha256(raw).hexdigest(),
        expected_request_count=scenario.request_count,
        model_profile_id=str(required["model_profile_id"]),
        model_digest=str(required["model_digest"]),
        expected_model_name=str(required["model_name"]),
        expected_slot_count=expected_slots,
    )


def run(
    arguments: argparse.Namespace,
    *,
    client_factory: Callable[[Path], LabAgentClient] = _client,
) -> int:
    try:
        client = client_factory(arguments.agent_profile)
        if arguments.runtime_command == "inspect":
            result = client.inspect_runtime()
        elif arguments.runtime_command == "attach":
            result = _attach(client, arguments.scenario)
        elif arguments.runtime_command == "status":
            result = client.status()
        elif arguments.runtime_command == "stop-and-seal":
            result = client.stop_and_seal(arguments.run_id)
        elif arguments.runtime_command == "detach":
            result = client.release(arguments.run_id)
        else:
            raise ValueError("runtime command is unsupported")
    except LabAgentClientError as error:
        print(f"{error.code}: {error.detail}", file=sys.stderr)
        return 2
    except (OSError, ValueError) as error:
        print(f"RUNTIME_CONTROL_ERROR: {error}", file=sys.stderr)
        return 2
    _write_result(result)
    return 0
