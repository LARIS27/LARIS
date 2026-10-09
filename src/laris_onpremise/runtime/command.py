from __future__ import annotations

import argparse
import sys

from laris_onpremise.runtime.elf import (
    ElfInspectionError,
    ProcessInspectionError,
    inspect_single_mapped_libllama,
    read_process_identity,
)
from laris_onpremise.runtime.provenance import (
    ProvenanceError,
    build_target_projection,
    canonical_json_bytes,
    collect_quiescent_snapshot,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect provenance for a live llama.cpp runtime."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    inspect_target = commands.add_parser(
        "inspect-target",
        help="Inspect one live process and its mapped libllama ELF.",
    )
    inspect_target.add_argument("--pid", required=True, type=int)
    quiescent = commands.add_parser(
        "snapshot-quiescent",
        help="Capture an address-free snapshot of an already stopped target.",
    )
    quiescent.add_argument("--pid", required=True, type=int)
    return parser


def _inspect_target(pid: int) -> dict[str, object]:
    process = read_process_identity(pid)
    library = inspect_single_mapped_libllama(pid)
    return {
        "schema_version": "1.0",
        "kind": "runtime_observer_target_inspection",
        "target": build_target_projection(process, library),
    }


def _snapshot_quiescent(pid: int) -> dict[str, object]:
    process = read_process_identity(pid)
    library = inspect_single_mapped_libllama(pid)
    snapshot = collect_quiescent_snapshot(
        pid,
        expected_process=process,
        expected_libllama=library,
    )
    return {
        "schema_version": "1.0",
        "kind": "runtime_observer_quiescent_snapshot",
        "target": build_target_projection(process, library),
        **snapshot.to_mapping(),
    }


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "inspect-target":
            result = _inspect_target(arguments.pid)
        else:
            result = _snapshot_quiescent(arguments.pid)
    except (ElfInspectionError, ProcessInspectionError, ProvenanceError, OSError) as error:
        print(f"runtime-observer: {error}", file=sys.stderr)
        return 2
    sys.stdout.buffer.write(canonical_json_bytes(result) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
