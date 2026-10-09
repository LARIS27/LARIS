from __future__ import annotations

import argparse
import json
from pathlib import Path

from laris_onpremise.reproduction.evaluation1.locations import (
    load_artifact_locations,
    verify_external_files,
)
from laris_onpremise.reproduction.evaluation1.verifier import verify_public_release


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify an external Evaluation1 KV release without extracting it."
    )
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--locations", type=Path)
    parser.add_argument(
        "--license-registry",
        type=Path,
        default=Path("evaluation/experiment_setup/model_licenses.yaml"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.locations is not None:
        locations = load_artifact_locations(args.locations)
        verify_external_files(locations, args.release_root.resolve())
    receipt = verify_public_release(args.release_root, args.license_registry)
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
