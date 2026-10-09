from __future__ import annotations

import argparse
import json
from pathlib import Path

from laris_onpremise.reproduction.evaluation1.packager import (
    audit_release_sources_from_spec,
    prepare_release_from_spec,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the curated external Evaluation1 KV release from sealed sources."
    )
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--destination", type=Path)
    parser.add_argument(
        "--inventory-only",
        action="store_true",
        help="Verify the four sealed source paths and digests without writing a release.",
    )
    parser.add_argument(
        "--license-registry",
        type=Path,
        default=Path("evaluation/experiment_setup/model_licenses.yaml"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.inventory_only:
        receipt = audit_release_sources_from_spec(args.spec)
    else:
        if args.destination is None:
            build_parser().error("--destination is required unless --inventory-only is used")
        receipt = prepare_release_from_spec(
            args.spec,
            args.destination,
            args.license_registry,
        )
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
