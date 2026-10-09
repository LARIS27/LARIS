from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from laris_onpremise.runtime.preflight import (
    HostProfile,
    collect_host_preflight,
    require_qualified_host,
)
from laris_onpremise.runtime.publication import (
    open_trusted_directory,
    publish_noreplace,
)


def build_parser() -> argparse.ArgumentParser:
    profiles = {profile.profile_id for profile in HostProfile.known_profiles()}
    parser = argparse.ArgumentParser(
        description="Collect and atomically publish one qualified host preflight."
    )
    parser.add_argument("--profile", required=True, choices=sorted(profiles))
    parser.add_argument("--output", type=Path, required=True)
    return parser


def run(arguments: argparse.Namespace) -> int:
    profiles = {
        profile.profile_id: profile for profile in HostProfile.known_profiles()
    }
    profile = profiles[arguments.profile]
    output = arguments.output.resolve()
    if not output.parent.is_dir():
        raise ValueError("preflight output parent must already exist")
    host = collect_host_preflight(profile)
    require_qualified_host(profile, host)
    descriptor = open_trusted_directory(output.parent)
    try:
        published = publish_noreplace(
            descriptor,
            output.name,
            host.canonical_bytes() + b"\n",
            mode=0o600,
        )
    finally:
        os.close(descriptor)
    print(
        json.dumps(
            {
                "profile_id": profile.profile_id,
                "output": str(output),
                "payload_digest": published.payload_digest,
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run(build_parser().parse_args(argv))
    except Exception as error:
        print(f"host-preflight: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
