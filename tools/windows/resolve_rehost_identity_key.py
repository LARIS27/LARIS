from __future__ import annotations

from pathlib import Path
import sys

from laris_onpremise.rehosting.catalog import PortableBundleCatalog
from laris_onpremise.rehosting.profile_command import required_identity_key_id


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 1:
        print("usage: resolve_rehost_identity_key.py BUNDLE_ROOT", file=sys.stderr)
        return 2
    try:
        catalog = PortableBundleCatalog.open(Path(arguments[0]).resolve())
        identity_key_id = required_identity_key_id(catalog)
    except Exception as error:
        print(f"rehost identity resolution failed: {error}", file=sys.stderr)
        return 2
    print(identity_key_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
