from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sys

from laris_onpremise.core.records import canonical_json


_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


def _receipt(path: Path) -> tuple[dict[str, object], bytes]:
    raw = path.read_bytes()
    if len(raw) < 2 or len(raw) > 1024 * 1024:
        raise ValueError("installation receipt size is invalid")

    def no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("installation receipt contains a duplicate key")
            result[key] = value
        return result

    value = json.loads(raw.decode("ascii"), object_pairs_hook=no_duplicates)
    if not isinstance(value, dict) or canonical_json(value) + b"\n" != raw:
        raise ValueError("installation receipt is not canonical")
    if value.get("schema_version") != "1.0" or value.get("phase") != "COMPLETE":
        raise ValueError("installation receipt is not COMPLETE")
    identity_key_id = value.get("identity_key_id")
    fingerprint = value.get("identity_key_fingerprint")
    if (
        not isinstance(identity_key_id, str)
        or _IDENTIFIER.fullmatch(identity_key_id) is None
        or not isinstance(fingerprint, str)
        or _DIGEST.fullmatch(fingerprint) is None
    ):
        raise ValueError("installation receipt identity binding is invalid")
    return value, raw


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 1:
        print(
            "usage: resolve_rehost_identity_requirements.py INSTALLATION_RECEIPT",
            file=sys.stderr,
        )
        return 2
    try:
        receipt, raw = _receipt(Path(arguments[0]).resolve())
        document = {
            "identity_key_id": receipt["identity_key_id"],
            "installation_receipt_sha256": "sha256:"
            + hashlib.sha256(raw).hexdigest(),
        }
    except Exception as error:
        print(f"rehost identity requirement resolution failed: {error}", file=sys.stderr)
        return 2
    sys.stdout.buffer.write(canonical_json(document) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
