from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path, PurePosixPath
import re
import sys

from laris_onpremise.core.records import canonical_json
from laris_onpremise.rehosting.catalog import PortableBundleCatalog
from laris_onpremise.rehosting.profile_command import required_identity_key_id


_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


def _canonical_receipt(path: Path) -> tuple[dict[str, object], bytes]:
    raw = path.read_bytes()
    if len(raw) < 2 or len(raw) > 1024 * 1024:
        raise ValueError("installation receipt size is invalid")

    def no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("installation receipt contains a duplicate key")
            value[key] = item
        return value

    value = json.loads(raw.decode("ascii"), object_pairs_hook=no_duplicates)
    if not isinstance(value, dict) or canonical_json(value) + b"\n" != raw:
        raise ValueError("installation receipt is not canonical")
    if value.get("schema_version") != "1.0" or value.get("phase") != "COMPLETE":
        raise ValueError("installation receipt is not COMPLETE")
    return value, raw


def _components(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list) or len(value) < 3:
        raise ValueError("installation receipt rehost components are invalid")
    results: list[dict[str, object]] = []
    paths: set[str] = set()
    roles: list[str] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {
            "canonical_path",
            "role",
            "sha256",
            "size_bytes",
        }:
            raise ValueError("installation receipt rehost component is invalid")
        role = item["role"]
        path = item["canonical_path"]
        digest = item["sha256"]
        size = item["size_bytes"]
        if role not in {"helper", "runtime_profile", "shared_library"}:
            raise ValueError("installation receipt rehost component role is invalid")
        if (
            not isinstance(path, str)
            or not path.startswith("/opt/laris/")
            or ".." in PurePosixPath(path).parts
            or path in paths
        ):
            raise ValueError("installation receipt rehost component path is invalid")
        if not isinstance(digest, str) or _DIGEST.fullmatch(digest) is None:
            raise ValueError("installation receipt rehost component digest is invalid")
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise ValueError("installation receipt rehost component size is invalid")
        paths.add(path)
        roles.append(role)
        results.append(dict(item))
    if roles.count("helper") != 1 or roles.count("runtime_profile") != 1:
        raise ValueError("installation receipt rehost component set is invalid")
    if "shared_library" not in roles:
        raise ValueError("installation receipt has no rehost libraries")
    return results


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 2:
        print(
            "usage: resolve_rehost_bundle_requirements.py BUNDLE_ROOT INSTALLATION_RECEIPT",
            file=sys.stderr,
        )
        return 2
    try:
        catalog = PortableBundleCatalog.open(Path(arguments[0]).resolve())
        receipt, receipt_raw = _canonical_receipt(Path(arguments[1]).resolve())
        model_bindings = {
            (entry.model_profile_id, entry.model_digest)
            for entry in catalog.entries
        }
        if len(model_bindings) != 1:
            raise ValueError("bundle must require a single model")
        model_profile_id, model_digest = next(iter(model_bindings))
        models = receipt.get("models")
        if not isinstance(models, list):
            raise ValueError("installation receipt model inventory is invalid")
        matches = [
            item
            for item in models
            if isinstance(item, dict)
            and item.get("model_profile_id") == model_profile_id
            and item.get("gguf_digest") == model_digest
        ]
        if len(matches) != 1:
            raise ValueError("bundle model is not uniquely installed on the selected node")
        model = matches[0]
        filename = model.get("filename")
        if (
            not isinstance(filename, str)
            or not filename.endswith(".gguf")
            or PurePosixPath(filename).name != filename
        ):
            raise ValueError("installation receipt model filename is invalid")
        identity_key_id = required_identity_key_id(catalog)
        receipt_key_id = receipt.get("identity_key_id")
        if (
            not isinstance(receipt_key_id, str)
            or _IDENTIFIER.fullmatch(receipt_key_id) is None
            or not hmac.compare_digest(identity_key_id, receipt_key_id)
        ):
            raise ValueError("bundle identity key does not match the selected node")
        document = {
            "identity_key_id": identity_key_id,
            "installation_receipt_sha256": "sha256:"
            + hashlib.sha256(receipt_raw).hexdigest(),
            "model_digest": model_digest,
            "model_filename": filename,
            "model_profile_id": model_profile_id,
            "rehost_components": _components(receipt.get("rehost_components")),
        }
    except Exception as error:
        print(f"rehost requirement resolution failed: {error}", file=sys.stderr)
        return 2
    sys.stdout.buffer.write(canonical_json(document) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
