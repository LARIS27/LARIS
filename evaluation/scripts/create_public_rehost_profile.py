from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path
import tempfile

from laris_onpremise.rehosting.profile_command import create_main
from laris_onpremise.reproduction.evaluation1.resolver import Evaluation1Release


def _release_configuration(release_root: Path) -> tuple[str, list[str]]:
    fixtures = release_root / "fixtures"
    identities = json.loads(
        (fixtures / "identity-fixtures.json").read_text(encoding="utf-8")
    )
    provenance = json.loads(
        (fixtures / "key-provenance.json").read_text(encoding="utf-8")
    )
    tenant_id = identities.get("tenant_id")
    if not isinstance(tenant_id, str) or not tenant_id:
        raise ValueError("public identity fixture has no tenant_id")
    keys: list[str] = []
    for item in provenance.get("keys", []):
        key_id = item.get("key_id")
        key_path = item.get("key_path")
        if not isinstance(key_id, str) or not isinstance(key_path, str):
            raise ValueError("public key provenance is invalid")
        resolved = (release_root / Path(key_path)).resolve()
        if resolved.parent.parent != fixtures.resolve() or resolved.parent.name != "keys":
            raise ValueError("public key path leaves fixtures/keys")
        keys.append(f"{key_id}={resolved}")
    if len(keys) != 2:
        raise ValueError("public release must contain exactly two replay keys")
    return tenant_id, sorted(keys)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a local investigator profile from a verified public release."
    )
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--wsl-distribution", required=True)
    parser.add_argument("--wsl-python", required=True)
    parser.add_argument("--wsl-runtime-profile", required=True)
    parser.add_argument("--wsl-helper", required=True)
    parser.add_argument("--wsl-helper-digest", required=True)
    parser.add_argument("--wsl-libllama", required=True)
    parser.add_argument("--wsl-work-root", required=True)
    parser.add_argument("--replace", action="store_true")
    arguments = parser.parse_args()

    release_root = arguments.release_root.resolve()
    release = Evaluation1Release.open(release_root)
    tenant_id, identity_keys = _release_configuration(release_root)
    with tempfile.TemporaryDirectory(prefix="laris-public-profile-") as temporary:
        bundle_root = Path(temporary) / "bundle"
        release.materialize(
            arguments.model_id,
            "PROXY",
            "C1",
            "U01",
            bundle_root,
        )
        return create_main(
            Namespace(
                bundle_root=bundle_root,
                profile=arguments.profile.resolve(),
                tenant_id=tenant_id,
                identity_key=identity_keys,
                model_root=[arguments.model_root.resolve()],
                analysis_root=arguments.analysis_root.resolve(),
                wsl_distribution=arguments.wsl_distribution,
                wsl_python=arguments.wsl_python,
                wsl_runtime_profile=arguments.wsl_runtime_profile,
                wsl_helper=arguments.wsl_helper,
                wsl_helper_digest=arguments.wsl_helper_digest,
                wsl_libllama=arguments.wsl_libllama,
                wsl_work_root=arguments.wsl_work_root,
                preserve_existing_identity_keys=False,
                replace=arguments.replace,
                json=True,
            )
        )


if __name__ == "__main__":
    raise SystemExit(main())
