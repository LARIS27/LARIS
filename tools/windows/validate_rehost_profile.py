from __future__ import annotations

import argparse
from pathlib import Path

from laris_onpremise.rehosting.catalog import PortableBundleCatalog
from laris_onpremise.rehosting.host_profile import InvestigatorProfile
from laris_onpremise.rehosting.identity import resolve_subject_sessions
from laris_onpremise.rehosting.runtime_catalog import resolve_model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--subject-id")
    arguments = parser.parse_args()

    profile = InvestigatorProfile.load(arguments.profile.resolve())
    catalog = PortableBundleCatalog.open(arguments.bundle_root.resolve())
    if arguments.subject_id is None:
        matches = catalog.entries
        for identity_key_id in sorted(
            {entry.identity_key_id for entry in catalog.entries}
        ):
            profile.identity_registry.resolve(identity_key_id)
    else:
        matches = resolve_subject_sessions(
            catalog,
            tenant_id=profile.tenant_id,
            registry=profile.identity_registry,
            raw_subject_id=arguments.subject_id,
        )
    models = {resolve_model(profile, entry).resolve() for entry in matches}
    if len(models) != 1:
        raise ValueError("selected subject resolves to multiple local models")
    print(
        "LARIS_REHOST_PROFILE_OK "
        f"entries={len(catalog.entries)} matches={len(matches)} model={models.pop()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
