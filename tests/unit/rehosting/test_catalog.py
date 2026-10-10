from __future__ import annotations

import pytest

from laris_onpremise.rehosting.catalog import (
    PortableBundleError,
    _validate_package,
)


def test_bundle_manifest_without_rehost_metadata_is_rejected() -> None:
    digest = "sha256:" + "a" * 64
    package = {
        "schema_version": "1.0",
        "package_type": "laris_rehost_bundle",
        "run_id": "run-evaluation1-test",
        "source": {
            "evidence_root_digest": digest,
            "gate_a_result_digest": digest,
        },
        "bundle_index": {
            "path": "bundle_index.json",
            "digest": digest,
        },
    }

    with pytest.raises(PortableBundleError, match="completion manifest"):
        _validate_package(package)
