from __future__ import annotations

from pathlib import Path

import pytest

from laris_onpremise.core.records import canonical_json, derive_user_hmac
from laris_onpremise.rehosting.host_profile import (
    InvestigatorProfile,
    InvestigatorProfileError,
)


def _profile_document(tmp_path: Path) -> dict[str, object]:
    key = (tmp_path / "identity-key-v1.bin").resolve()
    key.write_bytes(b"K" * 32)
    models = (tmp_path / "models").resolve()
    models.mkdir()
    analysis = (tmp_path / "analyses").resolve()
    return {
        "analysis_root": str(analysis),
        "identity_keys": [
            {"identity_key_id": "identity-key-v1", "secret_file": str(key)}
        ],
        "model_roots": [str(models)],
        "schema_version": "1.0",
        "tenant_id": "lab-tenant",
        "wsl": {
            "distribution": "Ubuntu-24.04",
            "helper": "/opt/laris/native/llama-runtime-payload-rehost",
            "helper_digest": "sha256:" + "a" * 64,
            "libllama": "/opt/laris/lib/libllama.so",
            "python": "/usr/bin/python3",
            "runtime_profile": "/etc/laris/rehost-runtime-profile.json",
            "work_root": "/var/tmp/laris-rehost",
        },
    }


def _write_profile(tmp_path: Path, document: dict[str, object]) -> Path:
    path = (tmp_path / "investigator-profile.json").resolve()
    path.write_bytes(canonical_json(document) + b"\n")
    return path


def test_profile_loads_external_identity_key_without_serializing_it(
    tmp_path: Path,
) -> None:
    profile = InvestigatorProfile.load(
        _write_profile(tmp_path, _profile_document(tmp_path))
    )

    assert profile.tenant_id == "lab-tenant"
    assert profile.model_roots == ((tmp_path / "models").resolve(),)
    assert profile.wsl.distribution == "Ubuntu-24.04"
    assert "KKKK" not in repr(profile)
    assert len(
        derive_user_hmac(
            profile.identity_registry,
            "identity-key-v1",
            profile.tenant_id,
            "U01",
        )
    ) == 43


def test_profile_load_with_bytes_returns_the_same_validated_canonical_document(
    tmp_path: Path,
) -> None:
    document = _profile_document(tmp_path)
    path = _write_profile(tmp_path, document)

    profile, raw = InvestigatorProfile.load_with_bytes(path)

    assert profile.tenant_id == "lab-tenant"
    assert raw == canonical_json(document) + b"\n"


def test_profile_rejects_a_non_32_byte_identity_key(tmp_path: Path) -> None:
    document = _profile_document(tmp_path)
    key_path = Path(document["identity_keys"][0]["secret_file"])
    key_path.write_bytes(b"short")

    with pytest.raises(InvestigatorProfileError, match="identity key"):
        InvestigatorProfile.load(_write_profile(tmp_path, document))
