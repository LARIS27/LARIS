from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from laris_onpremise.reproduction.evaluation1.contracts import (
    CaptureMode,
    ContractError,
    PublicBundleEntry,
    PublicBundleIndex,
    load_model_license_registry,
)


MODEL_ID = "llama-3.2-3b-instruct-q4-k-m"
DIGEST = "sha256:" + ("a" * 64)


def _entry(checkpoint: int, user: int) -> PublicBundleEntry:
    user_ref = f"U{user:02d}"
    return PublicBundleEntry(
        model_id=MODEL_ID,
        mode=CaptureMode.PROXY,
        checkpoint_id=f"C{checkpoint}",
        public_user_ref=user_ref,
        subject_ref_kind="USER_HMAC",
        subject_ref=f"user-hmac-{user_ref}",
        key_id="identity-epoch-example",
        session_ref=f"session-{checkpoint}-{user_ref}",
        bundle_id=f"bundle-{checkpoint}-{user_ref}",
        bundle_path=f"checkpoints/C{checkpoint}/users/{user_ref}/bundle",
        payload_sha256=DIGEST,
        payload_size_bytes=1024,
        source_bundle_manifest_sha256=DIGEST,
    )


def _complete_index() -> PublicBundleIndex:
    return PublicBundleIndex(
        schema_version="1.0",
        release_id="evaluation1-llama-proxy-v1",
        model_id=MODEL_ID,
        mode=CaptureMode.PROXY,
        subject_ref_kind="USER_HMAC",
        entries=tuple(
            _entry(checkpoint, user)
            for checkpoint in range(1, 7)
            for user in range(1, 11)
        ),
    )


def test_accepts_complete_sixty_entry_index() -> None:
    index = _complete_index()

    assert len(index.entries) == 60
    assert index.entries[-1].tuple_key == (MODEL_ID, "PROXY", "C6", "U10")


def test_rejects_duplicate_model_mode_checkpoint_user_tuple() -> None:
    index = _complete_index()

    with pytest.raises(ContractError, match="duplicate bundle tuple"):
        replace(index, entries=index.entries[:-1] + (index.entries[0],))


def test_rejects_subject_kind_that_does_not_match_mode() -> None:
    with pytest.raises(ContractError, match="subject_ref_kind"):
        replace(_entry(1, 1), subject_ref_kind="ENDPOINT_REF")


def test_rejects_zero_sized_bundle() -> None:
    with pytest.raises(ContractError, match="payload_size_bytes"):
        replace(_entry(1, 1), payload_size_bytes=0)


@pytest.mark.parametrize(
    "unsafe_path",
    (
        "../bundle",
        "/checkpoints/C1/users/U01/bundle",
        "C:\\bundles\\bundle",
    ),
)
def test_rejects_unsafe_relative_bundle_path(unsafe_path: str) -> None:
    with pytest.raises(ContractError, match="bundle_path"):
        replace(_entry(1, 1), bundle_path=unsafe_path)


def test_model_license_registry_contains_only_required_pinned_records() -> None:
    root = Path(__file__).resolve().parents[3]
    records = load_model_license_registry(
        root / "evaluation" / "experiment_setup" / "model_licenses.yaml"
    )

    assert set(records) == {
        "llama-3.2-3b-instruct-q4-k-m",
        "qwen2.5-7b-instruct-q4-k-m",
    }
    llama = records["llama-3.2-3b-instruct-q4-k-m"]
    assert llama.source_revision == "0cb88a4f764b7a12671c53f0838cd831a0843b95"
    assert llama.license_id == "llama-3.2-community"
    assert "Built with Llama" in llama.required_notices
    assert llama.redistributes_weights is False

    qwen = records["qwen2.5-7b-instruct-q4-k-m"]
    assert qwen.source_revision == "bb5d59e06d9551d752d08b292a50eb208b07ab1f"
    assert qwen.license_id == "Apache-2.0"
    assert qwen.redistributes_weights is False


def test_public_schemas_exist_and_are_nonempty() -> None:
    root = Path(__file__).resolve().parents[3]
    for filename in (
        "evaluation1_public_bundle_index.schema.json",
        "evaluation1_release_manifest.schema.json",
        "evaluation1_reproduction_receipt.schema.json",
    ):
        path = root / "schemas" / filename
        assert path.is_file()
        assert path.stat().st_size > 100
