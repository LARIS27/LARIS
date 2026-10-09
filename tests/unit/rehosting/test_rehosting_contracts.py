from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

import laris_onpremise.rehosting as rehosting
from laris_onpremise.rehosting.contracts import (
    HelperReceipt,
    InvestigatorSampler,
    RehostCase,
    RehostRuntimeProfile,
    RehostValidationError,
)


def _digest(payload: bytes = b"payload") -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def test_runtime_profile_is_available_from_rehosting_public_api() -> None:
    assert rehosting.RehostRuntimeProfile is RehostRuntimeProfile


def test_investigator_sampler_has_explicit_reproducible_defaults() -> None:
    sampler = InvestigatorSampler()

    assert sampler.to_mapping() == {
        "temperature": 0.8,
        "top_k": 40,
        "top_p": 0.95,
        "min_p": 0.05,
        "seed": 0,
    }


def _case(tmp_path: Path, **changes: object) -> RehostCase:
    payload = tmp_path / "runtime_state_payload.bin"
    payload.write_bytes(b"payload")
    values: dict[str, object] = {
        "run_id": "run-20260814-204253-KST_qwen_c5-attribution_abcd",
        "bundle_id": "rb_" + "a" * 64,
        "user_hmac": "A" * 43,
        "identity_key_id": "identity-key-v1",
        "session_ref": "sr_" + "b" * 64,
        "payload_path": payload,
        "payload_digest": _digest(),
        "payload_size_bytes": 7,
        "next_position": 2,
        "destination_seq_id": 1,
        "model_profile_id": "qwen2.5-0.5b-q4",
        "model_digest": "sha256:" + "c" * 64,
        "runtime_profile_digest": "sha256:" + "d" * 64,
        "memory_family": "simple_kv",
        "n_ctx": 32768,
        "n_seq_max": 4,
        "n_stream": 4,
        "n_layer": 2,
        "v_trans": True,
        "key_ggml_type_id": 1,
        "value_ggml_type_id": 1,
        "source_seq_id": 2,
        "source_stream_index": 1,
        "position_range": (0, 1),
    }
    values.update(changes)
    return RehostCase(**values)  # type: ignore[arg-type]


def test_rehost_case_round_trip_contains_only_runtime_restore_metadata(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)

    mapping = case.to_mapping()

    assert mapping["payload_path"].endswith("runtime_state_payload.bin")
    assert mapping["destination_seq_id"] == 1
    assert "checkpoint_path" not in mapping
    assert "actor_ref" not in mapping
    assert "prompt" not in mapping
    assert "token_history" not in mapping


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("user_hmac", "identity@example.invalid"),
        ("payload_digest", "bad"),
        ("destination_seq_id", 4),
        ("n_seq_max", 5),
        ("position_range", (1, 2)),
        ("next_position", 3),
    ],
)
def test_rehost_case_rejects_incompatible_or_identifying_values(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    with pytest.raises(RehostValidationError):
        _case(tmp_path, **{field: value})


def test_helper_receipt_has_no_loaded_token_history_field() -> None:
    receipt = HelperReceipt(
        exit_code=0,
        response="answer",
        stderr="",
        mapped_libllama_device=1,
        mapped_libllama_inode=2,
        restored_bytes=128,
        restored_pos_min=0,
        restored_pos_max=7,
        appended_token_count=5,
        logical_batch_count=1,
        effective_n_batch=128,
        effective_n_ubatch=128,
        generated_token_count=3,
        prompt_contract="gguf_default",
        sampler=InvestigatorSampler(),
        duration_ms=12,
    )

    mapping = receipt.to_mapping()

    assert mapping["restored_pos_max"] == 7
    assert "loaded_token_count" not in mapping


def test_runtime_profile_binds_payload_topology_and_runtime_artifacts(
    tmp_path: Path,
) -> None:
    profile = RehostRuntimeProfile(
        profile_digest="sha256:" + "d" * 64,
        runtime_binary_digest="sha256:" + "e" * 64,
        mapped_libllama_digest="sha256:" + "f" * 64,
        mapped_libllama_elf_build_id="1" * 40,
        n_seq_max=4,
        n_stream=4,
        n_layer=2,
        v_trans=True,
        key_ggml_type_id=1,
        value_ggml_type_id=1,
    )

    profile.verify_case(_case(tmp_path))

    with pytest.raises(RehostValidationError, match="v_trans"):
        profile.verify_case(_case(tmp_path, v_trans=False))


def test_dynamic_runtime_profile_binds_observed_bundle_topology_within_limits(
    tmp_path: Path,
) -> None:
    profile = RehostRuntimeProfile(
        profile_digest="sha256:" + "d" * 64,
        runtime_binary_digest="sha256:" + "e" * 64,
        mapped_libllama_digest="sha256:" + "f" * 64,
        mapped_libllama_elf_build_id="1" * 40,
        n_seq_max=4,
        n_stream=4,
        n_layer=0,
        v_trans=None,
        key_ggml_type_id=1,
        value_ggml_type_id=1,
        max_n_layer=512,
    )

    profile.verify_case(_case(tmp_path, n_layer=24, v_trans=True))

    with pytest.raises(RehostValidationError, match="n_layer"):
        profile.verify_case(_case(tmp_path, n_layer=513, v_trans=True))


def test_runtime_profile_digest_is_capture_provenance_not_local_identity(
    tmp_path: Path,
) -> None:
    profile = RehostRuntimeProfile(
        profile_digest="sha256:" + "8" * 64,
        runtime_binary_digest="sha256:" + "e" * 64,
        mapped_libllama_digest="sha256:" + "f" * 64,
        mapped_libllama_elf_build_id="1" * 40,
        n_seq_max=4,
        n_stream=4,
        n_layer=2,
        v_trans=True,
        key_ggml_type_id=1,
        value_ggml_type_id=1,
    )

    profile.verify_case(_case(tmp_path))
