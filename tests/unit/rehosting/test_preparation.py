from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path, PurePosixPath

import pytest

from laris_onpremise.acquisition.layout import plan_selected_rows
from laris_onpremise.acquisition.payload import encode_simple_kv_payload
from laris_onpremise.bundle.rehost_export import derive_rehost_bundle_id
from laris_onpremise.rehosting.preparation import (
    RehostPreparationError,
    prepare_rehost_cases,
)
from laris_onpremise.rehosting.contracts import RehostRuntimeProfile
from tests.fixtures.simple_kv_payload_fixtures import (
    rows_for,
    selected_cells,
    two_layer_layout,
)


RUN_ID = "run-20260814-204253-KST_qwen_c5-attribution_abcd"
MODEL_PROFILE_ID = "qwen2.5-0.5b-q4"
MODEL_DIGEST = "sha256:" + "a" * 64
RUNTIME_PROFILE_DIGEST = "sha256:" + "b" * 64


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _runtime_profile() -> RehostRuntimeProfile:
    return RehostRuntimeProfile(
        profile_digest=RUNTIME_PROFILE_DIGEST,
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


def _write_bundle(
    run_root: Path,
    *,
    user_hmac: str = "A" * 43,
    session_ref: str = "sr_" + "1" * 64,
    source_seq_id: int = 2,
    source_stream_index: int = 1,
) -> tuple[Path, Path]:
    layout = two_layer_layout(v_trans=True)
    cells = tuple(
        replace(
            cell,
            seq_id=source_seq_id,
            stream_index=source_stream_index,
            position=index,
        )
        for index, cell in enumerate(selected_cells())
    )
    spans = plan_selected_rows(layout, cells)
    payload = encode_simple_kv_payload(layout, cells, rows_for(spans))
    session_digest = hashlib.sha256(session_ref.encode()).hexdigest()[:16]
    relative_root = PurePosixPath(
        "rehost", f"uh_{user_hmac}", f"s_{session_digest}"
    )
    bundle_root = run_root / Path(*relative_root.parts)
    bundle_root.mkdir(parents=True)
    payload_path = bundle_root / "runtime_state_payload.bin"
    payload_path.write_bytes(payload)
    manifest = {
        "schema_version": "1.0",
        "run_id": RUN_ID,
        "bundle_id": "",
        "user_hmac": user_hmac,
        "identity_key_id": "identity-key-v1",
        "session_ref": session_ref,
        "application_context_mode": "cumulative_context",
        "last_turn_index": 5,
        "source_request_ref": "req-u01-t5",
        "source_lineage_digest": "sha256:" + "2" * 64,
        "model_profile_id": MODEL_PROFILE_ID,
        "model_digest": MODEL_DIGEST,
        "runtime_profile_digest": RUNTIME_PROFILE_DIGEST,
        "memory_family": "simple_kv",
        "n_ctx": 32768,
        "n_seq_max": 4,
        "n_stream": 4,
        "n_layer": 2,
        "v_trans": True,
        "key_ggml_type_id": 1,
        "value_ggml_type_id": 1,
        "source_seq_id": source_seq_id,
        "source_stream_index": source_stream_index,
        "destination_sequence_policy": "caller_selected_empty_sequence",
        "position_range": [0, 1],
        "kv_cell_ranges": [[1, 1], [6, 6]],
        "next_position": 2,
        "payload_format": "llama-simple-kv-seq-payload-v1",
        "payload_file": "runtime_state_payload.bin",
        "payload_size_bytes": len(payload),
        "payload_digest": _sha256(payload),
        "acquisition_manifest_digest": "sha256:" + "3" * 64,
        "gate_a_result_digest": "sha256:" + "4" * 64,
        "evidence_root_digest": "sha256:" + "5" * 64,
        "contains_raw_prompt": False,
        "contains_raw_response": False,
        "contains_explicit_token_history": False,
        "restore_api": "llama_state_seq_set_data_ext",
    }
    bundle_id = derive_rehost_bundle_id(
        run_id=RUN_ID,
        user_hmac=user_hmac,
        identity_key_id="identity-key-v1",
        session_ref=session_ref,
        last_turn_index=5,
        source_request_ref="req-u01-t5",
        source_lineage_digest="sha256:" + "2" * 64,
        model_profile_id=MODEL_PROFILE_ID,
        model_digest=MODEL_DIGEST,
        runtime_profile_digest=RUNTIME_PROFILE_DIGEST,
        payload_digest=_sha256(payload),
        acquisition_manifest_digest="sha256:" + "3" * 64,
        gate_a_digest="sha256:" + "4" * 64,
        evidence_root_digest="sha256:" + "5" * 64,
    )
    manifest["bundle_id"] = bundle_id
    manifest_path = bundle_root / "rehost_manifest.json"
    manifest_bytes = _canonical(manifest)
    manifest_path.write_bytes(manifest_bytes)
    index = {
        "schema_version": "1.0",
        "run_id": RUN_ID,
        "entries": [
            {
                "bundle_id": bundle_id,
                "user_hmac": user_hmac,
                "session_ref": session_ref,
                "state": "READY",
                "rehost_manifest_path": str(
                    relative_root / "rehost_manifest.json"
                ),
                "rehost_manifest_digest": _sha256(manifest_bytes),
                "payload_path": str(
                    relative_root / "runtime_state_payload.bin"
                ),
                "payload_digest": _sha256(payload),
                "payload_size_bytes": len(payload),
            }
        ],
    }
    (run_root / "rehost_bundle_index.json").write_bytes(_canonical(index))
    return payload_path, manifest_path


def _prepare(run_root: Path, **changes: object):
    index_bytes = (run_root / "rehost_bundle_index.json").read_bytes()
    values: dict[str, object] = {
        "user_hmac": "A" * 43,
        "session_ref": None,
        "bundle_id": None,
        "destination_seq_id": 1,
        "expected_model_profile_id": MODEL_PROFILE_ID,
        "expected_model_digest": MODEL_DIGEST,
        "expected_runtime_profile": _runtime_profile(),
        "expected_index_digest": _sha256(index_bytes),
    }
    values.update(changes)
    return prepare_rehost_cases(run_root, **values)  # type: ignore[arg-type]


def test_prepares_verified_payload_from_authoritative_completion_index(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / RUN_ID
    payload_path, _ = _write_bundle(run_root)

    cases = _prepare(run_root)

    assert len(cases) == 1
    assert cases[0].payload_path == payload_path
    assert cases[0].destination_seq_id == 1
    assert cases[0].next_position == 2
    assert cases[0].position_range == (0, 1)


def test_user_with_multiple_sessions_requires_explicit_session(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / RUN_ID
    _write_bundle(run_root)
    first_index = json.loads(
        (run_root / "rehost_bundle_index.json").read_text(encoding="utf-8")
    )
    (run_root / "rehost_bundle_index.json").unlink()
    second_session = "sr_" + "2" * 64
    _write_bundle(run_root, session_ref=second_session)
    second_index = json.loads(
        (run_root / "rehost_bundle_index.json").read_text(encoding="utf-8")
    )
    second_index["entries"] = first_index["entries"] + second_index["entries"]
    (run_root / "rehost_bundle_index.json").write_bytes(_canonical(second_index))

    with pytest.raises(RehostPreparationError, match="multiple sessions"):
        _prepare(run_root)

    cases = _prepare(run_root, session_ref=second_session)
    assert len(cases) == 1
    assert cases[0].session_ref == second_session


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("contains_raw_prompt", True, "privacy"),
        ("restore_api", "llama_state_seq_load_file", "restore"),
        ("model_digest", "sha256:" + "9" * 64, "model"),
    ],
)
def test_manifest_policy_or_compatibility_mismatch_is_rejected(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    run_root = tmp_path / RUN_ID
    _, manifest_path = _write_bundle(run_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = value
    manifest_bytes = _canonical(manifest)
    manifest_path.write_bytes(manifest_bytes)
    index_path = run_root / "rehost_bundle_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["entries"][0]["rehost_manifest_digest"] = _sha256(manifest_bytes)
    index_path.write_bytes(_canonical(index))

    with pytest.raises(RehostPreparationError, match=message):
        _prepare(run_root)


def test_payload_replacement_and_extra_bundle_file_are_fail_closed(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / RUN_ID
    payload_path, _ = _write_bundle(run_root)
    payload_path.write_bytes(b"replacement")
    with pytest.raises(RehostPreparationError, match="payload"):
        _prepare(run_root)

    payload_path, _ = _reset_run(tmp_path)
    (payload_path.parent / "raw_prompt.txt").write_text("secret", encoding="utf-8")
    with pytest.raises(RehostPreparationError, match="exactly"):
        _prepare(payload_path.parents[3])


def _reset_run(tmp_path: Path) -> tuple[Path, Path]:
    run_root = tmp_path / RUN_ID
    for path in sorted(run_root.rglob("*"), reverse=True):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            path.rmdir()
    run_root.rmdir()
    return _write_bundle(run_root)


def test_index_path_escape_is_rejected_before_artifact_read(tmp_path: Path) -> None:
    run_root = tmp_path / RUN_ID
    _write_bundle(run_root)
    index_path = run_root / "rehost_bundle_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["entries"][0]["payload_path"] = "../runtime_state_payload.bin"
    index_path.write_bytes(_canonical(index))

    with pytest.raises(RehostPreparationError, match="path"):
        _prepare(run_root)


def test_loader_accepts_collision_extended_session_digest_prefix(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / RUN_ID
    payload_path, _ = _write_bundle(run_root)
    session_ref = "sr_" + "1" * 64
    extended = hashlib.sha256(session_ref.encode()).hexdigest()[:20]
    old_root = payload_path.parent
    new_root = old_root.parent / f"s_{extended}"
    old_root.rename(new_root)
    index_path = run_root / "rehost_bundle_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["entries"][0]["rehost_manifest_path"] = (
        f"rehost/uh_{'A' * 43}/s_{extended}/rehost_manifest.json"
    )
    index["entries"][0]["payload_path"] = (
        f"rehost/uh_{'A' * 43}/s_{extended}/runtime_state_payload.bin"
    )
    index_path.write_bytes(_canonical(index))

    cases = _prepare(run_root)

    assert cases[0].payload_path.parent == new_root


def test_manifest_attribution_change_is_rejected_by_anchored_index(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / RUN_ID
    _, manifest_path = _write_bundle(run_root)
    anchored_index_digest = _sha256(
        (run_root / "rehost_bundle_index.json").read_bytes()
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["identity_key_id"] = "attacker-key"
    manifest_path.write_bytes(_canonical(manifest))

    with pytest.raises(RehostPreparationError, match="manifest digest|identity"):
        prepare_rehost_cases(
            run_root,
            user_hmac="A" * 43,
            session_ref=None,
            bundle_id=None,
            destination_seq_id=1,
            expected_model_profile_id=MODEL_PROFILE_ID,
            expected_model_digest=MODEL_DIGEST,
            expected_runtime_profile=_runtime_profile(),
            expected_index_digest=anchored_index_digest,
        )


def test_wrong_completion_index_anchor_is_rejected(tmp_path: Path) -> None:
    run_root = tmp_path / RUN_ID
    _write_bundle(run_root)

    with pytest.raises(RehostPreparationError, match="index digest"):
        prepare_rehost_cases(
            run_root,
            user_hmac="A" * 43,
            session_ref=None,
            bundle_id=None,
            destination_seq_id=1,
            expected_model_profile_id=MODEL_PROFILE_ID,
            expected_model_digest=MODEL_DIGEST,
            expected_runtime_profile=_runtime_profile(),
            expected_index_digest="sha256:" + "0" * 64,
        )


def test_capture_runtime_profile_digest_is_provenance_not_a_local_equality_gate(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / RUN_ID
    _write_bundle(run_root)
    local_profile = replace(
        _runtime_profile(),
        profile_digest="sha256:" + "8" * 64,
    )

    cases = prepare_rehost_cases(
        run_root,
        user_hmac="A" * 43,
        session_ref=None,
        bundle_id=None,
        destination_seq_id=1,
        expected_model_profile_id=MODEL_PROFILE_ID,
        expected_model_digest=MODEL_DIGEST,
        expected_runtime_profile=local_profile,
        expected_index_digest=_sha256(
            (run_root / "rehost_bundle_index.json").read_bytes()
        ),
    )

    assert cases[0].runtime_profile_digest == RUNTIME_PROFILE_DIGEST
    assert cases[0].runtime_profile_digest != local_profile.profile_digest
