from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess

import pytest

from laris_onpremise.bundle.rehost_metadata import RehostRuntimeMetadata
from laris_onpremise.rehosting.contracts import RehostCase, RehostRuntimeProfile
from laris_onpremise.rehosting.contracts import InvestigatorSampler
from laris_onpremise.rehosting.runner import (
    RehostExecutionError,
    _verified_run_root,
    invoke_helper,
)


RUN_ID = "run-20260814-204253-KST_qwen_c5-attribution_abcd"
USER_HMAC = "A" * 43
SESSION_REF = "sr_" + "b" * 64


def _digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _case(test_root: Path, *, v_trans: bool = True) -> RehostCase:
    session_digest = hashlib.sha256(SESSION_REF.encode("utf-8")).hexdigest()[:16]
    bundle_root = (
        test_root
        / RUN_ID
        / "rehost"
        / f"uh_{USER_HMAC}"
        / f"s_{session_digest}"
    )
    bundle_root.mkdir(parents=True)
    payload = bundle_root / "runtime_state_payload.bin"
    payload.write_bytes(b"payload")
    (bundle_root / "rehost_manifest.json").write_text("{}\n", encoding="utf-8")
    return RehostCase(
        run_id=RUN_ID,
        bundle_id="rb_" + "a" * 64,
        user_hmac=USER_HMAC,
        identity_key_id="identity-key-v1",
        session_ref=SESSION_REF,
        payload_path=payload,
        payload_digest=_digest(b"payload"),
        payload_size_bytes=7,
        next_position=2,
        destination_seq_id=1,
        model_profile_id="qwen2.5-0.5b-q4",
        model_digest=_digest(b"model"),
        runtime_profile_digest="sha256:" + "d" * 64,
        memory_family="simple_kv",
        n_ctx=32768,
        n_seq_max=4,
        n_stream=4,
        n_layer=2,
        v_trans=v_trans,
        key_ggml_type_id=1,
        value_ggml_type_id=1,
        source_seq_id=2,
        source_stream_index=1,
        position_range=(0, 1),
    )


def _portable_case(test_root: Path, *, v_trans: bool = True) -> RehostCase:
    case = _case(test_root, v_trans=v_trans)
    old_session_root = case.payload_path.parent
    portable_session_root = (
        test_root
        / RUN_ID
        / "bundle"
        / "users"
        / f"uh_{USER_HMAC}"
        / "sessions"
        / old_session_root.name
    )
    portable_session_root.parent.mkdir(parents=True)
    old_session_root.rename(portable_session_root)
    return RehostCase(
        **{
            **case.__dict__,
            "payload_path": portable_session_root / "runtime_state_payload.bin",
        }
    )


def test_verified_run_root_rejects_removed_run_root_layout(tmp_path: Path) -> None:
    case = _case(tmp_path)

    with pytest.raises(
        RehostExecutionError,
        match="payload path is not bound to the evidence run",
    ):
        _verified_run_root(case)


def test_invoke_helper_accepts_the_canonical_portable_bundle_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _portable_case(tmp_path)
    helper = tmp_path / "helper"
    model = tmp_path / "model.gguf"
    question = tmp_path / "question.txt"
    libllama = _libllama(tmp_path)
    helper.write_bytes(b"helper")
    model.write_bytes(b"model")
    question.write_text("question", encoding="utf-8")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command,
            0,
            stdout="answer\n",
            stderr=_success_metadata(libllama),
        ),
    )

    receipt = invoke_helper(
        helper,
        model,
        question,
        case,
        runtime_profile=_runtime_profile(),
        runtime=_runtime_metadata(),
        sampler=InvestigatorSampler(),
        libllama=libllama,
        expected_helper_digest=_digest(b"helper"),
        analysis_root=tmp_path / "analysis",
        timeout_seconds=10,
    )

    assert receipt.exit_code == 0


def _runtime_profile(*, v_trans: bool = True) -> RehostRuntimeProfile:
    return RehostRuntimeProfile(
        profile_digest="sha256:" + "d" * 64,
        runtime_binary_digest="sha256:" + "e" * 64,
        mapped_libllama_digest=_digest(b"libllama"),
        mapped_libllama_elf_build_id="1" * 40,
        n_seq_max=4,
        n_stream=4,
        n_layer=2,
        v_trans=v_trans,
        key_ggml_type_id=1,
        value_ggml_type_id=1,
    )


def _libllama(test_root: Path) -> Path:
    path = test_root / "libllama.so"
    path.write_bytes(b"libllama")
    return path


def _success_metadata(
    libllama: Path,
    *,
    generated: int = 3,
) -> str:
    identity = libllama.stat()
    return (
        f"LARIS_META mapped_libllama_device={identity.st_dev}\n"
        f"LARIS_META mapped_libllama_inode={identity.st_ino}\n"
        "LARIS_META restored_bytes=15\n"
        "LARIS_META restored_pos_min=0\n"
        "LARIS_META restored_pos_max=1\n"
        "LARIS_META appended_token_count=5\n"
        "LARIS_META logical_batch_count=1\n"
        "LARIS_META effective_n_batch=128\n"
        "LARIS_META effective_n_ubatch=128\n"
        "LARIS_META prompt_contract=gguf_default\n"
        "LARIS_META sampler_temperature=0.8\n"
        "LARIS_META sampler_top_k=40\n"
        "LARIS_META sampler_top_p=0.95\n"
        "LARIS_META sampler_min_p=0.05\n"
        "LARIS_META sampler_seed=0\n"
        f"LARIS_META generated_token_count={generated}\n"
    )


def _runtime_metadata() -> RehostRuntimeMetadata:
    return RehostRuntimeMetadata(
        source_runtime_profile_digest="sha256:" + "d" * 64,
        llama_cpp_commit="1" * 40,
        memory_family="simple_kv",
        n_batch=128,
        n_ubatch=128,
        cache_type_k="f16",
        cache_type_v="f16",
        flash_attention=False,
        n_gpu_layers=0,
        context_shift=False,
        cache_prompt=False,
        cache_idle_slots=False,
    )


@pytest.mark.parametrize(("v_trans", "wire_value"), [(True, "true"), (False, "false")])
def test_invoke_helper_materializes_wrapper_only_in_analysis_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    v_trans: bool,
    wire_value: str,
) -> None:
    case = _portable_case(tmp_path, v_trans=v_trans)
    bundle_root = case.payload_path.parent
    helper = tmp_path / "helper"
    model = tmp_path / "model.gguf"
    question = tmp_path / "question.txt"
    helper.write_bytes(b"helper")
    model.write_bytes(b"model")
    question.write_text("What do you remember?", encoding="utf-8")
    observed: dict[str, object] = {}
    libllama = _libllama(tmp_path)

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        observed["command"] = command
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="answer\n",
            stderr=_success_metadata(libllama),
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    receipt = invoke_helper(
        helper,
        model,
        question,
        case,
        runtime_profile=_runtime_profile(v_trans=v_trans),
        runtime=_runtime_metadata(),
        sampler=InvestigatorSampler(),
        libllama=libllama,
        expected_helper_digest=_digest(b"helper"),
        analysis_root=tmp_path / "analysis",
        timeout_seconds=10,
    )

    command = observed["command"]
    assert isinstance(command, list)
    assert "--wrapped-state" in command
    assert "--checkpoint" not in command
    assert "--destination-seq-id" in command
    assert command[command.index("--v-trans") + 1] == wire_value
    assert receipt.restored_bytes == 15
    assert receipt.restored_pos_min == 0
    assert receipt.restored_pos_max == 1
    assert receipt.appended_token_count == 5
    assert receipt.generated_token_count == 3
    assert set(path.name for path in bundle_root.iterdir()) == {
        "runtime_state_payload.bin",
        "rehost_manifest.json",
    }
    assert (
        tmp_path
        / "analysis"
        / case.bundle_id
        / "derived"
        / "laris_wrapped_seq_state.bin"
    ).is_file()


def test_success_receipt_rejects_legacy_loaded_token_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _portable_case(tmp_path)
    helper = tmp_path / "helper"
    model = tmp_path / "model.gguf"
    question = tmp_path / "question.txt"
    helper.write_bytes(b"helper")
    model.write_bytes(b"model")
    question.write_text("question", encoding="utf-8")

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0],
            0,
            stdout="answer\n",
            stderr="LARIS_META loaded_token_count=2\n",
        ),
    )

    with pytest.raises(RehostExecutionError, match="metadata"):
        invoke_helper(
            helper,
            model,
            question,
            case,
            runtime_profile=_runtime_profile(),
            runtime=_runtime_metadata(),
            sampler=InvestigatorSampler(),
            libllama=_libllama(tmp_path),
            expected_helper_digest=_digest(b"helper"),
            analysis_root=tmp_path / "analysis",
            timeout_seconds=10,
        )


def test_rejects_analysis_root_inside_evidence_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _portable_case(tmp_path)
    helper = tmp_path / "helper"
    model = tmp_path / "model.gguf"
    question = tmp_path / "question.txt"
    helper.write_bytes(b"helper")
    model.write_bytes(b"model")
    question.write_text("question", encoding="utf-8")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("helper must not execute"),
    )

    with pytest.raises(RehostExecutionError, match="analysis.*evidence"):
        invoke_helper(
            helper,
            model,
            question,
            case,
            runtime_profile=_runtime_profile(),
            runtime=_runtime_metadata(),
            sampler=InvestigatorSampler(),
            libllama=_libllama(tmp_path),
            expected_helper_digest=_digest(b"helper"),
            analysis_root=tmp_path / RUN_ID / "analysis",
            timeout_seconds=10,
        )


@pytest.mark.parametrize(
    ("response", "generated"),
    [("", 3), ("answer\n", 0)],
)
def test_success_requires_a_generated_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    response: str,
    generated: int,
) -> None:
    case = _portable_case(tmp_path)
    helper = tmp_path / "helper"
    model = tmp_path / "model.gguf"
    question = tmp_path / "question.txt"
    helper.write_bytes(b"helper")
    model.write_bytes(b"model")
    question.write_text("question", encoding="utf-8")
    libllama = _libllama(tmp_path)
    metadata = _success_metadata(libllama, generated=generated)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, stdout=response, stderr=metadata
        ),
    )

    with pytest.raises(RehostExecutionError, match="generated response"):
        invoke_helper(
            helper,
            model,
            question,
            case,
            runtime_profile=_runtime_profile(),
            runtime=_runtime_metadata(),
            sampler=InvestigatorSampler(),
            libllama=libllama,
            expected_helper_digest=_digest(b"helper"),
            analysis_root=tmp_path / "analysis",
            timeout_seconds=10,
        )


def test_helper_environment_removes_untrusted_loader_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _portable_case(tmp_path)
    helper = tmp_path / "helper"
    model = tmp_path / "model.gguf"
    question = tmp_path / "question.txt"
    helper.write_bytes(b"helper")
    model.write_bytes(b"model")
    question.write_text("question", encoding="utf-8")
    monkeypatch.setenv("LD_PRELOAD", "/tmp/attacker.so")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/tmp/attacker")
    monkeypatch.setenv("PYTHONPATH", "/tmp/attacker-python")
    observed: dict[str, object] = {}
    libllama = _libllama(tmp_path)

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        observed["env"] = kwargs["env"]
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="answer\n",
            stderr=_success_metadata(libllama),
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    invoke_helper(
        helper,
        model,
        question,
        case,
        runtime_profile=_runtime_profile(),
        runtime=_runtime_metadata(),
        sampler=InvestigatorSampler(),
        libllama=libllama,
        expected_helper_digest=_digest(b"helper"),
        analysis_root=tmp_path / "analysis",
        timeout_seconds=10,
    )

    environment = observed["env"]
    assert isinstance(environment, dict)
    assert "LD_PRELOAD" not in environment
    assert environment["LD_LIBRARY_PATH"] == str(libllama.parent)
    assert "PYTHONPATH" not in environment


def test_rejects_mapped_libllama_identity_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _portable_case(tmp_path)
    helper = tmp_path / "helper"
    model = tmp_path / "model.gguf"
    question = tmp_path / "question.txt"
    helper.write_bytes(b"helper")
    model.write_bytes(b"model")
    question.write_text("question", encoding="utf-8")
    libllama = _libllama(tmp_path)
    metadata = _success_metadata(libllama).replace(
        f"mapped_libllama_inode={libllama.stat().st_ino}",
        "mapped_libllama_inode=999999",
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, stdout="answer\n", stderr=metadata
        ),
    )

    with pytest.raises(RehostExecutionError, match="mapped libllama"):
        invoke_helper(
            helper,
            model,
            question,
            case,
            runtime_profile=_runtime_profile(),
            runtime=_runtime_metadata(),
            sampler=InvestigatorSampler(),
            libllama=libllama,
            expected_helper_digest=_digest(b"helper"),
            analysis_root=tmp_path / "analysis",
            timeout_seconds=10,
        )


@pytest.mark.parametrize(
    ("helper_bytes", "libllama_bytes", "message"),
    [
        (b"different-helper", b"libllama", "helper digest"),
        (b"helper", b"different-libllama", "libllama digest"),
    ],
)
def test_rejects_unqualified_helper_or_libllama(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    helper_bytes: bytes,
    libllama_bytes: bytes,
    message: str,
) -> None:
    case = _portable_case(tmp_path)
    helper = tmp_path / "helper"
    model = tmp_path / "model.gguf"
    question = tmp_path / "question.txt"
    libllama = tmp_path / "libllama.so"
    helper.write_bytes(helper_bytes)
    libllama.write_bytes(libllama_bytes)
    model.write_bytes(b"model")
    question.write_text("question", encoding="utf-8")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("helper must not execute"),
    )

    with pytest.raises(RehostExecutionError, match=message):
        invoke_helper(
            helper,
            model,
            question,
            case,
            runtime_profile=_runtime_profile(),
            runtime=_runtime_metadata(),
            sampler=InvestigatorSampler(),
            libllama=libllama,
            expected_helper_digest=_digest(b"helper"),
            analysis_root=tmp_path / "analysis",
            timeout_seconds=10,
        )
