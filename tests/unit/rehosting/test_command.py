from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import laris_onpremise.rehosting.command as command
from laris_onpremise.bundle.rehost_metadata import RehostRuntimeMetadata


def test_command_rejects_legacy_run_root_without_gguf_runtime_metadata(
    tmp_path: Path,
    monkeypatch: object,
    capsys: object,
) -> None:
    observer_profile = SimpleNamespace(
        profile_digest="sha256:" + "a" * 64,
        runtime_binary_sha256="sha256:" + "b" * 64,
        mapped_libllama_sha256="sha256:" + "c" * 64,
        mapped_libllama_elf_build_id="1" * 40,
        n_seq_max=4,
        n_stream=4,
        n_layer=0,
        v_trans=None,
        key_ggml_type_id=1,
        value_ggml_type_id=1,
        max_n_layer=512,
    )
    case = SimpleNamespace(
        model_digest="sha256:" + "e" * 64,
        to_mapping=lambda: {"bundle_id": "rb_" + "d" * 64},
    )
    receipt = SimpleNamespace(exit_code=0, to_mapping=lambda: {"exit_code": 0})

    monkeypatch.setattr(  # type: ignore[attr-defined]
        command,
        "load_runtime_state_observer_profile",
        lambda _path: observer_profile,
    )
    monkeypatch.setattr(  # type: ignore[attr-defined]
        command, "digest_file", lambda _path: "sha256:" + "e" * 64
    )

    monkeypatch.setattr(  # type: ignore[attr-defined]
        command, "invoke_helper", lambda *_args, **_kwargs: receipt
    )

    exit_code = command.main(
        [
            "--run-root",
            str(tmp_path / "run-test"),
            "--bundle-id",
            "rb_" + "d" * 64,
            "--helper",
            str(tmp_path / "helper"),
            "--helper-digest",
            "sha256:" + "f" * 64,
            "--libllama",
            str(tmp_path / "libllama.so"),
            "--model",
            str(tmp_path / "model.gguf"),
            "--model-profile-id",
            "qwen2.5-0.5b-q4",
            "--runtime-profile",
            str(tmp_path / "runtime-profile.json"),
            "--index-digest",
            "sha256:" + "9" * 64,
            "--messages-file",
            str(tmp_path / "messages.json"),
            "--analysis-root",
            str(tmp_path / "analysis"),
        ]
    )

    assert exit_code == 2
    assert "GGUF-default runtime metadata" in capsys.readouterr().err  # type: ignore[attr-defined]


def test_command_selects_only_the_explicit_portable_bundle(
    tmp_path: Path,
    monkeypatch: object,
    capsys: object,
) -> None:
    bundle_root = tmp_path / "selected" / "bundle"
    other_bundle = tmp_path / "other" / "bundle"
    case = SimpleNamespace(
        model_digest="sha256:" + "e" * 64,
        to_mapping=lambda: {"bundle_id": "rb_" + "d" * 64},
    )
    entry = SimpleNamespace(
        bundle_id="rb_" + "d" * 64,
        model_profile_id="qwen2.5-0.5b-q4",
        model_digest="sha256:" + "e" * 64,
        to_rehost_case=lambda **_kwargs: case,
    )
    catalog = SimpleNamespace(
        bundle_root=bundle_root,
        source_run_id="run-test",
        entries=(entry,),
        runtime_metadata=RehostRuntimeMetadata(
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
        ),
        prompt_contract="gguf_default",
    )
    observed: dict[str, object] = {}

    def open_catalog(path: Path, expected_manifest_digest: str | None = None) -> object:
        observed["path"] = path
        observed["digest"] = expected_manifest_digest
        assert path != other_bundle
        return catalog

    monkeypatch.setattr(command.PortableBundleCatalog, "open", open_catalog)  # type: ignore[attr-defined]
    monkeypatch.setattr(
        command,
        "_load_runtime_profile",
        lambda _path: SimpleNamespace(verify_case=lambda _case: None),
    )  # type: ignore[attr-defined]
    monkeypatch.setattr(command, "digest_file", lambda _path: entry.model_digest)  # type: ignore[attr-defined]
    monkeypatch.setattr(
        command,
        "invoke_helper",
        lambda *_args, **_kwargs: SimpleNamespace(
            exit_code=0,
            to_mapping=lambda: {"exit_code": 0},
        ),
    )  # type: ignore[attr-defined]

    exit_code = command.main(
        [
            "--bundle-root", str(bundle_root),
            "--bundle-manifest-digest", "sha256:" + "a" * 64,
            "--bundle-id", entry.bundle_id,
            "--helper", str(tmp_path / "helper"),
            "--helper-digest", "sha256:" + "f" * 64,
            "--libllama", str(tmp_path / "libllama.so"),
            "--model", str(tmp_path / "model.gguf"),
            "--model-profile-id", entry.model_profile_id,
            "--runtime-profile", str(tmp_path / "runtime-profile.json"),
            "--messages-file", str(tmp_path / "messages.json"),
            "--analysis-root", str(tmp_path / "analysis"),
        ]
    )

    assert exit_code == 0
    assert observed == {
        "path": bundle_root,
        "digest": "sha256:" + "a" * 64,
    }
    assert json.loads(capsys.readouterr().out)["status"] == "COMPLETE"  # type: ignore[attr-defined]
