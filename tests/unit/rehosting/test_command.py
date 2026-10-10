from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import laris_onpremise.rehosting.command as command
from laris_onpremise.bundle.rehost_metadata import RehostRuntimeMetadata


def _schema_1_4_runtime_profile_document() -> dict[str, object]:
    return {
        "schema_version": "1.4",
        "profile_id": "x86_64-cpu-simple-kv",
        "adapter_kind": "llama_cpp_simple_kv",
        "llama_cpp_commit": "daf6bc9f2d3516d7bda73d191c6a01388aff09c5",
        "os": "Linux",
        "architecture": "x86_64",
        "abi": "System V AMD64",
        "memory_family": "simple_kv",
        "qualification": {
            "state": "linux_live_qualified",
            "reason": "qualified schema 1.4 runtime",
        },
        "runtime_identity": {
            "runtime_binary_path": "/opt/laris/llama-runtime/bin/llama-server",
            "runtime_binary_sha256": "sha256:" + "a" * 64,
            "mapped_libllama_sha256": "sha256:" + "b" * 64,
            "mapped_libllama_elf_build_id": "c" * 40,
            "required_launch_arguments": ["--parallel", "4"],
        },
        "runtime_policy": {
            "cache_prompt": False,
            "cache_idle_slots": False,
            "context_shift": False,
            "requires_full_sequence_rewrite_per_lease": True,
        },
        "topology_expectations": {
            "n_stream": 4,
            "n_seq_max": 4,
            "max_n_layer": 512,
            "max_n_pos_per_embd": 16,
        },
        "kv_cache_types": {
            "key": "f16",
            "value": "f16",
            "key_ggml_type_id": 1,
            "value_ggml_type_id": 1,
        },
        # These fields belong to capture qualification, not rehosting.
        "abi_offsets": {
            "slot_info_strm": 8,
            "slot_info_idxs": 32,
            "ubatch_n_tokens": 4,
            "ubatch_pos": 40,
            "ubatch_n_seq_id": 48,
            "ubatch_seq_id": 56,
            "tensor_data": 248,
            "tensor_type": 0,
        },
        "symbols": {},
        "limits": {"max_apply_tokens": 128},
        "forbidden_mutations": [],
        "static_forbidden_mutation_attestations": [],
        "probe_targets": [],
        "abi_manifest": None,
    }


def test_load_runtime_profile_accepts_published_profile_without_new_capture_offsets(
    tmp_path: Path,
) -> None:
    path = tmp_path / "runtime-profile.json"
    path.write_text(
        json.dumps(
            _schema_1_4_runtime_profile_document(),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    profile = command._load_runtime_profile(path)

    assert profile.runtime_binary_digest == "sha256:" + "a" * 64
    assert profile.mapped_libllama_digest == "sha256:" + "b" * 64
    assert profile.mapped_libllama_elf_build_id == "c" * 40
    assert profile.n_seq_max == 4
    assert profile.n_stream == 4
    assert profile.n_layer == 0
    assert profile.v_trans is None
    assert profile.max_n_layer == 512
    assert profile.key_ggml_type_id == 1
    assert profile.value_ggml_type_id == 1


def test_load_runtime_profile_rejects_duplicate_runtime_identity_field(
    tmp_path: Path,
) -> None:
    path = tmp_path / "runtime-profile.json"
    document = json.dumps(
        _schema_1_4_runtime_profile_document(),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    document = document.replace(
        '"runtime_binary_sha256":"sha256:' + "a" * 64 + '"',
        '"runtime_binary_sha256":"sha256:'
        + "a" * 64
        + '","runtime_binary_sha256":"sha256:'
        + "a" * 64
        + '"',
    )
    path.write_text(document, encoding="utf-8")

    try:
        command._load_runtime_profile(path)
    except ValueError as error:
        assert "duplicate JSON key" in str(error)
    else:
        raise AssertionError("duplicate runtime identity field was accepted")


def test_load_runtime_profile_rejects_non_f16_ggml_type_id(tmp_path: Path) -> None:
    path = tmp_path / "runtime-profile.json"
    document = _schema_1_4_runtime_profile_document()
    cache_types = document["kv_cache_types"]
    assert isinstance(cache_types, dict)
    cache_types["value_ggml_type_id"] = 2
    path.write_text(
        json.dumps(document, ensure_ascii=True, separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
    )

    try:
        command._load_runtime_profile(path)
    except ValueError as error:
        assert "value_ggml_type_id must equal 1" in str(error)
    else:
        raise AssertionError("non-f16 ggml type id was accepted")


def test_load_runtime_profile_rejects_unqualified_schema_1_4_stream_width(
    tmp_path: Path,
) -> None:
    path = tmp_path / "runtime-profile.json"
    document = _schema_1_4_runtime_profile_document()
    topology = document["topology_expectations"]
    assert isinstance(topology, dict)
    topology["n_stream"] = 8
    topology["n_seq_max"] = 8
    path.write_text(
        json.dumps(document, ensure_ascii=True, separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
    )

    try:
        command._load_runtime_profile(path)
    except ValueError as error:
        assert "n_stream must equal 4" in str(error)
    else:
        raise AssertionError("unqualified schema 1.4 stream width was accepted")


def test_worker_parser_rejects_removed_run_root_source() -> None:
    try:
        command.build_parser().parse_args(["--run-root", "run-test"])
    except SystemExit as error:
        assert error.code == 2
    else:
        raise AssertionError("removed run-root source was accepted")


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
