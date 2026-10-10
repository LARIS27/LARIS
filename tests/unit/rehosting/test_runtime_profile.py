from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from laris_onpremise.kv_state.observation.runtime_state_profile import (
    PINNED_LLAMA_CPP_COMMIT,
    RuntimeStateProfileError,
)
from laris_onpremise.rehosting.runtime_profile import load_rehost_runtime_profile


ROOT = Path(__file__).resolve().parents[3]
GENERATOR = ROOT / "evaluation" / "scripts" / "build_public_rehost_profile.py"


def _generator_module():
    spec = importlib.util.spec_from_file_location("build_public_rehost_profile", GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generated_rehost_only_profile_is_accepted(tmp_path: Path) -> None:
    module = _generator_module()
    runtime = tmp_path / "llama-server"
    library = tmp_path / "libllama.so.0"
    runtime.write_bytes(b"\x7fELF-runtime")
    library.write_bytes(b"\x7fELF-library")
    output = tmp_path / "runtime-profile.json"

    document = module.build_profile(
        runtime,
        library,
        "a" * 40,
        runtime_binary_path="/opt/laris-rehost/bin/llama-server",
    )
    module.write_profile(output, document)

    loaded = load_rehost_runtime_profile(output)
    assert document["schema_version"] == "1.6"
    assert document["llama_cpp_commit"] == PINNED_LLAMA_CPP_COMMIT
    assert document["qualification"] == {
        "scope": "rehosting_only",
        "state": "local_rehost_qualified",
    }
    assert loaded.n_stream == 4
    assert loaded.max_n_layer == 64
    assert loaded.mapped_libllama_elf_build_id == "a" * 40


def test_rehost_only_schema_rejects_a_live_capture_claim(tmp_path: Path) -> None:
    module = _generator_module()
    runtime = tmp_path / "llama-server"
    library = tmp_path / "libllama.so.0"
    runtime.write_bytes(b"\x7fELF-runtime")
    library.write_bytes(b"\x7fELF-library")
    document = module.build_profile(
        runtime,
        library,
        "b" * 40,
        runtime_binary_path="/opt/laris-rehost/bin/llama-server",
    )
    document["qualification"]["state"] = "linux_live_qualified"
    output = tmp_path / "runtime-profile.json"
    output.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(RuntimeStateProfileError, match="qualification.state"):
        load_rehost_runtime_profile(output)
