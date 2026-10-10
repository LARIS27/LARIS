from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import tempfile

from laris_onpremise.kv_state.observation.runtime_state_profile import (
    PINNED_LLAMA_CPP_COMMIT,
)


_BUILD_ID = re.compile(r"[0-9a-f]{40}\Z")


def _regular_elf(path: Path, field: str) -> Path:
    if not isinstance(path, Path):
        raise ValueError(f"{field} must be a pathlib.Path")
    try:
        info = path.lstat()
    except OSError as error:
        raise ValueError(f"{field} is unavailable") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ValueError(f"{field} must be a regular non-symlink file")
    with path.open("rb") as stream:
        if stream.read(4) != b"\x7fELF":
            raise ValueError(f"{field} must be an ELF file")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def build_profile(
    runtime: Path,
    library: Path,
    build_id: str,
    *,
    runtime_binary_path: str,
) -> dict[str, object]:
    runtime = _regular_elf(runtime, "runtime")
    library = _regular_elf(library, "library")
    if _BUILD_ID.fullmatch(build_id) is None:
        raise ValueError("build_id must be 40 lowercase hexadecimal characters")
    if (
        not isinstance(runtime_binary_path, str)
        or not runtime_binary_path.startswith("/")
        or runtime_binary_path == "/"
        or "\\" in runtime_binary_path
        or "//" in runtime_binary_path
        or any(
            part in {"", ".", ".."}
            for part in runtime_binary_path.split("/")[1:]
        )
    ):
        raise ValueError("runtime_binary_path must be a canonical absolute POSIX path")
    return {
        "schema_version": "1.6",
        "profile_id": "x86_64-cpu-simple-kv-rehost",
        "adapter_kind": "llama_cpp_simple_kv",
        "llama_cpp_commit": PINNED_LLAMA_CPP_COMMIT,
        "os": "Linux",
        "architecture": "x86_64",
        "abi": "System V AMD64",
        "memory_family": "simple_kv",
        "qualification": {
            "scope": "rehosting_only",
            "state": "local_rehost_qualified",
        },
        "runtime_identity": {
            "runtime_binary_path": runtime_binary_path,
            "runtime_binary_sha256": _sha256(runtime),
            "mapped_libllama_sha256": _sha256(library),
            "mapped_libllama_elf_build_id": build_id,
        },
        "runtime_policy": {
            "cache_prompt": False,
            "cache_idle_slots": False,
            "context_shift": False,
            "requires_full_sequence_rewrite_per_lease": True,
        },
        "kv_cache_types": {
            "key": "f16",
            "value": "f16",
            "key_ggml_type_id": 1,
            "value_ggml_type_id": 1,
        },
        "topology_expectations": {
            "max_n_layer": 64,
            "max_n_pos_per_embd": 16,
            "n_stream": 4,
            "n_seq_max": 4,
        },
    }


def write_profile(output: Path, document: dict[str, object]) -> None:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            document,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    descriptor, raw_path = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    pending = Path(raw_path)
    try:
        try:
            os.fchmod(descriptor, 0o600)
        except (AttributeError, OSError):
            pass
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        pending.replace(output)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        pending.unlink(missing_ok=True)
        raise


def _read_build_id(library: Path) -> str:
    completed = subprocess.run(
        ["readelf", "-n", str(library)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    match = re.search(r"Build ID:\s*([0-9a-f]{40})\b", completed.stdout)
    if match is None:
        raise ValueError("libllama does not expose a 40-character GNU build ID")
    return match.group(1)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a local, rehosting-only LARIS runtime profile."
    )
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--libllama", type=Path, required=True)
    parser.add_argument("--runtime-binary-path", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    document = build_profile(
        arguments.runtime,
        arguments.libllama,
        _read_build_id(arguments.libllama),
        runtime_binary_path=arguments.runtime_binary_path,
    )
    write_profile(arguments.output, document)
    print(str(arguments.output.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
