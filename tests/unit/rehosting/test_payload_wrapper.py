from __future__ import annotations

import hashlib
import os
import struct
from pathlib import Path

import pytest

from laris_onpremise.rehosting.wrapper import (
    IO_MAGIC,
    DerivedStateError,
    materialize_derived_wrapper,
)


def _digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def test_materializes_exact_ephemeral_sequence_wrapper(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    payload = bundle / "runtime_state_payload.bin"
    payload_bytes = b"\x04\x00\x00\x00payload"
    payload.write_bytes(payload_bytes)

    derived = materialize_derived_wrapper(
        payload_path=payload,
        expected_payload_digest=_digest(payload_bytes),
        destination_seq_id=2,
        derived_dir=tmp_path / "analysis" / "derived",
    )

    expected = struct.pack("<Ii", IO_MAGIC, 2) + payload_bytes
    assert derived.path.name == "laris_wrapped_seq_state.bin"
    assert derived.path.read_bytes() == expected
    assert derived.size_bytes == len(expected)
    assert derived.digest == _digest(expected)
    assert derived.payload_digest == _digest(payload_bytes)
    assert set(path.name for path in bundle.iterdir()) == {
        "runtime_state_payload.bin"
    }


def test_rejects_payload_digest_mismatch_without_publishing(
    tmp_path: Path,
) -> None:
    payload = tmp_path / "bundle" / "runtime_state_payload.bin"
    payload.parent.mkdir()
    payload.write_bytes(b"payload")
    derived_dir = tmp_path / "analysis" / "derived"

    with pytest.raises(DerivedStateError, match="digest"):
        materialize_derived_wrapper(
            payload_path=payload,
            expected_payload_digest="sha256:" + "0" * 64,
            destination_seq_id=0,
            derived_dir=derived_dir,
        )

    assert not (derived_dir / "laris_wrapped_seq_state.bin").exists()


def test_refuses_to_overwrite_existing_derived_state(tmp_path: Path) -> None:
    payload = tmp_path / "bundle" / "runtime_state_payload.bin"
    payload.parent.mkdir()
    payload_bytes = b"payload"
    payload.write_bytes(payload_bytes)
    derived_dir = tmp_path / "analysis" / "derived"
    derived_dir.mkdir(parents=True)
    output = derived_dir / "laris_wrapped_seq_state.bin"
    output.write_bytes(b"existing")

    with pytest.raises(DerivedStateError, match="already exists"):
        materialize_derived_wrapper(
            payload_path=payload,
            expected_payload_digest=_digest(payload_bytes),
            destination_seq_id=0,
            derived_dir=derived_dir,
        )

    assert output.read_bytes() == b"existing"


@pytest.mark.parametrize("destination_seq_id", [-1, 2**31])
def test_rejects_destination_sequence_outside_int32(
    tmp_path: Path,
    destination_seq_id: int,
) -> None:
    payload = tmp_path / "bundle" / "runtime_state_payload.bin"
    payload.parent.mkdir()
    payload_bytes = b"payload"
    payload.write_bytes(payload_bytes)

    with pytest.raises(DerivedStateError, match="destination_seq_id"):
        materialize_derived_wrapper(
            payload_path=payload,
            expected_payload_digest=_digest(payload_bytes),
            destination_seq_id=destination_seq_id,
            derived_dir=tmp_path / "derived",
        )


def test_rejects_symbolic_link_payload(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"payload")
    payload = tmp_path / "runtime_state_payload.bin"
    try:
        payload.symlink_to(source)
    except OSError:
        pytest.skip("symbolic links are unavailable")

    with pytest.raises(DerivedStateError, match="symbolic link"):
        materialize_derived_wrapper(
            payload_path=payload,
            expected_payload_digest=_digest(b"payload"),
            destination_seq_id=0,
            derived_dir=tmp_path / "derived",
        )


def test_rejects_derived_output_inside_evidence_bundle(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    payload = bundle / "runtime_state_payload.bin"
    payload.write_bytes(b"payload")

    with pytest.raises(DerivedStateError, match="outside.*evidence"):
        materialize_derived_wrapper(
            payload_path=payload,
            expected_payload_digest=_digest(b"payload"),
            destination_seq_id=0,
            derived_dir=bundle / "derived",
        )

    assert set(path.name for path in bundle.iterdir()) == {
        "runtime_state_payload.bin"
    }


def test_retries_short_header_writes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = tmp_path / "bundle" / "runtime_state_payload.bin"
    payload.parent.mkdir()
    payload_bytes = b"payload"
    payload.write_bytes(payload_bytes)
    real_write = os.write

    def short_write(descriptor: int, data: object) -> int:
        view = memoryview(data)  # type: ignore[arg-type]
        return real_write(descriptor, view[: min(3, len(view))])

    monkeypatch.setattr(os, "write", short_write)
    derived = materialize_derived_wrapper(
        payload_path=payload,
        expected_payload_digest=_digest(payload_bytes),
        destination_seq_id=1,
        derived_dir=tmp_path / "analysis" / "derived",
    )

    assert derived.path.read_bytes() == struct.pack("<Ii", IO_MAGIC, 1) + payload_bytes
