from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from evaluation.scripts.assemble_paper_final_source import (
    PaperReleaseAssemblyError,
    assemble_source_tree,
)


def _row(path: str, payload: bytes, *, role: str = "test") -> dict[str, object]:
    return {
        "path": path,
        "role": role,
        "size_bytes": len(payload),
        "sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
    }


def _inventory(*rows: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "file_count": len(rows),
        "files": list(rows),
    }


def test_assembler_copies_only_inventory_regular_files(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    (source / "keep").mkdir()
    (source / "keep" / "a.txt").write_bytes(b"alpha\n")
    (source / "ignored.txt").write_bytes(b"not public\n")

    copied = assemble_source_tree(
        source,
        destination,
        _inventory(_row("keep/a.txt", b"alpha\n")),
    )

    assert copied == (destination / "keep" / "a.txt",)
    assert (destination / "keep" / "a.txt").read_bytes() == b"alpha\n"
    assert not (destination / "ignored.txt").exists()


def test_assembler_refuses_nonempty_or_symlink_destination(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.txt").write_bytes(b"a")
    inventory = _inventory(_row("a.txt", b"a"))

    nonempty = tmp_path / "nonempty"
    nonempty.mkdir()
    (nonempty / "unexpected.txt").write_bytes(b"x")
    with pytest.raises(PaperReleaseAssemblyError, match="destination"):
        assemble_source_tree(source, nonempty, inventory)

    symlink = tmp_path / "destination-link"
    try:
        symlink.symlink_to(tmp_path / "missing", target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    with pytest.raises(PaperReleaseAssemblyError, match="symlink"):
        assemble_source_tree(source, symlink, inventory)


def test_assembler_preserves_source_bytes_and_mtime(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    payload = b"verified bytes\r\n"
    path = source / "payload.bin"
    path.write_bytes(payload)
    expected_mtime_ns = 1_700_000_000_000_000_000
    os.utime(path, ns=(expected_mtime_ns, expected_mtime_ns))

    assemble_source_tree(
        source,
        destination,
        _inventory(_row("payload.bin", payload)),
    )

    copied = destination / "payload.bin"
    assert copied.read_bytes() == payload
    assert copied.stat().st_mtime_ns == path.stat().st_mtime_ns
