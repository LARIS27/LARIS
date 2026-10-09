from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from evaluation.scripts.stage_public_evaluation_release import (
    artifact_checklist,
    initialize_release,
    verify_staged_release,
)
from laris_onpremise.reproduction.evaluation1.locations import (
    REQUIRED_EXTERNAL_PATHS,
    ArtifactLocationError,
    load_artifact_locations,
    verify_external_files,
)


def _digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _document(*, paths: tuple[str, ...] = REQUIRED_EXTERNAL_PATHS) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "release_id": "evaluation1-release-v1",
        "artifacts": [
            {
                "relative_path": path,
                "size_bytes": 1,
                "sha256": _digest(path.encode("utf-8")),
                "download_url": "PENDING_PUBLICATION",
            }
            for path in paths
        ],
    }


def _write(path: Path, document: dict[str, object]) -> Path:
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_location_manifest_requires_exact_seven_release_files(tmp_path: Path) -> None:
    manifest = load_artifact_locations(_write(tmp_path / "locations.json", _document()))

    assert tuple(item.relative_path.as_posix() for item in manifest.artifacts) == (
        REQUIRED_EXTERNAL_PATHS
    )

    for paths in (REQUIRED_EXTERNAL_PATHS[:-1], REQUIRED_EXTERNAL_PATHS + ("extra.bin",)):
        with pytest.raises(ArtifactLocationError, match="exact external artifact set"):
            load_artifact_locations(_write(tmp_path / "invalid.json", _document(paths=paths)))


@pytest.mark.parametrize(
    "bad_path",
    ("../escape", "/absolute", "C:/absolute", "models\\bad.tar"),
)
def test_location_manifest_rejects_unsafe_duplicate_or_absolute_paths(
    tmp_path: Path,
    bad_path: str,
) -> None:
    document = _document()
    artifacts = document["artifacts"]
    assert isinstance(artifacts, list)
    artifacts[0]["relative_path"] = bad_path
    with pytest.raises(ArtifactLocationError):
        load_artifact_locations(_write(tmp_path / "unsafe.json", document))

    duplicate = _document()
    duplicate_artifacts = duplicate["artifacts"]
    assert isinstance(duplicate_artifacts, list)
    duplicate_artifacts[-1]["relative_path"] = duplicate_artifacts[0]["relative_path"]
    with pytest.raises(ArtifactLocationError, match="duplicate"):
        load_artifact_locations(_write(tmp_path / "duplicate.json", duplicate))


def test_pending_publication_is_valid_but_not_downloadable(tmp_path: Path) -> None:
    manifest = load_artifact_locations(_write(tmp_path / "locations.json", _document()))

    assert manifest.is_published is False
    assert all(item.download_url == "PENDING_PUBLICATION" for item in manifest.artifacts)


def test_external_file_verification_rejects_missing_size_or_digest_mismatch(
    tmp_path: Path,
) -> None:
    root = tmp_path / "release"
    document = _document()
    artifacts = document["artifacts"]
    assert isinstance(artifacts, list)
    for row in artifacts:
        payload = str(row["relative_path"]).encode("utf-8")
        destination = root / str(row["relative_path"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        row["size_bytes"] = len(payload)
        row["sha256"] = _digest(payload)
    manifest = load_artifact_locations(_write(tmp_path / "locations.json", document))
    assert verify_external_files(manifest, root) == tuple(
        root / path for path in REQUIRED_EXTERNAL_PATHS
    )

    (root / REQUIRED_EXTERNAL_PATHS[0]).unlink()
    with pytest.raises(ArtifactLocationError, match="missing"):
        verify_external_files(manifest, root)

    first = root / REQUIRED_EXTERNAL_PATHS[0]
    first.write_bytes(b"wrong-size")
    with pytest.raises(ArtifactLocationError, match="size"):
        verify_external_files(manifest, root)

    expected_size = manifest.artifacts[0].size_bytes
    first.write_bytes(b"x" * expected_size)
    with pytest.raises(ArtifactLocationError, match="digest"):
        verify_external_files(manifest, root)


def test_external_file_verification_rejects_symlinks(tmp_path: Path) -> None:
    root = tmp_path / "release"
    document = _document()
    artifacts = document["artifacts"]
    assert isinstance(artifacts, list)
    for row in artifacts:
        payload = str(row["relative_path"]).encode("utf-8")
        destination = root / str(row["relative_path"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        row["size_bytes"] = len(payload)
        row["sha256"] = _digest(payload)
    manifest = load_artifact_locations(_write(tmp_path / "locations.json", document))

    target = root / "target.bin"
    target.write_bytes(b"target")
    link = root / REQUIRED_EXTERNAL_PATHS[0]
    link.unlink()
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation is not available on this Windows host")

    with pytest.raises(ArtifactLocationError, match="symlink"):
        verify_external_files(manifest, root)


def _metadata_fixture(root: Path) -> tuple[Path, dict[str, bytes]]:
    root.mkdir()
    metadata = {
        "release-manifest.json": b'{"release_id":"evaluation1-release-v1"}\n',
        "fixtures/key-provenance.json": b'{"synthetic_only":true}\n',
    }
    external_payloads = {
        path: ("payload:" + path).encode("utf-8") for path in REQUIRED_EXTERNAL_PATHS
    }
    for relative, payload in metadata.items():
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    lines = []
    for relative, payload in sorted({**metadata, **external_payloads}.items()):
        lines.append(f"{hashlib.sha256(payload).hexdigest()}  {relative}\n")
    (root / "checksums.sha256").write_text("".join(lines), encoding="ascii")
    locations = _document()
    rows = locations["artifacts"]
    assert isinstance(rows, list)
    for row in rows:
        payload = external_payloads[str(row["relative_path"])]
        row["size_bytes"] = len(payload)
        row["sha256"] = _digest(payload)
    _write(root / "artifact-locations.json", locations)
    return root, external_payloads


def test_initialize_copies_only_tracked_metadata_and_never_moves_sources(
    tmp_path: Path,
) -> None:
    metadata_root, external_payloads = _metadata_fixture(tmp_path / "metadata")
    before = {
        path.relative_to(metadata_root).as_posix(): path.read_bytes()
        for path in metadata_root.rglob("*")
        if path.is_file()
    }

    copied = initialize_release(metadata_root, tmp_path / "external")

    assert tuple(path.relative_to(tmp_path / "external").as_posix() for path in copied) == (
        "checksums.sha256",
        "fixtures/key-provenance.json",
        "release-manifest.json",
    )
    assert not any((tmp_path / "external" / name).exists() for name in external_payloads)
    assert {
        path.relative_to(metadata_root).as_posix(): path.read_bytes()
        for path in metadata_root.rglob("*")
        if path.is_file()
    } == before


def test_initialize_rejects_same_source_and_destination(tmp_path: Path) -> None:
    metadata_root, _ = _metadata_fixture(tmp_path / "metadata")

    with pytest.raises(ArtifactLocationError, match="different"):
        initialize_release(metadata_root, metadata_root)


def test_initialize_rejects_conflicting_existing_metadata(tmp_path: Path) -> None:
    metadata_root, _ = _metadata_fixture(tmp_path / "metadata")
    external_root = tmp_path / "external"
    external_root.mkdir()
    (external_root / "release-manifest.json").write_bytes(b"conflict")

    with pytest.raises(ArtifactLocationError, match="conflicting"):
        initialize_release(metadata_root, external_root)


def test_checklist_reports_seven_relative_paths_without_key_bytes(tmp_path: Path) -> None:
    metadata_root, _ = _metadata_fixture(tmp_path / "metadata")

    rows = artifact_checklist(metadata_root)

    assert tuple(row["relative_path"] for row in rows) == REQUIRED_EXTERNAL_PATHS
    assert all(set(row) == {"relative_path", "size_bytes", "sha256", "publication_state"} for row in rows)
    assert "key_bytes" not in json.dumps(rows)


def test_verify_requires_all_external_files_then_calls_release_verifier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata_root, payloads = _metadata_fixture(tmp_path / "metadata")
    external_root = tmp_path / "external"
    initialize_release(metadata_root, external_root)
    called: list[Path] = []

    def fake_verify(root: Path, _registry: Path) -> dict[str, object]:
        called.append(root)
        return {
            "schema_version": "1.0",
            "release_id": "evaluation1-release-v1",
            "verification_status": "VERIFIED",
            "artifact_count": 4,
            "payload_count": 240,
            "checksum_count": 9,
            "license_review_count": 2,
        }

    monkeypatch.setattr(
        "evaluation.scripts.stage_public_evaluation_release.verify_public_release",
        fake_verify,
    )
    with pytest.raises(ArtifactLocationError, match="missing"):
        verify_staged_release(metadata_root, external_root)
    assert called == []

    for relative, payload in payloads.items():
        destination = external_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    receipt = verify_staged_release(metadata_root, external_root)

    assert called == [external_root.resolve()]
    assert receipt["verification_status"] == "VERIFIED"
    assert receipt["payload_count"] == 240
    assert len(receipt["external_files"]) == 7


def test_checked_in_location_manifest_matches_verified_release() -> None:
    root = Path(__file__).resolve().parents[3]
    path = (
        root
        / "evaluation"
        / "publication"
        / "releases"
        / "evaluation1-release-v1"
        / "artifact-locations.json"
    )
    manifest = load_artifact_locations(path)
    expected = {
        "fixtures/keys/75cbd8e47c923053.key": (
            32,
            "sha256:5aa31af06af1da600167acf7875b07748d813c46f7f1ed53b5069f406821cc8f",
        ),
        "fixtures/keys/da75811916eb78a6.key": (
            32,
            "sha256:5f0d02cf1b73a43db4577fb094496ad0ebdb2f88a8e9ab681ef7a18012292b9e",
        ),
        "models/llama-3.2-3b-instruct-q4-k-m/log-linked.tar": (
            4484464640,
            "sha256:2894353e5640e731cfd1fc00f12b64b6c86f0d17bb1cbdeb4e5f5115703e1e67",
        ),
        "models/llama-3.2-3b-instruct-q4-k-m/proxy.tar": (
            4484208640,
            "sha256:c35032aa65ac367e3242c1d9a3d2690bac6d6b39c3471ea0e2e5029ae8111af1",
        ),
        "models/qwen2.5-7b-instruct-q4-k-m/log-linked.tar": (
            2137180160,
            "sha256:d2c6f7f2b1b0bfbbeebba8d8242a91f1aa59b9f037f8fbc9868718239870195b",
        ),
        "models/qwen2.5-7b-instruct-q4-k-m/proxy.tar": (
            2136924160,
            "sha256:7820484a828bbea5c19dec6e935457482cbde28abdcadbd1d4ed9d19cc21a2e1",
        ),
        "shared-data.tar.gz": (
            17817,
            "sha256:314fdcce2249523ec713a428ca49262bb7219629f23a670bcb231208926f378a",
        ),
    }

    assert {
        item.relative_path.as_posix(): (item.size_bytes, item.sha256)
        for item in manifest.artifacts
    } == expected
    expected_urls = {
        "fixtures/keys/75cbd8e47c923053.key": (
            "https://drive.usercontent.google.com/download?"
            "id=1vcpEI_cdLXOQD2SV_RaD5PtuCylYFD1t&export=download&confirm=t"
        ),
        "fixtures/keys/da75811916eb78a6.key": (
            "https://drive.usercontent.google.com/download?"
            "id=1weyE4CE_WeEmG-K51XitQZwlvCscNchV&export=download&confirm=t"
        ),
        "models/llama-3.2-3b-instruct-q4-k-m/log-linked.tar": (
            "https://drive.usercontent.google.com/download?"
            "id=1ZQz1IK6vCSwcUy_VRN-DEhc_n8RTqfN3&export=download&confirm=t"
        ),
        "models/llama-3.2-3b-instruct-q4-k-m/proxy.tar": (
            "https://drive.usercontent.google.com/download?"
            "id=1aLVIRtfJoov7xpBhsEU1MKMHszkVqeGh&export=download&confirm=t"
        ),
        "models/qwen2.5-7b-instruct-q4-k-m/log-linked.tar": (
            "https://drive.usercontent.google.com/download?"
            "id=1ObsnfSCsHRtk65SYIymWV_CM40GH-7Fl&export=download&confirm=t"
        ),
        "models/qwen2.5-7b-instruct-q4-k-m/proxy.tar": (
            "https://drive.usercontent.google.com/download?"
            "id=1OyA209lQJORSIHwbjuXUyIOPfT6sTpAj&export=download&confirm=t"
        ),
        "shared-data.tar.gz": (
            "https://drive.usercontent.google.com/download?"
            "id=1OoRzilk6MUpbHhG99V79FIgbLflDqBgc&export=download&confirm=t"
        ),
    }
    assert manifest.is_published is True
    assert {
        item.relative_path.as_posix(): item.download_url
        for item in manifest.artifacts
    } == expected_urls
