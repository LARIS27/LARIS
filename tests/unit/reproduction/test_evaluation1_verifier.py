from __future__ import annotations

import io
import hashlib
import json
import tarfile
from pathlib import Path

import pytest

from laris_onpremise.reproduction.evaluation1.archive import (
    ArchiveError,
    inspect_safe_tar,
    sha256_file,
    write_checksums,
    write_deterministic_tar,
)
from laris_onpremise.reproduction.evaluation1.contracts import (
    ContractError,
    load_model_license_registry,
)
from laris_onpremise.reproduction.evaluation1.verifier import (
    ReleaseVerificationError,
    reject_forbidden_release_files,
    verify_checksums,
    verify_exact_release_indexes,
    verify_public_release,
    verify_required_notices,
)
from evaluation.scripts.verify_public_evaluation_release import build_parser


DIGEST = "sha256:" + ("a" * 64)
MODELS = (
    "llama-3.2-3b-instruct-q4-k-m",
    "qwen2.5-7b-instruct-q4-k-m",
)


def _registry_path() -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / "evaluation"
        / "experiment_setup"
        / "model_licenses.yaml"
    )


def _entry(model_id: str, mode: str, checkpoint: int, user: int) -> dict[str, object]:
    user_ref = f"U{user:02d}"
    subject_kind = "USER_HMAC" if mode == "PROXY" else "ENDPOINT_REF"
    return {
        "model_id": model_id,
        "mode": mode,
        "checkpoint_id": f"C{checkpoint}",
        "public_user_ref": user_ref,
        "subject_ref_kind": subject_kind,
        "subject_ref": f"subject-{mode.lower()}-{checkpoint}-{user_ref}",
        "key_id": "paper-key-v1",
        "session_ref": f"session-{checkpoint}-{user_ref}",
        "bundle_id": f"bundle-{checkpoint}-{user_ref}",
        "bundle_path": f"checkpoints/C{checkpoint}/users/{user_ref}/bundle",
        "payload_size_bytes": 1,
        "payload_sha256": DIGEST,
        "source_bundle_manifest_sha256": DIGEST,
    }


def _index(model_id: str, mode: str) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "release_id": "evaluation1-release-v1",
        "model_id": model_id,
        "mode": mode,
        "subject_ref_kind": "USER_HMAC" if mode == "PROXY" else "ENDPOINT_REF",
        "entries": [
            _entry(model_id, mode, checkpoint, user)
            for checkpoint in range(1, 7)
            for user in range(1, 11)
        ],
    }


def _unsafe_tar(path: Path, members: list[tarfile.TarInfo]) -> None:
    with tarfile.open(path, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for member in members:
            if member.isreg():
                member.size = 1
                archive.addfile(member, io.BytesIO(b"x"))
            else:
                archive.addfile(member)


def test_deterministic_tar_bytes_are_stable(tmp_path: Path) -> None:
    members = {"b.txt": b"second", "a.txt": b"first"}
    first = tmp_path / "first.tar"
    second = tmp_path / "second.tar"

    write_deterministic_tar(first, members)
    write_deterministic_tar(second, dict(reversed(tuple(members.items()))))

    assert first.read_bytes() == second.read_bytes()
    inspected = inspect_safe_tar(first)
    assert tuple(item.name for item in inspected) == ("a.txt", "b.txt")
    assert all(item.mtime == 0 and item.uid == 0 and item.gid == 0 for item in inspected)


@pytest.mark.parametrize(
    "members",
    (
        [tarfile.TarInfo("../escape")],
        [tarfile.TarInfo("/absolute")],
        [tarfile.TarInfo("duplicate"), tarfile.TarInfo("duplicate")],
        [tarfile.TarInfo("symlink")],
        [tarfile.TarInfo("hardlink")],
        [tarfile.TarInfo("device")],
    ),
)
def test_rejects_unsafe_tar_members(tmp_path: Path, members: list[tarfile.TarInfo]) -> None:
    name = members[0].name
    if name == "symlink":
        members[0].type = tarfile.SYMTYPE
        members[0].linkname = "target"
    elif name == "hardlink":
        members[0].type = tarfile.LNKTYPE
        members[0].linkname = "target"
    elif name == "device":
        members[0].type = tarfile.CHRTYPE
    path = tmp_path / "unsafe.tar"
    _unsafe_tar(path, members)

    with pytest.raises(ArchiveError):
        inspect_safe_tar(path)


def test_checksums_detect_tampering(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_bytes(b"original")
    (tmp_path / "b.txt").write_bytes(b"stable")
    write_checksums(tmp_path, ("a.txt", "b.txt"))

    verify_checksums(tmp_path)
    (tmp_path / "a.txt").write_bytes(b"tampered")

    with pytest.raises(ReleaseVerificationError, match="checksum"):
        verify_checksums(tmp_path)


@pytest.mark.parametrize("delta", (-1, 1))
def test_rejects_239_or_241_payload_inventory(delta: int) -> None:
    documents = [_index(model, mode) for model in MODELS for mode in ("PROXY", "LOG_LINKED")]
    entries = documents[-1]["entries"]
    assert isinstance(entries, list)
    if delta < 0:
        entries.pop()
    else:
        extra = dict(entries[-1])
        extra["public_user_ref"] = "U11"
        entries.append(extra)

    with pytest.raises((ContractError, ReleaseVerificationError), match="(?:60|240|U01)"):
        verify_exact_release_indexes(documents)


def test_rejects_missing_required_model_notice(tmp_path: Path) -> None:
    with pytest.raises(ReleaseVerificationError, match="Built with Llama"):
        verify_required_notices("Apache License 2.0", _registry_path())


@pytest.mark.parametrize("filename", ("model.gguf", "snapshot.vmdk", "guest.vmem", "state.vmsn"))
def test_rejects_model_weight_or_vm_files(tmp_path: Path, filename: str) -> None:
    (tmp_path / filename).write_bytes(b"forbidden")

    with pytest.raises(ReleaseVerificationError, match="forbidden"):
        reject_forbidden_release_files(tmp_path)


def test_checksum_file_is_canonical_and_sorted(tmp_path: Path) -> None:
    (tmp_path / "z.json").write_text(json.dumps({"z": 1}), encoding="utf-8")
    (tmp_path / "a.json").write_text(json.dumps({"a": 1}), encoding="utf-8")

    checksum_path = write_checksums(tmp_path, ("z.json", "a.json"))

    lines = checksum_path.read_text(encoding="ascii").splitlines()
    assert lines == sorted(lines, key=lambda line: line.split("  ", 1)[1])
    assert checksum_path.read_bytes().endswith(b"\n")


def test_complete_release_verifies_without_extracting(tmp_path: Path) -> None:
    artifacts: list[dict[str, object]] = []
    for model_id in MODELS:
        for mode in ("PROXY", "LOG_LINKED"):
            document = _index(model_id, mode)
            members: dict[str, bytes] = {}
            entries = document["entries"]
            assert isinstance(entries, list)
            for raw in entries:
                assert isinstance(raw, dict)
                prefix = str(raw["bundle_path"])
                payload = f"{model_id}:{mode}:{prefix}".encode("utf-8")
                bundle_manifest = json.dumps(
                    {"bundle_id": raw["bundle_id"]},
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8") + b"\n"
                raw["payload_size_bytes"] = len(payload)
                raw["payload_sha256"] = "sha256:" + hashlib.sha256(payload).hexdigest()
                raw["source_bundle_manifest_sha256"] = (
                    "sha256:" + hashlib.sha256(bundle_manifest).hexdigest()
                )
                members[prefix + "/bundle_manifest.json"] = bundle_manifest
                members[
                    prefix + "/users/selected/sessions/selected/runtime_state_payload.bin"
                ] = payload
            index_bytes = json.dumps(
                document,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8") + b"\n"
            members["bundle-index.json"] = index_bytes
            relative = f"models/{model_id}/{mode.lower().replace('_', '-')}.tar"
            archive_path = tmp_path / relative
            write_deterministic_tar(archive_path, members)
            artifacts.append(
                {
                    "model_id": model_id,
                    "mode": mode,
                    "archive_path": relative,
                    "archive_sha256": sha256_file(archive_path),
                    "archive_size_bytes": archive_path.stat().st_size,
                    "index_sha256": "sha256:" + hashlib.sha256(index_bytes).hexdigest(),
                    "bundle_count": 60,
                }
            )

    registry = load_model_license_registry(_registry_path())
    license_rows: list[dict[str, object]] = []
    notices: list[str] = []
    for record in registry.values():
        license_path = tmp_path / "licenses" / record.license_file_name
        license_path.parent.mkdir(parents=True, exist_ok=True)
        license_path.write_text(record.license_name + "\n", encoding="utf-8")
        notices.extend(record.required_notices)
        license_rows.append(
            {
                "model_id": record.model_id,
                "source_revision": record.source_revision,
                "license_id": record.license_id,
                "license_url": record.license_url,
                "license_path": license_path.relative_to(tmp_path).as_posix(),
                "license_sha256": sha256_file(license_path),
                "redistributes_weights": False,
                "kv_redistribution_decision": "PUBLIC_APPROVED",
                "reviewer": "test-reviewer",
            }
        )
    (tmp_path / "THIRD_PARTY_NOTICES.md").write_text(
        "\n\n".join(notices) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "shared-data.tar.gz").write_bytes(b"sealed shared data")
    fixtures_root = tmp_path / "fixtures"
    keys_root = fixtures_root / "keys"
    keys_root.mkdir(parents=True)
    (fixtures_root / "identity-fixtures.json").write_text(
        json.dumps({"schema_version": "1.0", "synthetic_only": True}) + "\n",
        encoding="utf-8",
    )
    key_path = keys_root / "paper-key-v1.key"
    key_path.write_bytes(b"k" * 32)
    (fixtures_root / "key-provenance.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "synthetic_only": True,
                "keys": [
                    {
                        "key_id": "paper-key-v1",
                        "status": "VERIFIED_SYNTHETIC_ONLY",
                        "fingerprint": "test",
                        "key_path": key_path.relative_to(tmp_path).as_posix(),
                        "key_sha256": sha256_file(key_path),
                    }
                ],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "1.0",
        "release_id": "evaluation1-release-v1",
        "source_evaluation": "Evaluation1",
        "weights_included": False,
        "artifacts": artifacts,
        "model_licenses": license_rows,
    }
    (tmp_path / "release-manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    paths = sorted(
        path.relative_to(tmp_path).as_posix()
        for path in tmp_path.rglob("*")
        if path.is_file()
    )
    write_checksums(tmp_path, paths)

    receipt = verify_public_release(tmp_path, _registry_path())

    assert receipt["verification_status"] == "VERIFIED"
    assert receipt["payload_count"] == 240


def test_verifier_wrapper_accepts_optional_location_manifest() -> None:
    args = build_parser().parse_args(
        [
            "--release-root",
            "release",
            "--locations",
            "metadata/artifact-locations.json",
        ]
    )

    assert args.locations == Path("metadata/artifact-locations.json")
