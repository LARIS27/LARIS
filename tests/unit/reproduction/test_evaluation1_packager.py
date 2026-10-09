from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
from pathlib import Path

import pytest

from evaluation.scripts.prepare_public_evaluation_release import main as prepare_release_main

from laris_onpremise.reproduction.evaluation1.archive import (
    read_safe_tar_member,
    sha256_file,
    write_deterministic_tar,
)
from laris_onpremise.reproduction.evaluation1.contracts import (
    CaptureMode,
    PublicBundleEntry,
    PublicBundleIndex,
    load_model_license_registry,
)
from laris_onpremise.reproduction.evaluation1.packager import (
    BundleSelection,
    PackageError,
    SourceMember,
    audit_release_sources_from_spec,
    atomic_publish,
    index_document,
    package_mode_archive,
    prepare_release_from_spec,
    validate_publication_gates,
)
from laris_onpremise.reproduction.evaluation1.resolver import Evaluation1Release
from laris_onpremise.reproduction.evaluation1.verifier import _verify_bundle_members


MODEL_ID = "llama-3.2-3b-instruct-q4-k-m"
QWEN_MODEL_ID = "qwen2.5-7b-instruct-q4-k-m"


def test_inventory_only_audits_four_sources_without_writing_release(
    tmp_path: Path,
) -> None:
    sources: list[dict[str, object]] = []
    for position in range(4):
        archive = tmp_path / f"source-{position}.tar.gz"
        archive.write_bytes(f"sealed-source-{position}".encode("ascii"))
        sources.append(
            {
                "model_id": MODEL_ID if position < 2 else QWEN_MODEL_ID,
                "mode": "PROXY" if position % 2 == 0 else "LOG_LINKED",
                "archive_path": str(archive),
                "archive_sha256": sha256_file(archive),
            }
        )
    spec = tmp_path / "inventory-spec.json"
    spec.write_text(
        json.dumps({"schema_version": "1.0", "sources": sources}),
        encoding="utf-8",
    )

    receipt = audit_release_sources_from_spec(spec)

    assert receipt["verification_status"] == "VERIFIED"
    assert receipt["source_count"] == 4
    assert receipt["total_source_bytes"] == sum(
        Path(str(source["archive_path"])).stat().st_size for source in sources
    )
    assert all(source["digest_matches"] is True for source in receipt["sources"])
    assert not (tmp_path / "evaluation1-release-v1").exists()


def test_inventory_only_rejects_changed_source_digest(tmp_path: Path) -> None:
    sources: list[dict[str, object]] = []
    for position in range(4):
        archive = tmp_path / f"source-{position}.tar.gz"
        archive.write_bytes(f"sealed-source-{position}".encode("ascii"))
        sources.append(
            {
                "model_id": MODEL_ID if position < 2 else QWEN_MODEL_ID,
                "mode": "PROXY" if position % 2 == 0 else "LOG_LINKED",
                "archive_path": str(archive),
                "archive_sha256": sha256_file(archive),
            }
        )
    sources[-1]["archive_sha256"] = "sha256:" + "0" * 64
    spec = tmp_path / "inventory-spec.json"
    spec.write_text(
        json.dumps({"schema_version": "1.0", "sources": sources}),
        encoding="utf-8",
    )

    with pytest.raises(PackageError, match="source archive digest"):
        audit_release_sources_from_spec(spec)


def test_inventory_only_cli_does_not_require_destination(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sources: list[dict[str, object]] = []
    for position in range(4):
        archive = tmp_path / f"source-{position}.tar.gz"
        archive.write_bytes(f"sealed-source-{position}".encode("ascii"))
        sources.append(
            {
                "model_id": MODEL_ID if position < 2 else QWEN_MODEL_ID,
                "mode": "PROXY" if position % 2 == 0 else "LOG_LINKED",
                "archive_path": str(archive),
                "archive_sha256": sha256_file(archive),
            }
        )
    spec = tmp_path / "inventory-spec.json"
    spec.write_text(
        json.dumps({"schema_version": "1.0", "sources": sources}),
        encoding="utf-8",
    )

    assert prepare_release_main(["--spec", str(spec), "--inventory-only"]) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["verification_status"] == "VERIFIED"
    assert receipt["source_count"] == 4


def _fixture(
    tmp_path: Path,
    *,
    model_id: str = MODEL_ID,
    mode: CaptureMode = CaptureMode.PROXY,
) -> tuple[Path, PublicBundleIndex, tuple[BundleSelection, ...]]:
    source_members: dict[str, bytes] = {"private/evidence.txt": b"must-not-publish"}
    entries: list[PublicBundleEntry] = []
    selections: list[BundleSelection] = []
    for checkpoint in range(1, 7):
        for user in range(1, 11):
            user_ref = f"U{user:02d}"
            source_prefix = f"runs/C{checkpoint}/{user_ref}/bundle"
            public_prefix = f"checkpoints/C{checkpoint}/users/{user_ref}/bundle"
            manifest = json.dumps(
                {"bundle_id": f"bundle-{checkpoint}-{user_ref}"},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8") + b"\n"
            payload = f"payload-{checkpoint}-{user_ref}".encode("ascii")
            manifest_source = source_prefix + "/bundle_manifest.json"
            payload_source = source_prefix + "/users/selected/sessions/selected/runtime_state_payload.bin"
            source_members[manifest_source] = manifest
            source_members[payload_source] = payload
            source_members[source_prefix + "/debug/evidence-copy.json"] = b"private"
            entry = PublicBundleEntry(
                model_id=model_id,
                mode=mode,
                checkpoint_id=f"C{checkpoint}",
                public_user_ref=user_ref,
                subject_ref_kind=mode.subject_kind,
                subject_ref=f"subject-{mode.value.lower()}-{checkpoint}-{user_ref}",
                key_id=(
                    "identity-epoch-test"
                    if mode is CaptureMode.PROXY
                    else "paper-eval-correlation-v1"
                ),
                session_ref=f"session-{checkpoint}-{user_ref}",
                bundle_id=f"bundle-{checkpoint}-{user_ref}",
                bundle_path=public_prefix,
                payload_sha256="sha256:" + hashlib.sha256(payload).hexdigest(),
                payload_size_bytes=len(payload),
                source_bundle_manifest_sha256=(
                    "sha256:" + hashlib.sha256(manifest).hexdigest()
                ),
            )
            entries.append(entry)
            selections.append(
                BundleSelection(
                    entry=entry,
                    members=(
                        SourceMember(
                            source_path=manifest_source,
                            bundle_relative_path="bundle_manifest.json",
                            role="BUNDLE_MANIFEST",
                        ),
                        SourceMember(
                            source_path=payload_source,
                            bundle_relative_path="users/selected/sessions/selected/runtime_state_payload.bin",
                            role="RUNTIME_PAYLOAD",
                        ),
                    ),
                )
            )
    source = tmp_path / f"private-source-{model_id}-{mode.value.lower()}.tar"
    write_deterministic_tar(source, source_members)
    index = PublicBundleIndex(
        schema_version="1.0",
        release_id="evaluation1-release-v1",
        model_id=model_id,
        mode=mode,
        subject_ref_kind=mode.subject_kind,
        entries=tuple(entries),
    )
    return source, index, tuple(selections)


def test_packages_only_explicit_sixty_bundles_and_preserves_source(tmp_path: Path) -> None:
    source, index, selections = _fixture(tmp_path)
    source_digest = sha256_file(source)
    destination = tmp_path / "public-proxy.tar"

    receipt = package_mode_archive(
        source_archive=source,
        expected_source_sha256=source_digest,
        destination=destination,
        index=index,
        selections=selections,
    )

    assert sha256_file(source) == source_digest
    assert receipt["bundle_count"] == 60
    assert receipt["source_archive_sha256"] == source_digest
    with tarfile.open(destination, mode="r:") as archive:
        names = archive.getnames()
    assert names[0] == "bundle-index.json"
    assert len(names) == 121
    assert not any("evidence" in name for name in names)
    embedded = json.loads(
        read_safe_tar_member(destination, "bundle-index.json", limit_bytes=1024 * 1024)
    )
    assert len(embedded["entries"]) == 60


def test_proxy_checkpoint_bundle_shared_by_ten_users_is_written_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_members: dict[str, bytes] = {}
    entries: list[PublicBundleEntry] = []
    selections: list[BundleSelection] = []
    for checkpoint in range(1, 7):
        source_prefix = f"runs/C{checkpoint}/bundle"
        public_prefix = f"checkpoints/C{checkpoint}/bundle"
        package = (f'{{"checkpoint":"C{checkpoint}"}}\n').encode("ascii")
        package_source = source_prefix + "/bundle_manifest.json"
        source_members[package_source] = package
        for user in range(1, 11):
            user_ref = f"U{user:02d}"
            subject_ref = f"subject-{checkpoint}-{user_ref}"
            session_ref = f"session-{checkpoint}-{user_ref}"
            session_slug = hashlib.sha256(session_ref.encode("utf-8")).hexdigest()[:16]
            session = f"users/uh_{subject_ref}/sessions/s_{session_slug}"
            payload = f"payload-{checkpoint}-{user_ref}".encode("ascii")
            payload_source = f"{source_prefix}/{session}/runtime_state_payload.bin"
            source_members[payload_source] = payload
            entry = PublicBundleEntry(
                model_id=MODEL_ID,
                mode=CaptureMode.PROXY,
                checkpoint_id=f"C{checkpoint}",
                public_user_ref=user_ref,
                subject_ref_kind="USER_HMAC",
                subject_ref=subject_ref,
                key_id="identity-epoch-test",
                session_ref=session_ref,
                bundle_id=f"bundle-{checkpoint}-{user_ref}",
                bundle_path=public_prefix,
                payload_sha256="sha256:" + hashlib.sha256(payload).hexdigest(),
                payload_size_bytes=len(payload),
                source_bundle_manifest_sha256=(
                    "sha256:" + hashlib.sha256(package).hexdigest()
                ),
            )
            entries.append(entry)
            selections.append(
                BundleSelection(
                    entry=entry,
                    members=(
                        SourceMember(
                            source_path=package_source,
                            bundle_relative_path="bundle_manifest.json",
                            role="BUNDLE_MANIFEST",
                        ),
                        SourceMember(
                            source_path=payload_source,
                            bundle_relative_path=f"{session}/runtime_state_payload.bin",
                            role="RUNTIME_PAYLOAD",
                        ),
                    ),
                )
            )
    source = tmp_path / "proxy-checkpoints.tar"
    write_deterministic_tar(source, source_members)
    index = PublicBundleIndex(
        schema_version="1.0",
        release_id="evaluation1-release-v1",
        model_id=MODEL_ID,
        mode=CaptureMode.PROXY,
        subject_ref_kind="USER_HMAC",
        entries=tuple(entries),
    )
    destination = tmp_path / "public-proxy.tar"

    receipt = package_mode_archive(
        source_archive=source,
        expected_source_sha256=sha256_file(source),
        destination=destination,
        index=index,
        selections=tuple(selections),
    )

    with tarfile.open(destination, mode="r:") as archive:
        names = archive.getnames()
    assert receipt["bundle_count"] == 60
    assert names.count("checkpoints/C1/bundle/bundle_manifest.json") == 1
    assert len([name for name in names if name.endswith("bundle_manifest.json")]) == 6
    assert len([name for name in names if name.endswith("runtime_state_payload.bin")]) == 60
    _verify_bundle_members(destination, index)

    release_root = tmp_path / "release"
    public_archive = release_root / "models" / MODEL_ID / "proxy.tar"
    public_archive.parent.mkdir(parents=True)
    shutil.copyfile(destination, public_archive)
    release = Evaluation1Release(
        release_root,
        {
            "artifacts": [
                {
                    "model_id": MODEL_ID,
                    "mode": "PROXY",
                    "archive_path": f"models/{MODEL_ID}/proxy.tar",
                    "archive_sha256": sha256_file(public_archive),
                    "archive_size_bytes": public_archive.stat().st_size,
                    "index_sha256": receipt["index_sha256"],
                }
            ]
        },
        {},
        {},
    )
    materialized = tmp_path / "materialized"
    monkeypatch.setattr(
        release,
        "verify_identifier",
        lambda _entry: {"status": "VERIFIED"},
    )

    selected = release.materialize(MODEL_ID, "PROXY", "C1", "U01", materialized)

    assert selected.public_user_ref == "U01"
    assert len(list(materialized.rglob("bundle_manifest.json"))) == 1
    assert len(list(materialized.rglob("runtime_state_payload.bin"))) == 10


def test_rejects_source_digest_change(tmp_path: Path) -> None:
    source, index, selections = _fixture(tmp_path)

    with pytest.raises(PackageError, match="source archive digest"):
        package_mode_archive(
            source_archive=source,
            expected_source_sha256="sha256:" + ("0" * 64),
            destination=tmp_path / "public.tar",
            index=index,
            selections=selections,
        )


def test_publication_gates_require_license_and_synthetic_key_provenance() -> None:
    with pytest.raises(PackageError, match="license review"):
        validate_publication_gates(
            model_ids=(MODEL_ID,),
            key_ids=("identity-epoch-test",),
            license_reviews={},
            key_provenance={"identity-epoch-test": "VERIFIED_SYNTHETIC_ONLY"},
        )

    with pytest.raises(PackageError, match="key provenance"):
        validate_publication_gates(
            model_ids=(MODEL_ID,),
            key_ids=("identity-epoch-test",),
            license_reviews={MODEL_ID: "PUBLIC_APPROVED"},
            key_provenance={},
        )


def test_interrupted_atomic_publish_removes_candidate(tmp_path: Path) -> None:
    destination = tmp_path / "evaluation1-release-v1"

    def fail(candidate: Path) -> None:
        (candidate / "partial.txt").write_text("partial", encoding="utf-8")
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        atomic_publish(destination, fail)

    assert not destination.exists()
    assert not list(tmp_path.glob("evaluation1-release-v1.incomplete-*"))


def test_prepare_release_from_four_sealed_sources(tmp_path: Path) -> None:
    registry_path = (
        Path(__file__).resolve().parents[3]
        / "evaluation"
        / "experiment_setup"
        / "model_licenses.yaml"
    )
    registry = load_model_license_registry(registry_path)
    source_specs: list[dict[str, object]] = []
    key_ids: set[str] = set()
    for model_id in (MODEL_ID, QWEN_MODEL_ID):
        for mode in (CaptureMode.PROXY, CaptureMode.LOG_LINKED):
            source_root = tmp_path / f"source-{model_id}-{mode.value}"
            source_root.mkdir()
            source, index, selections = _fixture(
                source_root,
                model_id=model_id,
                mode=mode,
            )
            key_ids.update(entry.key_id for entry in index.entries)
            source_specs.append(
                {
                    "archive_path": str(source),
                    "archive_sha256": sha256_file(source),
                    "index": index_document(index),
                    "selections": [
                        {
                            "checkpoint_id": selection.entry.checkpoint_id,
                            "public_user_ref": selection.entry.public_user_ref,
                            "members": [
                                {
                                    "source_path": member.source_path,
                                    "bundle_relative_path": member.bundle_relative_path,
                                    "role": member.role,
                                }
                                for member in selection.members
                            ],
                        }
                        for selection in selections
                    ],
                }
            )

    notices_path = tmp_path / "notices.md"
    notices_path.write_text(
        "\n\n".join(
            notice
            for record in registry.values()
            for notice in record.required_notices
        )
        + "\n",
        encoding="utf-8",
    )
    reviews: list[dict[str, object]] = []
    for record in registry.values():
        source = tmp_path / f"{record.model_id}.license.txt"
        source.write_text(record.license_name + "\n", encoding="utf-8")
        reviews.append(
            {
                "model_id": record.model_id,
                "kv_redistribution_decision": "PUBLIC_APPROVED",
                "reviewer": "test-reviewer",
                "license_source_path": str(source),
                "license_source_sha256": sha256_file(source),
            }
        )
    key_rows: list[dict[str, object]] = []
    for position, key_id in enumerate(sorted(key_ids), start=1):
        key_path = tmp_path / f"fixture-key-{position}.bin"
        key_path.write_bytes(bytes([position]) * 32)
        key_rows.append(
            {
                "key_id": key_id,
                "status": "VERIFIED_SYNTHETIC_ONLY",
                "key_path": str(key_path),
                "key_sha256": sha256_file(key_path),
                "fingerprint": hashlib.sha256(key_path.read_bytes()).hexdigest()[:16],
            }
        )
    identity_fixtures = tmp_path / "identity-fixtures.json"
    identity_fixtures.write_text(
        json.dumps({"schema_version": "1.0", "synthetic_only": True}) + "\n",
        encoding="utf-8",
    )
    shared_data = tmp_path / "shared-data.tar.gz"
    shared_data.write_bytes(b"sealed synthetic questions and oracle fixture")
    spec = {
        "schema_version": "1.0",
        "release_id": "evaluation1-release-v1",
        "sources": source_specs,
        "third_party_notices_path": str(notices_path),
        "identity_fixtures_path": str(identity_fixtures),
        "identity_fixtures_sha256": sha256_file(identity_fixtures),
        "shared_data_path": str(shared_data),
        "shared_data_sha256": sha256_file(shared_data),
        "license_reviews": reviews,
        "key_provenance": key_rows,
    }
    spec_path = tmp_path / "release-spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    destination = tmp_path / "evaluation1-release-v1"

    receipt = prepare_release_from_spec(spec_path, destination, registry_path)

    assert receipt["verification_status"] == "VERIFIED"
    assert receipt["payload_count"] == 240
    assert destination.is_dir()
    assert (destination / "fixtures" / "identity-fixtures.json").is_file()
    assert (destination / "fixtures" / "key-provenance.json").is_file()
    assert len(list((destination / "fixtures" / "keys").glob("*.key"))) == 2
    assert (destination / "shared-data.tar.gz").is_file()
    assert not list(tmp_path.glob("evaluation1-release-v1.incomplete-*"))
