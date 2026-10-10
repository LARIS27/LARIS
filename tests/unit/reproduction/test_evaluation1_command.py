from __future__ import annotations

import hashlib
import io
import json
import tarfile
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

import laris_onpremise.reproduction.evaluation1.command as evaluation1_command
from laris_onpremise.attribution.log_linked.endpoint import derive_endpoint_ref
from laris_onpremise.core.records import IdentityKey, IdentityKeyRegistry, derive_user_hmac
from laris_onpremise.reproduction.evaluation1.archive import sha256_file, write_deterministic_tar
from laris_onpremise.reproduction.evaluation1.command import main as evaluation1_main
from laris_onpremise.reproduction.evaluation1.resolver import (
    Evaluation1Release,
    ReleaseResolutionError,
)


MODEL_ID = "llama-3.2-3b-instruct-q4-k-m"
RUN_ID = "run-evaluation1-public-test"


def _release(tmp_path: Path) -> Path:
    release = tmp_path / "release"
    release.mkdir()
    proxy_key = b"p" * 32
    correlation_key = b"c" * 32
    tenant_id = "evaluation1-synthetic"
    subjects = []
    endpoints = []
    proxy_registry = IdentityKeyRegistry((IdentityKey("proxy-key", proxy_key),))
    for user in range(1, 11):
        user_ref = f"U{user:02d}"
        subject = f"synthetic-subject-{user_ref}"
        endpoint_value = f"192.0.2.{user}"
        subjects.append(
            {
                "public_user_ref": user_ref,
                "authenticated_subject_id": subject,
            }
        )
        endpoints.append(
            {
                "public_user_ref": user_ref,
                "endpoint_type": "IPV4",
                "endpoint_value": endpoint_value,
            }
        )

    artifacts = []
    for mode in ("PROXY", "LOG_LINKED"):
        entries = []
        members: dict[str, bytes] = {}
        for checkpoint in range(1, 7):
            for user in range(1, 11):
                user_ref = f"U{user:02d}"
                prefix = f"checkpoints/C{checkpoint}/users/{user_ref}/bundle"
                payload = f"{mode}-{checkpoint}-{user_ref}".encode("ascii")
                manifest = b'{"package_type":"laris_rehost_bundle"}\n'
                if mode == "PROXY":
                    subject_ref = derive_user_hmac(
                        proxy_registry,
                        "proxy-key",
                        tenant_id,
                        f"synthetic-subject-{user_ref}",
                    )
                    key_id = "proxy-key"
                    kind = "USER_HMAC"
                else:
                    subject_ref = derive_endpoint_ref(
                        correlation_key=correlation_key,
                        endpoint_type="IPV4",
                        endpoint_value=f"192.0.2.{user}",
                    )
                    key_id = "correlation-key"
                    kind = "ENDPOINT_REF"
                entries.append(
                    {
                        "model_id": MODEL_ID,
                        "mode": mode,
                        "checkpoint_id": f"C{checkpoint}",
                        "public_user_ref": user_ref,
                        "subject_ref_kind": kind,
                        "subject_ref": subject_ref,
                        "key_id": key_id,
                        "session_ref": f"session-{checkpoint}-{user_ref}",
                        "bundle_id": f"bundle-{mode}-{checkpoint}-{user_ref}",
                        "bundle_path": prefix,
                        "payload_size_bytes": len(payload),
                        "payload_sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
                        "source_bundle_manifest_sha256": (
                            "sha256:" + hashlib.sha256(manifest).hexdigest()
                        ),
                    }
                )
                members[prefix + "/bundle_manifest.json"] = manifest
                members[
                    prefix + "/users/selected/sessions/selected/runtime_state_payload.bin"
                ] = payload
        index = {
            "schema_version": "1.0",
            "release_id": "evaluation1-release-v1",
            "model_id": MODEL_ID,
            "mode": mode,
            "subject_ref_kind": "USER_HMAC" if mode == "PROXY" else "ENDPOINT_REF",
            "entries": entries,
        }
        index_bytes = json.dumps(index, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        members["bundle-index.json"] = index_bytes
        archive_rel = Path("models") / MODEL_ID / f"{mode.lower().replace('_', '-')}.tar"
        archive = release / archive_rel
        write_deterministic_tar(archive, members)
        artifacts.append(
            {
                "model_id": MODEL_ID,
                "mode": mode,
                "archive_path": archive_rel.as_posix(),
                "archive_sha256": sha256_file(archive),
                "archive_size_bytes": archive.stat().st_size,
                "index_sha256": "sha256:" + hashlib.sha256(index_bytes).hexdigest(),
                "bundle_count": 60,
            }
        )

    fixtures = release / "fixtures"
    keys = fixtures / "keys"
    keys.mkdir(parents=True)
    proxy_path = keys / "proxy.key"
    proxy_path.write_bytes(proxy_key)
    correlation_path = keys / "correlation.key"
    correlation_path.write_bytes(correlation_key)
    (fixtures / "identity-fixtures.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "synthetic_only": True,
                "tenant_id": tenant_id,
                "subjects": subjects,
                "endpoints": endpoints,
            }
        ),
        encoding="utf-8",
    )
    (fixtures / "key-provenance.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "synthetic_only": True,
                "keys": [
                    {
                        "key_id": "proxy-key",
                        "status": "VERIFIED_SYNTHETIC_ONLY",
                        "key_path": proxy_path.relative_to(release).as_posix(),
                        "key_sha256": sha256_file(proxy_path),
                    },
                    {
                        "key_id": "correlation-key",
                        "status": "VERIFIED_SYNTHETIC_ONLY",
                        "key_path": correlation_path.relative_to(release).as_posix(),
                        "key_sha256": sha256_file(correlation_path),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    shared = release / "shared-data.tar.gz"
    with tarfile.open(shared, mode="w:gz") as archive:
        for name, value in {
            "checkpoint-schedule.json": {
                "checkpoints": [
                    {
                        "checkpoint_id": f"C{checkpoint}",
                        "record_ids": [f"R{value:02d}" for value in range(3 * checkpoint - 2, 3 * checkpoint + 1)],
                    }
                    for checkpoint in range(1, 7)
                ]
            },
            "questions.jsonl": [
                {
                    "checkpoint_id": "C1",
                    "record_id": "R01",
                    "question": "Where was the item?\nAnswer using retained fields.\n",
                }
            ],
            "oracle.jsonl": [
                {
                    "checkpoint_id": "C1",
                    "public_user_ref": "U01",
                    "record_id": "R01",
                    "expected": "Lab A",
                },
                {
                    "checkpoint_id": "C1",
                    "public_user_ref": "U02",
                    "record_id": "R01",
                    "expected": "Lab B",
                },
            ],
        }.items():
            if name.endswith(".jsonl"):
                payload = b"".join(
                    json.dumps(row, sort_keys=True).encode() + b"\n" for row in value
                )
            else:
                payload = json.dumps(value, sort_keys=True).encode() + b"\n"
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mtime = 0
            archive.addfile(info, io.BytesIO(payload))
    (release / "release-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "release_id": "evaluation1-release-v1",
                "source_evaluation": "Evaluation1",
                "weights_included": False,
                "artifacts": artifacts,
                "model_licenses": [],
            }
        ),
        encoding="utf-8",
    )
    return release


def test_list_and_show_emit_canonical_public_tuple_data(tmp_path: Path) -> None:
    release = _release(tmp_path)
    output = io.StringIO()

    assert evaluation1_main(
        [
            "list",
            "--release-root",
            str(release),
            "--model-id",
            MODEL_ID,
            "--mode",
            "PROXY",
        ],
        output_stream=output,
    ) == 0
    document = json.loads(output.getvalue())
    assert document["count"] == 60
    assert document["entries"][0]["checkpoint_id"] == "C1"

    output = io.StringIO()
    assert evaluation1_main(
        [
            "show",
            "--release-root",
            str(release),
            "--model-id",
            MODEL_ID,
            "--mode",
            "LOG_LINKED",
            "--checkpoint-id",
            "C2",
            "--user-ref",
            "U03",
        ],
        output_stream=output,
    ) == 0
    shown = json.loads(output.getvalue())
    assert shown["identifier_verification"] == "VERIFIED"
    assert shown["subject_ref_kind"] == "ENDPOINT_REF"


def test_proxy_hmac_and_log_linked_endpoint_ref_are_recomputed(tmp_path: Path) -> None:
    release = Evaluation1Release.open(_release(tmp_path))

    proxy = release.resolve(MODEL_ID, "PROXY", "C4", "U07")
    linked = release.resolve(MODEL_ID, "LOG_LINKED", "C4", "U07")

    assert release.verify_identifier(proxy)["status"] == "VERIFIED"
    assert release.verify_identifier(linked)["status"] == "VERIFIED"


def test_verify_recall_selects_the_oracle_for_the_public_user(tmp_path: Path) -> None:
    release = _release(tmp_path)
    response = tmp_path / "response.txt"
    response.write_text("The answer is Lab B.", encoding="utf-8")
    output = io.StringIO()

    assert evaluation1_main(
        [
            "verify-recall",
            "--release-root",
            str(release),
            "--model-id",
            MODEL_ID,
            "--mode",
            "PROXY",
            "--checkpoint-id",
            "C1",
            "--user-ref",
            "U02",
            "--record-id",
            "R01",
            "--response-file",
            str(response),
        ],
        output_stream=output,
    ) == 0
    assert json.loads(output.getvalue())["match"] is True


def test_rejects_record_outside_checkpoint_cohort(tmp_path: Path) -> None:
    release = Evaluation1Release.open(_release(tmp_path))

    with pytest.raises(ReleaseResolutionError, match="checkpoint cohort"):
        release.validate_record("C1", "R04")


def test_archive_digest_is_checked_before_rehost_delegation(tmp_path: Path) -> None:
    root = _release(tmp_path)
    archive = root / "models" / MODEL_ID / "proxy.tar"
    archive.write_bytes(archive.read_bytes() + b"tamper")

    with pytest.raises(ReleaseResolutionError, match="digest"):
        Evaluation1Release.open(root).materialize(
            MODEL_ID,
            "PROXY",
            "C1",
            "U01",
            tmp_path / "selected",
        )


def test_rehost_forwards_current_open_arguments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _release(tmp_path)
    question = tmp_path / "question.txt"
    question.write_text("Where was the item?", encoding="utf-8")
    captured: list[Namespace] = []
    monkeypatch.setattr(
        evaluation1_command.PortableBundleCatalog,
        "open",
        lambda _root: SimpleNamespace(source_run_id=RUN_ID),
    )

    def delegate(arguments: Namespace) -> int:
        captured.append(arguments)
        assert arguments.bundle_root.is_dir()
        assert arguments.bundle_root.parent.name == RUN_ID
        assert arguments.bundle_root.name == "bundle"
        assert arguments.subject_id_file.read_text(encoding="utf-8") == "synthetic-subject-U01"
        assert arguments.question_file.read_text(encoding="utf-8") == "Where was the item?"
        return 0

    result = evaluation1_main(
        [
            "rehost",
            "--release-root",
            str(root),
            "--model-id",
            MODEL_ID,
            "--mode",
            "PROXY",
            "--checkpoint-id",
            "C1",
            "--user-ref",
            "U01",
            "--question-file",
            str(question),
            "--profile",
            str(tmp_path / "profile.yaml"),
        ],
        rehost_runner=delegate,
    )

    assert result == 0
    assert captured[0].endpoint_ref is None
    assert captured[0].session_ref == "session-1-U01"
    assert captured[0].temperature == 1.0
    assert captured[0].top_k == 1
    assert captured[0].top_p == 1.0
    assert captured[0].min_p == 1.0
    assert captured[0].seed == 20260907


def test_rehost_record_removes_only_terminal_line_endings_from_published_question(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _release(tmp_path)
    monkeypatch.setattr(
        evaluation1_command.PortableBundleCatalog,
        "open",
        lambda _root: SimpleNamespace(source_run_id=RUN_ID),
    )

    def delegate(arguments: Namespace) -> int:
        assert arguments.bundle_root.parent.name == RUN_ID
        assert arguments.bundle_root.name == "bundle"
        assert arguments.question_file.read_text(encoding="utf-8") == (
            "Where was the item?\nAnswer using retained fields."
        )
        return 0

    assert evaluation1_main(
        [
            "rehost",
            "--release-root",
            str(root),
            "--model-id",
            MODEL_ID,
            "--mode",
            "PROXY",
            "--checkpoint-id",
            "C1",
            "--user-ref",
            "U01",
            "--record-id",
            "R01",
            "--profile",
            str(tmp_path / "profile.yaml"),
        ],
        rehost_runner=delegate,
    ) == 0
