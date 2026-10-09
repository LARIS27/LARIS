from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EVALUATION_PAGE = ROOT / "evaluation" / "README.md"
REPRODUCTION_GUIDE = ROOT / "docs" / "reproducibility" / "evaluation1-release.md"
OVERHEAD_PAGE = ROOT / "evaluation" / "results" / "runtime-overhead" / "README.md"
COMPARISON_PAGE = (
    ROOT
    / "evaluation"
    / "results"
    / "attribution-accuracy-and-cross-user-isolation"
    / "comparison.md"
)
LICENSE_PAGE = ROOT / "evaluation" / "publication" / "LICENSES.md"
LLAMA_NOTICE = ROOT / "evaluation" / "publication" / "NOTICE.llama-3.2.txt"
QWEN_NOTICE = ROOT / "evaluation" / "publication" / "NOTICE.qwen2.5.txt"
SUMMARY = (
    ROOT
    / "evaluation"
    / "results"
    / "runtime-overhead"
    / "llama-3.2-3b"
    / "summary.json"
)
PUBLIC_RELEASE_RECEIPT = (
    ROOT / "evaluation" / "run_records" / "evaluation1-public-release.json"
)
ROOT_README = ROOT / "README.md"
QUICKSTART = ROOT / "docs" / "QUICKSTART.md"
RELEASE_TEMPLATE = (
    ROOT / "evaluation" / "publication" / "releases" / "evaluation1-release-v1"
)


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_evaluation_page_links_complete_publication_contract() -> None:
    page = _text(EVALUATION_PAGE)
    assert "docs/reproducibility/evaluation1-release.md" in page
    assert "results/runtime-overhead/README.md" in page
    assert "evaluation/publication/LICENSES.md" in page
    assert "25" in page and "COMPLETE" in page
    assert "model weights are not redistributed" in page


def test_evaluation_page_displays_only_required_model_notices() -> None:
    page = _text(EVALUATION_PAGE)
    normalized_page = " ".join(page.split())
    assert "Built with Llama" in page
    assert (
        "Llama 3.2 is licensed under the Llama 3.2 Community License, "
        "Copyright © Meta Platforms, Inc. All Rights Reserved."
        in normalized_page
    )
    assert "Qwen2.5" in page
    assert "Apache-2.0" in page
    assert "Built with Qwen" not in page


def test_reproduction_guide_publishes_exact_checkpoint_and_question_contract() -> None:
    guide = _text(REPRODUCTION_GUIDE)
    expected = {
        "C1": "R01–R03",
        "C2": "R04–R06",
        "C3": "R07–R09",
        "C4": "R10–R12",
        "C5": "R13–R15",
        "C6": "R16–R18",
    }
    for checkpoint, records in expected.items():
        assert checkpoint in guide
        assert records in guide
    assert '"This is record {record_id} that I organized."' in guide
    assert "Return exactly these five lines:" in guide
    assert "Material: <exact value or Unknown>" in guide
    assert "laris reproduce evaluation1 list" in guide
    assert "laris reproduce evaluation1 rehost" in guide
    assert "laris reproduce evaluation1 verify-recall" in guide
    assert "prepare_public_evaluation_release.py --spec" in guide
    assert "--inventory-only" in guide
    assert "240" in guide
    assert "PUBLIC_APPROVED" in guide


def test_model_revisions_and_only_needed_notices_are_published() -> None:
    licenses = _text(LICENSE_PAGE)
    llama_notice = _text(LLAMA_NOTICE)
    assert "0cb88a4f764b7a12671c53f0838cd831a0843b95" in licenses
    assert "bb5d59e06d9551d752d08b292a50eb208b07ab1f" in licenses
    assert "llama-3.2-community" in licenses
    assert "Apache-2.0" in licenses
    assert (
        "https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct/blob/main/LICENSE.txt"
        in licenses
    )
    assert (
        "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/blob/main/LICENSE"
        in licenses
    )
    assert "Built with Llama" in llama_notice
    assert (
        "Llama 3.2 is licensed under the Llama 3.2 Community License, "
        "Copyright © Meta Platforms, Inc. All Rights Reserved."
        in llama_notice
    )
    assert not QWEN_NOTICE.exists()
    assert "model weights" in licenses
    assert "not redistributed" in licenses


def test_overhead_page_is_bound_to_completed_machine_summary() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    page = _text(OVERHEAD_PAGE)
    normalized_page = " ".join(page.split())
    assert summary["state"] == "COMPLETE"
    assert summary["trial_count"] == 25
    assert summary["pair_count"] == 5
    assert "5 balanced blocks x 5 conditions" in normalized_page
    assert "123,529" in page
    assert "1,080" in page
    for mode, label in (("PROXY", "Proxy"), ("LOG_LINKED", "Log-linked")):
        result = summary["conditions"][mode]
        assert label in page
        assert f'{result["mean_kv_capture_work_seconds"]:.3f}' in page
        assert f'{result["mean_kv_overhead_percent"]:.3f}%' in page
        for value in result["kv_capture_work_ci95_seconds"]:
            assert f"{value:.3f}" in page
        for value in result["kv_overhead_ci95_percent"]:
            assert f"{value:.3f}" in page
        assert result["zero_loss"] is True
    assert "O(m,b) = 100 × (F(m,b) / C(m,b) − 1)" in page
    assert "not the end-to-end overhead by itself" in page
    assert "lost events: 0" in page


def test_accuracy_comparison_points_to_replay_guide_without_changing_results() -> None:
    comparison = _text(COMPARISON_PAGE)
    assert "../../../docs/reproducibility/evaluation1-release.md" in comparison
    assert "870/900 (96.67%)" in comparison
    assert "876/900 (97.33%)" in comparison


def test_public_release_receipt_is_sanitized_and_complete() -> None:
    raw = _text(PUBLIC_RELEASE_RECEIPT)
    receipt = json.loads(raw)
    assert receipt["release_id"] == "evaluation1-release-v1"
    assert receipt["publication_boundary"] == {
        "external_release_size_bytes": 13242821929,
        "file_count": 14,
        "model_weights_included": False,
        "payload_count": 240,
    }
    assert receipt["verification"] == {
        "independent_verifier": "VERIFIED",
        "materialize_smoke_cases": 4,
        "materialize_smoke_status": "VERIFIED",
        "packager_verification": "VERIFIED",
    }
    assert {
        (artifact["model_id"], artifact["mode"], artifact["bundle_count"])
        for artifact in receipt["artifacts"]
    } == {
        ("llama-3.2-3b-instruct-q4-k-m", "PROXY", 60),
        ("llama-3.2-3b-instruct-q4-k-m", "LOG_LINKED", 60),
        ("qwen2.5-7b-instruct-q4-k-m", "PROXY", 60),
        ("qwen2.5-7b-instruct-q4-k-m", "LOG_LINKED", 60),
    }
    assert {key["key_id"] for key in receipt["fixture_keys"]} == {
        "identity-epoch-20260915-5f0d02cf",
        "paper-eval-correlation-v1",
    }
    assert "C:\\Users\\" not in raw
    assert "D:\\" not in raw
    assert "private_seed" not in raw


def test_tracked_release_template_contains_only_verified_small_metadata() -> None:
    expected = {
        "checksums.sha256": "e46a62557f6e7394c4a9129fcc5cab6682360c69f59aa108527fd0d785dd9662",
        "release-manifest.json": "7751fee3665646d992305903dabe402e411f0b2ddd7674beafa7f9b52fe457e7",
        "THIRD_PARTY_NOTICES.md": "acd9433db6c08269ee1dcc2c21466ae31db9a973b84cd024a4a291d9f36961b5",
        "fixtures/identity-fixtures.json": "f148530afc1edff9d468a167c2d717bb77ebb90d77ae2f2c025999cd17998394",
        "fixtures/key-provenance.json": "d9b1bfde1ae8bd8c80bdbdfa8a5faabd45814da8f9c9bbfe81dfec57a8a5f174",
        "licenses/LICENSE.llama-3.2.txt": "8cc15535a8a34b41888f644b339a1a9eb428af793a4f5e24df58a3e5b1487d74",
        "licenses/LICENSE.qwen2.5-apache-2.0.txt": "832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e",
    }
    for relative, digest in expected.items():
        payload = (RELEASE_TEMPLATE / relative).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == digest
    assert (RELEASE_TEMPLATE / "README.md").is_file()
    assert (RELEASE_TEMPLATE / "artifact-locations.json").is_file()
    assert not (RELEASE_TEMPLATE / "models").exists()
    assert not (RELEASE_TEMPLATE / "shared-data.tar.gz").exists()
    assert not (RELEASE_TEMPLATE / "fixtures" / "keys").exists()


def test_public_release_receipt_binds_published_location_manifest() -> None:
    receipt = json.loads(PUBLIC_RELEASE_RECEIPT.read_text(encoding="utf-8"))
    location_path = RELEASE_TEMPLATE / "artifact-locations.json"

    assert receipt["publication_status"] == "PUBLISHED"
    assert receipt["artifact_locations_sha256"] == (
        "sha256:" + hashlib.sha256(location_path.read_bytes()).hexdigest()
    )


def test_verified_release_metadata_keeps_lf_bytes_on_checkout() -> None:
    paths = (
        "evaluation/publication/releases/evaluation1-release-v1/checksums.sha256",
        "evaluation/publication/releases/evaluation1-release-v1/licenses/LICENSE.llama-3.2.txt",
        "evaluation/publication/releases/evaluation1-release-v1/licenses/LICENSE.qwen2.5-apache-2.0.txt",
    )
    result = subprocess.run(
        ["git", "check-attr", "eol", "--", *paths],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.splitlines() == [f"{path}: eol: lf" for path in paths]


def test_public_entry_docs_define_the_complete_offline_reproduction_flow() -> None:
    combined = "\n".join(
        _text(path)
        for path in (ROOT_README, QUICKSTART, EVALUATION_PAGE, REPRODUCTION_GUIDE)
    )
    for required in (
        "evaluation/publication/releases/evaluation1-release-v1",
        "evaluation/publication/external/evaluation1-release-v1",
        "stage_public_evaluation_release.py initialize",
        "stage_public_evaluation_release.py checklist",
        "stage_public_evaluation_release.py verify",
        "laris reproduce evaluation1 list",
        "laris reproduce evaluation1 show",
        "laris reproduce evaluation1 rehost",
        "laris reproduce evaluation1 verify-recall",
    ):
        assert required in combined
    assert "paper-final-" + "release" not in combined
    assert "reviewer" not in combined.lower()
    quickstart = _text(QUICKSTART).lower()
    assert "dashboard" not in quickstart
    assert "run_dashboard" not in quickstart
