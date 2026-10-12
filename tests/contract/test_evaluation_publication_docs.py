from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
EVALUATION_PAGE = ROOT / "evaluation" / "README.md"
REPRODUCTION_GUIDE = ROOT / "docs" / "reproducibility" / "evaluation1-release.md"
OVERHEAD_PAGE = ROOT / "evaluation" / "results" / "runtime-overhead" / "README.md"
LOG_LINKED_TRACE_PAGE = (
    ROOT / "evaluation" / "results" / "log-linked-correlation" / "README.md"
)
LOG_LINKED_TRACE_ROOT = LOG_LINKED_TRACE_PAGE.parent
FIGURE3_APPLICATION_LOG = LOG_LINKED_TRACE_ROOT / "source" / "raw_llama_server.log"
FIGURE3_NORMALIZED_EVENTS = (
    LOG_LINKED_TRACE_ROOT / "source" / "normalized_application_events.jsonl"
)
FIGURE3_ATTRIBUTION_RESULTS = (
    LOG_LINKED_TRACE_ROOT / "source" / "attribution_results.jsonl"
)
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
ROOT_README = ROOT / "README.md"
QUICKSTART = ROOT / "docs" / "QUICKSTART.md"
REHOSTING_SETUP = ROOT / "docs" / "REHOSTING_SETUP.md"
MODEL_REGISTRY = ROOT / "evaluation" / "experiment_setup" / "models.yaml"
RELEASE_TEMPLATE = (
    ROOT / "evaluation" / "publication" / "releases" / "evaluation1-release-v1"
)


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_evaluation_page_links_complete_publication_contract() -> None:
    page = _text(EVALUATION_PAGE)
    for directory in (
        "experiment_setup/",
        "publication/",
        "results/",
        "scripts/",
    ):
        assert f"]({directory})" in page
    assert "docs/reproducibility/evaluation1-release.md" in page
    assert "results/runtime-overhead/README.md" in page
    assert "results/log-linked-correlation/README.md" in page
    assert "evaluation/publication/LICENSES.md" in page
    assert "25" in page and "COMPLETE" in page
    assert "model weights are not redistributed" in page


def test_log_linked_figure3_source_is_exactly_provenance_bound() -> None:
    page = _text(LOG_LINKED_TRACE_PAGE)
    normalized_page = " ".join(page.split())
    provenance = json.loads(_text(LOG_LINKED_TRACE_ROOT / "provenance.json"))

    assert page.startswith("# Figure 3 source records at TRACE (4)\n")
    assert "separate from the quantitative evaluation aggregates" in normalized_page
    assert "C1 Log-linked pilot" not in page
    assert provenance["record_purpose"] == "Figure 3 source records"
    assert provenance["log_verbosity"] == "TRACE (4)"
    assert provenance["model_id"] == "llama-3.2-3b-instruct-q4-k-m"
    assert "task 7" in page
    assert "task 6" in page
    assert "task 5" in page
    assert "execution termination" in page
    assert "HTTP completion" in page
    assert "correlation result" in page
    assert "original row" in page
    assert "bff29968a0779ba836a426c2327e9dd16f002edb7a5efaa91291a05a384171c9" in page
    assert "5424d629b6f275eba7d63015d48ef6931e3716bd23efe82ce8b1088eb5d61992" in page
    assert "7f34c7c15ac88b5087977d763357c52194521d7c226af42722506af6e878860f" in page
    for row in (52, 55, 60, 74, 75, 98, 99, 123, 124):
        assert f"row {row}" in page
    assert "AMBIGUOUS" not in page
    assert "```mermaid" not in page
    assert "Material:" not in page
    assert "Location:" not in page
    assert "Schedule:" not in page
    assert "Tool:" not in page
    assert "Route:" not in page

    expected_hashes = {
        FIGURE3_APPLICATION_LOG: "bff29968a0779ba836a426c2327e9dd16f002edb7a5efaa91291a05a384171c9",
        FIGURE3_NORMALIZED_EVENTS: "5424d629b6f275eba7d63015d48ef6931e3716bd23efe82ce8b1088eb5d61992",
        FIGURE3_ATTRIBUTION_RESULTS: "7f34c7c15ac88b5087977d763357c52194521d7c226af42722506af6e878860f",
    }
    for path, expected_hash in expected_hashes.items():
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash


def test_main_log_linked_sources_are_published_and_recompute_attribution() -> None:
    results_root = (
        ROOT
        / "evaluation"
        / "results"
        / "attribution-accuracy-and-cross-user-isolation"
    )
    totals = {"CERTIFIED": 0, "RANKED": 0}

    for model_dir in ("llama-3.2-3b", "qwen2.5-7b"):
        condition_root = results_root / model_dir / "log-linked"
        provenance = json.loads(_text(condition_root / "source-provenance.json"))
        summary = json.loads(_text(condition_root / "summary.json"))

        assert provenance["run_kind"] == "Evaluation1 main Log-linked"
        assert len(provenance["checkpoints"]) == 6
        model_counts = {"CERTIFIED": 0, "RANKED": 0}
        for checkpoint in provenance["checkpoints"]:
            assert checkpoint["checkpoint_id"] in {
                "C1",
                "C2",
                "C3",
                "C4",
                "C5",
                "C6",
            }
            for key in (
                "application_log",
                "normalized_application_events",
                "attribution_results",
            ):
                source = checkpoint[key]
                path = condition_root / source["path"]
                assert path.is_file()
                assert path.stat().st_size == source["size_bytes"]
                assert (
                    "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
                    == source["sha256"]
                )
                assert not source["original_relative_path"].startswith("/")

            attribution_path = condition_root / checkpoint["attribution_results"]["path"]
            rows = [
                json.loads(line)
                for line in _text(attribution_path).splitlines()
                if line.strip()
            ]
            assert len(rows) == 30
            for row in rows:
                decision = row["final_decision"]
                assert decision in model_counts
                model_counts[decision] += 1

        assert sum(model_counts.values()) == 180
        assert model_counts == {
            "CERTIFIED": summary["attribution"]["CERTIFIED"],
            "RANKED": summary["attribution"]["RANKED"],
        }
        assert summary["attribution"]["unresolved"] == 0
        assert summary["attribution"]["missing"] == 0
        for decision, count in model_counts.items():
            totals[decision] += count

    assert totals == {"CERTIFIED": 194, "RANKED": 166}


def test_public_docs_do_not_publish_local_rehost_profile_debt() -> None:
    combined = "\n".join(
        _text(path)
        for path in ROOT.glob("**/*.md")
        if ".git" not in path.parts
    ).lower()

    for stale_marker in (
        "20260824",
        "20260825",
        "20260827",
        "7a58c4c5d9084a8a",
        "outdated local profile",
        "older than the public bundle",
        "final smoke remains",
    ):
        assert stale_marker not in combined


def test_public_entry_points_match_the_paper_identity_and_remain_anonymous() -> None:
    readme = _text(ROOT_README)
    quickstart = _text(QUICKSTART)

    assert "Lock-on Acquisition of Runtime Inference State" in readme
    assert "DFC Europe 2027" in readme
    assert "github.com" not in readme.lower()
    assert "github.com" not in quickstart.lower()
    assert "git clone" not in readme.lower()
    assert "git clone" not in quickstart.lower()
    assert "python -m pip install -e ." in readme
    assert "python -m pip install -e ." in quickstart
    assert "papersubmit" not in readme
    assert "papersubmit" not in quickstart


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
    assert (
        "temperature `1.0`, top-k `1`, top-p `1.0`, min-p `1.0`, seed `20260907`"
        in " ".join(guide.split())
    )
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
    assert "model weights" in licenses.lower()
    assert "not redistributed" in licenses


def test_llama_gguf_derivation_is_explicitly_published() -> None:
    document = yaml.safe_load(MODEL_REGISTRY.read_text(encoding="utf-8"))
    llama = next(
        model
        for model in document["models"]
        if model["model_id"] == "llama-3.2-3b-instruct-q4-k-m"
    )

    assert llama["source_repository"] == "meta-llama/Llama-3.2-3B-Instruct"
    assert llama["source_revision"] == "0cb88a4f764b7a12671c53f0838cd831a0843b95"
    assert llama["source_snapshot"]["format"] == "HUGGING_FACE_SAFETENSORS"
    assert llama["source_snapshot"]["manifest_sha256"] == (
        "sha256:d3decedc83cddb3ae34f561c4284368c34778b142c60fc26a8248905ffa06473"
    )
    assert llama["artifact_derivation"] == "LLAMA_CPP_BF16_THEN_Q4_K_M"
    derivation = llama["derivation"]
    assert derivation["llama_cpp_revision"] == (
        "daf6bc9f2d3516d7bda73d191c6a01388aff09c5"
    )
    assert derivation["converter"] == "convert_hf_to_gguf.py"
    assert derivation["intermediate_format"] == "GGUF_BF16"
    assert derivation["intermediate_sha256"] == (
        "sha256:c33e9df3cb12e4b4656c624d16e14fff6e499eef097b3074c38bf1bb3ea9fc38"
    )
    assert derivation["quantizer"] == "llama-quantize"
    assert derivation["quantizer_sha256"] == (
        "sha256:198073827fd97658ff3442b278d6396a8e6710975b37064ac8b90fd2f0fb5acc"
    )
    assert derivation["quantization"] == "Q4_K_M"

    setup = " ".join(_text(REHOSTING_SETUP).split())
    assert "not distributed as GGUF by the pinned Meta repository" in setup
    assert "convert_hf_to_gguf.py" in setup
    assert "llama-quantize" in setup
    assert "Q4_K_M" in setup
    licenses = " ".join(_text(LICENSE_PAGE).split())
    assert "derive an evaluated artifact where recorded" in licenses


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
        assert f'{result["mean_kv_overhead_percent"]:.3f}%' in page
        for value in result["kv_overhead_ci95_percent"]:
            assert f"{value:.3f}" in page
        assert result["zero_loss"] is True

    expected_runtime_rows = (
        "| Native | None | 2284.32 s (2032.76–2535.87) | — | — |",
        "| Proxy control | Attribution only; KV copying disabled | 2401.69 s (2175.77–2627.61) | Native | — |",
        "| Proxy | Full | 2834.74 s (2649.91–3019.58) | Proxy control | 18.415% (7.655–29.174) |",
        "| Log-linked control | Attribution only; KV copying disabled | 2222.24 s (2132.07–2312.42) | Native | — |",
        "| Log-linked | Full | 4504.30 s (4275.73–4732.87) | Log-linked control | 102.769% (93.044–112.495) |",
    )
    for row in expected_runtime_rows:
        assert row in page
    assert "O(m,b) = 100 × (Tfull(m,b) / Tcontrol(m,b) − 1)" in page
    assert "KV capture work per run" not in page
    assert "332.190" not in page
    assert "149.704" not in page
    assert "lost events: 0" in page


def test_machine_summary_paper_table_matches_the_published_five_conditions() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))

    assert summary["paper_table"]["columns"] == [
        "Condition",
        "Capture policy",
        "Runtime mean seconds",
        "Runtime 95% CI seconds",
        "Matched reference",
        "Paired overhead mean percent",
        "Paired overhead 95% CI percent",
    ]
    assert summary["paper_table"]["rows"] == [
        {
            "capture_policy": "NONE",
            "condition": "Native",
            "matched_reference": None,
            "mean_paired_overhead_percent": None,
            "mean_runtime_seconds": 2284.32,
            "paired_overhead_ci95_percent": None,
            "runtime_ci95_seconds": [2032.76, 2535.87],
        },
        {
            "capture_policy": "CONTROL_NO_KV",
            "condition": "Proxy control",
            "matched_reference": "Native",
            "mean_paired_overhead_percent": None,
            "mean_runtime_seconds": 2401.69,
            "paired_overhead_ci95_percent": None,
            "runtime_ci95_seconds": [2175.77, 2627.61],
        },
        {
            "capture_policy": "FULL",
            "condition": "Proxy",
            "matched_reference": "Proxy control",
            "mean_paired_overhead_percent": 18.414661589633,
            "mean_runtime_seconds": 2834.74,
            "paired_overhead_ci95_percent": [7.655263188154, 29.174059991111],
            "runtime_ci95_seconds": [2649.91, 3019.58],
        },
        {
            "capture_policy": "CONTROL_NO_KV",
            "condition": "Log-linked control",
            "matched_reference": "Native",
            "mean_paired_overhead_percent": None,
            "mean_runtime_seconds": 2222.24,
            "paired_overhead_ci95_percent": None,
            "runtime_ci95_seconds": [2132.07, 2312.42],
        },
        {
            "capture_policy": "FULL",
            "condition": "Log-linked",
            "matched_reference": "Log-linked control",
            "mean_paired_overhead_percent": 102.769262638129,
            "mean_runtime_seconds": 4504.30,
            "paired_overhead_ci95_percent": [93.043753828068, 112.494771448191],
            "runtime_ci95_seconds": [4275.73, 4732.87],
        },
    ]


def test_evaluation1_summary_keeps_model_and_condition_results_separate() -> None:
    page = _text(EVALUATION_PAGE)
    for row in (
        "| Llama 3.2 3B Instruct Q4_K_M | Native | 870/900 (96.67%) |",
        "| Llama 3.2 3B Instruct Q4_K_M | Proxy | 855/900 (95.00%) |",
        "| Llama 3.2 3B Instruct Q4_K_M | Log-linked | 865/900 (96.11%) |",
        "| Qwen2.5 7B Instruct Q4_K_M | Native | 882/900 (98.00%) |",
        "| Qwen2.5 7B Instruct Q4_K_M | Proxy | 877/900 (97.44%) |",
        "| Qwen2.5 7B Instruct Q4_K_M | Log-linked | 876/900 (97.33%) |",
    ):
        assert row in page


def test_accuracy_comparison_points_to_replay_guide_without_changing_results() -> None:
    comparison = _text(COMPARISON_PAGE)
    assert "../../../docs/reproducibility/evaluation1-release.md" in comparison
    assert "870/900 (96.67%)" in comparison
    assert "876/900 (97.33%)" in comparison


def test_tracked_release_template_contains_only_verified_small_metadata() -> None:
    expected = {
        "checksums.sha256": "0624eab13c68ea59c49dd71454f193a7b9274f663c3eb93060ab4231c7ee0293",
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


def test_verified_release_metadata_keeps_lf_bytes_on_checkout() -> None:
    paths = (
        "evaluation/publication/releases/evaluation1-release-v1/checksums.sha256",
        "evaluation/publication/releases/evaluation1-release-v1/licenses/LICENSE.llama-3.2.txt",
        "evaluation/publication/releases/evaluation1-release-v1/licenses/LICENSE.qwen2.5-apache-2.0.txt",
        "evaluation/results/log-linked-correlation/source/raw_llama_server.log",
        "evaluation/results/log-linked-correlation/source/attribution_results.jsonl",
        "evaluation/results/attribution-accuracy-and-cross-user-isolation/llama-3.2-3b/log-linked/source/c1/raw_llama_server.log",
        "evaluation/results/attribution-accuracy-and-cross-user-isolation/qwen2.5-7b/log-linked/source/c1/normalized_application_events.jsonl",
    )
    result = subprocess.run(
        ["git", "check-attr", "eol", "--", *paths],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.splitlines() == [f"{path}: eol: lf" for path in paths]


def test_exact_source_logs_preserve_upstream_trailing_space() -> None:
    path = (
        "evaluation/results/log-linked-correlation/source/"
        "raw_llama_server.log"
    )
    result = subprocess.run(
        ["git", "check-attr", "whitespace", "--", path],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == f"{path}: whitespace: -trailing-space"


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


def test_public_replay_examples_use_concrete_release_and_model_values() -> None:
    combined = "\n".join(_text(path) for path in (QUICKSTART, REPRODUCTION_GUIDE))

    assert "llama-3.2-3b-instruct-q4-k-m" in combined
    assert "evaluation/publication/external/evaluation1-release-v1" in combined
    assert "daf6bc9f2d3516d7bda73d191c6a01388aff09c5" in combined
    assert "configs/runtime/runtime_state_observer/x86_64_cpu_simple_kv.json" in combined
    placeholder_free = combined.replace("LARIS_PUBLIC_RELEASE_ROOT", "")
    for placeholder in ("MODEL_ID", "PROFILE_JSON", "ANALYSIS_ROOT", "RELEASE_ROOT"):
        assert placeholder not in placeholder_free


def test_public_replay_docs_explain_user_ref_identity_derivation() -> None:
    combined = " ".join(
        "\n".join(_text(path) for path in (QUICKSTART, REPRODUCTION_GUIDE)).split()
    )

    assert "publication-safe alias" in combined
    assert "authenticated user or service-account identifier" in combined
    assert "endpoint type and IP address" in combined
    assert "preservation key" in combined
    assert "not itself a HMAC" in combined
