from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "evaluation" / "scripts" / "verify_published_results.py"


def _module():
    spec = importlib.util.spec_from_file_location("verify_published_results", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_verifier_recomputes_tracked_evaluation_results() -> None:
    module = _module()
    receipt = module.verify_repository(ROOT)

    assert receipt["state"] == "VERIFIED"
    assert receipt["evaluation1"] == {
        "log_linked": {"CERTIFIED": 194, "RANKED": 166},
        "models": 2,
        "requests_per_mode": 360,
        "unresolved": 0,
        "cross_user_mixing": 0,
    }
    assert receipt["evaluation2"]["trial_count"] == 25
    assert receipt["evaluation2"]["pair_count"] == 5
    assert receipt["evaluation2"]["proxy_overhead_percent"] == pytest.approx(
        18.414661589633
    )
    assert receipt["evaluation2"]["log_linked_overhead_percent"] == pytest.approx(
        102.769262638129
    )


def test_verifier_fails_closed_when_a_published_summary_drifts(
    tmp_path: Path,
) -> None:
    module = _module()
    source = (
        ROOT
        / "evaluation"
        / "results"
        / "runtime-overhead"
        / "llama-3.2-3b"
        / "summary.json"
    )
    document = json.loads(source.read_text(encoding="utf-8"))
    document["conditions"]["PROXY"]["mean_kv_overhead_percent"] = 0.0
    changed = tmp_path / "summary.json"
    changed.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(module.PublishedResultError, match="PROXY overhead"):
        module.verify_evaluation2(changed)
