#!/usr/bin/env python3
"""Recompute the small, Git-tracked Evaluation1 and Evaluation2 results."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Iterable


class PublishedResultError(ValueError):
    """Raised when a tracked result cannot be reproduced from its source rows."""


_T_95_DF4 = 2.7764451051977987


def _read_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PublishedResultError(f"invalid result file: {path.name}") from error
    if not isinstance(value, dict):
        raise PublishedResultError(f"result root must be an object: {path.name}")
    return value


def _close(observed: float, expected: float, label: str) -> None:
    if not math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-9):
        raise PublishedResultError(f"{label} differs from recomputation")


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        raise PublishedResultError("cannot compute an empty mean")
    return statistics.fmean(materialized)


def _ci95(values: Iterable[float]) -> list[float]:
    materialized = list(values)
    if len(materialized) != 5:
        raise PublishedResultError("Evaluation2 CI requires five balanced blocks")
    center = statistics.fmean(materialized)
    half = _T_95_DF4 * statistics.stdev(materialized) / math.sqrt(5)
    return [center - half, center + half]


def verify_evaluation1(results_root: Path) -> dict[str, object]:
    base = results_root / "attribution-accuracy-and-cross-user-isolation"
    model_roots = sorted(path for path in base.iterdir() if path.is_dir())
    if len(model_roots) != 2:
        raise PublishedResultError("Evaluation1 must contain exactly two models")

    certified = 0
    ranked = 0
    attributed = 0
    unresolved = 0
    cross_user = 0
    for model_root in model_roots:
        for mode in ("log-linked", "proxy"):
            document = _read_object(model_root / mode / "summary.json")
            attribution = document.get("attribution")
            errors = document.get("response_error_counts")
            if not isinstance(attribution, dict):
                raise PublishedResultError("Evaluation1 attribution fields are missing")
            request_count = document.get(
                "rehost_request_count", document.get("paired_recall_count")
            )
            if request_count != 180:
                raise PublishedResultError("Evaluation1 request count differs")
            mode_correct = attribution.get("correct", attribution.get("correct_count"))
            mode_incorrect = attribution.get(
                "incorrect", attribution.get("incorrect_count")
            )
            mode_unresolved_value = attribution.get(
                "unresolved", attribution.get("unresolved_count")
            )
            mode_missing_value = attribution.get(
                "missing", attribution.get("missing_result_count")
            )
            if mode_correct != 180 or mode_incorrect != 0:
                raise PublishedResultError("Evaluation1 attribution accuracy differs")
            mode_unresolved = int(mode_unresolved_value) + int(mode_missing_value)
            if mode_unresolved != 0:
                raise PublishedResultError("Evaluation1 has unresolved attribution")
            mode_certified = 0
            mode_ranked = 0
            if mode == "log-linked":
                mode_certified = int(attribution.get("CERTIFIED", -1))
                mode_ranked = int(attribution.get("RANKED", -1))
                if mode_certified + mode_ranked != 180:
                    raise PublishedResultError("Evaluation1 decision counts differ")
            if isinstance(errors, dict):
                mode_cross_user = int(errors.get("cross_actor_fields", -1)) + int(
                    errors.get("cross_actor_responses", -1)
                )
            else:
                rehost = document.get("rehost")
                if not isinstance(rehost, dict):
                    raise PublishedResultError(
                        "Evaluation1 cross-user fields are missing"
                    )
                mode_cross_user = int(
                    rehost.get("foreign_actor_value_hit_count", -1)
                ) + int(rehost.get("foreign_actor_response_count", -1))
            if mode_cross_user != 0:
                raise PublishedResultError("Evaluation1 cross-user mixing is non-zero")
            attributed += int(mode_correct)
            unresolved += mode_unresolved
            cross_user += mode_cross_user
            if mode == "log-linked":
                certified += mode_certified
                ranked += mode_ranked

    return {
        "log_linked": {"CERTIFIED": certified, "RANKED": ranked},
        "models": len(model_roots),
        "requests_per_mode": attributed // 2,
        "unresolved": unresolved,
        "cross_user_mixing": cross_user,
    }


def verify_evaluation2(summary_path: Path) -> dict[str, object]:
    document = _read_object(summary_path)
    trials = document.get("raw_trials")
    conditions = document.get("conditions")
    paper_table = document.get("paper_table")
    if not isinstance(trials, list) or not isinstance(conditions, dict):
        raise PublishedResultError("Evaluation2 source rows are missing")
    if document.get("state") != "COMPLETE" or len(trials) != 25:
        raise PublishedResultError("Evaluation2 trial set is incomplete")

    by_block: dict[int, dict[str, dict[str, object]]] = {}
    by_mode: dict[str, list[dict[str, object]]] = {}
    expected_modes = {
        "BASELINE",
        "PROXY_CONTROL",
        "PROXY",
        "LOG_LINKED_CONTROL",
        "LOG_LINKED",
    }
    for raw in trials:
        if not isinstance(raw, dict):
            raise PublishedResultError("Evaluation2 trial is not an object")
        block = raw.get("block")
        mode = raw.get("mode")
        if isinstance(block, bool) or not isinstance(block, int) or mode not in expected_modes:
            raise PublishedResultError("Evaluation2 trial identity is invalid")
        if raw.get("prompt_token_count") != 123529 or raw.get("generated_token_count") != 1080:
            raise PublishedResultError("Evaluation2 workload differs")
        if raw.get("lost_events") != 0:
            raise PublishedResultError("Evaluation2 lost-events count is non-zero")
        bucket = by_block.setdefault(block, {})
        if mode in bucket:
            raise PublishedResultError("Evaluation2 block contains a duplicate mode")
        bucket[str(mode)] = raw
        by_mode.setdefault(str(mode), []).append(raw)
    if set(by_block) != {1, 2, 3, 4, 5} or any(
        set(bucket) != expected_modes for bucket in by_block.values()
    ):
        raise PublishedResultError("Evaluation2 design is not five balanced blocks")

    overheads: dict[str, list[float]] = {"PROXY": [], "LOG_LINKED": []}
    work: dict[str, list[float]] = {"PROXY": [], "LOG_LINKED": []}
    bytes_by_mode: dict[str, int] = {"PROXY": 0, "LOG_LINKED": 0}
    controls = {"PROXY": "PROXY_CONTROL", "LOG_LINKED": "LOG_LINKED_CONTROL"}
    for block in sorted(by_block):
        bucket = by_block[block]
        for mode, control in controls.items():
            full_ns = int(bucket[mode]["timed_duration_ns"])
            control_ns = int(bucket[control]["timed_duration_ns"])
            overheads[mode].append(100.0 * (full_ns / control_ns - 1.0))
            work[mode].append(int(bucket[mode]["kv_capture_work_ns"]) / 1e9)
            bytes_by_mode[mode] += int(bucket[mode]["kv_capture_bytes"])

    for mode in ("PROXY", "LOG_LINKED"):
        published = conditions.get(mode)
        if not isinstance(published, dict):
            raise PublishedResultError(f"{mode} condition is missing")
        _close(
            float(published["mean_kv_overhead_percent"]),
            _mean(overheads[mode]),
            f"{mode} overhead",
        )
        _close(
            float(published["mean_kv_capture_work_seconds"]),
            _mean(work[mode]),
            f"{mode} capture work",
        )
        for observed, expected in zip(
            published["kv_overhead_ci95_percent"], _ci95(overheads[mode]), strict=True
        ):
            _close(float(observed), expected, f"{mode} overhead CI")
        for observed, expected in zip(
            published["kv_capture_work_ci95_seconds"], _ci95(work[mode]), strict=True
        ):
            _close(float(observed), expected, f"{mode} capture-work CI")
        if published.get("total_kv_capture_bytes") != bytes_by_mode[mode]:
            raise PublishedResultError(f"{mode} capture byte total differs")
        if published.get("zero_loss") is not True or published.get("pair_count") != 5:
            raise PublishedResultError(f"{mode} completion contract differs")

    if not isinstance(paper_table, dict) or not isinstance(paper_table.get("rows"), list):
        raise PublishedResultError("Evaluation2 paper table is missing")
    runtime_by_condition = {
        "Native": "BASELINE",
        "Proxy control": "PROXY_CONTROL",
        "Proxy": "PROXY",
        "Log-linked control": "LOG_LINKED_CONTROL",
        "Log-linked": "LOG_LINKED",
    }
    rows = {row["condition"]: row for row in paper_table["rows"]}
    if set(rows) != set(runtime_by_condition):
        raise PublishedResultError("Evaluation2 paper-table conditions differ")
    for condition, mode in runtime_by_condition.items():
        seconds = [int(row["timed_duration_ns"]) / 1e9 for row in by_mode[mode]]
        _close(float(rows[condition]["mean_runtime_seconds"]), round(_mean(seconds), 2), f"{condition} runtime")
        expected_ci = [round(value, 2) for value in _ci95(seconds)]
        for observed, expected in zip(rows[condition]["runtime_ci95_seconds"], expected_ci, strict=True):
            _close(float(observed), expected, f"{condition} runtime CI")

    return {
        "trial_count": len(trials),
        "pair_count": len(by_block),
        "proxy_overhead_percent": _mean(overheads["PROXY"]),
        "log_linked_overhead_percent": _mean(overheads["LOG_LINKED"]),
    }


def verify_repository(root: Path) -> dict[str, object]:
    root = Path(root)
    if not root.is_absolute() or not root.is_dir():
        raise PublishedResultError("repository root is unavailable")
    results = root / "evaluation" / "results"
    return {
        "schema_version": "1.0",
        "state": "VERIFIED",
        "evaluation1": verify_evaluation1(results),
        "evaluation2": verify_evaluation2(
            results / "runtime-overhead" / "llama-3.2-3b" / "summary.json"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Recompute the public Evaluation1 and Evaluation2 result summaries."
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    arguments = parser.parse_args(argv)
    try:
        receipt = verify_repository(arguments.repository_root)
    except PublishedResultError as error:
        print(
            json.dumps(
                {"schema_version": "1.0", "state": "REJECTED", "reason": str(error)},
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
