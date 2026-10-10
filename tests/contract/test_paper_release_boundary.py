from __future__ import annotations

import ast
import json
import subprocess
from pathlib import Path

import yaml

from evaluation.scripts.build_public_source_inventory import (
    build_inventory,
    canonical_inventory_bytes,
    load_allowlist,
)
ROOT = Path(__file__).resolve().parents[2]
ALLOWLIST = ROOT / "evaluation" / "publication" / "source-allowlist.yaml"


def _contains_hangul(text: str) -> bool:
    return any(
        0x1100 <= ord(char) <= 0x11FF
        or 0x3130 <= ord(char) <= 0x318F
        or 0xAC00 <= ord(char) <= 0xD7A3
        for char in text
    )


def _tracked_paths() -> list[str]:
    output = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
    )
    paths = sorted(item.decode("utf-8") for item in output.split(b"\0") if item)
    return [path for path in paths if (ROOT / path).is_file()]


def _inventory() -> dict[str, object]:
    return build_inventory(ROOT, load_allowlist(ALLOWLIST), _tracked_paths())


def _paths() -> dict[str, str]:
    return {
        str(entry["path"]): str(entry["role"])
        for entry in _inventory()["files"]  # type: ignore[index]
    }


def test_allowlist_assigns_role_complete_current_source() -> None:
    paths = _paths()
    assert paths["src/laris_onpremise/cli.py"] == "current-runtime"
    assert paths["src/laris_onpremise/attribution/proxy/core.py"] == "current-runtime"
    assert paths["src/laris_onpremise/attribution/log_linked/workflow.py"] == "current-runtime"
    assert paths["src/laris_onpremise/reproduction/evaluation1/verifier.py"] == "evaluation1-reproduction"
    assert paths["native/runtime_state_observer/src/runtime_state.bpf.c"] == "current-native"
    assert paths["native/rehosting/llama_runtime_payload_rehost.cpp"] == "current-native"
    assert paths["configs/runtime/log_linked/llama_server_daf6bc9.yaml"] == "current-operations"
    assert paths["tools/windows/validate_rehost_profile.py"] == "current-operations"
    assert paths["evaluation/scripts/stage_public_evaluation_release.py"] == "evaluation1-reproduction"
    assert paths["evaluation/results/runtime-overhead/llama-3.2-3b/summary.json"] == "public-evaluation"
    assert paths["evaluation/results/log-linked-correlation/README.md"] == "public-evaluation"
    assert paths["evaluation/run_records/evaluation1-public-rehost-smoke.json"] == "public-evaluation"
    assert paths["docs/reproducibility/evaluation1-release.md"] == "public-docs"


def test_final_inventory_has_no_unassigned_public_file() -> None:
    inventory = _inventory()
    rows = inventory["files"]
    paths = [str(row["path"]) for row in rows]  # type: ignore[index]
    assert inventory["file_count"] == len(rows)  # type: ignore[arg-type]
    assert paths == sorted(paths)
    assert len(paths) == len(set(paths))
    assert all(str(row["role"]) for row in rows)  # type: ignore[index]
    assert canonical_inventory_bytes(inventory) == canonical_inventory_bytes(_inventory())


def test_final_inventory_excludes_frozen_evaluation_backends_and_task_metadata() -> None:
    paths = set(_paths())
    forbidden_fragments = (
        "evaluation/scripts/e6_legacy_support/",
        "evaluation/scripts/e6_overhead_",
        "evaluation/scripts/evaluation1_c1_",
        "evaluation/scripts/run_checkpoint_",
        ".local-release/",
        ".local-evaluation/",
        ".release-staging/",
        ".worktrees/",
    )
    assert all(not any(fragment in path for fragment in forbidden_fragments) for path in paths)


def test_final_inventory_excludes_retired_native_observer_boundary() -> None:
    paths = set(_paths())
    retired_paths = {
        "src/laris_onpremise/runtime_inputs_reader.py",
        "src/laris_onpremise/rehosting/preparation.py",
        "tests/unit/rehosting/test_preparation.py",
        "tools/vm/build_clean_node_native.sh",
        "tools/vm/diagnose_uprobe_privilege.sh",
        "tools/vm/qualify_pinned_runtime.py",
        "tools/vm/run_cumulative_rehosting.sh",
        "src/laris_onpremise/core/policy.py",
        "src/laris_onpremise/core/standalone.py",
        "tools/verify_standalone.py",
    }
    retired_prefixes = ("configs/runtime/runtime_observer/", "native/runtime_observer/")
    assert retired_paths.isdisjoint(paths)
    assert all(not path.startswith(retired_prefixes) for path in paths)


def test_final_inventory_excludes_dashboard_ui_launchers_and_static_assets() -> None:
    paths = set(_paths())
    forbidden = (
        "dashboard/",
        "src/laris_onpremise/dashboard/",
        "tests/unit/dashboard/",
        "tests/integration/test_dashboard_server.py",
        "tools/run_dashboard_windows.cmd",
        "tools/windows/run_dashboard_windows.ps1",
    )
    assert all(not any(path.startswith(prefix) for prefix in forbidden) for path in paths)


def test_final_inventory_includes_current_config_and_reproduction_tools() -> None:
    paths = set(_paths())
    required = {
        "configs/runtime/runtime_state_observer/x86_64_cpu_simple_kv.json",
        "schemas/evaluation1_artifact_locations.schema.json",
        "evaluation/scripts/assemble_paper_final_source.py",
        "evaluation/scripts/build_public_source_inventory.py",
        "evaluation/scripts/stage_public_evaluation_release.py",
        "evaluation/scripts/verify_public_evaluation_release.py",
        "tools/run_portable_tests_windows.ps1",
        "tools/wsl/install_rehost_runtime.sh",
    }
    assert required <= paths


def test_final_inventory_excludes_repository_host_automation() -> None:
    assert all(not path.startswith(".github/") for path in _paths())


def test_public_cli_has_no_dashboard_or_experiment_control_plane_commands() -> None:
    from laris_onpremise.cli import _parser

    parser = _parser()
    action = next(
        item for item in parser._actions if getattr(item, "choices", None)
    )
    commands = set(action.choices)
    assert "dashboard" not in commands
    assert "experiment" not in commands
    assert "bundle" not in commands
    assert {"proxy", "log-linked", "rehost", "reproduce"} <= commands


def test_public_cli_exposes_only_portable_rehost_workflows() -> None:
    from laris_onpremise.cli import _parser

    parser = _parser()
    root_action = next(
        item for item in parser._actions if getattr(item, "choices", None)
    )
    rehost = root_action.choices["rehost"]
    rehost_action = next(
        item for item in rehost._actions if getattr(item, "choices", None)
    )

    assert set(rehost_action.choices) == {"open", "profile"}


def test_rehost_analysis_has_no_legacy_inventory_compatibility() -> None:
    source = (
        ROOT / "src" / "laris_onpremise" / "rehosting" / "analysis_session.py"
    ).read_text(encoding="utf-8")

    assert "_LEGACY_ANALYSIS_INVENTORIES" not in source
    assert "analysis_manifest.json" not in source


def test_non_ui_runtime_imports_resolve_without_dashboard_namespace() -> None:
    selected = _paths()
    for relative in selected:
        if not relative.startswith("src/laris_onpremise/") or not relative.endswith(".py"):
            continue
        source = (ROOT / relative).read_text(encoding="utf-8")
        tree = ast.parse(source, filename=relative)
        imported = [
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        ]
        assert all(not name.startswith("laris_onpremise.dashboard") for name in imported), relative


def test_final_first_party_public_text_is_english_only() -> None:
    exempt_prefixes = (
        "evaluation/publication/releases/evaluation1-release-v1/licenses/",
    )
    for relative in _paths():
        path = ROOT / relative
        if relative.startswith(exempt_prefixes) or path.suffix.lower() not in {
            "", ".c", ".cpp", ".h", ".ini", ".json", ".md", ".ps1",
            ".py", ".sh", ".toml", ".txt", ".yaml", ".yml",
        }:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        assert not _contains_hangul(text), relative


def test_final_paths_and_text_have_no_ai_workbench_markers() -> None:
    forbidden = tuple(
        marker.lower()
        for marker in (
            "co" + "dex",
            "chat" + "gpt",
            "open" + "ai",
            "super" + "powers",
            ".ag" + "ents",
            ".assist" + "ant-state",
            ".assist" + "ant-tmp",
            ".assist" + "ant_tmp",
            ".review" + "-tmp",
        )
    )
    for relative in _paths():
        assert all(marker not in relative.lower() for marker in forbidden), relative
        path = ROOT / relative
        if path.suffix.lower() not in {".json", ".md", ".py", ".toml", ".txt", ".yaml", ".yml"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        assert all(marker not in text.lower() for marker in forbidden), relative


def test_policy_has_unique_roles_and_required_forbidden_rules() -> None:
    document = yaml.safe_load(ALLOWLIST.read_text(encoding="utf-8"))
    role_names = [role["name"] for role in document["roles"]]
    assert len(role_names) == len(set(role_names))
    assert set(role_names) == {
        "release-root",
        "current-runtime",
        "current-native",
        "current-operations",
        "evaluation1-reproduction",
        "public-evaluation",
        "public-docs",
        "release-tests",
    }
    assert ".github/" in document["deny"]["prefixes"]
    assert "dashboard/" in document["deny"]["prefixes"]
    assert "src/laris_onpremise/dashboard/" in document["deny"]["prefixes"]
    assert ".gguf" in document["deny"]["suffixes"]
    assert ".vmdk" in document["deny"]["suffixes"]
    assert "C:\\Users\\" in document["deny"]["content_markers"]


def test_gitignore_keeps_private_and_generated_release_bytes_out_of_git() -> None:
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for marker in (
        "/evaluation/publication/generated/",
        "/evaluation/publication/external/",
        "/evaluation/publication/*.tar",
        "/evaluation/publication/*.tar.gz",
        "/evaluation/publication/source-inventory.json",
        "/.release-staging/",
        "/.worktrees/",
    ):
        assert marker in ignore


def test_inventory_is_sanitized_json() -> None:
    encoded = canonical_inventory_bytes(_inventory())
    parsed = json.loads(encoded)
    assert all(not Path(str(row["path"])).is_absolute() for row in parsed["files"])
    assert all(row["sha256"].startswith("sha256:") for row in parsed["files"])
