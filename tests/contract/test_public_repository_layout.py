from __future__ import annotations

import re
import subprocess
from pathlib import Path

from evaluation.scripts.build_public_source_inventory import build_inventory, load_allowlist


ROOT = Path(__file__).resolve().parents[2]
ALLOWLIST = ROOT / "evaluation" / "publication" / "source-allowlist.yaml"


def _public_paths() -> set[str]:
    output = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
    tracked = [item.decode("utf-8") for item in output.split(b"\0") if item]
    inventory = build_inventory(ROOT, load_allowlist(ALLOWLIST), tracked)
    return {str(row["path"]) for row in inventory["files"]}


def test_public_repository_has_one_authoritative_laris_package() -> None:
    paths = _public_paths()
    assert "src/laris_onpremise/__init__.py" in paths
    assert all(not path.startswith("src/llm_mem_forensics/") for path in paths)
    assert all(not path.startswith("LARIS_Onpremise/") for path in paths)
    for prefix in ("native/", "configs/", "schemas/", "tests/", "tools/", "docs/"):
        assert any(path.startswith(prefix) for path in paths), prefix


def test_public_repository_excludes_historical_and_ui_paths() -> None:
    paths = _public_paths()
    for prefix in (
        ".github/",
        "dashboard/",
        "src/laris_onpremise/dashboard/",
        "tools/legacy/",
        "docs/archive/",
        "evaluation/scripts/e6_legacy_support/",
    ):
        assert all(not path.startswith(prefix) for path in paths), prefix


def test_public_documentation_has_one_current_start_path() -> None:
    paths = _public_paths()
    for relative in (
        "README.md",
        "docs/QUICKSTART.md",
        "evaluation/README.md",
        "docs/reproducibility/evaluation1-release.md",
        "evaluation/results/runtime-overhead/README.md",
    ):
        assert relative in paths
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "docs/assets/LARIS.png" in paths
    assert "docs/assets/LARIS.png" in readme
    assert '<h1 align="center">LARIS</h1>' in readme
    assert 'src="docs/assets/LARIS.png"' in readme
    assert 'width="780"' in readme
    for label in (
        "ON-PREMISES LLM",
        "KV-CACHE FORENSICS",
        "eBPF OBSERVER",
        "LLAMA.CPP",
        "PYTHON 3.11+",
        "PROXY + LOG-LINKED",
        "SELECTIVE KV ACQUISITION",
        "SEALED BUNDLES",
        "FRESH-CONTEXT REHOSTING",
        "Official repository for the DFC EU 2027 submission",
    ):
        assert f'alt="{label}"' in readme
    assert readme.count("style=flat-square") == 10
    assert readme.count('height="20"') == 9
    assert readme.count('height="22"') == 1
    assert "style=for-the-badge" not in readme
    assert (
        '<h3 align="center">\n  '
        "LARIS: Attribution-Aware Selective KV-Cache Acquisition and Rehosting<br>\n"
        "  for Investigative Querying in Shared LLM Services\n"
        "</h3>"
    ) in readme
    assert re.search(
        r'alt="PYTHON 3\.11\+"[^>]+>\n  <br>\n  '
        r'<img alt="PROXY \+ LOG-LINKED"',
        readme,
    )
    assert (
        "DFC_EU_2027-OFFICIAL_SUBMISSION_REPOSITORY-f59e0b?style=flat-square"
        in readme
    )
    assert "## Getting started" in readme
    assert "Reviewer start path" not in readme
    assert "paper-final-" + "release" not in readme
    assert "cd LARIS_Onpremise" not in readme
    assert "REPOSITORY_URL" not in readme
    quickstart = (ROOT / "docs" / "QUICKSTART.md").read_text(encoding="utf-8")
    assert "REPOSITORY_URL" not in quickstart
    assert "docs/design/log_linked_runtime_mode.md" not in quickstart
    assert "docs/operations/clean_vm_to_rehost.md" not in quickstart


def test_public_markdown_local_links_resolve_inside_the_release() -> None:
    unresolved: list[str] = []
    for relative in _public_paths():
        if not relative.endswith(".md"):
            continue
        document = ROOT / relative
        text = document.read_text(encoding="utf-8")
        for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", text):
            if target.startswith(("https://", "http://", "#")):
                continue
            path_text = target.split("#", 1)[0]
            if path_text and not (document.parent / path_text).resolve().is_file():
                unresolved.append(f"{relative}: {target}")
    assert unresolved == []


def test_public_verification_entrypoints_exist_without_host_automation() -> None:
    paths = _public_paths()
    assert "tools/run_portable_tests_windows.ps1" in paths
    assert "evaluation/scripts/assemble_paper_final_source.py" in paths
    assert "evaluation/scripts/stage_public_evaluation_release.py" in paths
    assert all(not path.startswith(".github/") for path in paths)


def test_portable_verification_does_not_require_excluded_ui() -> None:
    script = (ROOT / "tools" / "run_portable_tests_windows.ps1").read_text(
        encoding="utf-8"
    )
    assert "dashboard/" not in script
    assert "dashboard\\" not in script


def test_public_text_excludes_local_developer_profile_paths() -> None:
    windows_profile = re.compile(r"[A-Za-z]:\\Users\\[^\\/\s\"']+")
    linux_profile = re.compile("/" + r"home/([^/\s\"']+)/")
    allowed_linux_users = {"laris"}
    leaked: list[str] = []
    for relative in _public_paths():
        path = ROOT / relative
        if path.suffix.lower() not in {
            "", ".c", ".cpp", ".h", ".ini", ".json", ".md", ".ps1",
            ".py", ".sh", ".toml", ".txt", ".yaml", ".yml",
        }:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        linux_users = set(linux_profile.findall(text))
        if windows_profile.search(text) or linux_users.difference(allowed_linux_users):
            leaked.append(relative)
    assert leaked == []
