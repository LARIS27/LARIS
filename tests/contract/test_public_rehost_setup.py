from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _text(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_new_machine_rehost_setup_has_a_bounded_public_path() -> None:
    guide = _text("docs/REHOSTING_SETUP.md")
    powershell = _text("tools/windows/setup_public_rehost.ps1")
    shell = _text("tools/wsl/setup_public_rehost_runtime.sh")

    assert "C:\\LARIS-Evaluation1\\evaluation1-release-v1" in guide
    assert "LARIS_Evaluation\\evaluation1-release-v1" in guide
    assert "D:\\" not in guide
    assert "rehosting only" in guide.lower()
    assert "not" in guide.lower() and "acquisition" in guide.lower()
    assert "daf6bc9f2d3516d7bda73d191c6a01388aff09c5" in guide
    assert "setup_public_rehost.ps1" in guide
    assert "verify_published_results.py" in guide
    assert "python -m pip install -e ." in guide

    assert "VmHost" not in powershell
    assert "ssh.exe" not in powershell
    assert " locate --download-root " in powershell
    assert "stage_public_evaluation_release.py" in powershell
    assert " initialize --metadata-root " in powershell
    assert "setup_public_rehost_runtime.sh" in powershell
    assert "sh -lc 'wslpath -a -u \"$1\"' sh $repository" in powershell
    assert "sh -lc 'wslpath -a -u \"$1\"' sh $llamaSourceResolved" in powershell
    assert "LARIS_PUBLIC_RELEASE_ROOT=$releaseRoot" in powershell
    assert "LARIS_PUBLIC_REHOST_READY=$profile" in powershell

    assert "git rev-parse HEAD" in shell
    assert "cmake" in shell
    assert "native/rehosting" in shell
    assert "build_public_rehost_profile.py" in shell
    assert "local_rehost_qualified" not in shell
