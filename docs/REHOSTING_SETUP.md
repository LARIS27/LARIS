# Public rehosting setup

This guide prepares a new Windows PC to query the published Evaluation1 KV bundles.
WSL provides only the Linux execution layer required by llama.cpp and the native
rehosting helper. It is **not** the acquisition environment, and this procedure does
not qualify the PC for live Proxy or Log-linked capture.

## Required inputs

1. Clone this repository on Windows, install Python 3.11 or newer, and install the
   current checkout from the repository root:

   ```text
   python -m pip install -e .
   ```
2. Download the complete `LARIS-Evaluation1-Public-v1` folder. This guide uses
   `C:\LARIS-Evaluation1-Public-v1` as an example; any local path is accepted.
   The setup command accepts the downloaded `evaluation1-release-v1` directory
   itself, its immediate parent, or a wrapper folder containing
   `LARIS_Evaluation\evaluation1-release-v1`.
3. Install WSL 2 with an Ubuntu distribution and the packages `build-essential`,
   `cmake`, `git`, `python3`, and `python3-venv`.
4. Obtain the GGUF model files separately and place them in one Windows model
   directory. Their exact filenames, sizes, source revisions, and SHA-256 digests are
   listed in `evaluation/experiment_setup/models.yaml`.
5. Clone llama.cpp into a Windows path accessible to WSL (the example below uses
   `C:\src\llama.cpp`) and check out exactly
   `daf6bc9f2d3516d7bda73d191c6a01388aff09c5`. The setup command converts that
   Windows path to its WSL path before building.

Model weights are not part of the public KV release. The Llama and Qwen notices that
govern the upstream models and the synthetic release are recorded in
`evaluation/publication/LICENSES.md`.

## One-command preparation

From PowerShell in the repository root:

```powershell
.\tools\windows\setup_public_rehost.ps1 `
  -DownloadRoot 'C:\LARIS-Evaluation1-Public-v1' `
  -ModelRoot 'C:\Models\LARIS' `
  -LlamaSource 'C:\src\llama.cpp' `
  -Distribution 'Ubuntu'
```

The command performs five fail-closed steps:

1. It locates exactly one complete seven-file external release tree and rejects
   ambiguous duplicate copies.
2. `stage_public_evaluation_release.py initialize` copies the small, Git-tracked
   manifest, license, and checksum metadata into that tree without altering the seven
   downloaded files. Its `verify` command then checks every file and all 240 public
   tuples.
3. `setup_public_rehost_runtime.sh` confirms the pinned llama.cpp Git commit, builds
   llama.cpp and `native/rehosting`, and installs them in a private WSL directory.
4. `build_public_rehost_profile.py` records the locally built runtime, `libllama`
   SHA-256 and ELF build ID in a profile whose scope is **rehosting only**.
5. The Windows investigator profile binds that WSL runtime to the downloaded synthetic
   keys and the locally supplied model directory.

The final two lines report `LARIS_PUBLIC_RELEASE_ROOT=<verified download path>` and
`LARIS_PUBLIC_REHOST_READY=<profile path>`. Use those concrete paths in the replay
commands below. No VM, SSH connection, capture daemon, or legacy Evaluation1 campaign
runner is used.

## Replay and verification

Follow `docs/reproducibility/evaluation1-release.md` to list, inspect, and query either
the Proxy or Log-linked bundles. The paper replay tuple is deterministic:
temperature `1.0`, top-k `1`, top-p `1.0`, min-p `1.0`, seed `20260907`, and a
128-token generation limit.

To verify the compact result files independently of large-bundle replay, run:

```text
python evaluation/scripts/verify_published_results.py
```

The command exits non-zero if tracked Evaluation1 counts or the 25-run Evaluation2
summary no longer recompute from their machine-readable sources.
