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
2. Download the complete public `evaluation1-release-v1` folder. This guide uses
   `C:\LARIS-Evaluation1\evaluation1-release-v1` as an example; any local path is
   accepted. The setup command accepts the downloaded release directory itself or
   its immediate parent. For compatibility, it also recognizes a wrapper containing
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

### Llama model derivation

The evaluated `Llama-3.2-3B-Instruct-Q4_K_M.gguf` is not distributed as GGUF by the
pinned Meta repository. It was derived locally from the gated Hugging Face
Safetensors snapshot at revision `0cb88a4f764b7a12671c53f0838cd831a0843b95`.
Using llama.cpp revision `daf6bc9f2d3516d7bda73d191c6a01388aff09c5`, the snapshot
was first converted to BF16 GGUF with `convert_hf_to_gguf.py` and then quantized to
`Q4_K_M` with `llama-quantize`. The source-manifest, BF16 intermediate, quantizer,
and final artifact digests are recorded in `evaluation/experiment_setup/models.yaml`.
A reconstructed model is accepted for replay only when its final byte size and
SHA-256 digest match that registry.

A minimal reconstruction is:

1. Download the gated Meta Safetensors snapshot at the revision above.
2. Check out the pinned llama.cpp revision and build its `llama-quantize` target.
3. Convert the snapshot to BF16 GGUF, then quantize that intermediate to `Q4_K_M`:

   ```sh
   python <llama.cpp>/convert_hf_to_gguf.py <snapshot-dir> \
     --outfile <llama-bf16.gguf> --outtype bf16
   <llama.cpp-build>/bin/llama-quantize \
     <llama-bf16.gguf> <Llama-3.2-3B-Instruct-Q4_K_M.gguf> Q4_K_M
   sha256sum <llama-bf16.gguf> <llama.cpp-build>/bin/llama-quantize \
     <Llama-3.2-3B-Instruct-Q4_K_M.gguf>
   ```

Replace the angle-bracketed paths with local paths and compare all reported digests,
as well as the final file size, with `evaluation/experiment_setup/models.yaml`. Do not
use a reconstructed model for public-bundle replay when the final identity differs.

## One-command preparation

From PowerShell in the repository root:

```powershell
.\tools\windows\setup_public_rehost.ps1 `
  -DownloadRoot 'C:\LARIS-Evaluation1\evaluation1-release-v1' `
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
