# Quickstart

This quickstart covers the public repository and the Evaluation1 KV release. It does
not require the removed interactive UI or any legacy checkpoint backend.

## 1. Install LARIS

From the repository root:

```text
python -m pip install -e .
```

For native capture, use a supported Linux host with clang, libbpf, and the required
kernel permissions. For replay, obtain the compatible GGUF model weights separately.

## 2. Prepare the Evaluation1 artifact directory

The tracked metadata root is:

```text
evaluation/publication/releases/evaluation1-release-v1
```

The ignored external artifact root is:

```text
evaluation/publication/external/evaluation1-release-v1
```

Create the external directory skeleton without copying or changing tracked metadata:

```text
python evaluation/scripts/stage_public_evaluation_release.py initialize --metadata-root evaluation/publication/releases/evaluation1-release-v1 --external-root evaluation/publication/external/evaluation1-release-v1
```

Print the exact seven-file download checklist:

```text
python evaluation/scripts/stage_public_evaluation_release.py checklist --metadata-root evaluation/publication/releases/evaluation1-release-v1
```

Download each file from its file-specific public location into the matching relative
path under the external root. `PENDING_PUBLICATION` means that the external release is
not yet available; it is not a valid download location.

Verify the completed external tree offline before replay:

```text
python evaluation/scripts/stage_public_evaluation_release.py verify --metadata-root evaluation/publication/releases/evaluation1-release-v1 --external-root evaluation/publication/external/evaluation1-release-v1
```

Verification is fail-closed: unexpected files, missing files, symlinks, size changes,
or digest changes are rejected.

## 3. Prepare the qualified rehosting runtime

Listing and inspecting tuples requires only the verified external release. Executing a
rehost additionally requires an investigator profile that points to the exact model,
native helper, shared libraries, and WSL runtime installed on the local host. The
profile itself is not portable and is therefore not stored in Git. On a new Windows
PC, follow the bounded [public rehosting setup](REHOSTING_SETUP.md); it verifies a
whole-folder download, builds the pinned Linux runtime in WSL, and creates the local
investigator profile. WSL is used for rehosting only, not live acquisition.

Evaluation1 used llama.cpp commit
`daf6bc9f2d3516d7bda73d191c6a01388aff09c5`. The observer and launch contract is
[`configs/runtime/runtime_state_observer/x86_64_cpu_simple_kv.json`](../configs/runtime/runtime_state_observer/x86_64_cpu_simple_kv.json),
and exact model revisions, filenames, sizes, and SHA-256 digests are pinned in
[`evaluation/experiment_setup/models.yaml`](../evaluation/experiment_setup/models.yaml).
Build the rehosting helper with the
[`native/rehosting/Makefile`](../native/rehosting/Makefile) against that pinned
llama.cpp build. Operators using a qualified capture node can generate and
validate the local Windows/WSL profile with
[`tools/windows/install_rehost_profile.ps1`](../tools/windows/install_rehost_profile.ps1).

The commands below use the default Windows profile and analysis locations. They fail
closed if the model digest, runtime profile, helper, identity key, or bundle binding
does not match.

## 4. Inspect and replay a published tuple

Set the concrete Llama example once in PowerShell. On a new PC, use the path printed
as `LARIS_PUBLIC_RELEASE_ROOT` by `setup_public_rehost.ps1`; the value below is the
equivalent ignored in-repository staging path:

```powershell
$releaseRoot = 'evaluation/publication/external/evaluation1-release-v1'
$modelId = 'llama-3.2-3b-instruct-q4-k-m'
$profile = Join-Path $env:APPDATA 'LARIS\rehost-profile.json'
$analysisRoot = Join-Path $env:APPDATA 'LARIS\analyses'
```

`U01` is a publication-safe alias, not itself a HMAC or a production identity. In
Proxy mode, it resolves to the synthetic authenticated user or service-account
identifier used in the paper. In Log-linked mode, it resolves to the synthetic
endpoint type and IP address observed at service access. LARIS uses the corresponding
preservation key to recompute the HMAC reference and compare it with the selected
bundle before rehosting. The public alias therefore exposes the paper's identity
selection step without requiring a reader to copy a 43-character HMAC manually.

List available tuples:

```text
laris reproduce evaluation1 list --release-root $releaseRoot --model-id $modelId --mode PROXY
```

Inspect one tuple without materializing values:

```text
laris reproduce evaluation1 show --release-root $releaseRoot --model-id $modelId --mode PROXY --checkpoint-id C1 --user-ref U01
```

Replay the published recall question:

```text
laris reproduce evaluation1 rehost --release-root $releaseRoot --model-id $modelId --mode PROXY --checkpoint-id C1 --user-ref U01 --record-id R01 --profile $profile --analysis-root $analysisRoot --json
```

Verify the response against the published exact-value oracle:

```text
laris reproduce evaluation1 verify-recall --release-root $releaseRoot --model-id $modelId --mode PROXY --checkpoint-id C1 --user-ref U01 --record-id R01 --response-file response.txt
```

For Log-linked replay, use `--mode LOG_LINKED`. To ask a custom question, replace the
published record query with `--question-file QUESTION.txt`; that is an arbitrary
qualitative query and is not scored as a published recall trial.

## 5. Live-capture boundary

Current capture profiles identify the live target with PID/start ticks so that PID reuse
cannot silently redirect evidence. The public event and acquisition contracts are the
[runtime-state event schema](../schemas/runtime_state_observer_event.schema.json) and
[acquisition-manifest schema](../schemas/runtime_state_acquisition_manifest.schema.json).
The [operator guide](OPERATOR_GUIDE.md) describes the current Proxy and Log-linked
lifecycles and their fail-closed evidence boundary.

## 6. Read the paper results

- Evaluation1 accuracy and isolation: `evaluation/results/attribution-accuracy-and-cross-user-isolation/comparison.md`
- Evaluation2 overhead: `evaluation/results/runtime-overhead/README.md`
- Complete replay protocol: `docs/reproducibility/evaluation1-release.md`
