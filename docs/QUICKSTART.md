# Quickstart

This quickstart covers the public repository and the Evaluation1 KV release. It does
not require the removed interactive UI or any legacy checkpoint backend.

## 1. Install LARIS

```text
git clone https://github.com/LARIS27/papersubmit.git
cd papersubmit
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

## 3. Inspect and replay a published tuple

List available tuples:

```text
laris reproduce evaluation1 list --release-root evaluation/publication/external/evaluation1-release-v1 --model-id MODEL_ID --mode PROXY
```

Inspect one tuple without materializing values:

```text
laris reproduce evaluation1 show --release-root evaluation/publication/external/evaluation1-release-v1 --model-id MODEL_ID --mode PROXY --checkpoint-id C1 --user-ref U01
```

Replay the published recall question:

```text
laris reproduce evaluation1 rehost --release-root evaluation/publication/external/evaluation1-release-v1 --model-id MODEL_ID --mode PROXY --checkpoint-id C1 --user-ref U01 --record-id R01 --profile PROFILE_JSON --analysis-root ANALYSIS_ROOT --json
```

Verify the response against the published exact-value oracle:

```text
laris reproduce evaluation1 verify-recall --release-root evaluation/publication/external/evaluation1-release-v1 --model-id MODEL_ID --mode PROXY --checkpoint-id C1 --user-ref U01 --record-id R01 --response-file response.txt
```

For Log-linked replay, use `--mode LOG_LINKED`. To ask a custom question, replace the
published record query with `--question-file QUESTION.txt`; that is an arbitrary
qualitative query and is not scored as a published recall trial.

## 4. Live-capture boundary

Current capture profiles identify the live target with PID/start ticks so that PID reuse
cannot silently redirect evidence. The public event and acquisition contracts are the
[runtime-state event schema](../schemas/runtime_state_observer_event.schema.json) and
[acquisition-manifest schema](../schemas/runtime_state_acquisition_manifest.schema.json).

## 5. Read the paper results

- Evaluation1 accuracy and isolation: `evaluation/results/attribution-accuracy-and-cross-user-isolation/comparison.md`
- Evaluation2 overhead: `evaluation/results/runtime-overhead/README.md`
- Complete replay protocol: `docs/reproducibility/evaluation1-release.md`
