# Reproducing Evaluation1 from the public KV release

This guide defines the public artifact boundary and the exact replay protocol for the
paper's attribution-accuracy and cross-user non-mixing evaluation. The release contains
current LARIS code and externally hosted KV artifacts; it does not restore or depend on
the removed legacy checkpoint implementation.

Verified small metadata is tracked at
`evaluation/publication/releases/evaluation1-release-v1`. The downloaded external
release may be kept anywhere outside the repository. The new-machine setup guide uses
`C:\LARIS-Evaluation1\evaluation1-release-v1` as an example and passes that downloaded
release directory directly; see
[`docs/REHOSTING_SETUP.md`](../REHOSTING_SETUP.md). Repository maintainers may instead
stage the same verified files at the ignored development path
`evaluation/publication/external/evaluation1-release-v1`.

## Artifact boundary

The complete public candidate contains exactly **240** addressable KV entries:

```text
2 models × 2 capture modes × 6 checkpoints × 10 users = 240 entries
```

The modes are Proxy and Log-linked. Native is the matched comparison condition and has
no captured KV artifact. Each model/mode archive contains exactly 60 tuples. Proxy keeps
the original complete ten-user sealed package for each checkpoint and addresses a user
through its index; Log-linked keeps the original per-user sealed package. The release
does not split either source package into a new evidentiary object. Every tuple
is addressed by `(model_id, mode, checkpoint_id, public_user_ref)` and is bound to its
payload size, payload digest, source manifest digest, session reference and attribution
identifier in the public index.

Proxy entries publish the exact synthetic runtime identity key, key ID and authenticated
subject ID used by the sealed Evaluation1 run. For example, public user `U01` is addressed
with authenticated subject `U01` under key ID
`identity-epoch-20260915-5f0d02cf`; the resolver recomputes and checks its `USER_HMAC`
before materializing a bundle. Log-linked entries similarly publish the synthetic runtime
endpoint fixture used by the sealed run: `U01` maps to `127.0.0.2`, and the resolver
recomputes its endpoint reference with key ID `paper-eval-correlation-v1`.

The shared campaign data also retains the earlier corpus-normalization labels as
provenance: Proxy `U01` was represented there as `paper-eval:u01`, and the corresponding
Log-linked source endpoint was `127.20.0.1`. Those campaign labels are not the runtime
identifiers that address the sealed bundles. Keeping both fields explicit prevents
accidental derivation of an identifier with the wrong identity boundary.
All of these values and keys are Evaluation1-only synthetic fixtures, not production
credentials. The verifier rejects a release that omits a runtime fixture, changes a
derived identifier, or publishes a key that no public index entry uses.

KV archives are external artifacts, not Git objects. Model weights, VM disks, source
archives, private host paths, raw service logs and unrelated evidence copies are not in
the release. Publication is fail-closed: every model's release input must carry an
explicit human `PUBLIC_APPROVED` license-review record. Absence of that record means the
candidate is not approved for distribution.

## Checkpoint contract

Each user supplied 18 complete record turns in order. A checkpoint was captured after
the third acknowledgement in each group; no filler or padding was inserted.

| Checkpoint | Target records | Capture boundary |
|---|---|---|
| C1 | R01–R03 | immediately after the R03 acknowledgement |
| C2 | R04–R06 | immediately after the R06 acknowledgement |
| C3 | R07–R09 | immediately after the R09 acknowledgement |
| C4 | R10–R12 | immediately after the R12 acknowledgement |
| C5 | R13–R15 | immediately after the R15 acknowledgement |
| C6 | R16–R18 | immediately after the R18 acknowledgement |

The source record prompt used this exact template, with sealed synthetic values supplied
by the public artifact's shared evaluation data:

```text
This is record {record_id} that I organized.
Please remember these five fields together.

Material: {material_value}
Location: {location_value}
Schedule: {schedule_value}
Tool: {tool_value}
Route: {route_value}

Do not repeat any value now. Reply only with "Recorded {record_id}."
```

The acknowledgement is exactly `Recorded {record_id}.` and is part of the captured
conversation state.

## Published recall question

Each target record was queried in a separate fresh process from the same immutable
checkpoint bundle. The exact question was:

```text
Find the earlier user message whose first line is exactly:
"This is record {record_id} that I organized."

Treat only that single message as record {record_id}.
Copy each value from the matching field line in that same message.
Do not infer values. Do not combine values from any other record.

The required prefixes are:
Material: MAT-
Location: LOC-
Schedule: SCH-
Tool: TOL-
Route: RTE-

Return exactly these five lines:
Material: <exact value or Unknown>
Location: <exact value or Unknown>
Schedule: <exact value or Unknown>
Tool: <exact value or Unknown>
Route: <exact value or Unknown>
```

Published replay defaults are temperature `1.0`, top-k `1`, top-p `1.0`, min-p `1.0`, seed
`20260907`, and a 128-token generation upper bound. A custom question file may be supplied
instead; such output is a qualitative replay and is not scored as the published recall
trial. Published exact-value answers are keyed by the complete
`(checkpoint_id, public_user_ref, record_id)` tuple; the verifier never reuses one
user's oracle row for another user's bundle.

## Release operator preflight

Before creating any external artifact, verify the four sealed source archives without
writing to the source trees or creating a partial release:

```text
python evaluation/scripts/prepare_public_evaluation_release.py --spec RELEASE_SPEC.json --inventory-only
```

The command must report `verification_status=VERIFIED`, `source_count=4`, and matching
digests. Building the release is a separate invocation that additionally requires exactly
the two runtime keys referenced by the public indexes (Proxy identity and Log-linked
correlation), the pinned upstream license files, explicit human `PUBLIC_APPROVED` records
and an external destination. The campaign-normalization identity key is not an execution
key and is not shipped. A candidate is atomically renamed to its final name only after the
independent verifier succeeds.

## Reproduction commands

Create the external directory skeleton without modifying tracked metadata:

```text
python evaluation/scripts/stage_public_evaluation_release.py initialize --metadata-root evaluation/publication/releases/evaluation1-release-v1 --external-root evaluation/publication/external/evaluation1-release-v1
```

Print the exact seven-file download checklist:

```text
python evaluation/scripts/stage_public_evaluation_release.py checklist --metadata-root evaluation/publication/releases/evaluation1-release-v1
```

Download every file into its listed relative path under the external root. A location
value of `PENDING_PUBLICATION` means that the public artifact is not yet available and
must not be interpreted as a URL. Then run the offline verifier:

```text
python evaluation/scripts/stage_public_evaluation_release.py verify --metadata-root evaluation/publication/releases/evaluation1-release-v1 --external-root evaluation/publication/external/evaluation1-release-v1
```

The command checks the release manifest, all four indexes, archive membership, every
bundle payload and source-manifest digest, the synthetic identity/key fixtures, license
notices, exact paths, sizes and SHA-256 digests, and the complete 240-tuple inventory.
The underlying verifier can also be invoked directly after staging:

```text
python evaluation/scripts/verify_public_evaluation_release.py --release-root evaluation/publication/external/evaluation1-release-v1 --locations evaluation/publication/releases/evaluation1-release-v1/artifact-locations.json
```

The example below uses the published Llama model ID. On a new PC, copy the concrete
path printed as `LARIS_PUBLIC_RELEASE_ROOT` by `setup_public_rehost.ps1`. A maintainer
who staged the files inside the repository may use the ignored development path shown
below. In PowerShell, define the concrete local paths once:

```powershell
$releaseRoot = 'evaluation/publication/external/evaluation1-release-v1'
$modelId = 'llama-3.2-3b-instruct-q4-k-m'
$profile = Join-Path $env:APPDATA 'LARIS\rehost-profile.json'
$analysisRoot = Join-Path $env:APPDATA 'LARIS\analyses'
```

Here, `U01` is a publication-safe alias, not itself a HMAC or a production identity.
For Proxy, the alias selects the synthetic authenticated user or service-account
identifier; for Log-linked, it selects the synthetic endpoint type and IP address.
The resolver combines that mode-specific identity evidence with the corresponding
preservation key, recomputes the HMAC reference, and requires an exact match with the
sealed bundle index before materialization. This is the public reproduction form of
the identity-selection procedure described in the paper.

The rehosting runtime must be built against llama.cpp commit
`daf6bc9f2d3516d7bda73d191c6a01388aff09c5` and conform to
[`configs/runtime/runtime_state_observer/x86_64_cpu_simple_kv.json`](../../configs/runtime/runtime_state_observer/x86_64_cpu_simple_kv.json).
Exact Llama and Qwen model artifacts and digests are pinned in
[`evaluation/experiment_setup/models.yaml`](../../evaluation/experiment_setup/models.yaml).
The default profile path is local and host-specific; it is intentionally not published
as a transferable artifact.

List or inspect the 60 tuples for one model/mode:

```text
laris reproduce evaluation1 list --release-root $releaseRoot --model-id $modelId --mode PROXY
laris reproduce evaluation1 show --release-root $releaseRoot --model-id $modelId --mode PROXY --checkpoint-id C1 --user-ref U01
```

Replay one published question:

```text
laris reproduce evaluation1 rehost --release-root $releaseRoot --model-id $modelId --mode PROXY --checkpoint-id C1 --user-ref U01 --record-id R01 --profile $profile --analysis-root $analysisRoot --json
```

For Log-linked, change `--mode PROXY` to `--mode LOG_LINKED`; the resolver supplies the
verified endpoint reference. To ask a custom question, replace `--record-id` with
`--question-file QUESTION.txt`.

Save the response and verify the published exact-value oracle without printing synthetic
values into the receipt:

```text
laris reproduce evaluation1 verify-recall --release-root $releaseRoot --model-id $modelId --mode PROXY --checkpoint-id C1 --user-ref U01 --record-id R01 --response-file response.txt
```

The command emits canonical JSON containing only the match result and answer/response
digests. The aggregate paper results remain in
[`evaluation/results/attribution-accuracy-and-cross-user-isolation/comparison.md`](../../evaluation/results/attribution-accuracy-and-cross-user-isolation/comparison.md).
