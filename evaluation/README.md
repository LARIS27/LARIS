# Evaluation

Status: `TWO_MODEL_K4_ATTRIBUTION_ACCURACY_AND_E6_OVERHEAD_COMPLETE`

The public paper snapshot contains the finalized protocols and compact, machine-readable
results for Evaluation1 and Evaluation2. Raw VM disks, model weights, and large KV
archives are outside Git.

## Directory guide

| Directory | Contents |
|---|---|
| [`experiment_setup/`](experiment_setup/) | Evaluation protocols, schedules, and fixed inputs |
| [`publication/`](publication/) | Public-release metadata, checksums, licenses, and external-bundle layout |
| [`results/`](results/) | Evaluation summaries, machine-readable results, and figure source records |
| [`scripts/`](scripts/) | Public result-verification and reproduction utilities |

Within `results/`,
[`attribution-accuracy-and-cross-user-isolation/`](results/attribution-accuracy-and-cross-user-isolation/)
contains Evaluation1 evidence,
[`runtime-overhead/`](results/runtime-overhead/) contains Evaluation2 evidence, and
[`log-linked-correlation/`](results/log-linked-correlation/) contains the source records
underlying Figure 3.

## Evaluation1: attribution accuracy and cross-user isolation

Evaluation1 covers two models, two capture modes, six checkpoints, and ten synthetic
users:

```text
2 models × 2 capture modes × 6 checkpoints × 10 users = 240 entries
```

Native is the comparison condition and does not produce a KV bundle. Proxy and
Log-linked bundles retain their original evidentiary package boundaries and are selected
through public checkpoint/user references.

Published exact-field recovery, kept separate by evaluation condition:

| Model | Condition | Exact field recovery |
|---|---|---:|
| Llama 3.2 3B Instruct Q4_K_M | Native | 870/900 (96.67%) |
| Llama 3.2 3B Instruct Q4_K_M | Proxy | 855/900 (95.00%) |
| Llama 3.2 3B Instruct Q4_K_M | Log-linked | 865/900 (96.11%) |
| Qwen2.5 7B Instruct Q4_K_M | Native | 882/900 (98.00%) |
| Qwen2.5 7B Instruct Q4_K_M | Proxy | 877/900 (97.44%) |
| Qwen2.5 7B Instruct Q4_K_M | Log-linked | 876/900 (97.33%) |

Read the [comparison](results/attribution-accuracy-and-cross-user-isolation/comparison.md)
and the complete [Evaluation1 replay guide](../docs/reproducibility/evaluation1-release.md).
The [Figure 3 source records](results/log-linked-correlation/README.md) publish the
original `TRACE (4)` application log and its corresponding normalized and attribution
records. These records underlie the paper figure and are separate from the quantitative
evaluation aggregates.

Verified small metadata is tracked at
`evaluation/publication/releases/evaluation1-release-v1`. The seven large public files
belong under the ignored path
`evaluation/publication/external/evaluation1-release-v1` and must pass the documented
offline size and SHA-256 checks before use.

## Evaluation2: runtime overhead

Evaluation2 contains 25 completed runs arranged as five balanced blocks across five
conditions. Every run issued 180 requests with 123,529 prompt tokens and 1,080 generated
tokens from the same restored VM snapshot. The reported paired metric isolates the
KV-enabled path against its matched control.

See the completed [runtime overhead result](results/runtime-overhead/README.md) and its
machine-readable summary.

Recompute the compact Evaluation1 and Evaluation2 headline values from the tracked
machine-readable results:

```text
python evaluation/scripts/verify_published_results.py
```

The command emits a canonical `VERIFIED` receipt and exits non-zero if the source
results drift from the published aggregates.

## Distribution and model notices

The public release distributes LARIS-generated synthetic KV artifacts and the minimum
synthetic replay keys required by their sealed indexes; model weights are not redistributed.
Review the pinned upstream terms and release boundary in
[`evaluation/publication/LICENSES.md`](publication/LICENSES.md).

**Built with Llama.** Llama 3.2 is licensed under the Llama 3.2 Community License,
Copyright © Meta Platforms, Inc. All Rights Reserved.

Qwen2.5 is used under the Apache-2.0 license. The repository includes only the required
license reference and does not redistribute the Qwen model weights.
