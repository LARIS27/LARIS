# Attribution Accuracy and Cross-User Non-Mixing Verification

The comparison uses the same sealed corpus, K4 configuration, ten synthetic users,
and C1–C6 checkpoints. Each model's Native baseline contains 180
`HISTORICAL_MATCHED_CHECKPOINT` trials. Log-linked and Proxy each contain 180 separate
rehosting calls that ask the same questions against the corresponding checkpoint
bundles. Responses from different models are not compared for literal equality.
The model-specific Log-linked reports publish the exact C1–C6 application logs,
normalized events, attribution results, and archive-bound provenance used to recompute
the 360 final attribution decisions.

## Llama 3.2 3B

| Metric | Native | Log-linked | Proxy |
|---|---:|---:|---:|
| Correct-field recall | 870/900 (96.67%) | 865/900 (96.11%) | 855/900 (95.00%) |
| Complete 5/5 recall | 174/180 (96.67%) | 173/180 (96.11%) | 171/180 (95.00%) |
| Exact Native-response reproduction | Baseline | 179/180 (99.44%) | 177/180 (98.33%) |
| Native correct-field preservation | Baseline | 865/870 (99.43%) | 855/870 (98.28%) |
| Request-attribution accuracy | Not applicable | 180/180 | 180/180 |
| Observed cross-user code mixing | 0 | 0 | 0 |

### Complete 5/5 recall by checkpoint

| Checkpoint | Native | Log-linked | Proxy | Exact Log-linked response match | Exact Proxy response match |
|---|---:|---:|---:|---:|---:|
| C1 | 28/30 | 28/30 | 28/30 | 30/30 | 30/30 |
| C2 | 30/30 | 30/30 | 30/30 | 30/30 | 30/30 |
| C3 | 28/30 | 28/30 | 26/30 | 30/30 | 28/30 |
| C4 | 30/30 | 29/30 | 30/30 | 29/30 | 30/30 |
| C5 | 29/30 | 29/30 | 29/30 | 30/30 | 30/30 |
| C6 | 29/30 | 29/30 | 28/30 | 30/30 | 29/30 |
| **Total** | **174/180** | **173/180** | **171/180** | **179/180** | **177/180** |

Relative to Native, Log-linked adds one failure and Proxy adds three failures.

## Qwen2.5 7B

| Metric | Native | Log-linked | Proxy |
|---|---:|---:|---:|
| Correct-field recall | 882/900 (98.00%) | 876/900 (97.33%) | 877/900 (97.44%) |
| Complete 5/5 recall | 173/180 (96.11%) | 171/180 (95.00%) | 172/180 (95.56%) |
| Exact Native-response reproduction | Baseline | 178/180 (98.89%) | 179/180 (99.44%) |
| Native correct-field preservation | Baseline | 876/882 (99.32%) | 877/882 (99.43%) |
| Request-attribution accuracy | Not applicable | 180/180 | 180/180 |
| Observed cross-user code mixing | 0 | 0 | 0 |

### Complete 5/5 recall by checkpoint

| Checkpoint | Native | Log-linked | Proxy | Exact Log-linked response match | Exact Proxy response match |
|---|---:|---:|---:|---:|---:|
| C1 | 26/30 | 25/30 | 26/30 | 29/30 | 30/30 |
| C2 | 30/30 | 30/30 | 30/30 | 30/30 | 30/30 |
| C3 | 30/30 | 30/30 | 30/30 | 30/30 | 30/30 |
| C4 | 27/30 | 27/30 | 26/30 | 30/30 | 29/30 |
| C5 | 30/30 | 30/30 | 30/30 | 30/30 | 30/30 |
| C6 | 30/30 | 29/30 | 30/30 | 29/30 | 30/30 |
| **Total** | **173/180** | **171/180** | **172/180** | **178/180** | **179/180** |

Relative to Native, Log-linked adds two failures and Proxy adds one failure. The first
Log-linked rehosting attempt stopped after 14 trials. The table uses only the completed
180-trial rerun against the same preserved 60 bundles; the interrupted trials are not
mixed into the result.

Both LARIS modes achieved 180/180 request attribution for both models, with zero
observed cross-user code mixing. Different or failed-recall responses were not removed.
These are K4 single-cell results and do not include K8 or the separate overhead study.

Follow the [Evaluation1 release and replay guide](../../../docs/reproducibility/evaluation1-release.md)
to select the same C1–C6/U01–U10 tuples from the external KV release and submit the
paper question or a custom question. The public tools do not alter these
aggregate values: they reverify the selected tuple's digest and attribution identifier
before passing it to the current rehosting engine.
