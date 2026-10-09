# Llama 3.2 3B · K4 · Log-linked evaluation

Ten users produced 60 C1–C6 bundles from 180 R01–R18 requests, followed by 180
independent-process queries after bundle creation completed. The aggregate was computed
locally from sealed records without rerunning model queries. Only the 180 Native
`HISTORICAL_MATCHED_CHECKPOINT` trials are compared; the 180 C6 full-history controls
are excluded.

## Results by checkpoint

| Checkpoint | Native 5/5 recall | Rehosted 5/5 recall | Exact response match |
|---|---:|---:|---:|
| C1 | 28/30 (93.33%) | 28/30 (93.33%) | 30/30 |
| C2 | 30/30 (100%) | 30/30 (100%) | 30/30 |
| C3 | 28/30 (93.33%) | 28/30 (93.33%) | 30/30 |
| C4 | 30/30 (100%) | 29/30 (96.67%) | 29/30 |
| C5 | 29/30 (96.67%) | 29/30 (96.67%) | 30/30 |
| C6 | 29/30 (96.67%) | 29/30 (96.67%) | 30/30 |
| Total | 174/180 (96.67%) | 173/180 (96.11%) | 179/180 (99.44%) |

## Reproduction and preservation

- Exact question text: 180/180.
- Exact checkpoint input/response context: 180/180.
- Prompt hashes and response history through the checkpoint: 180/180.
- These checks establish recorded context identity, not direct Native/KV byte equality.
- Exact response string match: 179/180 (99.44%), unchanged after whitespace removal.
- Field-value agreement: 895/900 (99.44%). Matching wrong answers count as reproduced
  output but not as correct recall.
- Preservation of 870 Native-correct fields: 865 (99.43%); five were lost and none
  became newly correct.
- Correct/correct: 865; correct/incorrect: 5; incorrect/incorrect: 30.

All six Native failures also failed after rehosting. The one added mismatch is
`U08/C4/R11` (Native 5/5, rehosted 0/5); R10 and R12 from the same user and bundle
matched Native. Three rehosts completed without error with identical restored ranges
and byte counts. The aggregate does not distinguish numerical variation from a bundle
or rehosting defect, and no result was excluded or rerun because of its score.

## Attribution and mixing

An independent post-run oracle joined client completion time and returned slot to server
task-completion logs without using the endpoint selected by attribution or model recall.
Request attribution was 180/180: 93 `CERTIFIED` and 87 `RANKED`.

No cross-user code was observed under Native or rehosting. Both emitted ten fields from
another record for the same user and no unregistered complete code form. Other incorrect
fields classified as `UNRECOGNIZED` numbered 20 for Native and 25 for rehosting. They
are not uniformly relabeled as hallucination or `Unknown`, and zero observed cross-user
mixing does not establish absence of every hallucination form.

Recall uses the same target-code-presence rule as qualification and pilot runs. A
separate label-by-label field comparison produces the same correct-field count.

## Preservation and measurement scope

Public JSON contains only value-free counts, rates, and input digests. Original bundles,
requests, responses, logs, and queries are retained outside Git. After exhaustive host
archive verification and user approval, only this VM run directory was removed; corpus,
model, and installed LARIS remained. Free VM space was 49,964,773,376 bytes.

The 52,936,364,278-byte source archive contains 15,905 exhaustively verified files.
Declared payload sizes/digests for all 60 bundles, bundle-index digests, and rehost
manifest digests matched the source inventory. Each of 7,979 analysis copies matched
its source digest.

Archive SHA-256: `6cb8ada372580d9677a56e8afad0975143af98d43ce115796403822c64dad428`.

Exhaustive manifest SHA-256: `093244afc62f359edaf9358b606d7a380c6eeb750cccb96f59c8de9474f38f8b`.

Stored timestamps permit post-run request/query timing, but this accuracy run includes
concurrent queuing, capture, sealing, export, and rehosting. It is not used to claim a
separate overhead result or speed improvement.
