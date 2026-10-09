# Llama 3.2 3B · K4 · Proxy evaluation

Ten users produced 60 C1–C6 bundles from 180 R01–R18 requests, followed by 180
independent-process queries after all bundles were complete. The aggregate was computed
locally from sealed records without rerunning model queries. Only the 180 Native
`HISTORICAL_MATCHED_CHECKPOINT` trials are compared; the 180 C6 final-window controls
are excluded.

## Results by checkpoint

| Checkpoint | Native 5/5 recall | Proxy 5/5 recall | Exact response match |
|---|---:|---:|---:|
| C1 | 28/30 (93.33%) | 28/30 (93.33%) | 30/30 |
| C2 | 30/30 (100%) | 30/30 (100%) | 30/30 |
| C3 | 28/30 (93.33%) | 26/30 (86.67%) | 28/30 |
| C4 | 30/30 (100%) | 30/30 (100%) | 30/30 |
| C5 | 29/30 (96.67%) | 29/30 (96.67%) | 30/30 |
| C6 | 29/30 (96.67%) | 28/30 (93.33%) | 29/30 |
| Total | 174/180 (96.67%) | 171/180 (95.00%) | 177/180 (98.33%) |

## Reproduction and preservation

- Exact question text: 180/180.
- Exact checkpoint input/response context: 180/180.
- Prompt hashes and response history through the checkpoint: 180/180.
- Exact response string match: 177/180 (98.33%), unchanged after whitespace removal.
- Field-value agreement: 885/900 (98.33%). Matching wrong answers count as reproduced
  output but not as correct recall.
- Preservation of 870 Native-correct fields: 855 (98.28%); 15 were lost and none
  became newly correct.
- Correct/correct: 855; correct/incorrect: 15; incorrect/incorrect: 30.

All six Native failures also failed after Proxy rehosting. The added mismatches are
`U03/C3/R08`, `U06/C3/R08`, and `U04/C6/R16`; each changed from Native 5/5 to
rehosted 0/5. No error termination, cross-user value, same-user/different-record value,
or unregistered complete code form was detected. No result was excluded or rerun because
of its score, and the aggregate does not establish numerical variation or a bundle defect
as the cause.

## Direct attribution and mixing

Unlike Log-linked correlation, Proxy directly joins an authenticated request identifier
to a committed lease. Stored Proxy bindings matched source user, record, request ID,
user HMAC, identity-key epoch, lease, session, and slot epoch for all 180 requests.

- Direct request attribution: 180/180.
- Per-user C1–C6 sealing decisions: `CERTIFIED` 60/60.
- Missing, unresolved, or incorrect attribution: zero each.
- Cross-user owner-HMAC reuse: zero.
- Cross-user code mixing: zero.
- Same-user/different-record output: ten fields across two responses under both Native
  and Proxy.
- Unregistered complete code forms: zero.

The remaining 35 incorrect Proxy fields are `UNRECOGNIZED`; they are not uniformly
relabeled as hallucination or `Unknown`. Zero cross-user mixing does not establish the
absence of every hallucination form.

## Bundle and source preservation

The 60 bundle payloads total 4,483,894,400 bytes: 378,631,700 at C1,
727,090,300 at C2, and approximately 842–846 MB at C3–C6, consistent with a K4
active window that plateaus after filling.

The retained archive is 38,858,116,953 bytes. Counts, sizes, and SHA-256 values for
all 5,903 source files were exhaustively verified; the VM source was not deleted.

- Archive SHA-256: `e189f525ed9abc09322ee3902b6c84c6d7e8665e0a68c8cf2abea58065a7b3b4`
- Exhaustive manifest SHA-256: `f5118d8ac14773b590727b2e029dc28f46e5d765444c37cfc90e91b820a7a186`

Stored timestamps permit post-run request/query timing, but this accuracy run includes
concurrent queuing, capture, sealing, export, and rehosting. It is not used to claim a
separate overhead result or speed improvement.
