# Qwen2.5 7B · K4 · Log-linked evaluation

Ten users produced 60 C1–C6 bundles from R01–R18. The first run stopped after 14
rehosting trials. Without recreating source requests or bundles, all 180 trials were
rerun in fresh processes against the same preserved 60 bundles. The incomplete 14 trials
are not mixed into the result below.

| Checkpoint | Native 5/5 recall | Rehosted 5/5 recall | Exact response match |
|---|---:|---:|---:|
| C1 | 26/30 | 25/30 | 29/30 |
| C2 | 30/30 | 30/30 | 30/30 |
| C3 | 30/30 | 30/30 | 30/30 |
| C4 | 27/30 | 27/30 | 30/30 |
| C5 | 30/30 | 30/30 | 30/30 |
| C6 | 30/30 | 29/30 | 29/30 |
| Total | 173/180 (96.11%) | 171/180 (95.00%) | 178/180 (98.89%) |

Correct-field recall is **882/900 (98.00%)** for Native and **876/900 (97.33%)**
for rehosting. Label-level output agreement is **894/900 (99.33%)**, and preservation
of fields correct under Native is **876/882 (99.32%)**. Questions, checkpoint context,
and source history all matched 180/180. This checks recorded context identity, not direct
byte equality between Native and captured KV.

The two additional mismatches are `U10/C1/R02` (Native 5/5, rehosted 4/5) and
`U05/C6/R18` (Native 5/5, rehosted 0/5). The aggregate does not establish their cause.
All seven existing Native failures also failed after rehosting, and no field became newly
correct after rehosting.

Independent client completion/slot records and server task-completion logs establish
180/180 request attribution: 101 `CERTIFIED` and 79 `RANKED`. No cross-user code was
observed. Two responses contained five fields from another record for the same user,
and nine unregistered code forms were observed. Zero observed mixing does not prove the
absence of every unobserved disclosure.

The exact C1–C6 application logs, normalized events, and attribution results supporting
that aggregate are published under `source/`. Their original archive-relative
paths, byte sizes, and SHA-256 digests are bound in
[`source-provenance.json`](source-provenance.json).

The retained source archive contains 17,194 files; the completed rerun archive contains
183 files. Each inventory, size, and SHA-256 was exhaustively verified. Source transcript
and 60-bundle-set digests were recomputed against retained metadata. Only the two verified
VM run directories were removed afterward; free VM space was 78,590,353,408 bytes.

Timing includes capture, sealing, bundle generation, and rehosting and is not interpreted
as a separate overhead measurement. Compare the [Proxy result](../proxy/report.md) as a
separate condition.
