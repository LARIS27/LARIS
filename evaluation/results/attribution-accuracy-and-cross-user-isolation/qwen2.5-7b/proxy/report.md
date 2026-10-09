# Qwen2.5 7B · K4 · Proxy evaluation

Ten users ran R01–R18 against the same sealed corpus. Proxy directly attributed 180
source requests and produced 60 per-user C1–C6 bundles. After all bundles were sealed
and exported, 180 questions ran as separate rehosting calls. Stored records were compared
offline with the matching Native checkpoint trials without result exclusion or rescoring.

| Checkpoint | Native 5/5 recall | Proxy 5/5 recall | Exact response match |
|---|---:|---:|---:|
| C1 | 26/30 | 26/30 | 30/30 |
| C2 | 30/30 | 30/30 | 30/30 |
| C3 | 30/30 | 30/30 | 30/30 |
| C4 | 27/30 | 26/30 | 29/30 |
| C5 | 30/30 | 30/30 | 30/30 |
| C6 | 30/30 | 30/30 | 30/30 |
| Total | 173/180 (96.11%) | 172/180 (95.56%) | 179/180 (99.44%) |

Correct-field recall is **882/900 (98.00%)** for Native and **877/900 (97.44%)**
for Proxy rehosting. Label-level agreement is **895/900 (99.44%)**, and preservation
of Native-correct fields is **877/882 (99.43%)**. Questions, checkpoint context, and
source history matched **180/180**. This verifies recorded context, not direct Native/KV
byte equality.

The only different response is `U07/C4/R12`: Native recalled 5/5 target fields and
rehosting recalled 0/5. Question and source history matched; this aggregate does not
distinguish numerical variation from a bundle or rehosting defect. All seven existing
Native failures also failed after rehosting, with no newly correct field.

Authenticated requests, committed leases, and retained source records matched
**180/180**. Per-user C1–C6 seals were **CERTIFIED 60/60**. Misattribution, unresolved
or missing attribution, cross-user owner reuse, and cross-user code output were all zero.
Native and Proxy each emitted five same-user/different-record fields and eight
unregistered code-form fields. Other unrecognized fields numbered 13 for Native and 18
for Proxy. Zero observed mixing does not prove absence of every unobserved disclosure.

The retained archive contains 6,297 source-run, bundle, and log files. Inventory, size,
and SHA-256 were exhaustively verified before the corresponding VM run directory was
removed. The 60 bundle payloads total **2,136,610,356 bytes**; the archive is
**19,760,612,213 bytes**.

- Archive SHA-256: `48322f248c3ea472c41a6cbcdc41178a5ef54575b2c143487dd8721ff706728e`
- Exhaustive manifest SHA-256: `044e56c909244b32fddccbaab7f1df63ef9a70a36e497074eed4aedd9792be91`

Timing includes capture, sealing, export, and rehosting and is not interpreted as a
separate overhead benchmark.
