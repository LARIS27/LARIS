# Llama 3.2 3B · Native baseline

The baseline uses 180 `HISTORICAL_MATCHED_CHECKPOINT` trials that replay the same
cumulative source path as K4 C1–C6 from an empty logical slot. The separately collected
180 C6 final-window control trials are not included.

## Results by checkpoint

| Checkpoint | Complete 5/5 recall | Correct fields |
|---|---:|---:|
| C1 | 28/30 (93.33%) | 140/150 |
| C2 | 30/30 (100%) | 150/150 |
| C3 | 28/30 (93.33%) | 140/150 |
| C4 | 30/30 (100%) | 150/150 |
| C5 | 29/30 (96.67%) | 145/150 |
| C6 | 29/30 (96.67%) | 145/150 |
| **Total** | **174/180 (96.67%)** | **870/900 (96.67%)** |

There was no observed cross-user code mixing. Two responses emitted ten fields from a
different record belonging to the same user; no unregistered complete code form was
observed. The remaining 20 incorrect fields are classified as `UNRECOGNIZED`.

Native has no LARIS attribution or rehosting bundle. It isolates the model's recall
limit at the same checkpoints. Five source transcript and run-record files were retained
and verified by SHA-256. The compressed archive is 3,134,061 bytes.

- Archive SHA-256: `aebb91c185696e8a253052ba02a9ee42cf61ff1f271eb0a437aa8be4cb9396a8`
- Manifest SHA-256: `9bec7cd8baeca261ed62961df0ce3faf9ad16ddfcfc9fe6df212ef65ab3eb2be`

Recorded timestamps are operational evidence for the accuracy run, not a separate
overhead measurement.
