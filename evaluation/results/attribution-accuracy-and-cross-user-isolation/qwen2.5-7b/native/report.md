# Qwen2.5 7B · Native baseline

The baseline uses 180 `HISTORICAL_MATCHED_CHECKPOINT` trials that replay the same
cumulative source path as K4 C1–C6 from an empty logical slot. The separately collected
180 C6 final-window control trials are not included.

## Results by checkpoint

| Checkpoint | Complete 5/5 recall | Correct fields |
|---|---:|---:|
| C1 | 26/30 (86.67%) | 142/150 (94.67%) |
| C2 | 30/30 (100%) | 150/150 (100%) |
| C3 | 30/30 (100%) | 150/150 (100%) |
| C4 | 27/30 (90.00%) | 140/150 (93.33%) |
| C5 | 30/30 (100%) | 150/150 (100%) |
| C6 | 30/30 (100%) | 150/150 (100%) |
| **Total** | **173/180 (96.11%)** | **882/900 (98.00%)** |

There was no observed cross-user code mixing. Two responses emitted five fields from a
different record belonging to the same user. Five responses contained eight
unregistered complete code forms, and five other incorrect fields were classified as
`UNRECOGNIZED`. These counts were computed after execution from retained responses.

Native has no LARIS attribution or rehosting bundle. It isolates Qwen2.5 7B's recall
limit at the same checkpoints. Five source transcript and run-record files were retained
and verified by SHA-256. The compressed archive is 3,102,486 bytes.

- Archive SHA-256: `043a3ef702e6afba6b388d75c67808a011dafee2aa99cb29ad6f693008c1e34d`
- Manifest SHA-256: `3cb832022922a23e147a1d008348703066d136eedca4f40ba90cfa3abe14de3f`

Recorded timestamps are operational evidence for the accuracy run, not a separate
overhead measurement.
