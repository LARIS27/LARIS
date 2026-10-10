# Figure 3 source records at TRACE (4)

Figure 3 is based on an actual `llama-server` log recorded at `TRACE (4)`. This
directory publishes the original log and its corresponding normalized and attribution
records as the source records underlying the figure. These records are separate from
the quantitative evaluation aggregates reported in the paper.

The figure selects two overlapping requests from the trace and preserves their
timestamps, slot IDs, task IDs, token counts, event order, and HTTP status. Only the
endpoint labels shown in the paper were replaced with documentation-only addresses.

This directory publishes the complete source triplet used to verify the figure:

| File | Role | SHA-256 |
|---|---|---|
| [`source/raw_llama_server.log`](source/raw_llama_server.log) | Original application log | `bff29968a0779ba836a426c2327e9dd16f002edb7a5efaa91291a05a384171c9` |
| [`source/normalized_application_events.jsonl`](source/normalized_application_events.jsonl) | Normalized application events | `5424d629b6f275eba7d63015d48ef6931e3716bd23efe82ce8b1088eb5d61992` |
| [`source/attribution_results.jsonl`](source/attribution_results.jsonl) | Final correlation results | `7f34c7c15ac88b5087977d763357c52194521d7c226af42722506af6e878860f` |

## Rows used in Figure 3

| Request | Slot / task | Start | First observed completion signal | Second observed completion signal | Final correlation result |
|---|---|---|---|---|---|
| A | slot 0 / task 7 | row 52, `0.12.309.817` | execution termination, row 74, `0.39.067.124` | HTTP completion, row 75, `0.39.067.304` | `CERTIFIED` |
| B | slot 2 / task 6 | row 60, `0.15.390.585` | HTTP completion, row 123, `0.54.656.583` | execution termination, row 124, `0.54.656.687` | `CERTIFIED` |

The table retains the source ordering and the final recorded correlation result for
each selected request.

## Third request present in the same trace

The paper shows two requests for legibility. The same source interval also contains
a third overlapping request. These are original rows, not a constructed schematic:

| Slot / task | Start | execution termination | HTTP completion | Final correlation result |
|---|---|---|---|---|
| slot 0 / task 7 | row 52 | row 74 | row 75 | `CERTIFIED` |
| slot 1 / task 5 | row 55 | row 98 | row 99 | `CERTIFIED` |
| slot 2 / task 6 | row 60 | row 124 | row 123 | `CERTIFIED` |

The source log contains no prompt or response body. Its `127.0.0.x` endpoint values
are synthetic loopback identities used by the controlled evaluation. Figure 3 maps
the selected endpoint labels to documentation ranges; no timing or ordering field is
changed.

## Scope

These files document the source records underlying Figure 3 only. Quantitative
Evaluation1 results and their model-specific source records are published in the
[attribution results](../attribution-accuracy-and-cross-user-isolation/).
