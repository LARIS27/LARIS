# Evaluation2: KV capture overhead

Status: `COMPLETE`

This is the finalized Evaluation2 result for this public release. Start at
the [evaluation index](../../README.md) or the
[repository quickstart](../../../docs/QUICKSTART.md) for the complete reproduction path.

This evaluation used Llama 3.2 3B Q4_K_M and a balanced paired design: **5 balanced
blocks x 5 conditions = 25 runs**. Every run issued 180 requests with 123,529 prompt
tokens and 1,080 generated tokens. Each run started from the same VM snapshot. Order
was balanced across blocks rather than fixed. The machine-readable source is
[`llama-3.2-3b/summary.json`](llama-3.2-3b/summary.json).

## End-to-end result

| Condition | Capture policy | Runtime, mean (95% CI) | Matched reference | Paired overhead, mean (95% CI) |
|---|---|---:|---|---:|
| Native | None | 2284.32 s (2032.76–2535.87) | — | — |
| Proxy control | Attribution only; KV copying disabled | 2401.69 s (2175.77–2627.61) | Native | — |
| Proxy | Full | 2834.74 s (2649.91–3019.58) | Proxy control | 18.415% (7.655–29.174) |
| Log-linked control | Attribution only; KV copying disabled | 2222.24 s (2132.07–2312.42) | Native | — |
| Log-linked | Full | 4504.30 s (4275.73–4732.87) | Log-linked control | 102.769% (93.044–112.495) |

All 25 runs completed 180/180 requests and passed host verification. Both capture modes
reported **lost events: 0**. Each full run captured the same 14,270,513,152 KV bytes.

## Paired metric

For mode `m` and balanced block `b`, `Tfull(m,b)` is the full condition runtime and
`Tcontrol(m,b)` is the runtime of the same attribution path with KV copying disabled.
The block-level overhead is:

```text
O(m,b) = 100 × (Tfull(m,b) / Tcontrol(m,b) − 1)
```

The table reports the mean of the five block values and its 95% confidence interval.
Native and both attribution-only controls are shown so that the absolute runtime and
the matched comparison used by each full mode remain explicit. The overhead value is
the full/control runtime change within the same balanced block.

The result supports a moderate Proxy overhead in this environment. It does not support
a low-overhead claim for the current Log-linked implementation: its paired mean is
102.769%. Accuracy, attribution correctness and zero loss are reported separately from
performance.
