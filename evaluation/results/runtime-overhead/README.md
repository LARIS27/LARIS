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

## Paper result

| Mode | KV capture work per run, mean (95% CI) | Normalized KV-enabled overhead, mean (95% CI) |
|---|---:|---:|
| Native | 0.000 s (0.000–0.000) | 0.000% (0.000–0.000) |
| Proxy | 332.190 s (298.942–365.439) | 18.415% (7.655–29.174) |
| Log-linked | 149.704 s (128.692–170.716) | 102.769% (93.044–112.495) |

All 25 runs completed 180/180 requests and passed host verification. Both capture modes
reported **lost events: 0**. Each full run captured 14,270,513,152 KV bytes; each mode
therefore captured 71,352,565,760 bytes across its five full runs.

## What the two columns mean

`KV capture work per run` is the time accumulated by the instrumented capture work in
the full run. It is a direct-work counter and **not the end-to-end overhead by itself**.
It must not be substituted for the change in application runtime.

For mode `m` and balanced block `b`, let:

- `Tnative(b)` be the Native duration;
- `Tfull(m,b)` be the duration with that attribution path and KV capture enabled;
- `Tcontrol(m,b)` be the duration of the same attribution path with KV copying disabled;
- `F(m,b) = Tfull(m,b) / Tnative(b)`; and
- `C(m,b) = Tcontrol(m,b) / Tnative(b)`.

The reported paired overhead is:

```text
O(m,b) = 100 × (F(m,b) / C(m,b) − 1)
```

The table reports the mean of the five block values and its 95% confidence interval.
The same-block Native term makes the normalization and pairing explicit; algebraically,
the ratio is the full/control runtime change within the block. It therefore includes
capture-induced stalls, scheduling effects, encoding and persistence, not only the
direct-work counter.

The result supports a moderate Proxy overhead in this environment. It does not support
a low-overhead claim for the current Log-linked implementation: its paired mean is
102.769%. Accuracy, attribution correctness and zero loss are reported separately from
performance.
