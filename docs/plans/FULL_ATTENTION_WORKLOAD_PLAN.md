# Full Attention Workload Plan

Qingchen's next direction is to move from the QK^T microbenchmark toward full fused attention and workload shapes collected from real LLM applications.

## Goal

```text
Reproduce a full Triton fused-attention implementation and test it on a small set of realistic LLM attention shapes.
```

The purpose is not to immediately propose a paper topic. The purpose is to gather enough evidence to decide which direction is strongest:

- faster Triton tuning
- kernel-level speed versus application-level speed
- memory prediction when hardware counters are unavailable

## Why The QK Experiment Is Not Enough

The QK experiment isolated:

```text
S = QK^T
```

But full attention is:

```text
O = softmax(QK^T / sqrt(d)) V
```

A full fused-attention kernel must handle:

- tiled QK computation
- online softmax
- numerical stability
- optional causal masking
- V accumulation
- dtype behavior
- real query/KV length differences

The QK microbenchmark was useful for learning Triton and tile sensitivity, but it cannot explain full FlashAttention behavior by itself.

## Initial Implementation Target

Start from the official Triton fused-attention tutorial:

```text
https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html
```

Use it as a baseline implementation rather than writing full attention from scratch.

I should first run it with small fixed shapes, then adapt the benchmark harness to output the same style of metrics used in this repo:

- correctness against PyTorch SDPA
- median latency
- variation
- estimated FLOPs
- median TFLOP/s
- peak memory
- environment metadata

## Current Implementation Status

I added an initial forward-only Triton fused-attention benchmark:

```text
src/benchmark_triton_fused_attention.py
```

This script compares:

- PyTorch SDPA
- a small Triton fused-attention forward kernel

The initial kernel supports contiguous causal self-attention where:

```text
query_length == kv_length
query_heads == kv_heads
```

This means it is enough for the first full-attention prefill experiment, but it is not yet a complete reproduction of production FlashAttention. Decode shapes and GQA/MQA shapes are intentionally recorded as skipped for now instead of being benchmarked incorrectly.

The purpose of this first implementation is to move one step beyond the QK^T microbenchmark and start measuring the full operation:

```text
O = softmax(QK^T / sqrt(d)) V
```

I should treat the first results as an implementation smoke test, then use them to decide whether the next useful step is:

- add GQA support
- add decode support
- add tile/autotune experiments
- compare with the official Triton tutorial kernel
- collect application-derived shapes from small LLM inference workloads

## Small Workload Table

Start with a small hand-built workload table:

| model/source | phase | batch | query length | KV length | query heads | KV heads | head_dim | dtype | causal |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| Qwen2.5-1.5B-like | prefill | 1 | 512 | 512 | 12 | 12 | 128 | fp16 | true |
| Qwen2.5-1.5B-like | prefill | 1 | 2048 | 2048 | 12 | 12 | 128 | fp16 | true |
| Qwen2.5-1.5B-like | decode | 1 | 1 | 2048 | 12 | 12 | 128 | fp16 | true |
| Qwen2.5-7B-like | prefill | 1 | 512 | 512 | 28 | 4 | 128 | fp16 | true |
| Qwen2.5-7B-like | prefill | 1 | 2048 | 2048 | 28 | 4 | 128 | fp16 | true |
| Qwen2.5-7B-like | decode | 1 | 1 | 2048 | 28 | 4 | 128 | fp16 | true |

These shapes should be verified against model configs before I use them in a report. For the first implementation pass, they are a starting workload table.

The same starter table is also saved as:

```text
workloads/small_attention_workloads.csv
```

## First Experiment

1. Copy or wrap the official Triton fused-attention tutorial implementation.
2. Run correctness checks against PyTorch SDPA on small shapes.
3. Run the small workload table above.
4. Compare:
   - PyTorch SDPA auto
   - Triton fused attention with a fixed configuration
   - Triton fused attention with a small autotune set, if available

## Evaluation Logic

Continue the tuning direction only if:

```text
different workload shapes prefer different configurations,
and exhaustive tuning has a meaningful cost.
```

If all shapes prefer the same configuration, or if tuning cost is negligible, then the stronger direction may be application-level evaluation instead of tuning prediction.

## Next Concrete Command Goal

The next server run should start with a smoke test:

```bash
python src/benchmark_triton_fused_attention.py \
  --workloads workloads/small_attention_workloads.csv \
  --warmup 2 \
  --trials 5 \
  --block-m 64 \
  --block-n 64 \
  --output results/triton_fused_attention_smoke.csv \
  --metadata-output results/triton_fused_attention_smoke_metadata.json
```

If the smoke test passes, run a more stable measurement:

```bash
python src/benchmark_triton_fused_attention.py \
  --workloads workloads/small_attention_workloads.csv \
  --warmup 20 \
  --trials 100 \
  --block-m 64 \
  --block-n 64 \
  --output results/triton_fused_attention.csv \
  --metadata-output results/triton_fused_attention_metadata.json
```

The expected first output is not that Triton beats PyTorch SDPA. PyTorch SDPA is already highly optimized. The useful first signal is whether my Triton implementation is correct, how far it is from PyTorch, and which workload shapes expose the biggest gap.
