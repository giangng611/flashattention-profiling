# Fused Attention Config Sweep Report

## Purpose

This experiment is a small step from simply reproducing a Triton fused-attention kernel toward understanding workload-aware kernel configuration.

The earlier fused-attention benchmark showed that my Triton kernel is numerically correct, but still slower than PyTorch SDPA on the tested workloads. This report asks a narrower question:

```text
Do different attention workloads prefer different Triton BLOCK_M/BLOCK_N configurations?
```

This question connects the code in this repo to the paper-reading direction:

- The Anatomy of a Triton Attention Kernel motivates workload-specific attention tuning.
- tritonBLAS motivates reducing expensive exhaustive tuning through configuration selection.
- FlashInfer motivates testing realistic LLM-like attention shapes instead of only toy shapes.

## Setup

Script:

```text
src/sweep_triton_fused_attention_configs.py
```

Workload file:

```text
workloads/small_attention_workloads.csv
```

Stable command:

```bash
python src/sweep_triton_fused_attention_configs.py \
  --workloads workloads/small_attention_workloads.csv \
  --block-ms 32 64 128 \
  --block-ns 32 64 128 \
  --warmup 20 \
  --trials 100 \
  --output results/triton_fused_attention_config_sweep.csv \
  --summary-output results/triton_fused_attention_config_sweep_summary.csv \
  --metadata-output results/triton_fused_attention_config_sweep_metadata.json
```

Environment:

- GPU: NVIDIA GeForce RTX 3090
- CUDA: 13.0 through PyTorch
- PyTorch: 2.14.0+cu130
- Triton: 3.8.0
- dtype: float16
- causal attention: true

## Stable Sweep Summary

The sweep tested `BLOCK_M x BLOCK_N` values from:

```text
BLOCK_M in {32, 64, 128}
BLOCK_N in {32, 64, 128}
```

For each supported workload, the fastest correct Triton result within this small search space is treated as the local oracle.

| Workload | Status | PyTorch SDPA median ms | Best Triton config | Best Triton median ms | Best Triton TFLOP/s | Speedup over PyTorch SDPA | Valid Triton configs |
|---|---|---:|---|---:|---:|---:|---:|
| `toy-prefill-d64` | ok | 0.0707 | `bm64_bn32` | 0.0798 | 3.37 | 0.885x | 9 |
| `toy-prefill-d128` | ok | 0.0747 | `bm32_bn32` | 0.0819 | 6.57 | 0.912x | 6 |
| `qwen2.5-1.5b-like-prefill-512` | ok | 0.0963 | `bm32_bn32` | 0.1004 | 8.04 | 0.959x | 6 |
| `qwen2.5-1.5b-like-prefill-2048` | ok | 0.3133 | `bm32_bn32` | 0.5560 | 23.18 | 0.564x | 6 |
| `qwen2.5-1.5b-like-decode-2048` | skipped | | | | | | 0 |
| `qwen2.5-7b-like-prefill-512-gqa` | skipped | | | | | | 0 |

## Observations

### 1. The best Triton config is workload-dependent

The same Triton kernel does not choose the same tile configuration for every supported workload:

| Workload | Best config |
|---|---|
| `toy-prefill-d64` | `bm64_bn32` |
| `toy-prefill-d128` | `bm32_bn32` |
| `qwen2.5-1.5b-like-prefill-512` | `bm32_bn32` |
| `qwen2.5-1.5b-like-prefill-2048` | `bm32_bn32` |

This is the main useful result from this sweep. Even in a small search space, the best tile choice is not completely universal. The `head_dim=64` toy workload preferred `BLOCK_M=64, BLOCK_N=32`, while the `head_dim=128` workloads preferred `BLOCK_M=32, BLOCK_N=32`.

This supports the idea that attention kernel configuration should be considered workload-aware rather than fixed once for all shapes.

### 2. Some tile configurations are infeasible before performance is measured

Several configurations with `BLOCK_N=128` failed on `head_dim=128` workloads with an out-of-resources error:

```text
OutOfResources: out of resource: shared memory
Required: 147456 or 163840
Hardware limit: 101376
```

The failed configurations were:

| Workload | Failed configs | Reason |
|---|---|---|
| `toy-prefill-d128` | `bm32_bn128`, `bm64_bn128`, `bm128_bn128` | shared memory requirement exceeds hardware limit |
| `qwen2.5-1.5b-like-prefill-512` | `bm32_bn128`, `bm64_bn128`, `bm128_bn128` | shared memory requirement exceeds hardware limit |
| `qwen2.5-1.5b-like-prefill-2048` | `bm32_bn128`, `bm64_bn128`, `bm128_bn128` | shared memory requirement exceeds hardware limit |

This separates the selection problem into two parts:

```text
1. Feasibility filtering: can this config run on this GPU for this workload?
2. Performance selection: among valid configs, which one is fastest?
```

This is important because a practical config selector should not only predict fast configurations. It should also avoid configurations that exceed hardware resources.

### 3. The Triton kernel is still slower than PyTorch SDPA

All best Triton results are still slower than PyTorch SDPA for the tested workloads. The closest case is `qwen2.5-1.5b-like-prefill-512`, where best Triton reaches about `0.959x` of PyTorch SDPA. The largest gap is `qwen2.5-1.5b-like-prefill-2048`, where best Triton reaches about `0.564x` of PyTorch SDPA.

This should not be interpreted as a failure of Triton. PyTorch SDPA is a strong production baseline and may dispatch to highly optimized FlashAttention-style kernels. My current Triton kernel is still a simple forward-only baseline.

The useful conclusion is narrower:

```text
The Triton kernel is correct and exposes workload-dependent tile behavior,
but it is not yet competitive with PyTorch SDPA on long prefill.
```

### 4. Decode and GQA are still outside the current kernel scope

Two workload rows were skipped intentionally:

| Workload | Reason |
|---|---|
| `qwen2.5-1.5b-like-decode-2048` | current Triton kernel supports self-attention only, so `q_len` must equal `kv_len` |
| `qwen2.5-7b-like-prefill-512-gqa` | current Triton kernel does not support GQA/MQA yet |

This limitation is important for the next research step. Real LLM serving depends heavily on decode and GQA. The current experiment is only about dense prefill self-attention.

## What This Means For The Research Direction

This sweep gives a concrete, small-scale version of the paper-reading idea:

```text
same attention math
same GPU
same dtype
different workload shape
different feasible/best tile behavior
```

A possible research direction is therefore:

```text
workload-aware configuration selection for Triton fused attention
```

The current result does not prove that this is a full paper topic yet. It only shows that:

- tile choices matter;
- not every tile choice is feasible;
- different workloads can prefer different tile choices;
- a simple baseline Triton kernel is not enough to match PyTorch SDPA;
- more realistic attention features, especially decode and GQA, are needed before making stronger claims.

## Next Steps

The most useful next steps are:

1. Add `num_warps` and `num_stages` to the sweep, not only `BLOCK_M/BLOCK_N`.
2. Add a feasibility summary that predicts or records why configs fail.
3. Add GQA support so Qwen-like workloads are not skipped.
4. Add a decode-specific path for `q_len=1, kv_len >> 1`.
5. Compare three selectors:
   - fixed config;
   - exhaustive sweep oracle;
   - simple workload-aware rule.

The next hypothesis to test could be:

```text
A simple rule based on head_dim and phase can select a near-oracle Triton
configuration while testing fewer configs than exhaustive search.
```

## Current Interpretation In One Paragraph

This experiment shows that my Triton fused-attention baseline is correct but not yet as fast as PyTorch SDPA. More importantly, it shows that tile configuration is not a cosmetic detail: some configurations are invalid because they exceed shared-memory limits, and the fastest valid configuration can vary across workload shapes. This gives a concrete reason to study workload-aware configuration selection before moving to a larger serving-level evaluation.
