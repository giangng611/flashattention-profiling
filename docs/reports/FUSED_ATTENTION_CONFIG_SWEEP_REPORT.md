# Fused Attention Config Sweep Report

## Purpose

This experiment is a small step from simply reproducing a Triton fused-attention kernel toward understanding workload-aware kernel configuration.

The earlier fused-attention benchmark showed that my Triton kernel is numerically correct, but the fixed configuration was slower than PyTorch SDPA on the tested workloads. This report asks a narrower question:

```text
Do different attention workloads prefer different Triton tile and launch configurations?
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
workloads/week3_config_sweep_workloads.csv
```

The Week 3 run originally used `workloads/small_attention_workloads.csv`. The snapshot above preserves exactly the six shapes from that run while applying the corrected workload labels described below.

Reproduction command with the explicit Week 3 seed:

```bash
python src/sweep_triton_fused_attention_configs.py \
  --workloads workloads/week3_config_sweep_workloads.csv \
  --block-ms 32 64 \
  --block-ns 32 64 \
  --num-warps-values 4 8 \
  --num-stages-values 2 3 \
  --warmup 20 \
  --trials 100 \
  --seed 0 \
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
- seed: 0

The report and workload snapshot are preserved in this repository. The detailed CSV, summary CSV, and metadata JSON named by the command were not present on the local machine during the Week 4 repository audit; they must be copied from the CUDA server if still available. They have not been reconstructed from this summary table.

## Stable Sweep Summary

The expanded sweep tested:

```text
BLOCK_M in {32, 64}
BLOCK_N in {32, 64}
num_warps in {4, 8}
num_stages in {2, 3}
```

This gives 16 Triton configurations per supported workload. For each supported workload, the fastest correct Triton result within this small search space is treated as the local oracle.

| Workload | Status | PyTorch SDPA median ms | Best Triton config | Best Triton median ms | Best Triton TFLOP/s | Speedup over PyTorch SDPA | Valid Triton configs |
|---|---|---:|---|---:|---:|---:|---:|
| `toy-prefill-d64` | ok | 0.0420 | `bm64_bn64_w4_s3` | 0.0440 | 6.11 | 0.953x | 16 |
| `toy-prefill-d128` | ok | 0.0840 | `bm32_bn32_w4_s3` | 0.0594 | 9.06 | 1.414x | 16 |
| `dense-qwen-shaped-proxy-prefill-512` | ok | 0.0988 | `bm32_bn64_w4_s2` | 0.0788 | 10.23 | 1.253x | 16 |
| `dense-qwen-shaped-proxy-prefill-2048` | ok | 0.2970 | `bm32_bn32_w4_s2` | 0.5130 | 25.13 | 0.579x | 16 |
| `dense-qwen-shaped-proxy-decode-2048` | skipped | | | | | | 0 |
| `qwen2.5-7b-prefill-512-gqa` | skipped | | | | | | 0 |

The two supported `12 Q heads / 12 KV heads` workloads are relabeled post-hoc as dense Qwen-shaped proxies. Their shapes and measurements are unchanged. They must not be presented as actual Qwen GQA workloads.

The run also checked for configuration errors and correctness failures. No Triton configuration in this expanded sweep failed to run, and no valid configuration failed the correctness check.

## Observations

### 1. The best Triton config is workload-dependent

The same Triton kernel does not choose the same configuration for every supported workload:

| Workload | Best config |
|---|---|
| `toy-prefill-d64` | `bm64_bn64_w4_s3` |
| `toy-prefill-d128` | `bm32_bn32_w4_s3` |
| `dense-qwen-shaped-proxy-prefill-512` | `bm32_bn64_w4_s2` |
| `dense-qwen-shaped-proxy-prefill-2048` | `bm32_bn32_w4_s2` |

This is the main useful result from this sweep. Even in a small search space, the best choice is not universal. The `head_dim=64` toy workload preferred a larger tile, while the `head_dim=128` workloads preferred smaller `BLOCK_M=32` tiles. The launch settings also mattered: all best configs used `num_warps=4`, but the toy workloads selected `num_stages=3`, while the dense Qwen-shaped proxy workloads selected `num_stages=2`.

This supports the idea that attention kernel configuration should be considered workload-aware rather than fixed once for all shapes.

### 2. Expanding the config space changed the conclusion

The previous tile-only sweep suggested that the Triton baseline was consistently slower than PyTorch SDPA. After adding `num_warps` and `num_stages`, the result became more nuanced:

| Workload | Best Triton vs PyTorch SDPA |
|---|---:|
| `toy-prefill-d64` | 0.953x |
| `toy-prefill-d128` | 1.414x |
| `dense-qwen-shaped-proxy-prefill-512` | 1.253x |
| `dense-qwen-shaped-proxy-prefill-2048` | 0.579x |

This means the baseline Triton kernel can outperform PyTorch SDPA for some short or medium prefill shapes, but it still trails badly on the longer 2048-token prefill workload. The result is therefore not "Triton is faster" or "Triton is slower." The better statement is:

```text
The relative performance depends strongly on workload shape and kernel configuration.
```

### 3. The long prefill workload remains the largest gap

The 2048-token dense Qwen-shaped proxy is still much slower than PyTorch SDPA:

```text
PyTorch SDPA: 0.2970 ms
Best Triton:  0.5130 ms
Speedup:      0.579x
```

This suggests that my simple forward-only Triton kernel is not yet efficient for long prefill, where a production attention kernel likely benefits from more mature scheduling, memory reuse, and backend-specific optimizations.

### 4. Decode and GQA are still outside the current kernel scope

Two workload rows were skipped intentionally:

| Workload | Reason |
|---|---|
| `dense-qwen-shaped-proxy-decode-2048` | current Triton kernel supports self-attention only, so `q_len` must equal `kv_len` |
| `qwen2.5-7b-prefill-512-gqa` | current Triton kernel does not support GQA/MQA yet |

The corrected main workload table also contains actual Qwen2.5-1.5B reference rows with `12 Q heads / 2 KV heads`. They are explicitly labeled `unsupported` and were not part of the Week 3 measurements.

This limitation is important for the next research step. Real LLM serving depends heavily on decode and GQA. The current experiment is only about dense prefill self-attention.

## What This Means For The Research Direction

This sweep gives a concrete, small-scale version of the paper-reading idea:

```text
same attention math
same GPU
same dtype
different workload shape
different best Triton configuration
```

A possible research direction is therefore:

```text
workload-aware configuration selection for Triton fused attention
```

The current result does not prove that this is a full paper topic yet. It only shows that:

- tile choices matter;
- launch choices such as `num_warps` and `num_stages` matter;
- different workloads can prefer different configurations;
- tuning can change whether Triton trails or beats PyTorch SDPA for short and medium prefill;
- a simple baseline Triton kernel is still not enough to match PyTorch SDPA for long prefill;
- more realistic attention features, especially decode and GQA, are needed before making stronger claims.

## Next Steps

The most useful next steps are:

1. Start the CURO proposal outline now using this sweep only as preliminary evidence.
2. Expand to 12–20 supported dense prefill workloads with FP16 and BF16.
3. Expand to approximately 32–64 valid configurations and measure compilation and benchmarking cost separately.
4. Split workloads into development and held-out sets.
5. Compare fixed, nearest-workload, budgeted-random, simple-rule/decision-tree, and exhaustive-oracle baselines before evaluating a learned top-k selector.

GQA and decode are later extensions rather than the next implementation target.

The next hypothesis to test could be:

```text
A lightweight selector can approach exhaustive-tuning performance on unseen
attention workloads while testing far fewer configurations.
```

## Current Interpretation In One Paragraph

This experiment shows that my Triton fused-attention baseline is correct and that configuration choice materially changes the conclusion. With a fixed configuration, the Triton kernel trailed PyTorch SDPA. With a small expanded sweep over tile and launch settings, Triton became faster than PyTorch SDPA on two short or medium prefill workloads, while still trailing on long prefill. The fastest configuration varied across workload shapes, which gives a concrete reason to study workload-aware configuration selection before moving to a larger serving-level evaluation.
