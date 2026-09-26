# Head Dimension Results Report

This report summarizes the controlled head-dimension sweep after Qingchen's feedback.

## Goal

The goal was to move from:

```text
latency changes when head_dim changes
```

to:

```text
does latency change mainly because there is more work, or because kernel efficiency drops?
```

## Environment

```text
GPU: NVIDIA GeForce RTX 3090
PyTorch: 2.14.0+cu130
PyTorch CUDA: 13.0
seq_len: 4096
batch size: 1
heads: 16
dtype: float16
warmup: 30
trials: 200
```

I also checked lower-level profiler availability:

```text
nsys: not available on PATH
ncu: not available on PATH
```

## Main Result

The controlled run does not support the earlier worry that the head_dim 64 to 128 transition causes a major loss in FlashAttention efficiency.

For non-causal FP16 forced FlashAttention:

| head_dim | median latency | estimated FLOPs | median TFLOP/s |
|---:|---:|---:|---:|
| 32 | 0.635 ms | 34.36 GFLOPs | 54.12 |
| 64 | 1.150 ms | 68.72 GFLOPs | 59.76 |
| 128 | 2.128 ms | 137.44 GFLOPs | 64.59 |
| 256 | 4.177 ms | 274.88 GFLOPs | 65.81 |

Latency increases as head dimension grows, but the normalized throughput does not drop. It actually improves slightly from head_dim 32 to 256.

This suggests that the latency growth is mostly explained by increased arithmetic work, not by a clear kernel-efficiency collapse.

## Head Dimension 64 To 128

The key transition was head_dim 64 to 128.

For forced FlashAttention, non-causal:

```text
head_dim 64 latency: 1.150 ms
head_dim 128 latency: 2.128 ms
latency ratio: about 1.85x
estimated FLOP ratio: 2.00x
median TFLOP/s: 59.76 -> 64.59
```

This is not evidence of lower kernel efficiency. If anything, the kernel gets slightly better normalized throughput at head_dim 128 in this controlled run.

## Causal Attention

For causal FP16 forced FlashAttention:

| head_dim | median latency | estimated FLOPs | median TFLOP/s |
|---:|---:|---:|---:|
| 32 | 0.399 ms | 17.18 GFLOPs | 43.09 |
| 64 | 0.693 ms | 34.37 GFLOPs | 49.58 |
| 128 | 1.185 ms | 68.74 GFLOPs | 57.99 |
| 256 | 2.383 ms | 137.47 GFLOPs | 57.69 |

Causal attention uses roughly half as many attention pairs as non-causal attention, so its latency is lower. Its estimated TFLOP/s is also lower than non-causal in smaller head dimensions, but it rises with head_dim and stabilizes around head_dim 128 to 256.

## SDPA Auto vs Flash Forced

`sdpa_auto` and `flash_forced` remain very close in this controlled sweep.

The largest visible gap is still small compared with the standard-vs-FlashAttention gap from the earlier CUDA baseline. This supports the current conclusion:

```text
PyTorch SDPA auto is not the main performance issue for these tested shapes.
```

## Interpretation

The main interpretation is now:

```text
For seq_len 4096 on RTX 3090, increasing head_dim increases latency mostly because the attention operation has more work. The controlled normalized metrics do not show a major loss of FlashAttention kernel efficiency at head_dim 128.
```

This changes the next research question. Instead of asking whether head_dim 128 is a failure point, I should ask:

```text
How do tiling, arithmetic intensity, and masking affect the normalized performance of attention kernels?
```

That question is a better bridge into Triton.

## Next Step

The next step is a small Triton experiment, but it should be scoped carefully.

I should not try to reimplement full FlashAttention immediately. A better first Triton step is:

```text
write a small tiled matmul-style kernel that resembles the QK^T work,
measure it across head_dim 32, 64, 128, and 256,
and compare normalized throughput trends with PyTorch SDPA/FlashAttention.
```

This would help me understand whether the observed scaling is mostly a tiling/arithmetic-intensity story before attempting anything closer to full attention.

Raw outputs:

```text
results/cuda_head_dim_controlled_seq4096_fp16.csv
results/cuda_head_dim_controlled_seq4096_fp16_metadata.json
```
