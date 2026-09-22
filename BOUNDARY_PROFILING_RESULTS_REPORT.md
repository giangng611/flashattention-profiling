# Boundary Profiling Results Report

This note follows up on the boundary sweep candidate:

```text
seq_len = 4096
head_dim = 256
dtype = float16
causal = true
```

The boundary sweep showed a large median latency gap for this case:

| Method | Sweep median latency |
|---|---:|
| `sdpa_auto` | 6.754 ms |
| `flash_forced` | 4.738 ms |

At first, this looked like a possible backend-selection boundary. The goal of this profiling pass was to check whether `sdpa_auto` and `flash_forced` were actually using different kernels.

## Profiling Setup

Command:

```bash
python src/profile_attention.py --device cuda --seq-lengths 4096 --head-dim 256 --dtype float16 --causal --methods sdpa_auto flash_forced --warmup 5 --trials 10 --output-dir profiling/pytorch_boundary_seq4096_hd256_causal_fp16
```

Environment:

```text
GPU: NVIDIA GeForce RTX 3090
PyTorch: 2.14.0+cu130
PyTorch CUDA: 13.0
batch size: 1
heads: 16
head dimension: 256
dtype: float16
causal: true
warmup: 5
trials: 10
```

## Main Finding

Both `sdpa_auto` and `flash_forced` used the same PyTorch FlashAttention path:

```text
aten::scaled_dot_product_attention
aten::_scaled_dot_product_flash_attention
aten::_flash_attention_forward
pytorch_flash::flash_fwd_kernel
```

The CUDA kernel time was nearly identical:

| Method | Flash kernel CUDA time across 10 trials | Per-trial average |
|---|---:|---:|
| `sdpa_auto` | 23.989 ms | 2.399 ms |
| `flash_forced` | 23.966 ms | 2.397 ms |

This is not evidence of a different backend being selected.

## What This Changes

The earlier sweep result should be interpreted more carefully.

The sweep found a real measurement anomaly:

```text
sdpa_auto looked much slower than flash_forced for the 4096/head_dim 256/FP16/causal case.
```

But the profiler shows:

```text
both methods use the same fused FlashAttention kernel and have essentially the same kernel time.
```

So I should not frame this case as a confirmed backend-selection boundary. A better interpretation is:

```text
The boundary sweep found a candidate anomaly, but PyTorch Profiler suggests the anomaly is probably caused by timing variability, run order, profiler overhead differences, or runtime effects outside the FlashAttention kernel itself.
```

## Supported Conclusion

The current supported result is:

```text
For the tested RTX 3090 configurations, PyTorch SDPA auto generally selects the same FlashAttention path as forced FlashAttention. Even the strongest boundary candidate from the sweep used the same kernel when profiled.
```

I also repeated the strongest anomaly with 30 warmup runs and 200 timed trials:

| Method | Repeat mean latency | Repeat median latency | Min latency | Max latency | Std latency | Peak memory |
|---|---:|---:|---:|---:|---:|---:|
| `sdpa_auto` | 4.638 ms | 4.678 ms | 2.574 ms | 5.052 ms | 0.321 ms | 128.3 MiB |
| `flash_forced` | 4.694 ms | 4.668 ms | 2.579 ms | 5.094 ms | 0.272 ms | 128.3 MiB |

The repeated median gap is about 0.2%, with identical peak memory. This confirms that the earlier 42.6% gap was not stable.

This strengthens the earlier conclusion from the head_dim 64 profiler run. The main difference I have solid evidence for is still:

```text
explicit standard attention vs fused FlashAttention
```

not:

```text
SDPA auto vs forced FlashAttention
```

## Remaining Interesting Pattern

The profiler weakens the auto-vs-forced hypothesis, but it does not make the sweep useless.

Two patterns are still worth studying:

1. Large head dimensions create sharp latency jumps.

   At sequence length 4096 and non-causal FP16, the median latency jumped strongly from head dimension 64 to 128:

   | Head dimension | SDPA auto median latency | Flash forced median latency |
   |---:|---:|---:|
   | 32 | 0.561 ms | 0.561 ms |
   | 64 | 1.095 ms | 1.089 ms |
   | 128 | 6.787 ms | 6.654 ms |
   | 256 | 13.407 ms | 13.289 ms |

2. Causal attention was often much faster than non-causal attention at large sequence lengths and head dimensions.

These are now better research leads than the auto-vs-forced gap.

## Revised Hypothesis

```text
On this RTX 3090 and PyTorch 2.14 CUDA build, SDPA auto reliably dispatches to FlashAttention for the tested FP16/BF16 attention shapes. The more interesting performance question is how FlashAttention kernel behavior changes across head dimension and causal masking, especially around the head_dim 64 to 128 transition.
```

## Next Step

The strongest anomaly has now been repeated and does not hold up. The next useful direction is to profile the head dimension transition:

```bash
python src/profile_attention.py --device cuda --seq-lengths 4096 --head-dim 64 --dtype float16 --methods sdpa_auto flash_forced --warmup 5 --trials 10 --output-dir profiling/pytorch_hd64_seq4096_noncausal_fp16
python src/profile_attention.py --device cuda --seq-lengths 4096 --head-dim 128 --dtype float16 --methods sdpa_auto flash_forced --warmup 5 --trials 10 --output-dir profiling/pytorch_hd128_seq4096_noncausal_fp16
```

I should compare the profiler tables for these two cases to see whether the latency jump from head dimension 64 to 128 corresponds to different FlashAttention kernel behavior, different memory allocation, or simply more work per token.

Raw repeat outputs:

```text
results/cuda_boundary_repeat_seq4096_hd256_causal_fp16.csv
results/cuda_boundary_repeat_seq4096_hd256_causal_fp16_metadata.json
```
