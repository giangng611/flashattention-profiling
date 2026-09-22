# Boundary Sweep Results Report

This report summarizes the first boundary sweep across sequence length, head dimension, dtype, and causal mode.

## Sweep Setup

Environment:

```text
GPU: NVIDIA GeForce RTX 3090
PyTorch: 2.14.0+cu130
PyTorch CUDA: 13.0
dtype sweep: FP16, BF16
batch size: 1
heads: 16
sequence lengths: 128, 512, 2048, 4096
head dimensions: 32, 64, 128, 256
causal values: false, true
warmup: 5
trials: 20
```

Methods:

- `sdpa_auto`
- `flash_forced`

I excluded explicit standard attention from this sweep because the CUDA baseline already showed its memory-heavy behavior, and the goal here was to look for backend-selection boundaries.

Raw outputs:

```text
results/cuda_boundary_sweep.csv
results/cuda_boundary_sweep_metadata.json
```

## Main Result

There were no support failures in this sweep.

```text
64 configurations x 2 methods = 128 successful rows
```

Both `sdpa_auto` and `flash_forced` succeeded for every tested configuration.

This suggests that, on this RTX 3090 with this PyTorch build, FlashAttention support is broad across:

- FP16 and BF16
- causal and non-causal attention
- head dimensions 32, 64, 128, and 256
- sequence lengths up to 4096

## Main Boundary Found

The most interesting performance boundary appears at:

```text
seq_len = 4096
head_dim = 256
dtype = float16
causal = true
```

Median latency:

| Method | Median latency |
|---|---:|
| `sdpa_auto` | 6.754 ms |
| `flash_forced` | 4.738 ms |

`sdpa_auto` was about:

```text
42.6% slower than flash_forced
```

This is the clearest candidate configuration for deeper profiling.

The same row also showed high variability for `sdpa_auto`:

```text
mean: 5.965 ms
median: 6.754 ms
min: 4.574 ms
max: 6.778 ms
std: 1.008 ms
```

This suggests that the result should be repeated before making a strong claim, but it is still the most interesting boundary candidate found so far.

## Other Observations

Most configurations showed very small differences between `sdpa_auto` and `flash_forced`.

Only one case had `sdpa_auto` more than 5% slower than `flash_forced`:

```text
seq_len=4096, head_dim=256, dtype=float16, causal=true
```

Only one case had `flash_forced` more than 5% slower than `sdpa_auto`:

```text
seq_len=2048, head_dim=256, dtype=bfloat16, causal=false
```

In that case:

| Method | Median latency |
|---|---:|
| `sdpa_auto` | 1.113 ms |
| `flash_forced` | 1.178 ms |

The gap is only about 5.5%, so it is less compelling than the 4096/head_dim 256/causal FP16 case.

## Head Dimension Effect

At sequence length 4096, increasing head dimension creates large latency changes.

For non-causal FP16:

| Head dimension | SDPA auto median latency | Flash forced median latency |
|---:|---:|---:|
| 32 | 0.561 ms | 0.561 ms |
| 64 | 1.095 ms | 1.089 ms |
| 128 | 6.787 ms | 6.654 ms |
| 256 | 13.407 ms | 13.289 ms |

The jump from head dimension 64 to 128 is especially large. This may indicate a different kernel regime, lower hardware efficiency, or a less favorable tiling/memory behavior. This needs profiling before I can explain it.

## Causal Effect

Causal attention was often faster than non-causal attention at larger sequence lengths and larger head dimensions.

For sequence length 4096 and head dimension 128:

| dtype | method | non-causal median | causal median |
|---|---|---:|---:|
| FP16 | `sdpa_auto` | 6.787 ms | 1.128 ms |
| FP16 | `flash_forced` | 6.654 ms | 1.149 ms |
| BF16 | `sdpa_auto` | 6.817 ms | 1.126 ms |
| BF16 | `flash_forced` | 6.692 ms | 1.135 ms |

This is plausible because causal attention can exploit triangular structure, but I should verify the kernel behavior with profiler output.

## Interpretation

The boundary sweep changes my current view:

```text
PyTorch SDPA auto is broadly reliable for these tested shapes, but there may be a performance anomaly at long sequence length, large head dimension, FP16, and causal attention.
```

The strongest candidate for deeper analysis is:

```text
seq_len=4096, head_dim=256, dtype=float16, causal=true
```

The next question is whether `sdpa_auto` and `flash_forced` are truly using the same kernel in this case. If they are using different kernels or different launch/configuration paths, this is a real backend-selection boundary. If they use the same kernel, the gap may be measurement instability or runtime overhead.

## Revised Hypothesis

```text
For most tested attention shapes on RTX 3090, PyTorch SDPA auto and forced FlashAttention have nearly identical performance, suggesting that automatic backend selection is generally reliable. However, large-head causal configurations may expose a performance instability or dispatch/configuration difference that deserves targeted profiling.
```

## Next Step

Profile the strongest boundary candidate:

```bash
python src/profile_attention.py --device cuda --seq-lengths 4096 --head-dim 256 --dtype float16 --causal --methods sdpa_auto flash_forced --warmup 5 --trials 10 --output-dir profiling/pytorch_boundary_seq4096_hd256_causal_fp16
```

If needed, repeat the benchmark with more trials:

```bash
python src/sweep_attention_boundaries.py --device cuda --seq-lengths 4096 --head-dims 256 --dtypes float16 --causal-values true --warmup 10 --trials 100 --output results/cuda_boundary_repeat_seq4096_hd256_causal_fp16.csv --metadata-output results/cuda_boundary_repeat_seq4096_hd256_causal_fp16_metadata.json
```
