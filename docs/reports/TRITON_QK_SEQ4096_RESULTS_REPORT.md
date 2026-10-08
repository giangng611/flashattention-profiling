# Triton QK Sequence Length 4096 Results Report

This report records the focused QK-only Triton runs at sequence length 4096.

## Setup

```text
GPU: NVIDIA GeForce RTX 3090
PyTorch: 2.14.0+cu130
PyTorch CUDA: 13.0
Triton: 3.8.0
seq_len: 4096
head_dim: 32, 64, 128
warmup: 10
trials: 30
```

The tile choices came from the previous seq_len 2048 tile sweep:

| head_dim | Triton tile |
|---:|---:|
| 32 | 16x16 |
| 64 | 32x32 |
| 128 | 16x16 |

This is still a QK^T microbenchmark, not full FlashAttention.

## Main Result

At sequence length 4096, PyTorch matmul is still substantially faster than my first Triton QK kernel.

| head_dim | PyTorch median | PyTorch TFLOP/s | Triton median | Triton TFLOP/s |
|---:|---:|---:|---:|---:|
| 32 | 0.0645 ms | 16.64 | 0.1352 ms | 7.94 |
| 64 | 0.0635 ms | 33.83 | 0.1126 ms | 19.07 |
| 128 | 0.0932 ms | 46.09 | 0.2417 ms | 17.77 |

The Triton kernel is correct enough for this FP16 smoke benchmark, with max absolute difference around 0.014 to 0.016.

## Scaling Pattern

PyTorch matmul gets much higher normalized throughput as head_dim increases:

```text
head_dim 32: 16.64 TFLOP/s
head_dim 64: 33.83 TFLOP/s
head_dim 128: 46.09 TFLOP/s
```

The custom Triton kernel also improves from head_dim 32 to 64:

```text
head_dim 32: 7.94 TFLOP/s
head_dim 64: 19.07 TFLOP/s
head_dim 128: 17.77 TFLOP/s
```

But it does not keep scaling at head_dim 128. This suggests that the simple kernel is hitting a tuning or resource limit that the PyTorch matmul path handles much better.

## Interpretation

This is useful evidence, but I should be careful about the claim.

Supported:

- Triton is working on the CUDA server.
- A small QK^T Triton kernel can reproduce the broad idea that head_dim affects normalized throughput.
- Tile choice matters.
- PyTorch matmul is far more optimized than this first Triton kernel.

Not supported:

- This Triton kernel does not explain full FlashAttention behavior.
- This kernel is not yet tuned enough to compare directly against PyTorch SDPA or FlashAttention.

The current interpretation is:

```text
The Triton experiment is useful as a controlled learning tool. It shows that tiled QK performance depends strongly on tile shape and head_dim, but the first implementation is still too simple to serve as a competitive or explanatory FlashAttention model.
```

## What I Can Tell Qingchen

This is enough for a short progress update:

```text
I checked nsys/ncu availability but they are not exposed on the server. In parallel, I started a small Triton QK^T microbenchmark. The kernel runs correctly, but it is slower than PyTorch matmul. Tile size affects performance significantly, and the first kernel shows that head_dim changes normalized throughput, but it needs more tuning before it can explain FlashAttention-level behavior.
```

## Next Step

The next technical step is to improve the Triton kernel rather than running many more shapes. Possible changes:

- try rectangular tiles such as 16x32, 32x16, 32x64, and 64x32
- add `num_warps` as a tunable parameter
- add `num_stages` as a tunable parameter
- avoid over-interpreting results until tile and launch parameters are better controlled

Raw outputs:

```text
results/triton_qk_seq4096_hd32_tile16.csv
results/triton_qk_seq4096_hd64_tile32.csv
results/triton_qk_seq4096_hd128_tile16.csv
```
