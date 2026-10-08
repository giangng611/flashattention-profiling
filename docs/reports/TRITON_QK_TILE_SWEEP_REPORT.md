# Triton QK Tile Sweep Report

This report records the first Triton QK tile-size sweep.

## Setup

```text
GPU: NVIDIA GeForce RTX 3090
PyTorch: 2.14.0+cu130
PyTorch CUDA: 13.0
Triton: 3.8.0
seq_len: 2048
head_dim: 32, 64, 128
tile sizes: 16x16, 32x32, 64x64
warmup: 10
trials: 30
```

This is still a QK^T microbenchmark, not full FlashAttention.

## Main Result

The Triton kernel works, but the first tiled implementation is still clearly below PyTorch matmul performance.

Best median TFLOP/s by head dimension:

| head_dim | PyTorch matmul | Best Triton tile | Best Triton TFLOP/s |
|---:|---:|---:|---:|
| 32 | 9.04 | 16x16 | 4.95 |
| 64 | 14.57 | 32x32 | 10.92 |
| 128 | 18.08 | 16x16 | 9.36 |

The best Triton tile changes with head dimension. This is useful because it shows that the tile configuration matters, but it also means I should not over-interpret a single tile-size result.

## Tile-Size Behavior

For head_dim 32:

```text
16x16: 4.95 TFLOP/s
32x32: 1.30 TFLOP/s
64x64: 3.36 TFLOP/s
```

For head_dim 64:

```text
16x16: 9.04 TFLOP/s
32x32: 10.92 TFLOP/s
64x64: 4.10 TFLOP/s
```

For head_dim 128:

```text
16x16: 9.36 TFLOP/s
32x32: 8.07 TFLOP/s
64x64: 8.13 TFLOP/s
```

The 64x64 tile is not consistently better, even though it launches fewer programs. That suggests the simple kernel may be running into register pressure, less favorable scheduling, or other resource limits. I need lower-level metrics to say which one, but Nsight is not currently available on the server.

## Interpretation

This result is a useful Triton learning step, not a final performance result.

The main supported points are:

- A small custom Triton QK kernel runs correctly on the CUDA server.
- Tile size materially affects performance.
- PyTorch matmul remains much faster, which is expected because it uses a highly optimized library path.
- The current Triton kernel is too simple to explain FlashAttention performance by itself.

The current conclusion is:

```text
Triton is useful here as a controlled microscope, but the first QK kernel needs more tuning before it can be used as a meaningful performance comparison against PyTorch.
```

## Next Step

Before running the full seq_len 4096 benchmark, I should run one focused 4096 test using the best tile seen so far for each head_dim:

```bash
python src/benchmark_triton_qk.py --seq-len 4096 --head-dims 32 --block-sizes 16 --warmup 10 --trials 30 --output results/triton_qk_seq4096_hd32_tile16.csv --metadata-output results/triton_qk_seq4096_hd32_tile16_metadata.json
python src/benchmark_triton_qk.py --seq-len 4096 --head-dims 64 --block-sizes 32 --warmup 10 --trials 30 --output results/triton_qk_seq4096_hd64_tile32.csv --metadata-output results/triton_qk_seq4096_hd64_tile32_metadata.json
python src/benchmark_triton_qk.py --seq-len 4096 --head-dims 128 --block-sizes 16 --warmup 10 --trials 30 --output results/triton_qk_seq4096_hd128_tile16.csv --metadata-output results/triton_qk_seq4096_hd128_tile16_metadata.json
```

If those runs are stable, I can then compare the QK-only scaling trend against the PyTorch SDPA/FlashAttention head-dimension results.

Raw outputs:

```text
results/triton_qk_tile_sweep_seq2048.csv
results/triton_qk_tile_sweep_seq2048_metadata.json
```
