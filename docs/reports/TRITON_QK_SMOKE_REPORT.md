# Triton QK Smoke Report

This report records the first Triton microbenchmark smoke test on the CUDA server.

## Setup

```text
GPU: NVIDIA GeForce RTX 3090
PyTorch: 2.14.0+cu130
PyTorch CUDA: 13.0
Triton: 3.8.0
seq_len: 2048
head_dim: 32, 64, 128
warmup: 5
trials: 10
Triton tile: 16 x 16
```

Nsight status:

```text
nsys: not found in PATH, $HOME, /usr/local, or /opt
ncu: not found in PATH, $HOME, /usr/local, or /opt
```

## Smoke Result

The Triton kernel compiled and ran successfully.

Correctness differences were small for FP16 matmul:

| head_dim | max abs diff | mean abs diff |
|---:|---:|---:|
| 32 | 0.0106 | 0.00079 |
| 64 | 0.0156 | 0.00112 |
| 128 | 0.0156 | 0.00159 |

Performance:

| head_dim | PyTorch median | PyTorch TFLOP/s | Triton median | Triton TFLOP/s |
|---:|---:|---:|---:|---:|
| 32 | 0.0317 ms | 8.46 | 0.0558 ms | 4.81 |
| 64 | 0.0906 ms | 5.92 | 0.1136 ms | 4.72 |
| 128 | 0.0901 ms | 11.92 | 0.1475 ms | 7.28 |

## Interpretation

This is a successful first Triton smoke test, but not yet a meaningful performance comparison.

The Triton kernel uses a very small `16 x 16` output tile. That creates many small programs and likely high launch/scheduling overhead relative to useful work. PyTorch matmul is also using a highly optimized library path, so I should not expect this first Triton kernel to beat it.

The important result is:

```text
Triton works on the server, the QK-style kernel produces reasonable output, and the benchmark pipeline now records normalized metrics.
```

## Next Step

Before running the full sequence length 4096 benchmark, I should sweep tile sizes at sequence length 2048:

```bash
python src/benchmark_triton_qk.py --seq-len 2048 --head-dims 32 64 128 --block-sizes 16 32 64 --warmup 10 --trials 30 --output results/triton_qk_tile_sweep_seq2048.csv --metadata-output results/triton_qk_tile_sweep_seq2048_metadata.json
```

If `64 x 64` is too slow, fails to compile, or uses too many resources, I can reduce to:

```bash
python src/benchmark_triton_qk.py --seq-len 2048 --head-dims 32 64 128 --block-sizes 16 32 --warmup 10 --trials 30 --output results/triton_qk_tile_sweep_seq2048.csv --metadata-output results/triton_qk_tile_sweep_seq2048_metadata.json
```

After finding a reasonable tile size, I can run the sequence length 4096 benchmark.
