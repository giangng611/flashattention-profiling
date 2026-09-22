# CUDA Results Report Draft

This report summarizes the first CUDA baseline run on the UGA CUDA server.

## Summary

I extended the local Mac/MPS benchmark to the CUDA server and ran the first CUDA baseline. The benchmark compares three attention paths:

- `standard`: explicit scaled dot-product attention using matmul, softmax, matmul
- `sdpa_auto`: PyTorch `scaled_dot_product_attention` with automatic backend selection
- `flash_forced`: PyTorch `scaled_dot_product_attention` with FlashAttention forced

All three paths completed successfully for the tested sequence lengths.

The main pattern is that both SDPA paths are substantially faster and use much less peak GPU memory than the explicit standard attention implementation, especially at long sequence lengths. At sequence length 4096, standard attention used about 1.05 GiB peak allocated memory, while the SDPA and forced FlashAttention paths used about 56.4 MiB.

## Environment

Server:

```text
cuda3.cs.uga.edu
```

Hardware and software:

```text
GPU: NVIDIA GeForce RTX 3090
GPU memory: 24,576 MiB
Driver: 580.119.02
System CUDA reported by nvidia-smi: 13.0
Python: 3.10.12
PyTorch: 2.14.0+cu130
PyTorch CUDA: 13.0
cuDNN: 92400
Compute capability: 8.6
```

Benchmark configuration:

```text
batch size: 1
heads: 16
head dimension: 64
dtype: float16
causal: false
warmup iterations: 20
timed trials: 50
seed: 0
sequence lengths: 128, 256, 512, 1024, 2048, 4096
```

Reproduction commands:

```bash
python src/check_environment.py
python src/benchmark_attention.py --device cuda --seq-lengths 128 256 512 1024 2048 4096 --warmup 20 --trials 50 --output results/cuda_attention_benchmark.csv --metadata-output results/cuda_environment_metadata.json
python src/plot_results.py --input results/cuda_attention_benchmark.csv --output-dir plots/cuda
```

Generated outputs:

```text
results/cuda_attention_benchmark.csv
results/cuda_environment_metadata.json
plots/cuda/latency_vs_sequence_length.png
plots/cuda/peak_memory_vs_sequence_length.png
plots/cuda/speedup_vs_sequence_length.png
```

## Latency Results

Median latency in milliseconds:

| Sequence length | Standard | SDPA auto | Flash forced | Best latency backend |
|---:|---:|---:|---:|---|
| 128 | 0.130 | 0.031 | 0.053 | SDPA auto |
| 256 | 0.127 | 0.032 | 0.053 | SDPA auto |
| 512 | 0.129 | 0.043 | 0.058 | SDPA auto |
| 1024 | 0.322 | 0.109 | 0.123 | SDPA auto |
| 2048 | 1.108 | 0.353 | 0.368 | SDPA auto |
| 4096 | 4.992 | 1.118 | 1.108 | Flash forced |

Speedup over standard attention:

| Sequence length | SDPA auto speedup | Flash forced speedup |
|---:|---:|---:|
| 128 | 4.23x | 2.44x |
| 256 | 3.94x | 2.38x |
| 512 | 3.00x | 2.23x |
| 1024 | 2.96x | 2.61x |
| 2048 | 3.14x | 3.01x |
| 4096 | 4.46x | 4.51x |

## Peak Memory Results

Peak allocated memory in MiB:

| Sequence length | Standard | SDPA auto | Flash forced |
|---:|---:|---:|---:|
| 128 | 10.38 | 9.63 | 9.63 |
| 256 | 14.88 | 11.14 | 11.14 |
| 512 | 29.62 | 14.16 | 14.16 |
| 1024 | 83.13 | 20.19 | 20.19 |
| 2048 | 286.13 | 32.25 | 32.25 |
| 4096 | 1076.12 | 56.38 | 56.38 |

## Most Significant Pattern

The most significant pattern is that the explicit standard attention implementation becomes much more memory intensive as sequence length increases, while the SDPA and forced FlashAttention paths keep memory usage much lower.

At sequence length 4096:

- `standard` median latency: 4.992 ms
- `sdpa_auto` median latency: 1.118 ms
- `flash_forced` median latency: 1.108 ms
- `standard` peak memory: about 1076 MiB
- `sdpa_auto` and `flash_forced` peak memory: about 56 MiB

This is consistent with the expected memory pressure of materializing the full attention score matrix in standard attention.

## Backend Selection Observation

For sequence lengths 128 through 2048, `sdpa_auto` is faster than `flash_forced`. At sequence length 4096, `flash_forced` becomes slightly faster than `sdpa_auto`.

The follow-up PyTorch Profiler pass shows that this should not be over-interpreted as a clear backend crossover. At sequence lengths 512 and 4096, `sdpa_auto` and `flash_forced` both use the PyTorch FlashAttention path:

```text
aten::_scaled_dot_product_flash_attention
pytorch_flash::flash_fwd_kernel
```

This means the strongest finding is the gap between explicit standard attention and fused FlashAttention, not a confirmed difference between two distinct optimized backends.

The small timing differences between `sdpa_auto` and `flash_forced` may come from measurement noise, dispatch/context overhead, or other small runtime effects.

## Supported Measurements

Measured facts:

- CUDA is available on the server.
- The benchmark ran successfully on an RTX 3090.
- All three methods passed correctness checks for all tested sequence lengths.
- SDPA auto and forced FlashAttention were faster than explicit standard attention for all tested sequence lengths.
- Standard attention used much more peak allocated memory as sequence length increased.
- The best backend changed from `sdpa_auto` at smaller sequence lengths to `flash_forced` at sequence length 4096.

## Hypotheses

Preliminary hypothesis:

```text
The best attention implementation depends on workload shape. A lightweight workload-aware selection policy based on sequence length, head dimension, dtype, and GPU properties can choose between SDPA auto, forced FlashAttention, and future Triton kernels with performance close to exhaustive selection, while avoiding expensive tuning.
```

Refined hypothesis after PyTorch profiling:

```text
For FP16 attention on an RTX 3090 with head dimension 64, PyTorch SDPA auto already selects FlashAttention for the tested representative sequence lengths. The next useful question is where automatic attention backend selection stops being reliable, such as different head dimensions, dtypes, causal settings, or longer sequence lengths.
```

This refined hypothesis should be tested with broader workload sweeps.

## Limitations

This first CUDA run only varies sequence length. It keeps batch size, number of heads, head dimension, dtype, and causal mode fixed.

The benchmark measures latency and peak allocated memory, but it does not yet measure:

- kernel duration by kernel type
- DRAM bytes read and written
- memory throughput
- compute throughput
- occupancy
- kernel launch counts

Therefore, explanations about memory traffic and backend behavior are still hypotheses.

## Next Experiment

The next step is to profile representative cases with PyTorch Profiler first, since Nsight Systems and Nsight Compute are not currently available in the server shell path:

- sequence length 512
- sequence length 4096

Command:

```bash
python src/profile_attention.py --device cuda --seq-lengths 512 4096 --warmup 5 --trials 10 --output-dir profiling/pytorch
```

The profiler results should help answer:

- Which kernels are launched by `sdpa_auto` and `flash_forced`?
- Does forced FlashAttention reduce DRAM traffic compared with standard attention?
- Is the 4096 result related to memory traffic, kernel selection, or launch overhead?
- Why is `sdpa_auto` faster than `flash_forced` at smaller sequence lengths?

See `PROFILING_RESULTS_REPORT.md` for the PyTorch Profiler interpretation.

After that, I can start exploring Triton through the official tutorial sequence.
