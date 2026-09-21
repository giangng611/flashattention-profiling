# Next Report Draft

This is a template for the next update to Qingchen after running the CUDA baseline.

## Summary

I extended the benchmark from the local Mac/MPS practice run to the CUDA machine. The CUDA baseline compares:

- explicit standard attention
- PyTorch SDPA automatic backend selection
- PyTorch SDPA with FlashAttention forced where supported

The goal is to understand how latency and memory behavior change with sequence length, and whether the fastest attention path changes across workload shapes.

## Environment

Fill in after running on the server:

```text
Server:
GPU:
Driver:
Python:
PyTorch:
PyTorch CUDA:
CUDA available:
dtype:
batch size:
heads:
head dimension:
sequence lengths:
warmup:
trials:
```

Environment command:

```bash
.venv/bin/python src/check_environment.py
```

Benchmark command:

```bash
.venv/bin/python src/benchmark_attention.py --device cuda --seq-lengths 128 256 512 1024 2048 4096 --warmup 20 --trials 50 --output results/cuda_attention_benchmark.csv --metadata-output results/cuda_environment_metadata.json
```

Plot command:

```bash
.venv/bin/python src/plot_results.py --input results/cuda_attention_benchmark.csv --output-dir plots/cuda
```

## Results

Add the CUDA table here after running:

| Sequence length | Standard median latency | SDPA auto median latency | Flash forced median latency | Best backend | Notes |
|---:|---:|---:|---:|---|---|
| 128 | | | | | |
| 256 | | | | | |
| 512 | | | | | |
| 1024 | | | | | |
| 2048 | | | | | |
| 4096 | | | | | |

Attach or link:

- `results/cuda_attention_benchmark.csv`
- `results/cuda_environment_metadata.json`
- `plots/cuda/latency_vs_sequence_length.png`
- `plots/cuda/peak_memory_vs_sequence_length.png`
- `plots/cuda/speedup_vs_sequence_length.png`

## Most Significant Pattern Or Limitation

Fill this in after inspecting the CUDA data.

Possible pattern types:

- FlashAttention is consistently faster as sequence length increases.
- SDPA auto and forced FlashAttention differ, suggesting backend selection matters.
- A small sequence length has little or no benefit from FlashAttention, possibly due to overhead.
- One backend fails for a configuration, which limits portability.
- Peak memory grows sharply for explicit standard attention.

Measured observation:

```text
TODO
```

Possible explanation:

```text
TODO
```

Evidence still missing:

```text
TODO
```

## Preliminary Research Problem

Draft:

```text
How can an attention system choose an efficient implementation across workload shapes and GPU characteristics without relying on exhaustive tuning?
```

## Testable Hypothesis

Draft:

```text
A lightweight workload-aware selection rule based on sequence length, head dimension, dtype, and GPU properties can choose between explicit attention, SDPA auto, FlashAttention, and later Triton kernels with performance close to exhaustive selection, while requiring much less tuning time.
```

This is only a preliminary hypothesis. It should be revised after the CUDA measurements.

## Next Experiment

After the CUDA baseline:

1. Profile representative sequence lengths, likely 512 and 4096.
2. Record kernel duration, DRAM traffic, memory throughput, compute throughput, occupancy, and kernel launches if available.
3. Check whether profiler data supports the latency and memory explanation.
4. Begin Triton tutorials after the CUDA baseline is stable.
