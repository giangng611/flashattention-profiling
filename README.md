# flashattention-profiling

This repository is my first hands-on experiment for understanding the systems idea behind FlashAttention and FlexInfer: actual performance is not determined by FLOP count alone. Memory movement, hardware utilization, kernel behavior, and runtime decisions can dominate the execution time.

The current goal is intentionally modest. I am not trying to propose a new research problem yet. I am first building a small, reproducible benchmark so that I can collect measurements and discuss the results with Qingchen.

## Motivation

The two papers I started from operate at different levels of the ML systems stack.

FlashAttention focuses on attention inside the GPU memory hierarchy. Its main point is that standard attention materializes large intermediate tensors, especially the attention score matrix, and this can create expensive memory traffic between GPU HBM and on-chip memory. The paper argues that an IO-aware implementation can reduce memory movement and improve wall-clock time.

FlexInfer focuses on a higher-level CPU-GPU inference setting. Its question is not only which device has more compute, but whether moving data between CPU memory and GPU memory is worth the communication cost.

The common theme I want to study is:

```text
code -> kernels -> memory behavior -> hardware utilization -> latency
```

For the first step, I am narrowing this down to attention only.

## Current Hardware Constraint

My local machine is a Mac. It does not have an NVIDIA GPU, so it cannot run CUDA FlashAttention or Nsight CUDA profiling locally.

Because of that, this repo separates the work into two stages:

1. Mac/MPS practice benchmark
2. CUDA FlashAttention benchmark when I have access to an NVIDIA GPU machine

The Mac benchmark is not meant to reproduce the FlashAttention paper directly. It is a way to validate the benchmark pipeline, practice measuring latency, generate CSV results, make plots, and check that the two attention implementations have matching semantics.

The CUDA benchmark is the version that should be used for the actual FlashAttention comparison and deeper profiling.

## Experiment Design

The first experiment varies only sequence length.

Fixed configuration:

- batch size: 1
- number of heads: 16
- head dimension: 64
- dtype: FP16 where supported

Sequence lengths:

- 128
- 256
- 512
- 1024
- 2048
- 4096

The benchmark records:

- mean latency after warmup
- median latency after warmup
- min and max latency
- standard deviation
- recorded allocated memory
- speedup over the explicit standard attention implementation
- environment metadata

## Implementations Compared

The `standard` implementation is a straightforward scaled dot-product attention implementation:

```text
scores = QK^T / sqrt(d)
probabilities = softmax(scores)
output = probabilities V
```

This implementation materializes the full attention score tensor:

```text
batch x heads x sequence length x sequence length
```

That tensor grows quadratically with sequence length, so it is useful for observing why memory behavior can matter.

On CUDA, the comparison target is:

```text
torch.nn.functional.scaled_dot_product_attention
```

with the FlashAttention backend forced when available.

On Mac MPS, the comparison target is also PyTorch SDPA, but with the backend available on Apple MPS. I will label this result as `sdpa`, not `flash`, because it is not CUDA FlashAttention.

Both paths use the same `q`, `k`, and `v` tensors and the same non-causal attention semantics.

## Current Local Environment

The local development environment has been created at:

```text
.venv/bin/python
```

Current local status:

- Python 3.11
- PyTorch installed
- pandas installed
- matplotlib installed
- Apple MPS available
- CUDA not available
- `nvidia-smi` not available

This means I can develop and run the Mac/MPS practice benchmark locally. I still need an NVIDIA GPU machine for the CUDA FlashAttention benchmark and Nsight profiling.

## How To Run On My Mac

Environment check:

```bash
.venv/bin/python src/check_environment.py
```

Mac/MPS practice benchmark:

```bash
.venv/bin/python src/benchmark_attention.py --device mps --seq-lengths 128 256 512 1024 --warmup 10 --trials 30 --output results/mps_attention_benchmark.csv --metadata-output results/mps_environment_metadata.json
```

Generate plots:

```bash
.venv/bin/python src/plot_results.py --input results/mps_attention_benchmark.csv --output-dir plots/mps
```

These results should be interpreted as a local workflow test, not as evidence about CUDA FlashAttention performance.

## How To Run Later On NVIDIA GPU

On a CUDA-capable machine with CUDA-enabled PyTorch installed:

```bash
python3 src/check_environment.py
python3 src/benchmark_attention.py --device cuda
python3 src/plot_results.py
```

A smaller smoke test:

```bash
python3 src/benchmark_attention.py --device cuda --seq-lengths 128 256 --warmup 5 --trials 10
```

Expected outputs:

```text
results/attention_benchmark.csv
results/environment_metadata.json
plots/latency_vs_sequence_length.png
plots/peak_memory_vs_sequence_length.png
plots/speedup_vs_sequence_length.png
```

## Methodological Notes

For CUDA, timing uses CUDA events and explicit synchronization. This is more appropriate for GPU kernels than CPU wall-clock timing because CUDA kernel launches are asynchronous.

For MPS, timing uses wall-clock time with explicit MPS synchronization before and after each trial. This is less detailed than CUDA events, but it is sufficient for validating the workflow on Mac.

The benchmark includes a correctness check before timing. The purpose is to avoid comparing two implementations that silently compute different attention outputs.

I am currently using latency and allocated memory as the first measurements. For the CUDA stage, I plan to add profiler evidence for representative sequence lengths, likely 512 and 4096.

## What I Can Discuss After Initial Results

After collecting the first benchmark results, I want to answer:

- How does latency change as sequence length increases?
- Does the gap between standard attention and SDPA/FlashAttention grow with sequence length?
- Does memory usage show the expected quadratic pressure in the standard implementation?
- Which observations are directly supported by measurements?
- Which explanations are still hypotheses?

For the CUDA stage, the next questions are:

- How much time is spent in attention kernels?
- How much DRAM traffic is observed?
- Is performance more limited by memory bandwidth or compute throughput?
- Are there crossover points where the optimized backend gives less benefit?

## Repository Layout

```text
flashattention-profiling/
├── README.md
├── requirements.txt
├── src/
│   ├── check_environment.py
│   ├── attention.py
│   ├── benchmark_attention.py
│   └── plot_results.py
├── results/
├── plots/
└── profiling/
    └── README.md
```

## Current Status

The repository is ready for local Mac/MPS testing and later CUDA testing. The next concrete step is to run the MPS benchmark locally, inspect the CSV and plots, and then repeat the experiment on an NVIDIA GPU machine when one is available.
