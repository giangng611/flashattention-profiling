# flashattention-profiling

Small reproduction and profiling project for understanding why FlashAttention can be faster than straightforward attention on real GPU hardware.

This repo is intentionally modest. The current goal is not to claim a new research contribution. The goal is to produce a technically defensible first artifact for discussion with Qingchen:

- environment and hardware description
- reproducible benchmark code
- CSV results
- 2-3 plots
- short observations connecting latency, memory use, and GPU behavior

## Research Context

The starting point is the systems principle from the selected papers and Qingchen's note:

> Theoretical FLOPs are not equivalent to actual execution time.

FlashAttention studies attention inside the GPU memory hierarchy. Its main idea is IO-awareness: reduce reads and writes between GPU high bandwidth memory and on-chip memory instead of only counting arithmetic operations.

FlexInfer studies a higher-level heterogeneous CPU-GPU inference problem. It asks when work or data should stay on CPU memory versus move through PCIe to GPU memory.

For this first project, we focus on the smaller and more direct experiment:

```text
standard attention vs PyTorch FlashAttention backend
```

We vary only sequence length at first, while keeping the other dimensions fixed.

## Initial Experiment

Default configuration:

- batch size: 1
- heads: 16
- head dimension: 64
- dtype: FP16
- sequence lengths: 128, 256, 512, 1024, 2048, 4096

Measurements:

- mean latency after warmup
- median latency after warmup
- peak GPU memory allocated
- speedup relative to standard attention
- environment metadata

The benchmark compares two implementations with matching semantics:

- `standard`: explicit scaled dot-product attention that materializes the attention score matrix
- `flash`: `torch.nn.functional.scaled_dot_product_attention` forced to use the FlashAttention backend when available

## Setup

Create an environment in PyCharm or a terminal, then install dependencies appropriate for the machine.

For an NVIDIA CUDA machine, follow the official PyTorch install selector for the correct command:

https://pytorch.org/get-started/locally/

Then install the plotting dependencies:

```bash
python3 -m pip install -r requirements.txt
```

The local development machine does not need a compatible NVIDIA GPU just to edit the repo. The benchmark itself requires CUDA.

## Check The Environment

Run:

```bash
python3 src/check_environment.py
```

This reports:

- Python version
- PyTorch version if installed
- CUDA availability
- CUDA version known to PyTorch
- GPU model
- compute capability
- GPU memory
- whether `nvidia-smi` is available

If PyTorch or CUDA is missing, this script explains that clearly instead of crashing.

## Run One Benchmark

On an NVIDIA GPU machine with a CUDA-enabled PyTorch install:

```bash
python3 src/benchmark_attention.py
```

Useful smaller smoke test:

```bash
python3 src/benchmark_attention.py --seq-lengths 128 256 --warmup 5 --trials 10
```

Output:

```text
results/attention_benchmark.csv
```

## Make Plots

After producing the CSV:

```bash
python3 src/plot_results.py
```

Output:

```text
plots/latency_vs_sequence_length.png
plots/peak_memory_vs_sequence_length.png
plots/speedup_vs_sequence_length.png
```

## Profiling

After the basic benchmark works, choose about two representative sequence lengths, such as 512 and 4096, for deeper profiling with Nsight Systems or Nsight Compute.

See [profiling/README.md](/Users/giangnguyendohoang/PycharmProjects/flashattention-profiling/profiling/README.md).

## Method Notes

GPU timing uses CUDA events and explicit synchronization. This avoids measuring only CPU launch overhead.

The standard attention implementation intentionally materializes the full attention score tensor with shape:

```text
batch x heads x sequence length x sequence length
```

That tensor grows quadratically with sequence length. This is the memory behavior the first experiment is meant to expose.

The FlashAttention path uses the same `q`, `k`, and `v` tensors and the same non-causal attention semantics. It should not be interpreted as a different model or approximation.

## Current Status

The repository is ready for local development. The next required step is to run the environment check and benchmark on a CUDA-capable NVIDIA machine.
