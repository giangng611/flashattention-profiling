# CUDA Server Runbook

This file records the steps for moving from the local Mac/MPS practice benchmark to the CUDA baseline requested by Qingchen.

## Goal

Run the attention benchmark on the UGA CUDA machine and collect:

- exact environment information
- reproducible CUDA results
- raw CSV results
- latency, memory, and speedup plots
- notes about the most significant performance pattern or limitation

## Connect To The Server

Use the UGA MyID account:

```bash
ssh <UGA-MyID>@cuda3.cs.uga.edu
```

After connecting:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

If `torch` is not installed yet, create a virtual environment first.

## Prepare The Repository

Clone the repo or copy it to the server. Then enter the repo:

```bash
cd flashattention-profiling
```

Create a virtual environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
```

Install PyTorch using the CUDA command recommended by the official PyTorch install selector for the server's CUDA version:

```text
https://pytorch.org/get-started/locally/
```

Then install the remaining dependencies:

```bash
.venv/bin/python -m pip install pandas matplotlib
```

Check the environment:

```bash
.venv/bin/python src/check_environment.py
```

Save the output for the report.

## CUDA Smoke Test

Run a small CUDA test first:

```bash
.venv/bin/python src/benchmark_attention.py --device cuda --seq-lengths 128 256 --warmup 5 --trials 10 --output results/cuda_smoke.csv --metadata-output results/cuda_smoke_metadata.json
```

This should produce rows for:

- `standard`
- `sdpa_auto`
- `flash_forced`

If `flash_forced` fails, keep the CSV row and error message. That is useful information because the report should record unsupported configurations or unavailable CUDA backends.

## Full CUDA Baseline

After the smoke test works:

```bash
.venv/bin/python src/benchmark_attention.py --device cuda --seq-lengths 128 256 512 1024 2048 4096 --warmup 20 --trials 50 --output results/cuda_attention_benchmark.csv --metadata-output results/cuda_environment_metadata.json
```

Generate plots:

```bash
.venv/bin/python src/plot_results.py --input results/cuda_attention_benchmark.csv --output-dir plots/cuda
```

Expected outputs:

```text
results/cuda_attention_benchmark.csv
results/cuda_environment_metadata.json
plots/cuda/latency_vs_sequence_length.png
plots/cuda/peak_memory_vs_sequence_length.png
plots/cuda/speedup_vs_sequence_length.png
```

## Representative Profiling Cases

Only profile a small number of cases first:

- sequence length 512
- sequence length 4096

Suggested starting point:

```bash
nsys profile -o profiling/cuda_seq512 .venv/bin/python src/benchmark_attention.py --device cuda --seq-lengths 512 --warmup 5 --trials 10 --output results/cuda_seq512_profile_run.csv --metadata-output results/cuda_seq512_profile_metadata.json
```

```bash
nsys profile -o profiling/cuda_seq4096 .venv/bin/python src/benchmark_attention.py --device cuda --seq-lengths 4096 --warmup 5 --trials 10 --output results/cuda_seq4096_profile_run.csv --metadata-output results/cuda_seq4096_profile_metadata.json
```

Use Nsight Compute later if kernel-level metrics are needed:

```bash
ncu -o profiling/cuda_seq4096 .venv/bin/python src/benchmark_attention.py --device cuda --seq-lengths 4096 --warmup 3 --trials 5 --output results/cuda_seq4096_ncu_run.csv --metadata-output results/cuda_seq4096_ncu_metadata.json
```

## What To Record

For the next report, record:

- GPU model
- driver version from `nvidia-smi`
- Python version
- PyTorch version
- PyTorch CUDA version
- dtype
- batch size
- number of heads
- head dimension
- sequence lengths
- exact benchmark commands
- whether each backend succeeded or failed
- median latency and variability
- peak memory
- most significant observed pattern
- one testable hypothesis

## If CUDA Is Unavailable

If the server login works but CUDA is unavailable:

1. Save the output of `nvidia-smi`.
2. Save the output of `src/check_environment.py`.
3. Send Qingchen the exact failure message.
