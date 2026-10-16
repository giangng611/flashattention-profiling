# CUDA Server Runbook

This file records the reproducible CUDA workflow, from the original PyTorch baseline through the expanded Triton sweep and direct selector timing.

## Goal

Run or reproduce the attention experiments on the UGA CUDA machine and preserve:

- exact environment information
- reproducible CUDA results
- raw CSV results
- latency, memory, and speedup plots
- notes about the most significant performance pattern or limitation

The current research pipeline uses 16 supported dense-prefill workloads, 2 unsupported GQA references, 48 candidate configurations, three exhaustive sweeps, CPU-only selector analysis, and a direct tree top-4 versus exhaustive control.

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

Then install the remaining repository dependencies. CUDA-enabled PyTorch should still come from the official selector rather than being replaced by another wheel:

```bash
.venv/bin/python -m pip install pandas matplotlib
```

Check the environment:

```bash
.venv/bin/python src/check_environment.py
```

Save the output for the report.

Validate the expanded input files before using the GPU:

```bash
.venv/bin/python src/validate_expanded_sweep_inputs.py
```

Expected output:

```text
Validated 16 supported workloads, 2 unsupported references, and 48 candidate configurations.
```

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

The server currently may not expose `nsys` or `ncu` in the shell path. If those tools are unavailable, use PyTorch Profiler first:

```bash
python src/profile_attention.py --device cuda --seq-lengths 512 4096 --warmup 5 --trials 10 --output-dir profiling/pytorch
```

This writes text summaries, machine-readable event CSV files, and Chrome trace JSON files under:

```text
profiling/pytorch/
```

If Nsight Systems is available later, use it for a timeline:

```bash
nsys profile -o profiling/cuda_seq512 .venv/bin/python src/benchmark_attention.py --device cuda --seq-lengths 512 --warmup 5 --trials 10 --output results/cuda_seq512_profile_run.csv --metadata-output results/cuda_seq512_profile_metadata.json
```

```bash
nsys profile -o profiling/cuda_seq4096 .venv/bin/python src/benchmark_attention.py --device cuda --seq-lengths 4096 --warmup 5 --trials 10 --output results/cuda_seq4096_profile_run.csv --metadata-output results/cuda_seq4096_profile_metadata.json
```

Use Nsight Compute later if kernel-level metrics are available and needed:

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
3. Save the exact failure message for the next update.

## Expanded Triton Configuration Sweep

The canonical smoke and stable commands are maintained in [FUSED_ATTENTION_CONFIG_SWEEP_PLAN.md](../plans/FUSED_ATTENTION_CONFIG_SWEEP_PLAN.md). The stable run uses:

```bash
.venv/bin/python src/sweep_triton_fused_attention_configs.py \
  --workloads workloads/expanded_dense_prefill_workloads.csv \
  --configs configs/attention_config_space.csv \
  --warmup 20 \
  --trials 100 \
  --seed 0 \
  --atol 0.08 \
  --rtol 0.02 \
  --output results/expanded_attention_sweep.csv \
  --summary-output results/expanded_attention_sweep_summary.csv \
  --metadata-output results/expanded_attention_sweep_metadata.json
```

Repeat the complete run with seeds 1 and 2 and seed-specific output names. Do not overwrite the seed-0 artifacts. The preserved runs should each contain 16 supported workloads, 696 valid Triton results, 72 resource errors, zero correctness failures, and at least 40 valid configurations per workload.

After all three sweeps are present, run the CPU-only repeatability, selector, and split analyses described in [ATTENTION_SELECTOR_BASELINE_PLAN.md](../plans/ATTENTION_SELECTOR_BASELINE_PLAN.md). These analysis steps can run locally and do not require CUDA.

## Direct Online Selector Timing

After the repeated exhaustive sweeps and offline split-sensitivity analysis are committed, measure the direct warm-cache cost of the robust tree top-4 policy and an exhaustive 48-config control. Run them as separate Python processes so their in-memory Triton states do not overlap.

Tree top-4:

```bash
.venv/bin/python src/benchmark_attention_selector_online.py \
  --mode tree \
  --budget 4 \
  --warmup 20 \
  --trials 100 \
  --seed 3 \
  --output results/online_selector_top4_measurements.csv \
  --summary-output results/online_selector_top4_summary.csv \
  --metadata-output results/online_selector_top4_metadata.json \
  2>&1 | tee results/online_selector_top4.log
```

Exhaustive control:

```bash
.venv/bin/python src/benchmark_attention_selector_online.py \
  --mode exhaustive \
  --budget 4 \
  --warmup 20 \
  --trials 100 \
  --seed 3 \
  --output results/online_selector_exhaustive_measurements.csv \
  --summary-output results/online_selector_exhaustive_summary.csv \
  --metadata-output results/online_selector_exhaustive_metadata.json \
  2>&1 | tee results/online_selector_exhaustive.log
```

Expected output for each command:

- six held-out workloads;
- four attempted configurations per workload for tree top-4;
- 48 attempted configurations per workload for exhaustive mode;
- zero correctness failures;
- three data files plus one log per mode.

These commands retain the existing Triton disk cache. They measure direct warm-cache online tuning cost, not cold compilation cost.

The preserved RTX 3090 run completed all six held-out workloads in both modes. Top-4 attempted 24 candidates with no failures and took 1.987 seconds of total tuning wall time. Exhaustive search attempted 288 candidates, encountered 32 expected resource failures, and took 12.600 seconds. Both modes had zero correctness failures. The CPU-only comparison is reproduced with:

```bash
.venv/bin/python src/analyze_online_selector_timing.py
```
