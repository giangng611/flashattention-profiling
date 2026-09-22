# Boundary Sweep Plan

The first CUDA baseline and PyTorch Profiler pass showed that PyTorch SDPA auto already selects FlashAttention for the representative FP16, head dimension 64 cases I profiled.

The next step is to look for boundary conditions where this stops being true, or where the performance behavior changes enough to become interesting.

## Question

```text
When does automatic attention backend selection work well, and when does it fail or leave performance on the table?
```

## Sweep Dimensions

Initial sweep:

- sequence length: 128, 512, 2048, 4096
- head dimension: 32, 64, 128, 256
- dtype: FP16, BF16
- causal: false, true
- batch size: 1
- heads: 16

The first sweep excludes explicit standard attention by default, because standard attention can use much more memory and is already well understood from the baseline. The sweep focuses on:

- `sdpa_auto`
- `flash_forced`

I can include standard attention later for smaller shapes only.

## Command

Run this on the CUDA server:

```bash
python src/sweep_attention_boundaries.py --device cuda --seq-lengths 128 512 2048 4096 --head-dims 32 64 128 256 --dtypes float16 bfloat16 --causal-values false true --warmup 5 --trials 20 --output results/cuda_boundary_sweep.csv --metadata-output results/cuda_boundary_sweep_metadata.json
```

Smoke test:

```bash
python src/sweep_attention_boundaries.py --device cuda --seq-lengths 128 --head-dims 32 64 --dtypes float16 --causal-values false true --warmup 2 --trials 5 --stop-after 4 --output results/cuda_boundary_sweep_smoke.csv --metadata-output results/cuda_boundary_sweep_smoke_metadata.json
```

## What To Look For

Important cases:

- `flash_forced` fails but `sdpa_auto` succeeds
- `sdpa_auto` is much faster than `flash_forced`
- `flash_forced` is much faster than `sdpa_auto`
- memory differs significantly between the two
- BF16 behaves differently from FP16
- causal attention behaves differently from non-causal attention
- large head dimensions change support or performance

## Interpreting Results

If `sdpa_auto` and `flash_forced` have similar latency and memory across the sweep, then PyTorch's automatic selection is probably reliable for these shapes.

If there are failures or large gaps, those become candidate boundary cases for deeper profiling.

Potential report sentence:

```text
The initial CUDA baseline did not reveal a clear SDPA auto vs FlashAttention backend difference for head_dim=64. I therefore expanded the experiment across head dimension, dtype, and causal mode to search for configurations where backend selection changes or fails.
```

## After The Sweep

After running the sweep, inspect:

```bash
cat results/cuda_boundary_sweep.csv
cat results/cuda_boundary_sweep_metadata.json
```

Then identify 2-3 interesting configurations and run PyTorch Profiler on those specific cases.

The first completed CUDA sweep is summarized in:

- [BOUNDARY_SWEEP_RESULTS_REPORT.md](/Users/giangnguyendohoang/PycharmProjects/flashattention-profiling/BOUNDARY_SWEEP_RESULTS_REPORT.md)
