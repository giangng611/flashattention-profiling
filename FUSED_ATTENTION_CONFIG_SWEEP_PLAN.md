# Fused Attention Config Sweep Plan

## Why This Experiment Exists

The first Triton fused-attention kernel is correct, but it is slower than PyTorch SDPA on the longer prefill workload. This is expected for an initial kernel. The next question is not whether Triton is "good" or "bad", but whether different workload shapes prefer different Triton configurations.

This experiment is a small bridge between the paper-reading task and the code in this repo:

- The Anatomy of a Triton Attention Kernel motivates workload-specific attention optimization.
- tritonBLAS motivates avoiding expensive exhaustive tuning when a smaller selector may be enough.
- FlashInfer motivates using realistic LLM attention shapes rather than only toy shapes.

## Experiment Question

```text
Do different attention workloads prefer different Triton tile and launch configurations?
```

If the answer is yes, then a future research direction could be:

```text
workload-aware configuration selection for Triton attention
```

If the answer is no, then the stronger direction may be elsewhere, such as decode support, GQA support, or application-level serving evaluation.

## Script

```text
src/sweep_triton_fused_attention_configs.py
```

The script:

- loads the same workload table used by the fused-attention benchmark;
- runs PyTorch SDPA once per supported workload;
- sweeps several Triton `BLOCK_M`, `BLOCK_N`, `num_warps`, and `num_stages` values;
- checks correctness for every Triton config;
- marks the fastest correct Triton config per workload as the local oracle;
- writes detailed results, summary results, and metadata.

## Smoke Run

```bash
python src/sweep_triton_fused_attention_configs.py \
  --workloads workloads/small_attention_workloads.csv \
  --block-ms 32 64 \
  --block-ns 32 64 \
  --num-warps-values 4 \
  --num-stages-values 2 3 \
  --warmup 5 \
  --trials 10 \
  --output results/triton_fused_attention_config_sweep_smoke.csv \
  --summary-output results/triton_fused_attention_config_sweep_smoke_summary.csv \
  --metadata-output results/triton_fused_attention_config_sweep_smoke_metadata.json
```

## Stable Run

```bash
python src/sweep_triton_fused_attention_configs.py \
  --workloads workloads/small_attention_workloads.csv \
  --block-ms 32 64 \
  --block-ns 32 64 \
  --num-warps-values 4 8 \
  --num-stages-values 2 3 \
  --warmup 20 \
  --trials 100 \
  --output results/triton_fused_attention_config_sweep.csv \
  --summary-output results/triton_fused_attention_config_sweep_summary.csv \
  --metadata-output results/triton_fused_attention_config_sweep_metadata.json
```

## How I Should Interpret The Result

The main output to read first is:

```text
results/triton_fused_attention_config_sweep_summary.csv
```

Important columns:

- `best_config_id`: fastest correct Triton config within the small sweep.
- `best_num_warps` and `best_num_stages`: launch settings for the fastest correct Triton config.
- `best_speedup_over_torch_sdpa`: whether the best Triton config beats or trails PyTorch SDPA.
- `tested_triton_configs`: number of valid configs tested.

The detailed output also includes:

- `oracle_gap_pct`: how far each config is from the fastest correct Triton config for that workload.
- `latency_gap_to_torch_pct`: how far each config is from PyTorch SDPA.
- `correctness_passed`: whether the config produced numerically acceptable output.

The result should not be overclaimed. This is a local oracle over a small search space, not a proof of global optimality.
