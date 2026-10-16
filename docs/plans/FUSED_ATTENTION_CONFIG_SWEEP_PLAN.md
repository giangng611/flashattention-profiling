# Fused Attention Config Sweep Plan

## Why This Experiment Exists

The first Triton fused-attention kernel is correct, but it is slower than PyTorch SDPA on the longer prefill workload. This is expected for an initial kernel. The next question is not whether Triton is "good" or "bad", but whether different workload shapes prefer different Triton configurations.

This experiment is a small bridge between the paper-reading task and the code in this repo:

- The Anatomy of a Triton Attention Kernel motivates workload-specific attention optimization.
- tritonBLAS motivates avoiding expensive exhaustive tuning when a smaller selector may be enough.
- FlashInfer motivates using realistic LLM attention shapes rather than only toy shapes.

## Experiment Questions

```text
1. Do different dense-prefill workloads still prefer different Triton configurations?
2. Is exhaustive compilation and benchmarking expensive enough to justify a selector?
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

The expanded experiment uses:

```text
workloads/expanded_dense_prefill_workloads.csv
configs/attention_config_space.csv
```

It contains 16 supported dense-prefill workloads:

- 10 development and 6 held-out workloads;
- FP16 and BF16;
- causal and non-causal attention;
- sequence lengths from 256 to 2048;
- head dimensions 64 and 128;
- head counts 8, 12, and 16.

The held-out set includes dense shapes derived from [GPT-2](https://huggingface.co/openai-community/gpt2/blob/main/config.json) (`12` heads, head dimension `64`) and [BERT-large](https://huggingface.co/google-bert/bert-large-uncased/blob/main/config.json) (`16` heads, head dimension `64`, maximum sequence length `512`). Other supported rows are labeled synthetic or synthetic dense proxies.

Two model-derived Qwen GQA rows are included as explicit unsupported references and are excluded from selector splits. The candidate file contains 48 configurations over `BLOCK_M`, `BLOCK_N`, `num_warps`, and `num_stages`. Runtime failures are retained as errors; the first CUDA run must confirm that at least 32 candidates are valid for each supported workload.

The script:

- loads the same workload table used by the fused-attention benchmark;
- runs PyTorch SDPA once per supported workload;
- sweeps several Triton `BLOCK_M`, `BLOCK_N`, `num_warps`, and `num_stages` values;
- checks every Triton config using absolute and relative tolerances;
- measures JIT compile/cache preparation separately from steady-state benchmarking;
- marks the fastest correct Triton config per workload as the local oracle;
- writes detailed results, summary results, and metadata.

## Smoke Run

```bash
python src/sweep_triton_fused_attention_configs.py \
  --workloads workloads/expanded_dense_prefill_workloads.csv \
  --block-ms 32 64 \
  --block-ns 32 64 \
  --num-warps-values 4 \
  --num-stages-values 2 \
  --warmup 2 \
  --trials 5 \
  --seed 0 \
  --atol 0.08 \
  --rtol 0.02 \
  --output results/expanded_attention_sweep_smoke.csv \
  --summary-output results/expanded_attention_sweep_smoke_summary.csv \
  --metadata-output results/expanded_attention_sweep_smoke_metadata.json
```

## Stable Run

```bash
python src/sweep_triton_fused_attention_configs.py \
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

## How I Should Interpret The Result

The main output to read first is:

```text
results/expanded_attention_sweep_summary.csv
```

Important columns:

- `best_config_id`: fastest correct Triton config within the small sweep.
- `best_num_warps` and `best_num_stages`: launch settings for the fastest correct Triton config.
- `best_speedup_over_torch_sdpa`: whether the best Triton config beats or trails PyTorch SDPA.
- `tested_triton_configs`: number of valid configs tested.
- `candidate_triton_configs`, `runtime_error_configs`, and `correctness_failed_configs`: whether the declared space actually produced at least 32 valid correct candidates.
- `compile_or_cache_prepare_total_ms`: total time spent in Triton's compile-only `warmup` calls for correct configs.
- `benchmark_wall_time_total_ms`: total wall time for steady-state warmups and timed trials.
- `sweep_wall_time_ms`: end-to-end configuration-sweep time for the workload.

The detailed output also includes:

- `oracle_gap_pct`: how far each config is from the fastest correct Triton config for that workload.
- `latency_gap_to_torch_pct`: how far each config is from PyTorch SDPA.
- `correctness_passed`: whether the config produced numerically acceptable output.
- `correctness_max_abs_diff` and `correctness_max_rel_diff`: diagnostic error magnitudes.
- `compile_or_cache_prepare_ms`: time for `JITFunction.warmup`, which does not launch the kernel.

`compile_or_cache_prepare_ms` is cache-aware: it can include an in-memory or disk-cache lookup and is not guaranteed to be cold compilation time. A cold-cache study should run in a fresh process with a controlled Triton cache directory; it must be reported separately from this main sweep.

The result should not be overclaimed. This is a local oracle over a declared candidate space, not a proof of global optimality. Selector work should proceed only if configuration preferences remain workload-dependent and the measured tuning cost is meaningful.
