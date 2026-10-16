# FlashAttention Profiling and Configuration Selection

This repository studies how workload shape affects the performance of a forward Triton dense-prefill attention kernel. The current research question is:

> Can a lightweight workload-aware selector approach exhaustive-tuning performance on unseen attention workloads while benchmarking far fewer configurations?

The project began as a PyTorch SDPA/FlashAttention profiling exercise and developed into a small configuration-selection prototype. It is scoped to dense self-attention prefill. GQA, decode, cross-GPU transfer, and end-to-end serving are later extensions.

## Current evidence

- 16 supported dense-prefill workloads and 2 unsupported model-derived Qwen GQA references.
- FP16/BF16, causal/non-causal, head dimensions 64/128, multiple sequence lengths, and multiple head counts.
- 48 Triton tile/launch configurations.
- Three complete RTX 3090 sweeps with 696 valid Triton measurements per run and no correctness failures among successful executions.
- Required baselines: tutorial-derived default, best fixed development config, nearest-workload reuse, random search, shallow decision-tree top-k, and exhaustive local oracle.
- Original 10/6 development/held-out evaluation plus 99 additional balanced splits.
- Direct warm-cache top-4 versus exhaustive timing on the six held-out workloads.

The direct top-4 experiment retained 96.97% geometric-mean oracle-relative performance while reducing tuning wall time from 12.600 seconds to 1.987 seconds, an 84.23% reduction. This result is preliminary and limited to the current GPU, kernel, workloads, candidate space, and cache conditions.

## Repository structure

```text
flashattention-profiling/
├── configs/       # fixed Triton candidate space
├── docs/          # notes, plans, proposals, reports, and runbooks
├── profiling/     # profiler artifacts and notes
├── results/       # preserved raw and derived experiment data
├── src/           # benchmarks and CPU-only analysis scripts
├── tests/         # reproducibility and analysis tests
└── workloads/     # Week 3 and expanded workload definitions
```

## Local setup and verification

The local Mac is used for development and CPU-only analysis; it cannot produce CUDA evidence.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python src/validate_expanded_sweep_inputs.py
.venv/bin/python -m unittest discover -s tests -v
```

Expected input validation:

```text
Validated 16 supported workloads, 2 unsupported references, and 48 candidate configurations.
```

Reproduce the CPU-only selector analyses from the preserved CUDA measurements:

```bash
.venv/bin/python src/analyze_attention_sweep_repetitions.py
.venv/bin/python src/evaluate_attention_config_selectors.py \
  --sweep \
    results/expanded_attention_sweep.csv \
    results/expanded_attention_sweep_seed1.csv \
    results/expanded_attention_sweep_seed2.csv \
  --sweep-summary results/expanded_attention_sweep_summary.csv \
  --configs configs/attention_config_space.csv \
  --tutorial-config bm64_bn64_w4_s3 \
  --budgets 1 2 4 8 \
  --random-seeds 100 \
  --seed 0 \
  --tree-max-depth 2 \
  --tree-min-leaf 2 \
  --trials-output results/attention_selector_repeated_trials.csv \
  --summary-output results/attention_selector_repeated_summary.csv \
  --metadata-output results/attention_selector_repeated_metadata.json
.venv/bin/python src/evaluate_attention_selector_split_sensitivity.py
.venv/bin/python src/analyze_online_selector_timing.py
```

## CUDA execution

CUDA benchmarks require an NVIDIA GPU, CUDA-enabled PyTorch, and Triton. The tested environment is an NVIDIA GeForce RTX 3090. Use the [CUDA server runbook](docs/setup/CUDA_SERVER_RUNBOOK.md) for the exact environment, sweep, repetition, transfer, and direct online timing workflow.

The main experiment definitions are:

- [expanded workload/config sweep](docs/plans/FUSED_ATTENTION_CONFIG_SWEEP_PLAN.md)
- [selector baselines and metrics](docs/plans/ATTENTION_SELECTOR_BASELINE_PLAN.md)
- [Week 4 completion audit](docs/plans/WEEK4_COMPLETION_AUDIT.md)

## Main outputs

- [selector and repeatability report](docs/reports/ATTENTION_SELECTOR_BASELINE_REPORT.md)
- [weekly research tracker](docs/notes/WEEKLY_PROGRESS.md)
- [paper review table](docs/reading/PAPER_REVIEW_TABLE.md)
- [CURO proposal first draft](docs/proposals/CURO_PROPOSAL_FIRST_DRAFT.md)
- [technical proposal working draft](docs/proposals/CURO_PROPOSAL_DRAFT.md)
- [sendable proposal PDF](output/pdf/CURO_Proposal_First_Draft.pdf)
- [sendable Week 4 update PDF](output/pdf/Week4_Research_Update.pdf)

Communication drafts are preserved separately for [Qingchen](docs/emails/EMAIL_TO_QINGCHEN_WEEK4.md) and [Dr. Wenwen Wang](docs/emails/EMAIL_TO_WENWEN_CURO.md).

The raw exhaustive runs, repeated runs, online timing outputs, and derived comparison tables are preserved under `results/` with metadata and input hashes.

## Interpretation limits

- The oracle is local to the declared 48-configuration space.
- The kernel is forward-only dense self-attention and is not a production FlashAttention implementation.
- Resource-invalid candidates count as tuning attempts but are excluded from valid latency comparisons.
- Later repetitions and direct online timing retain Triton's disk cache and do not measure controlled cold compilation.
- PyTorch SDPA is faster than this Triton kernel on most current workloads; the research question concerns selection within the Triton configuration space.
- Exact top-1 labels are noisy, so oracle-relative top-k performance is the primary outcome.

Earlier SDPA, boundary, head-dimension, Triton QK, and Nsight experiments remain documented under `docs/reports/`.
