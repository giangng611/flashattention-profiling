# Attention Configuration Selector Baseline Plan

## Question

Can a lightweight selector approach exhaustive-tuning performance on unseen dense-prefill attention workloads while measuring far fewer configurations?

## Data boundary

- Fit and choose policies using only the 10 `development` workloads.
- Evaluate only on the 6 `held_out` workloads.
- Exclude the two unsupported Qwen GQA reference rows.
- Use the same 48 declared configuration candidates for every method.
- Treat runtime/resource failures as attempted configurations, not as valid latency measurements.

The evaluation is offline: it replays the preserved exhaustive sweep and therefore does not require CUDA. This makes method comparisons deterministic while retaining the measured latency and failure outcomes from the RTX 3090 run.

## Baselines

1. `tutorial_derived_default`: `bm64_bn64_w4_s3`, the fixed default used by this project's tutorial-derived benchmark. The official Triton tutorial does not define this as a universal default.
2. `best_fixed_development`: the one configuration with the lowest geometric-mean latency ratio to the local oracle across all development workloads.
3. `nearest_workload_reuse`: normalize workload features using development ranges, find the nearest development workload, and reuse its marked oracle config.
4. `random_search`: sample without replacement at budgets 1, 2, 4, and 8. Report variation across 100 seeds.
5. `decision_tree_ranker`: fit a depth-2 workload tree on development measurements. Each leaf ranks configs by geometric-mean oracle-relative latency and returns top-k candidates.
6. `exhaustive_local_oracle`: attempt all 48 declared candidates and select the fastest valid correct result.

Workload features are log2 query length, query-head count, head dimension, BF16 indicator, and causal indicator. The tree is deliberately small and transparent; it is a baseline rather than a claimed contribution.

## Metrics

- oracle-relative performance: `oracle_latency / selected_latency`;
- regret: `(selected_latency / oracle_latency - 1) * 100`;
- exact oracle and within-1% oracle coverage;
- configurations compiled or measured and candidate-space fraction;
- selection success and invalid candidates attempted;
- observed compile/cache preparation and benchmark wall time;
- estimated total tuning time from the measured exhaustive wall time scaled by the candidate fraction;
- estimated break-even calls versus the tutorial-derived default.

The target from the Week 4 instructions is 90-95% of local-oracle performance while measuring no more than 20-25% of candidates. This is an evaluation target, not an assumed result.

## Command

```bash
python src/evaluate_attention_config_selectors.py \
  --sweep results/expanded_attention_sweep.csv \
  --sweep-summary results/expanded_attention_sweep_summary.csv \
  --configs configs/attention_config_space.csv \
  --tutorial-config bm64_bn64_w4_s3 \
  --budgets 1 2 4 8 \
  --random-seeds 100 \
  --seed 0 \
  --tree-max-depth 2 \
  --tree-min-leaf 2 \
  --trials-output results/attention_selector_trials.csv \
  --summary-output results/attention_selector_summary.csv \
  --metadata-output results/attention_selector_metadata.json
```

## Interpretation limits

- The oracle is local to the declared 48-candidate space.
- The workload split is small and comes from one GPU.
- The input contains one stable benchmark run, so near-tied oracle labels may not be repeatable.
- Compile/cache preparation is not a controlled cold-cache measurement.
- Scaled tuning time is an estimate; only a future online selector run can provide direct selector wall time.
