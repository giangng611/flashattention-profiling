# Attention Selector Baseline and Repeatability Results

## Experimental result

Three exhaustive sweeps were completed on the same RTX 3090 with the same 16 supported workloads, 48 declared configurations, 20 warmup iterations, and 100 timed trials. Each run produced 696 valid Triton measurements and 72 resource failures. Every successfully executed configuration passed the numerical correctness check.

The resource failures were consistent across all three runs: eight `BLOCK_N=128, num_stages=3` configurations exceeded shared-memory capacity on nine `head_dim=128` workloads. Every workload retained at least 40 valid candidates.

The first sweep took 347.6 seconds across the workload set. Repetitions 1 and 2 took 24.15 and 24.20 seconds because Triton disk-cache artifacts were already available. The later wall times are useful for repeatability analysis but do not measure cold compilation cost. The original end-to-end wall time remains the safer tuning-cost reference.

## Ranking stability

Exact fastest-config labels are not stable enough to serve as the only target:

- all three runs selected the same deterministic top-1 label on 4 of 16 workloads; after accounting for exact latency ties, 5 workloads had at least one common exact-oracle config across all runs;
- mean pairwise Spearman rank correlation across the full valid config ranking was 0.836;
- mean pairwise top-4 Jaccard overlap was 0.485;
- the aggregate oracle was within 1% of each run's oracle on 6 workloads, within 3% on 7, and within 5% on 11;
- held-out workloads were more stable than development workloads, with mean rank correlation 0.919 and all aggregate oracles within 5% in every run.

One development workload, `dense-prefill-s512-h12-d128-fp16-noncausal`, is a clear outlier: the aggregate oracle is 28.8% slower than the run-0 oracle in that run. Short workloads also show weaker top-k overlap because small absolute latency differences can change the ordering.

These results support oracle-relative performance and top-k coverage as primary metrics. Exact top-1 accuracy should be reported only as a secondary diagnostic.

## Held-out selector comparison using all three runs

For each workload/config pair, the repeated evaluation uses the geometric mean of the three per-run median latencies. Policies are fit on the same 10 development workloads and evaluated on the same 6 held-out workloads. Timing-cost columns still use the first sweep so the warm-cache repetitions do not understate tuning cost.

| Method | Configs measured | Candidate fraction | Success | Geometric-mean oracle performance | Mean successful regret | Estimated tuning-time saving | Meets 90% / 25% target |
|---|---:|---:|---:|---:|---:|---:|---|
| Tutorial-derived project default | 1 | 2.1% | 100% | 84.7% | 18.2% | 97.9% | No |
| Best fixed development config | 1 | 2.1% | 100% | 87.6% | 14.2% | 97.9% | No |
| Nearest-workload reuse | 1 | 2.1% | 100% | 91.2% | 9.8% | 97.9% | Yes |
| Random search, budget 1 | 1 | 2.1% | 88.5% | 0% including failures | 41.1% | 97.9% | No |
| Random search, budget 2 | 2 | 4.2% | 98.5% | 0% including failures | 28.7% | 95.8% | No |
| Random search, budget 4 | 4 | 8.3% | 100% | 87.2% | 15.6% | 91.7% | No |
| Random search, budget 8 | 8 | 16.7% | 100% | 93.2% | 7.6% | 83.3% | Yes |
| Decision-tree ranker, top-2 | 2 | 4.2% | 100% | 94.0% | 6.5% | 95.8% | Yes |
| Decision-tree ranker, top-4 | 4 | 8.3% | 100% | 96.0% | 4.2% | 91.7% | Yes |
| Decision-tree ranker, top-8 | 8 | 16.7% | 100% | 98.0% | 2.1% | 83.3% | Yes |
| Exhaustive local oracle | 48 | 100% | 100% | 100% | 0.0% | 0.0% | Not applicable |

Random budget-1 and budget-2 have zero aggregate geometric mean because some trials sampled only invalid configurations. Random results use 100 seeds and preserve every workload/seed outcome.

The repeated result is more conservative than the first-run result, but the main finding survives. The decision-tree top-2 policy reaches 94.0% of the aggregate local oracle with 2 of 48 configurations. At the same or nearby budgets, it outperforms best fixed and random search. Nearest-workload reuse is also a strong one-config baseline and must remain in future comparisons.

## Split sensitivity

The original 10/6 split was compared with 99 deterministic, unique random 10/6 splits. Every split preserves basic categorical coverage on both sides: FP16/BF16, head dimensions 64/128, causal/non-causal workloads, and at least two query-head counts. These constraints avoid evaluating a selector when a major feature value is entirely absent from training, but they also mean the result is not an average over every possible 10/6 partition.

| Method | Configs measured | Mean split performance | P10 split performance | Worst split | Splits at least 90% | Splits at least 95% |
|---|---:|---:|---:|---:|---:|---:|
| Tutorial-derived project default | 1 | 85.5% | 83.3% | 80.8% | 0% | 0% |
| Best fixed development config | 1 | 90.9% | 88.3% | 83.3% | 70% | 1% |
| Nearest-workload reuse | 1 | 91.2% | 88.5% | 84.6% | 70% | 1% |
| Decision-tree ranker, top-1 | 1 | 89.7% | 87.5% | 0% | 69% | 8% |
| Decision-tree ranker, top-2 | 2 | 93.7% | 90.6% | 87.4% | 92% | 33% |
| Decision-tree ranker, top-4 | 4 | 96.3% | 94.0% | 90.6% | 100% | 82% |
| Decision-tree ranker, top-8 | 8 | 98.2% | 96.5% | 95.8% | 100% | 100% |

Top-1 fails on two splits because the single tree-selected candidate is invalid for some `head_dim=128` held-out workloads. Top-k selection avoids this failure mode. Top-2 supports a cautious 90% claim for most, but not all, balanced splits. Top-4 reaches at least 90% on every tested split, while top-8 reaches at least 95% on every tested split.

Random search is not repeated in this analysis because it does not learn from the development partition. Its earlier 100-seed held-out result remains the direct random baseline.

## Direct online tuning cost

The robust tree top-4 policy and exhaustive 48-config control were also run directly on the six held-out workloads. Each mode ran in a separate Python process on the same RTX 3090, with the existing Triton disk cache retained. The measured wall time includes cache/compile preparation, correctness checking, and steady-state benchmarking for the candidate set; it excludes input construction and the PyTorch reference timing.

| Method | Config attempts | Valid results | Resource errors | Correctness failures | Total tuning wall time | Direct geometric-mean oracle performance |
|---|---:|---:|---:|---:|---:|---:|
| Decision-tree ranker, top-4 | 24 | 24 | 0 | 0 | 1.987 s | 96.97% |
| Exhaustive local oracle | 288 | 256 | 32 | 0 | 12.600 s | 100% |

Top-4 reduced direct warm-cache tuning wall time by 84.23%, or 6.34x, while testing 4 rather than 48 configurations per workload. Its oracle-relative performance ranged from 93.30% to 98.58%; the worst direct regret was 7.19% on the dense Qwen-shaped proxy. Selector inference itself totaled 0.050 ms across all six workloads and was negligible relative to candidate measurement.

The exhaustive search's additional tuning cost would be recovered only after approximately 93,777 to 672,104 invocations of the same workload, based on the directly measured latency advantage for each workload. This break-even calculation excludes application-level overhead and assumes that the selected configuration is reused without retuning.

The first workload in each fresh process paid about one second of cache/compile preparation, so its top-4 wall-time saving was only 29.6%. The other five workloads saved 92.0-93.3%. This fixed startup effect makes the aggregate 84.23% saving more conservative than the simple 4/48 candidate ratio, but the result remains a warm-disk-cache measurement rather than controlled cold compilation.

## Kernel comparison

The fastest Triton configuration beat PyTorch SDPA on only 3 of the 16 supported workloads in the original sweep. The selector experiment therefore answers a within-Triton configuration-selection question. It does not show that the current Triton kernel is generally faster than PyTorch.

## Limitations

- The three repetitions share one GPU, software environment, workload split, and configuration space.
- Seeds 1 and 2 reuse Triton disk-cache artifacts and cannot estimate cold compilation cost.
- The 10/6 split is small. The shallow tree may be sensitive to this particular split.
- The split-sensitivity set is balanced by categorical coverage and therefore does not represent all possible partitions.
- The aggregate geometric mean reduces single-run noise but does not resolve the large outlier noted above.
- The direct online timing retains the existing Triton disk cache. It measures warm-cache selection in separate fresh Python processes, not controlled cold compilation.
- Break-even counts assume repeated use of an identical workload and omit end-to-end application overhead.

## Interpretation

The evidence supports continuing the selector direction and writing the first CURO proposal draft. The defensible claim is that a lightweight top-k ranker can reduce empirical search while retaining high oracle-relative performance across the tested balanced splits. Top-4 is the smallest tested budget that stays above 90% on every split, and its direct run retained 96.97% of exhaustive performance while reducing warm-cache tuning time by 84.23%. Top-8 is the smallest tested budget that stays above 95% on every split. The current evidence does not support a universal performance guarantee, stable exact-oracle prediction, cold-cache savings, cross-GPU generalization, or Triton superiority over PyTorch SDPA.
