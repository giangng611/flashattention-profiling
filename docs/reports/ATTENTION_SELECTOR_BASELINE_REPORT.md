# Attention Selector Baseline Results

## Result

The expanded sweep provides preliminary evidence for workload-aware selection. All 696 successfully executed Triton configurations passed correctness, while 72 attempts failed because eight `BLOCK_N=128, num_stages=3` configurations exceeded RTX 3090 shared-memory capacity on nine `head_dim=128` workloads. Every supported workload retained at least 40 valid candidates.

The 16 supported workloads selected nine marked local-oracle configurations. Exhaustive tuning across the workload set took 347.6 seconds. Of that total, 273.8 seconds were recorded in successful compile/cache preparation calls and 25.6 seconds in steady-state benchmark calls. These components do not include per-config failed-compile time, and compile preparation can reuse caches, so end-to-end sweep wall time is the safer tuning-cost measurement.

## Held-out comparison

The selector policies were fit only on 10 development workloads and evaluated on 6 held-out workloads.

| Method | Configs measured | Candidate fraction | Success | Geometric-mean oracle performance | Mean successful regret | Exact oracle coverage | Estimated tuning-time saving | Meets 90% / 25% target |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Tutorial-derived project default | 1 | 2.1% | 100% | 77.7% | 30.1% | 0.0% | 97.9% | No |
| Best fixed development config | 1 | 2.1% | 100% | 84.1% | 19.3% | 0.0% | 97.9% | No |
| Nearest-workload reuse | 1 | 2.1% | 100% | 85.7% | 17.2% | 0.0% | 97.9% | No |
| Random search, budget 1 | 1 | 2.1% | 88.5% | 0% including failures | 51.7% | 3.2% | 97.9% | No |
| Random search, budget 2 | 2 | 4.2% | 98.5% | 0% including failures | 37.7% | 4.5% | 95.8% | No |
| Random search, budget 4 | 4 | 8.3% | 100% | 81.9% | 23.8% | 8.8% | 91.7% | No |
| Random search, budget 8 | 8 | 16.7% | 100% | 88.5% | 13.8% | 17.0% | 83.3% | No |
| Decision-tree ranker, top-2 | 2 | 4.2% | 100% | 92.6% | 8.2% | 16.7% | 95.8% | Yes |
| Decision-tree ranker, top-4 | 4 | 8.3% | 100% | 96.5% | 3.9% | 50.0% | 91.7% | Yes |
| Decision-tree ranker, top-8 | 8 | 16.7% | 100% | 99.2% | 0.8% | 66.7% | 83.3% | Yes |
| Exhaustive local oracle | 48 | 100% | 100% | 100% | 0.0% | 100% | 0.0% | Not applicable |

Random budget-1 and budget-2 have a zero aggregate geometric mean because at least one held-out trial sampled only invalid configurations. Their selection success rates were 88.5% and 98.5%, respectively. Random results use 100 seeds; the detailed trial file preserves every workload/seed outcome.

The decision-tree top-1 choice is the same as the best fixed development config. Its value begins at top-2, where leaf-specific ranking adds a second candidate. Top-2 reaches the Week 4 target on this split with only 2 of 48 configurations, while top-4 and top-8 reduce regret further.

## Kernel comparison

The fastest Triton configuration beat PyTorch SDPA on only 3 of the 16 supported workloads. The geometric-mean selector results therefore answer a within-Triton configuration-selection question. They do not show that the current Triton kernel is generally faster than PyTorch.

## Limitations

- Two workloads have exact median-latency ties, and several top-two gaps are only 1-3%.
- Oracle labels come from one benchmark run on one RTX 3090.
- The 10/6 workload split is small, and the depth-2 tree may be sensitive to the chosen split.
- Per-config compile measurements are cache-aware and depend on sweep order.
- Estimated selector tuning time scales the observed exhaustive wall time by configuration fraction; it is not a direct online-selector timing.

## Interpretation

The result supports continuing the research direction: a small ranked set clearly outperforms fixed, nearest-workload, and random baselines at comparable budgets on this held-out split. It remains preliminary evidence. Repeated CUDA measurements should verify whether near-tied oracle rankings are stable before the proposal claims generalization or a reliable 90-95% performance guarantee.
