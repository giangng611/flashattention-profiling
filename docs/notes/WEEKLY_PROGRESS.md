# Weekly Progress

This file keeps a short record of what I did, what I observed, what did not work, and what I plan to do next.

## Week 1

### Main Results And Observations

- Set up the initial attention benchmarking workflow.
- Compared explicit standard attention against PyTorch SDPA/FlashAttention-style paths.
- Confirmed that standard attention becomes expensive at longer sequence lengths because it materializes large intermediate attention-score tensors.

### Problems Or Limitations

- Local Mac environment does not have CUDA or an NVIDIA GPU.
- MPS runs are useful for workflow testing, but not valid FlashAttention reproduction.

### Plan For Following Week

- Move experiments to the CUDA server.
- Collect CUDA environment metadata and baseline results.

## Week 2

### Main Results And Observations

- Ran CUDA baseline on RTX 3090.
- Verified that PyTorch SDPA auto and forced FlashAttention use the FlashAttention kernel path for tested cases.
- Ran boundary sweeps across sequence length, head dimension, dtype, and causal mode.
- Repeated the strongest anomaly and found that it did not reproduce.
- Added normalized metrics for head-dimension experiments.
- Installed and tested NVIDIA profiling tools:
  - `ncu` installed, but hardware counters are blocked by `ERR_NVGPUCTRPERM`.
  - `nsys` installed and can generate `.nsys-rep` timeline reports.
- Built a first Triton QK^T microbenchmark.

### Problems Or Limitations

- `ncu` cannot collect useful hardware-counter metrics without administrator support.
- The first Triton QK kernel is much slower than PyTorch matmul and is not yet a useful performance model for full FlashAttention.
- QK^T alone does not include online softmax, masking, or V accumulation.

### Plan For Following Week

- Read the week 3 papers and update the paper review table.
- Move from QK^T microbenchmark to a full fused-attention implementation using the official Triton tutorial as the starting point.
- Build a small table of real LLM workload shapes.
- Evaluate correctness and latency against PyTorch SDPA for those shapes.

## Week 3 Research Tracker

### Question investigated

- Do different dense prefill workloads prefer different Triton tile and launch configurations?

### What I implemented

- Built a forward-only fused-attention Triton kernel with correctness checks against PyTorch SDPA.
- Swept 16 combinations of `BLOCK_M`, `BLOCK_N`, `num_warps`, and `num_stages` on four supported workloads.

### Main findings

- Each supported workload selected a different fastest configuration within the tested space.
- The best Triton configuration beat PyTorch SDPA on two short/medium cases but remained slower on the 2048-token case.
- These results are preliminary evidence that configuration choice is workload-dependent; they do not show that a selector generalizes.

### Unsuccessful attempts / limitations

- The kernel supports dense self-attention only; decode and GQA are unsupported.
- The experiment contains only four supported FP16 workloads and 16 configurations.
- The original `12 Q heads / 12 KV heads` workloads were incorrectly described as Qwen-like and are now labeled dense Qwen-shaped proxies.
- The raw detailed CSV, summary CSV, and metadata JSON were not found locally during the Week 4 audit and must be recovered from the CUDA server if available.

### Interpretation

- The result is sufficient to motivate a larger experiment, but not to claim that autotuning is expensive enough or that a selector will work.

### Next step

- Start the CURO proposal outline and expand the dense-prefill experiment to test tuning cost and held-out selection.

## Week 4 Research Tracker

### Question investigated

- Can a lightweight selector approach exhaustive-tuning performance on unseen attention workloads while testing far fewer configurations?

### What I implemented

- Corrected the Week 3 workload taxonomy and added explicit model-derived versus synthetic and supported versus unsupported metadata.
- Preserved a corrected-label snapshot of the Week 3 workload set and documented the missing raw artifacts without reconstructing them.
- Expanded the paper table using the comparison fields requested for configuration-selection work.
- Defined 16 supported dense-prefill workloads with development/held-out splits, FP16/BF16, causal/non-causal modes, and varied sequence length, head count, and head dimension.
- Defined 48 candidate Triton configurations and added separate compile/cache-preparation and steady-state benchmark timing.
- Replaced the absolute-only correctness threshold with combined absolute and relative tolerances.
- Ran the 48-configuration sweep on an RTX 3090 and preserved the detailed results, summary, and environment metadata.
- Added offline held-out evaluation for the tutorial-derived project default, best fixed development config, nearest-workload reuse, random search at budgets 1/2/4/8, a shallow decision-tree ranker, and the exhaustive local oracle.
- Repeated the complete CUDA sweep twice with seeds 1 and 2, preserving detailed results, summaries, logs, and metadata.
- Added reproducible ranking-stability analysis and reran selector baselines using the geometric mean of three per-run median latencies.

### Main findings

- All 696 successfully executed Triton measurements passed correctness; 72 attempts failed because eight `BLOCK_N=128, num_stages=3` configurations exceeded shared-memory capacity on nine `head_dim=128` workloads.
- The 16 supported workloads selected nine marked oracle configurations. The full sweep took 347.6 seconds across the workload set, although compile/cache timing is order- and cache-dependent.
- The best fixed development configuration reached 84.1% geometric-mean oracle-relative performance on held-out workloads. The decision-tree ranker reached 92.6% with top-2, 96.5% with top-4, and 99.2% with top-8 while considering at most 16.7% of the 48 candidates.
- The Triton local oracle beat PyTorch SDPA on only 3 of 16 workloads, so the result supports configuration selection but not a claim that this kernel is generally faster than PyTorch.
- Deterministic top-1 configs agreed across all three runs on only 4 of 16 workloads; accounting for exact ties raises common-oracle agreement to 5 of 16. The full rankings had mean pairwise Spearman correlation 0.836, so exact winner labels are noisy even when the broader ranking is fairly consistent.
- On repeated aggregate latency, nearest-workload reuse reached 91.2% of oracle with one config. The decision-tree ranker reached 94.0% with top-2, 96.0% with top-4, and 98.0% with top-8.

### Unsuccessful attempts / limitations

- Two workloads have exact median-latency ties, and several top configurations differ by only 1-3%; repeated CUDA runs are needed to test oracle-label stability.
- The dataset has only 10 development and 6 held-out workloads on one RTX 3090. The shallow tree result is preliminary and may be sensitive to the split.
- Compile preparation is cache-aware rather than a controlled cold-cache measurement, and failed compile attempts have no per-config timing.
- The two repetitions completed in about 24.2 seconds each because they reused the Triton disk cache; they cannot be used as cold compile-cost measurements.
- One development workload has a 28.8% worst-run regret for the aggregate oracle, so aggregation does not eliminate every unstable case.

### Interpretation

- A small ranked candidate set remains close to the measured local oracle after repeated measurement. Exact top-1 prediction is not a reliable research target; oracle-relative performance from a small top-k set is better supported. This justifies a CURO proposal while keeping the claims limited to the current GPU, workload split, and candidate space.

### Next step

- Draft the CURO proposal around top-k configuration selection, then test split sensitivity and direct online selection cost before claiming generalization.
