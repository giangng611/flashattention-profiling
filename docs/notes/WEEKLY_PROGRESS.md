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

### Main findings

- In progress.

### Unsuccessful attempts / limitations

- No expanded sweep or selector result exists yet.

### Interpretation

- In progress.

### Next step

- Add absolute and relative correctness tolerances, BF16 cases, and separate compilation/benchmark timing before the expanded sweep.
