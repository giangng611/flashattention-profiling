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

## Week 3 Plan

### Main Results And Observations

- To be filled after the full fused-attention experiments.

### Problems Or Limitations

- To be filled.

### Plan For Following Week

- To be filled.

