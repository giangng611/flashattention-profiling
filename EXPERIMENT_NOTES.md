# Experiment Notes

These notes summarize the first local benchmark run on my Mac. This is a progress note for myself, not a final research result.

## What Was Tested

I compared two implementations of the same attention operation:

- `standard`: explicit scaled dot-product attention
- `sdpa`: PyTorch `scaled_dot_product_attention` on Apple MPS

This is not comparing two models. The input tensors `q`, `k`, and `v` have the same shapes and values for both implementations. The comparison is about implementation behavior, not model quality.

The standard implementation explicitly computes:

```text
scores = QK^T / sqrt(d)
probabilities = softmax(scores)
output = probabilities V
```

The important part is that `scores` has shape:

```text
batch x heads x sequence length x sequence length
```

So as sequence length increases, this intermediate tensor grows quadratically.

## Hardware And Software Environment

This run was done locally on a Mac using Apple MPS.

Environment:

- Python: 3.11.15
- PyTorch: 2.14.0
- device: Apple MPS
- CUDA: not available
- benchmark kind: MPS practice SDPA
- dtype: float16
- batch size: 1
- heads: 16
- head dimension: 64
- warmup iterations: 10
- timed trials: 30

Because this machine does not have an NVIDIA GPU, these results should not be described as CUDA FlashAttention results. They are useful for validating the benchmark pipeline and for practicing how to interpret performance data.

## Results

Raw outputs:

- `results/mps_attention_benchmark.csv`
- `results/mps_environment_metadata.json`

Plots:

- `plots/mps/latency_vs_sequence_length.png`
- `plots/mps/peak_memory_vs_sequence_length.png`
- `plots/mps/speedup_vs_sequence_length.png`

Summary of median latency:

| Sequence length | Standard median latency (ms) | SDPA median latency (ms) | Speedup over standard |
|---:|---:|---:|---:|
| 128 | 0.406 | 0.266 | 1.52x |
| 256 | 0.582 | 0.311 | 1.87x |
| 512 | 1.166 | 0.630 | 1.85x |
| 1024 | 4.206 | 1.647 | 2.55x |
| 2048 | 12.337 | 6.131 | 2.01x |
| 4096 | 48.348 | 22.931 | 2.11x |

Correctness checks passed for all sequence lengths. The maximum absolute difference between standard attention and SDPA decreased from about `1.46e-3` at sequence length 128 to about `3.66e-4` at sequence length 4096.

## Interpretation

The latency plot has two lines because it compares two implementations:

- `standard`
- `sdpa`

Both lines increase as sequence length increases. This is expected because attention becomes more expensive at longer sequence lengths.

The speedup plot has one line because speedup is already a ratio:

```text
standard median latency / sdpa median latency
```

So there is only one speedup value per sequence length.

In this local MPS run, SDPA is faster than the explicit standard implementation for every tested sequence length. The speedup ranges from about `1.52x` to `2.55x`.

The largest observed speedup in this run is at sequence length 1024. The speedup does not increase monotonically after that. It drops to about `2.01x` at 2048 and `2.11x` at 4096. This may be due to MPS backend behavior, memory allocation effects, or different kernel choices inside PyTorch, but I cannot conclude that from latency alone.

## Memory Plot Caveat

The memory plot currently shows only one visible line because the recorded MPS memory values for `standard` and `sdpa` are the same in this script.

This does not mean both implementations have the same true peak intermediate memory behavior. The current MPS measurement uses:

```text
torch.mps.current_allocated_memory()
```

This records allocated memory after each benchmark section, not detailed peak memory during each attention operation. Since temporary tensors can be allocated and freed inside the operation, this metric can miss the short-lived intermediate memory pressure that matters for attention.

On CUDA, PyTorch exposes better peak memory tracking through:

```text
torch.cuda.reset_peak_memory_stats()
torch.cuda.max_memory_allocated()
```

Therefore, the memory plot is mainly a placeholder for the workflow on Mac. The CUDA run should provide a more meaningful memory comparison.

## What Is Supported By Measurement

Measured facts from this run:

- The benchmark runs successfully on Apple MPS.
- The correctness checks passed for all tested sequence lengths.
- SDPA has lower median latency than the explicit standard implementation for all tested sequence lengths.
- Median latency increases with sequence length for both implementations.
- The speedup is not monotonic across sequence length.

## What Is Still A Hypothesis

The following explanations are plausible but not proven by this Mac run:

- SDPA may use a more optimized backend than the explicit Python-level composition of matmul, softmax, and matmul.
- The larger latency gap at some sequence lengths may be related to reduced intermediate materialization or better kernel behavior.
- The non-monotonic speedup may reflect backend-specific kernel selection or memory behavior.

These require profiler data or a CUDA run before making stronger claims.

## Next Steps

1. Run the same benchmark on a CUDA-capable NVIDIA GPU.
2. Compare CUDA `standard`, `sdpa_auto`, and `flash_forced`.
3. Recreate the latency, memory, and speedup plots from CUDA data.
4. Profile two representative sequence lengths, likely 512 and 4096.
5. Use profiler metrics to separate measured facts from explanations.

The next report should include CUDA results, exact environment and reproduction steps, the most significant performance pattern or limitation, and a preliminary research problem with a testable hypothesis. I added:

- `CUDA_SERVER_RUNBOOK.md`
- `NEXT_REPORT_TEMPLATE.md`
