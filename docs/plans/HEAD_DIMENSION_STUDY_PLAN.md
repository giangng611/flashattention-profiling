# Head Dimension Study Plan

Qingchen's feedback is that I should move from observing a latency difference to understanding why it happens. The next target is the latency jump around head dimension 64 to 128 at sequence length 4096.

## Question

```text
When head dimension increases, is the latency increase mostly explained by more arithmetic work, or does FlashAttention lose kernel efficiency?
```

## Why This Matters

The boundary sweep showed that PyTorch SDPA auto and forced FlashAttention usually behave the same. The earlier auto-vs-forced anomaly did not reproduce. The more interesting signal is now inside FlashAttention itself:

```text
seq_len = 4096
head_dim = 64 -> 128
non-causal FP16
```

The first sweep showed a large latency jump there. To interpret it correctly, I need normalized metrics, not just latency.

## Metrics To Report

For each configuration, I will report:

- median latency
- mean latency
- latency variability
- peak allocated memory
- estimated attention FLOPs
- median TFLOP/s
- median microseconds per output token
- median nanoseconds per attention pair

The normalized metrics are approximate, but they are useful for separating two explanations:

```text
latency increases because the operation has more work
```

versus:

```text
latency increases more than expected because the kernel is less efficient
```

## Controlled Measurement Command

Run this on the CUDA server after pulling the latest repo:

```bash
git pull
python src/check_cuda_profilers.py
python src/sweep_attention_boundaries.py --device cuda --seq-lengths 4096 --head-dims 32 64 128 256 --dtypes float16 --causal-values false true --warmup 30 --trials 200 --output results/cuda_head_dim_controlled_seq4096_fp16.csv --metadata-output results/cuda_head_dim_controlled_seq4096_fp16_metadata.json
```

This repeats the head-dimension sweep with more stable timing and adds normalized metrics directly to the CSV.

## Profiler Follow-Up

After the controlled sweep, profile the two key non-causal cases:

```bash
python src/profile_attention.py --device cuda --seq-lengths 4096 --head-dim 64 --dtype float16 --methods sdpa_auto flash_forced --warmup 5 --trials 10 --output-dir profiling/pytorch_hd64_seq4096_noncausal_fp16
python src/profile_attention.py --device cuda --seq-lengths 4096 --head-dim 128 --dtype float16 --methods sdpa_auto flash_forced --warmup 5 --trials 10 --output-dir profiling/pytorch_hd128_seq4096_noncausal_fp16
```

If `nsys` or `ncu` is available, I should then run one lower-level profiler pass on the same two configurations. If neither tool is available, I should document that and continue with PyTorch Profiler for now.

## What Would Count As Evidence

If latency roughly doubles from head dimension 64 to 128 and TFLOP/s stays similar, then the transition is probably mostly increased work.

If latency grows much more than the estimated FLOPs and TFLOP/s drops sharply, then the transition likely reflects lower kernel efficiency, less favorable tiling, memory behavior, occupancy, or another hardware-level effect.

## Current Hypothesis

```text
PyTorch SDPA auto is not the main issue for the tested shapes. The next useful research question is whether FlashAttention efficiency changes sharply across head dimension and masking regimes, and whether this can be explained by kernel-level resource usage rather than only FLOP count.
```
