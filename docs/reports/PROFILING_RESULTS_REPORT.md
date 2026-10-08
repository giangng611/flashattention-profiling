# PyTorch Profiler Results Report

This report summarizes the first PyTorch Profiler pass for representative CUDA cases.

## Why I Ran This

`nsys` and `ncu` were not available in the CUDA server shell path, so I used PyTorch Profiler as a first profiling step.

The goal was to inspect operator-level behavior for two representative sequence lengths:

- 512
- 4096

For each sequence length, I profiled:

- `standard`
- `sdpa_auto`
- `flash_forced`

The profiler ran 10 trials per method after warmup.

## Main Finding

The most important finding is that `sdpa_auto` and `flash_forced` both used the PyTorch FlashAttention path in these configurations.

Both showed:

```text
aten::_scaled_dot_product_flash_attention
aten::_flash_attention_forward
pytorch_flash::flash_fwd_kernel
```

This means my earlier interpretation needs to be refined. The small latency differences between `sdpa_auto` and `flash_forced` are probably not caused by PyTorch selecting a fundamentally different CUDA backend. They are more likely due to measurement noise, wrapper/context overhead, or small dispatch differences.

The stronger supported result is:

```text
standard attention uses separate matmul, scaling, softmax, and matmul operations, while SDPA auto and forced FlashAttention both use the fused FlashAttention kernel.
```

## Sequence Length 512

### Standard Attention

Dominant CUDA work:

- `aten::bmm`
- `aten::_softmax`
- `aten::mul`
- GEMM kernels
- softmax kernel
- elementwise multiply kernel

Total self CUDA time:

```text
817.423 us across 10 trials
about 81.7 us per trial
```

Memory shown by profiler:

- `aten::bmm`: 90.00 MB CUDA memory across 20 calls
- `aten::_softmax`: 80.00 MB across 10 calls
- `aten::mul`: 80.00 MB across 10 calls

Interpretation:

At sequence length 512, standard attention already shows the expected pattern of separate kernels and intermediate tensors, but the absolute cost is still small.

### SDPA Auto

Dominant CUDA work:

```text
pytorch_flash::flash_fwd_kernel
```

Total self CUDA time:

```text
263.151 us across 10 trials
about 26.3 us per trial
```

The profiler shows:

```text
aten::_scaled_dot_product_flash_attention
aten::_flash_attention_forward
```

### Flash Forced

Dominant CUDA work:

```text
pytorch_flash::flash_fwd_kernel
```

Total self CUDA time:

```text
263.725 us across 10 trials
about 26.4 us per trial
```

Interpretation:

At sequence length 512, `sdpa_auto` and `flash_forced` use essentially the same CUDA kernel and have nearly identical CUDA kernel time in the profiler. The benchmark latency gap between them is likely not due to different GPU kernels.

## Sequence Length 4096

### Standard Attention

Dominant CUDA work:

- `aten::_softmax`
- `aten::bmm`
- `aten::mul`
- GEMM kernels
- softmax kernel
- elementwise multiply kernel

Total self CUDA time:

```text
50.228 ms across 10 trials
about 5.023 ms per trial
```

Breakdown:

- softmax: 23.610 ms across 10 trials
- bmm: 13.877 ms across 20 calls
- mul: 12.741 ms across 10 trials

Memory shown by profiler:

- `aten::_softmax`: 5.00 GB across 10 calls
- `aten::mul`: 5.00 GB across 10 calls
- `aten::bmm`: 5.08 GB across 20 calls

Interpretation:

This strongly supports the memory-pressure explanation for standard attention. At sequence length 4096, the explicit implementation materializes and transforms large attention-score-shaped tensors. The cost of softmax and elementwise scaling becomes very visible.

### SDPA Auto

Dominant CUDA work:

```text
pytorch_flash::flash_fwd_kernel
```

Total self CUDA time:

```text
11.491 ms across 10 trials
about 1.149 ms per trial
```

The profiler shows the FlashAttention path:

```text
aten::_scaled_dot_product_flash_attention
aten::_flash_attention_forward
```

### Flash Forced

Dominant CUDA work:

```text
pytorch_flash::flash_fwd_kernel
```

Total self CUDA time:

```text
11.523 ms across 10 trials
about 1.152 ms per trial
```

Interpretation:

At sequence length 4096, `sdpa_auto` and `flash_forced` again use the same FlashAttention kernel and have essentially the same CUDA time. This supports the view that PyTorch SDPA auto is already selecting FlashAttention for this workload.

## Updated Interpretation

The original CUDA benchmark showed:

- `sdpa_auto` faster than `flash_forced` from sequence length 128 to 2048
- `flash_forced` slightly faster at 4096

After profiling, I should be careful not to over-interpret that as a backend crossover. For the profiled cases, both `sdpa_auto` and `flash_forced` used the FlashAttention kernel.

The better interpretation is:

```text
For this RTX 3090 FP16 workload, PyTorch SDPA auto already selects FlashAttention at the profiled sequence lengths. The main performance difference is between explicit standard attention and fused FlashAttention, not between SDPA auto and forced FlashAttention.
```

## What Is Supported By Measurement

Measured facts:

- `standard` launches separate matmul, softmax, scaling, and matmul-related work.
- `standard` has much higher CUDA memory activity in profiler summaries, especially at sequence length 4096.
- `sdpa_auto` uses the FlashAttention path at sequence lengths 512 and 4096.
- `flash_forced` uses the same FlashAttention path at sequence lengths 512 and 4096.
- At sequence length 4096, standard attention spends about 50.2 ms total CUDA time across 10 trials, while SDPA auto and forced FlashAttention each spend about 11.5 ms.

## Revised Hypothesis

The preliminary hypothesis should be narrowed.

Earlier hypothesis:

```text
The best backend changes between SDPA auto and forced FlashAttention depending on sequence length.
```

This is not strongly supported by the profiler for the cases tested.

Revised hypothesis:

```text
For FP16 attention on an RTX 3090 with head dimension 64, PyTorch SDPA auto selects FlashAttention for the tested sequence lengths. The main performance benefit comes from using the fused FlashAttention kernel instead of explicit standard attention. The next useful question is where this automatic selection stops being reliable, such as different head dimensions, dtypes, causal settings, or sequence lengths.
```

## Next Experiments

I should now look for a real boundary condition where backend selection or kernel behavior changes.

Possible next sweeps:

1. Vary head dimension:
   - 32
   - 64
   - 128
   - 256

2. Compare causal vs non-causal attention.

3. Compare FP16 and BF16 if supported.

4. Add longer sequence lengths if memory allows:
   - 8192
   - possibly 16384 for SDPA/FlashAttention only

5. Run the boundary sweep in `BOUNDARY_SWEEP_PLAN.md`, then start Triton tutorials after identifying which operation or backend boundary is worth studying.

The most natural research direction remains:

```text
When does automatic attention backend selection work well, and when does it fail or leave performance on the table?
```
