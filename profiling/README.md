# Profiling Plan

Do this only after the basic benchmark is correct and produces CSV results.

## Representative Cases

Start with two sequence lengths:

- 512: a smaller case where overheads may matter
- 4096: a larger case where attention memory behavior should be more visible

Keep the same baseline settings as the benchmark:

- batch size: 1
- heads: 16
- head dimension: 64
- dtype: FP16
- causal: false unless intentionally changed

## What To Look For

For the first discussion with Qingchen, do not collect dozens of metrics. Focus on a small set that connects to the architectural story.

- kernel duration: how much time the GPU spends inside the attention kernels
- GPU DRAM bytes read and written: how much data moves through high bandwidth memory
- DRAM throughput: how close the kernel gets to available memory bandwidth
- compute throughput: whether arithmetic units are heavily used
- occupancy: whether enough work is active on the GPU to hide latency
- kernel launches: whether launch overhead or many small kernels are visible

## Interpretation Discipline

Separate measured facts from hypotheses.

Example measured fact:

```text
At sequence length 4096, standard attention had higher peak allocated memory and higher median latency than flash attention.
```

Example hypothesis:

```text
The larger gap is likely related to avoiding materialization of the full attention matrix and reducing HBM traffic.
```

The hypothesis should be supported later with profiler metrics such as DRAM bytes read/write or memory throughput.

## Suggested Commands

These commands are placeholders because exact profiler paths differ by machine.

Nsight Systems can give a timeline:

```bash
nsys profile -o profiling/seq4096_flash python3 src/benchmark_attention.py --seq-lengths 4096 --trials 20
```

Nsight Compute can inspect kernel-level metrics:

```bash
ncu -o profiling/seq4096_flash python3 src/benchmark_attention.py --seq-lengths 4096 --trials 5
```

Use profiler results to explain the benchmark, not as a separate pile of numbers.

For the UGA CUDA server workflow, see `CUDA_SERVER_RUNBOOK.md`.
