# PyTorch Profiler Notes

The CUDA server currently has PyTorch CUDA support, but `nsys` and `ncu` are not available in the shell path. To keep moving without admin privileges, I added a PyTorch Profiler workflow.

## Purpose

The PyTorch Profiler pass is meant to answer a narrower question than Nsight:

```text
Which PyTorch/CUDA operators dominate each attention implementation?
```

It is useful for checking:

- whether standard attention launches separate matmul, softmax, and matmul work
- whether SDPA and forced FlashAttention appear as fused or specialized attention kernels
- how CPU time and CUDA time are distributed across operators
- whether the 512 and 4096 cases look qualitatively different

It does not replace Nsight Compute for low-level metrics such as DRAM bytes, memory throughput, compute throughput, or occupancy.

## Command

Run this on the CUDA server:

```bash
python src/profile_attention.py --device cuda --seq-lengths 512 4096 --warmup 5 --trials 10 --output-dir profiling/pytorch
```

Expected outputs:

```text
profiling/pytorch/metadata.json
profiling/pytorch/summary.csv
profiling/pytorch/seq512_standard.txt
profiling/pytorch/seq512_standard.events.csv
profiling/pytorch/seq512_standard.trace.json
profiling/pytorch/seq512_sdpa_auto.txt
profiling/pytorch/seq512_sdpa_auto.events.csv
profiling/pytorch/seq512_sdpa_auto.trace.json
profiling/pytorch/seq512_flash_forced.txt
profiling/pytorch/seq512_flash_forced.events.csv
profiling/pytorch/seq512_flash_forced.trace.json
profiling/pytorch/seq4096_*.txt
profiling/pytorch/seq4096_*.events.csv
profiling/pytorch/seq4096_*.trace.json
```

## How To Read The Output

Start with the `.txt` summary files. They are sorted by CUDA self time on CUDA devices.

Things to look for:

- For `standard`, I expect to see separate matrix multiplication and softmax-related operations.
- For `sdpa_auto`, I want to see which scaled-dot-product attention path PyTorch chose.
- For `flash_forced`, I want to confirm whether a flash attention kernel appears.
- Compare seq 512 and seq 4096 to see whether the dominant operators change.

The `.events.csv` files are machine-readable versions of the profiler tables.

The `.trace.json` files are Chrome trace files. They can be opened with a trace viewer if needed, but the text summaries are the fastest first pass.

## Interpretation Rule

This profiler can support statements like:

```text
The standard implementation spends most CUDA time in separate matmul and softmax operations.
```

It should not be used alone for statements like:

```text
FlashAttention reduces DRAM traffic by a specific number of bytes.
```

That requires Nsight Compute or another lower-level profiler.
