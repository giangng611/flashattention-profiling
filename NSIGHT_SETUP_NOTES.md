# Nsight Setup Notes

Qingchen suggested checking whether lower-level NVIDIA profiling tools are available. The first check showed:

```text
nsys: not available on PATH
ncu: not available on PATH
```

This does not necessarily mean the tools cannot be used. They may be installed somewhere else, or I may need to install them under my home directory.

## Check Current Availability

Run:

```bash
python src/check_cuda_profilers.py
which nsys
which ncu
find $HOME -type f \( -name nsys -o -name ncu \) 2>/dev/null | head
find /usr/local /opt -type f \( -name nsys -o -name ncu \) 2>/dev/null | head
```

If either tool appears in a directory like `.../bin`, temporarily add that directory to `PATH`:

```bash
export PATH=/path/to/nsight/bin:$PATH
```

Then rerun:

```bash
python src/check_cuda_profilers.py
```

## If I Need Local Install

If the tools are not already installed, I should try a user-local install instead of system install. I should not use `apt` unless the system administrators explicitly allow it.

The rough approach is:

```text
download NVIDIA Nsight Systems or Nsight Compute Linux installer
install/extract under $HOME/local or $HOME/nsight
add the tool's bin directory to PATH
rerun src/check_cuda_profilers.py
```

I should record:

- exact installer source or module path
- exact install directory
- `nsys --version`
- `ncu --version`
- any permission or unsupported-platform errors

## What I Would Use Each Tool For

`nsys` is useful first because it gives a timeline-level view:

- kernel launches
- CUDA API overhead
- synchronization
- whether runtime behavior differs between configurations

`ncu` is more detailed and kernel-specific:

- memory throughput
- achieved occupancy
- warp-level behavior
- tensor core usage
- cache and DRAM metrics

For this project, I should try `nsys` first, then `ncu` only for one or two representative kernels.

## Candidate Profiling Cases

Use the controlled head-dimension cases:

```text
seq_len = 4096
head_dim = 64 and 128
dtype = float16
non-causal
method = sdpa_auto or flash_forced
```

If Nsight works, profile a small run, not the full 200-trial benchmark.

Nsight Systems candidate:

```bash
nsys profile -o profiling/nsys_hd64_seq4096 python src/profile_attention.py --device cuda --seq-lengths 4096 --head-dim 64 --dtype float16 --methods flash_forced --warmup 3 --trials 5 --output-dir profiling/pytorch_hd64_seq4096_noncausal_fp16_nsys
nsys profile -o profiling/nsys_hd128_seq4096 python src/profile_attention.py --device cuda --seq-lengths 4096 --head-dim 128 --dtype float16 --methods flash_forced --warmup 3 --trials 5 --output-dir profiling/pytorch_hd128_seq4096_noncausal_fp16_nsys
```

Nsight Compute candidate:

```bash
ncu -o profiling/ncu_hd64_seq4096 python src/profile_attention.py --device cuda --seq-lengths 4096 --head-dim 64 --dtype float16 --methods flash_forced --warmup 2 --trials 3 --output-dir profiling/pytorch_hd64_seq4096_noncausal_fp16_ncu
ncu -o profiling/ncu_hd128_seq4096 python src/profile_attention.py --device cuda --seq-lengths 4096 --head-dim 128 --dtype float16 --methods flash_forced --warmup 2 --trials 3 --output-dir profiling/pytorch_hd128_seq4096_noncausal_fp16_ncu
```

If these tools are unavailable after reasonable effort, I should document that and continue with PyTorch Profiler plus the Triton microbenchmark.

## Current Status After Local Install Attempt

I successfully installed Nsight Compute under my home directory:

```text
$HOME/local/nsight/nsight_compute-linux-x86_64-2026.3.1.2-archive
```

The correct binary for this server is:

```text
target/linux-desktop-glibc_2_11_3-x64/ncu
```

The archive also contains an ARM binary:

```text
target/linux-desktop-t210-a64/ncu
```

That binary should not be used on this x86_64 server.

After adding the correct binary directory to `PATH`, `ncu` is callable. However, useful hardware-counter collection is currently blocked:

```text
ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters on the target device 0.
```

The default `ncu` set can attach to the process and produce a `.ncu-repz` file, but it does not provide the hardware-counter metrics I wanted for roofline or memory-throughput analysis.

Detailed notes are in:

```text
NSIGHT_NCU_RESULTS_REPORT.md
```
