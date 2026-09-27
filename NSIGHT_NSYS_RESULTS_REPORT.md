# Nsight Systems Results Report

This report records the first successful Nsight Systems smoke test on the CUDA server.

## Install Status

I installed Nsight Systems under my home directory:

```text
$HOME/local/nsight/nsight_systems-linux-x86_64-2026.3.2.476-archive
```

The executable is:

```text
target-linux-x64/nsys
```

Version:

```text
NVIDIA Nsight Systems version 2026.3.2.476-263238834031v0
```

After adding the binary directory to `PATH`, `src/check_cuda_profilers.py` detected both tools:

```text
nsys: available
ncu: available
```

## Smoke Command

```bash
nsys profile -o profiling/nsys_qk_smoke \
python src/benchmark_triton_qk.py --seq-len 512 --head-dims 64 --block-sizes 32 --warmup 2 --trials 3 --output results/nsys_qk_smoke.csv --metadata-output results/nsys_qk_smoke_metadata.json
```

Nsight Systems produced:

```text
profiling/nsys_qk_smoke.nsys-rep
```

The report file size was about:

```text
353 KB
```

## Warning

The run printed:

```text
The version of the system or its configuration does not allow enabling CPU profiling:
- CPU IP/backtrace sampling will be disabled.
- CPU context switch tracing will be disabled.
```

This is not a CUDA profiling failure. It means CPU sampling and CPU context-switch tracing are unavailable in the current server configuration. The CUDA timeline report was still generated.

## Interpretation

Nsight Systems is usable for this project as a timeline-level profiler. It can generate `.nsys-rep` files without the hardware-counter permission issue that blocked Nsight Compute roofline profiling.

The latency numbers from this smoke run should not be interpreted as benchmark results because the run was under profiler overhead and used a very small test case.

Useful status:

```text
nsys: installed and able to generate timeline reports
ncu: installed, but useful hardware-counter collection is blocked by ERR_NVGPUCTRPERM
```

## Next Step

Use `nsys` on one or two representative cases only:

```bash
nsys profile -o profiling/nsys_triton_qk_seq4096_hd64 \
python src/benchmark_triton_qk.py --seq-len 4096 --head-dims 64 --block-sizes 32 --warmup 5 --trials 10 --output results/nsys_triton_qk_seq4096_hd64.csv --metadata-output results/nsys_triton_qk_seq4096_hd64_metadata.json
```

I should use the `.nsys-rep` file for timeline evidence, not for standalone latency conclusions.

Raw smoke outputs:

```text
results/nsys_qk_smoke.csv
results/nsys_qk_smoke_metadata.json
```
