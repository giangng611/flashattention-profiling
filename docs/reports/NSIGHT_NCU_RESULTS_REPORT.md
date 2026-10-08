# Nsight Compute Results Report

This report records the first attempt to install and use NVIDIA Nsight Compute on the CUDA server.

## Install Status

Initial check:

```text
nsys: not available on PATH
ncu: not available on PATH
```

I downloaded and extracted Nsight Compute under my home directory:

```text
$HOME/local/nsight/nsight_compute-linux-x86_64-2026.3.1.2-archive
```

The archive contained multiple target binaries. The first automatic path accidentally selected the ARM binary:

```text
target/linux-desktop-t210-a64/ncu
```

That failed with:

```text
Exec format error
```

The correct x86_64 binary is:

```text
target/linux-desktop-glibc_2_11_3-x64/ncu
```

After adding that directory to `PATH`, `src/check_cuda_profilers.py` reported:

```text
ncu: available
nsys: not available
```

## Roofline Attempt

I first tried:

```bash
ncu --target-processes all --set roofline -o profiling/ncu_qk_smoke \
python src/benchmark_triton_qk.py --seq-len 512 --head-dims 64 --block-sizes 32 --warmup 2 --trials 3 --output results/ncu_qk_smoke.csv --metadata-output results/ncu_qk_smoke_metadata.json
```

This failed to collect the requested hardware counters:

```text
ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters on the target device 0.
```

This means the server currently blocks normal users from accessing NVIDIA GPU performance counters. I cannot fix that without administrator support.

## Default Attempt

I then tried:

```bash
ncu --target-processes all --set default -o profiling/ncu_qk_smoke_default \
python src/benchmark_triton_qk.py --seq-len 512 --head-dims 64 --block-sizes 32 --warmup 2 --trials 3 --output results/ncu_qk_smoke_default.csv --metadata-output results/ncu_qk_smoke_default_metadata.json
```

This produced a report file:

```text
profiling/ncu_qk_smoke_default.ncu-repz
```

However, it also printed:

```text
No metrics to collect found in sections.
```

So this run shows that `ncu` can attach to the process and enumerate kernels, but it does not yet provide useful hardware-counter metrics for this project.

The latency numbers from runs under `ncu` should not be interpreted as benchmark results because profiler overhead is large.

## Current Interpretation

The useful conclusion is:

```text
Nsight Compute was installed successfully under my home directory, but useful hardware-counter profiling is currently blocked or unavailable under the server's user permissions/configuration.
```

This is better than the earlier status. I no longer need to say simply that `ncu` is unavailable. The precise status is:

```text
ncu installed and callable, but hardware-counter collection is blocked by ERR_NVGPUCTRPERM.
```

`nsys` has not been installed yet.

## Next Step

I should report this limitation to Qingchen and ask whether it is worth trying to get administrator support for profiler permissions. In the meantime, I can continue with:

- PyTorch Profiler
- CUDA-event timing
- normalized metrics
- Triton microbenchmarks

Raw smoke outputs:

```text
results/ncu_qk_smoke.csv
results/ncu_qk_smoke_metadata.json
```
