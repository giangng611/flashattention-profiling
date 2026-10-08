# Triton Experiment Plan

The controlled head-dimension sweep suggests that the latency increase from head_dim 64 to 128 is mostly explained by increased arithmetic work, not by a clear FlashAttention efficiency collapse.

This means the Triton step should not start by trying to "fix" a PyTorch problem. Instead, the goal should be educational and diagnostic:

```text
use a small Triton kernel to understand how tiled GPU work scales with head dimension
```

## First Question

```text
Can a simple tiled Triton kernel reproduce the same broad head_dim scaling trend seen in PyTorch SDPA/FlashAttention?
```

## Scope

Start with a small QK-style matmul microbenchmark, not full FlashAttention.

For each head:

```text
Q: [seq_len, head_dim]
K: [seq_len, head_dim]
S = QK^T
```

This isolates the first large matrix multiplication inside attention. It does not include online softmax, masking, or the P/V multiplication, so it is not a FlashAttention replacement. It is a controlled stepping stone.

## Why This Scope

Full FlashAttention requires:

- tiled QK computation
- numerically stable online softmax
- causal masking support
- accumulation into V
- careful SRAM/register use
- backward pass if training is considered

That is too much for the first Triton step.

A QK-style tiled matmul is smaller and still directly connected to the current question:

```text
how does tiled GPU work scale as head_dim changes?
```

## Initial CUDA Server Steps

Check whether Triton is installed:

```bash
python -c "import triton; print(triton.__version__)"
```

If Triton is missing, install it in the project virtual environment:

```bash
python -m pip install triton
```

Then create a small benchmark that compares:

- PyTorch `torch.matmul(q, k.transpose(-2, -1))`
- a Triton tiled QK matmul kernel

The first version is implemented in:

```text
src/benchmark_triton_qk.py
```

Initial shapes:

```text
seq_len: 4096
head_dim: 32, 64, 128, 256
batch size: 1
heads: start with 1 head, then expand to 16 heads
dtype: float16
```

## Metrics

Use the same normalized metrics style as the controlled head-dim sweep:

- median latency
- estimated FLOPs
- median TFLOP/s
- latency per token
- latency per QK pair

## First Run Command

```bash
python src/benchmark_triton_qk.py --seq-len 4096 --head-dims 32 64 128 256 --warmup 20 --trials 100 --output results/triton_qk_benchmark.csv --metadata-output results/triton_qk_benchmark_metadata.json
```

If this is too slow or memory-heavy, reduce the sequence length first:

```bash
python src/benchmark_triton_qk.py --seq-len 2048 --head-dims 32 64 128 256 --warmup 10 --trials 50 --output results/triton_qk_benchmark_seq2048.csv --metadata-output results/triton_qk_benchmark_seq2048_metadata.json
```

After the first smoke test, tune tile sizes before interpreting performance:

```bash
python src/benchmark_triton_qk.py --seq-len 2048 --head-dims 32 64 128 --block-sizes 16 32 64 --warmup 10 --trials 30 --output results/triton_qk_tile_sweep_seq2048.csv --metadata-output results/triton_qk_tile_sweep_seq2048_metadata.json
```

## Expected Value

If Triton matmul shows similar normalized scaling, then the PyTorch/FlashAttention head_dim trend is likely a general tiled-matmul arithmetic-intensity effect.

If Triton matmul shows a very different scaling pattern, then the attention-specific pieces such as softmax, masking, launch configuration, or memory layout may be more important.

## Research Framing

The framing for now is:

```text
I am using Triton as a controlled microscope for one piece of attention, not as an immediate attempt to outperform FlashAttention.
```

That is a better match for the current stage of the project.
