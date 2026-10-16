# Week 4 Research Update

## Question investigated

Can a lightweight workload-aware selector approach exhaustive-tuning performance on unseen dense-prefill attention workloads while testing far fewer configurations?

## What I implemented

- Corrected the Qwen workload taxonomy: runnable 12-query-head/12-KV-head cases are dense proxies, while actual 12/2 GQA shapes remain unsupported references.
- Expanded the study to 16 supported FP16/BF16 workloads, a 10/6 development/held-out split, and 48 Triton configurations.
- Completed three exhaustive RTX 3090 sweeps with separate cache/compile-preparation and benchmark timing.
- Evaluated the required baselines, 100 balanced workload splits, and direct warm-cache tree top-4 versus exhaustive tuning.

## Main findings

- Every successful Triton execution passed correctness; each exhaustive sweep produced 696 valid measurements and 72 repeatable resource failures.
- Tree top-4 reached 96.0% geometric-mean oracle-relative performance on the six held-out workloads and remained above 90% on all 100 balanced splits.
- In the direct run, top-4 achieved 96.97% of exhaustive performance while reducing tuning wall time from 12.600 seconds to 1.987 seconds, an 84.23% reduction or 6.34x speedup.
- Direct top-4 attempted 24 candidates with no failures; exhaustive search attempted 288, including 32 expected resource failures.

## Unsuccessful attempts / limitations

- Exact fastest-configuration labels remain noisy, so exact top-1 prediction is not a reliable primary target.
- Results cover one RTX 3090, one forward-only dense-attention kernel, 16 workloads, and 48 configurations.
- Direct timing retains the Triton disk cache; official-Triton comparison and a stronger learned ranker remain future work.

## Interpretation

A small ranked candidate set appears to provide a useful tuning-time/performance tradeoff within the tested space. The defensible current claim is limited to warm-cache selection on the present GPU and workload set; it is not a universal guarantee or a claim that this Triton kernel is generally faster than PyTorch SDPA.

## Next step

- Review the first CURO proposal draft with Qingchen and reconfirm Spring 2027 CURO supervision with Dr. Wenwen Wang.
- After feedback, prioritize timing stability, a separate kernel comparison, and cross-GPU evaluation if available.

Repository: <https://github.com/giangng611/flashattention-profiling>
