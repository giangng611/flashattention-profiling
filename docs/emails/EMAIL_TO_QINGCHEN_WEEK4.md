# Email to Qingchen

**Subject:** Week 4 update and first CURO proposal draft

Dear Qingchen,

Thank you again for the detailed Week 4 guidance. I completed the expanded dense-prefill study and prepared a first CURO proposal draft.

The main updates are:

- I corrected the Qwen workload labels and kept the actual 12-query-head/2-KV-head GQA shapes as unsupported references.
- I expanded the experiment to 16 supported FP16/BF16 workloads and 48 Triton configurations, with a 10/6 development/held-out split.
- Three complete RTX 3090 sweeps produced 696 valid Triton measurements per run, with no correctness failures among successful executions.
- I evaluated the fixed, nearest-workload, random-search, decision-tree top-k, and exhaustive-oracle baselines, and tested the selector across 100 balanced splits.
- In direct warm-cache measurement, tree top-4 retained 96.97% of exhaustive performance while reducing tuning wall time from 12.600 seconds to 1.987 seconds, an 84.23% reduction.
- The main limitations are the single GPU, small workload set, forward-only dense-attention scope, and retained Triton disk cache.

I attached the short Week 4 update and the proposal draft as separate files. I have left the faculty-supervisor field blank until I reconfirm the Spring 2027 CURO arrangement with Dr. Wenwen Wang.

Could you please review whether the research framing, current evidence, and proposed next experiments are appropriate for the first draft? After incorporating your feedback, I plan to share the revised proposal with Dr. Wang, as you suggested.

The repository is available at: <https://github.com/giangng611/flashattention-profiling>

Best,

Giang
