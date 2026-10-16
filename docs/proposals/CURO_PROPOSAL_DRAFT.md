# CURO Proposal Working Draft

## Working title

Reducing Triton Attention Autotuning Cost with Workload-Aware Top-k Configuration Selection

## Research question

Can a lightweight workload-aware selector approach exhaustive-tuning performance on unseen dense-prefill attention workloads while benchmarking no more than 20-25% of a fixed Triton configuration space?

## Motivation

Triton kernels expose tile and launch parameters that can materially affect latency. Exhaustive autotuning can find a strong configuration for each workload, but it pays compilation and benchmarking cost whenever a new tuning key is encountered. A practical selector could use workload features to rank a small candidate set, benchmark only those candidates, and retain most of the performance of exhaustive search.

This project focuses on exact dense prefill attention. It does not attempt to solve decode, GQA, sparse attention, or end-to-end serving optimization in the first prototype.

## Preliminary evidence

I implemented a forward-only fused-attention Triton kernel and evaluated 48 tile/launch configurations on 16 supported dense-prefill workloads. The workload set varies sequence length, query-head count, head dimension, dtype, and causal mode. Ten workloads form the development set and six form a held-out set. Two true Qwen GQA shapes are recorded as unsupported references because the current kernel requires equal query and key/value head counts.

Three complete RTX 3090 sweeps produced 696 valid Triton measurements and 72 consistent shared-memory resource failures per run. All successfully executed configurations passed correctness. Deterministic fastest-config labels agreed across all three runs on only 4 of 16 workloads, or 5 of 16 after allowing exact latency ties, but the mean pairwise Spearman correlation of the full rankings was 0.836. This suggests that exact winner prediction is noisy while the broader configuration ordering remains informative.

Using the geometric mean of three per-run median latencies, a nearest-workload baseline reached 91.2% of the held-out local oracle with one candidate. A shallow decision-tree ranker reached 94.0% with two candidates, 96.0% with four, and 98.0% with eight. The two-candidate result measures 4.2% of the 48-config space and satisfies the initial 90% performance / 25% search-budget target on the current split.

These measurements are preliminary. They establish feasibility on one GPU and one workload split, not a general performance guarantee.

## Proposed approach

The prototype will separate the fixed configuration space from the selection policy.

1. Define a reproducible dense-prefill workload set and a valid Triton configuration space.
2. Collect repeated exhaustive measurements to provide local-oracle labels and quantify measurement noise.
3. Train or construct lightweight selectors only from development workloads.
4. Return a ranked top-k candidate set for each unseen workload.
5. Benchmark only that set and choose its fastest valid configuration.

The first selector will remain deliberately simple and interpretable. Candidate workload features are log2 sequence length, query-head count, head dimension, BF16 indicator, and causal indicator. The current depth-2 tree is a baseline; the project contribution must come from the evaluated top-k selection workflow and evidence, not from claiming that decision trees are new.

## Baselines

- tutorial-derived project default;
- best single configuration selected on the development set;
- nearest-development-workload reuse;
- random search without replacement at budgets 1, 2, 4, and 8;
- shallow decision-tree ranking;
- exhaustive local oracle over all 48 candidates.

The final study may add a small analytical rule or another lightweight ranker if the existing baselines leave a clear gap. A larger learned cost model is outside the initial scope unless the dataset grows enough to support it.

## Evaluation

Primary metrics:

- geometric-mean oracle-relative performance on unseen workloads;
- regret relative to the local exhaustive oracle;
- configurations compiled or measured and fraction of the search space;
- selection success when some candidates exceed hardware resources;
- direct or estimated tuning wall time.

Secondary diagnostics:

- exact-oracle and within-1% coverage;
- rank correlation and top-k overlap across repetitions;
- sensitivity to development/held-out split;
- break-even invocation count versus a fixed default.

The initial success target is at least 90-95% of local-oracle performance while measuring at most 20-25% of the candidates. This is an evaluation threshold, not an assumed result.

## Next experiments

1. Evaluate split sensitivity with multiple predefined development/held-out partitions or leave-group-out tests.
2. Measure the selector's direct online wall time instead of relying only on a scaled exhaustive-sweep estimate.
3. Repeat representative unstable workloads and investigate whether longer timing windows improve ranking stability.
4. Test whether the same selection policy transfers to another NVIDIA GPU if access is available.
5. Add GQA only after the dense-prefill prototype and claims are complete.

## Expected outcome

The project will deliver a reproducible attention configuration dataset, a compact selector prototype, and an evaluation of the tradeoff between tuning work and achieved latency. A useful negative result is also possible: if simple workload features do not transfer across splits or hardware, the study will identify where empirical autotuning remains necessary.

## Risks and limitations

- The current kernel is forward-only dense self-attention and is not a production FlashAttention implementation.
- The current evidence uses one RTX 3090 and a small synthetic/model-shaped workload set.
- Later repetitions reuse Triton disk-cache artifacts, so they do not measure cold compilation cost.
- Near-tied configurations and short kernel runtimes make exact top-1 labels sensitive to measurement noise.
- PyTorch SDPA is faster than the Triton local oracle on most current workloads; the research question concerns selection within this Triton kernel, not superiority over PyTorch.

## Related-work position

Triton autotune provides the exhaustive empirical reference. Ansor and TVM MetaSchedule motivate learned search and cost models at a broader compiler scope. TenSet and TLP show the value and data requirements of learned transfer. ROLLER and tritonBLAS motivate low-search, hardware-aware selection. The Anatomy of a Triton Attention Kernel makes a decision tree an important baseline rather than a novel contribution. This proposal therefore focuses on a small, measured top-k selector and an explicit tuning-cost/performance tradeoff for dense-prefill attention.
