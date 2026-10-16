# Week 4 Completion Audit

This audit maps Qingchen's Week 4 instructions to the preserved repository artifacts. It distinguishes completed prototype evidence from optional or later extensions.

## Completed core work

| Requested item | Status | Evidence |
|---|---|---|
| Correct the Qwen workload description | Complete | Runnable 12/12 rows are labeled dense Qwen-shaped proxies; actual 12/2 GQA rows remain unsupported references. |
| Preserve commands, seeds, environment, and results | Complete | CUDA runbook, three exhaustive sweeps, logs, summaries, and metadata are preserved. |
| Use absolute and relative correctness tolerances | Complete | Expanded sweeps use both tolerances and report zero correctness failures among successful executions. |
| Add BF16 and vary workload features | Complete | Sixteen supported workloads vary sequence length, head count, head dimension, dtype, and causal mode. |
| Label provenance and create a development/held-out split | Complete | Workload CSV records model-derived versus synthetic provenance and a 10/6 split. |
| Expand the configuration space | Complete | Forty-eight configurations vary `BLOCK_M`, `BLOCK_N`, `num_warps`, and `num_stages`. |
| Build a local oracle and record tuning work | Complete | Three exhaustive RTX 3090 sweeps preserve correctness, latency, cache/compile preparation, benchmark time, and wall time. |
| Compare required baselines | Complete | Tutorial-derived default, best fixed, nearest-workload reuse, random 1/2/4/8, shallow tree top-k, and exhaustive oracle are evaluated. |
| Evaluate held-out performance and seed variation | Complete | Six held-out workloads, 100 random-search seeds, three exhaustive repetitions, and 100 balanced workload splits are reported. |
| Build an initial top-k selector | Complete | The depth-2 tree ranks candidates; direct top-4 evaluation benchmarks only four candidates per workload. |
| Report direct tuning cost and break-even | Complete | Top-4 versus exhaustive warm-cache timing and per-workload break-even counts are preserved. |
| Draft the CURO proposal | Complete for mentor review | The first general-audience draft is in `docs/proposals/CURO_PROPOSAL_FIRST_DRAFT.md`; the technical working draft remains available separately. |

## Remaining work that does not block the first proposal

- Add a separate kernel-performance table containing PyTorch SDPA, forced FlashAttention where supported, the official Triton attention implementation, and this project's selected and oracle configurations. Current evidence only fully covers PyTorch SDPA and this project's Triton kernel.
- Evaluate a stronger learned ranker, such as a random forest or gradient-boosted model, if the dataset is expanded enough to justify it. The present decision tree is intentionally a transparent baseline.
- Consider Triton `prune_configs_by` integration only after the standalone selection workflow and claims are stable.

## Optional or later extensions

- Bayesian optimization or TPE.
- Controlled cold-cache compilation experiments.
- Transfer to another NVIDIA GPU.
- GQA, decode, and end-to-end inference evaluation.

The current evidence is sufficient for a first proposal because it demonstrates feasibility, measures the main tradeoff directly, and identifies the remaining experiments without claiming they are already complete.
