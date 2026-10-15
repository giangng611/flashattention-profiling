# Paper Review Table

## Week 4: Configuration Selection

This table records the comparison fields requested for the Week 4 direction: search space, selection method, tuning cost, workloads, and hardware. Reported tuning-cost claims come from the linked paper or official documentation; they are not assumed to transfer directly to this attention prototype.

| Paper/system | Search space | Selection method | Tuning cost | Workloads | Hardware | Implication for this project |
|---|---|---|---|---|---|---|
| [Triton autotune](https://triton-lang.org/main/python-api/generated/triton.autotune.html) | User-provided `triton.Config` objects, including tile and launch meta-parameters | Empirically benchmarks all candidates for each key; `prune_configs_by` can use a performance model, `top_k`, or early pruning | Compiles and benchmarks every unpruned candidate when an autotune key changes; the API can report total autotuning time | Generic Triton kernels; the official fused-attention tutorial tunes attention tiles and launch parameters | Triton-supported GPU backends | Use as the exhaustive empirical baseline and later integration target; first measure compile and benchmark cost outside the decorator |
| [Ansor](https://www.usenix.org/conference/osdi20/presentation/zheng) | Hierarchically generated tensor programs with broad schedule-transformation combinations | Learned cost model plus evolutionary search; task scheduler allocates trials across graph subproblems | Requires iterative candidate generation and on-device measurements; broader and more expensive than this project's fixed 32–64-config space | Tensor operators and end-to-end deep-learning networks | Intel CPU, ARM CPU, NVIDIA GPU | Motivates learned ranking, but is too general to serve as a direct implementation baseline here |
| [TVM MetaSchedule](https://arxiv.org/abs/2205.13603) | Modular probabilistic TIR schedule transformations defined by domain experts | Learning-driven search over sampled schedules with hardware measurements | Trades compilation and measurement time for better schedules; tuning records can be stored and reused | Tensor programs and end-to-end deep-learning workloads | CPU and GPU targets in the evaluation | Separate search-space design from selection policy; this project should freeze a small attention space before comparing selectors |
| [TenSet](https://openreview.net/forum?id=aIfp8kLuvc9) | Tensor programs collected from search-based compilation | Pretrained learned cost models rank candidates and transfer across tasks/hardware | Dataset collection is large: 52 million performance records from six hardware platforms; pretraining can reduce later search time by up to 10× | Tensor operators derived from deep-learning models | Six CPU/GPU hardware platforms | Motivates reusable measurements and strict development/held-out splits; the current project is intentionally much smaller |
| [TLP / MTL-TLP](https://arxiv.org/abs/2211.03578) | Ansor tensor-program candidates represented by sequences of schedule primitives | Transformer-style latency regression; multi-task learning improves cross-hardware transfer | Reports average search-time speedups of 9.1× on CPU and 3.0× on GPU; MTL-TLP uses 7% of target-hardware data in its transfer setting | Ansor tensor-program tuning workloads | Multiple CPU and GPU platforms | Shows what a data-heavy learned cost model can do; use a small tree-based ranker first because this project has limited data |
| [ROLLER](https://www.usenix.org/conference/osdi22/presentation/zhu) | Programs constructed from hardware-aligned `rTile` choices rather than a broad unconstrained schedule space | Recursive construction plus a micro-performance model; avoids evaluating every candidate on the device | Generates kernels in seconds rather than the hours often required by search-based compilers | Common tensor operators and DNN workloads | GPUs and Graphcore IPUs | Supports a hardware-aware restricted-space baseline with little empirical search |
| [tritonBLAS](https://arxiv.org/abs/2512.04226) | Triton GEMM blocking and launch configurations parameterized by matrix shape and architecture | Deterministic analytical model using cache hierarchy, topology, and blocking behavior | Reports more than 95% of autotuned performance while reducing empirical autotuning time to zero | Diverse GEMM sizes and ML/HPC GEMM workloads | Modern AMD GPUs in the released implementation | Closest analytical-selection analogy, but GEMM assumptions must not be claimed to apply to attention without evaluation |
| [The Anatomy of a Triton Attention Kernel](https://arxiv.org/abs/2511.11581) | Paged-attention algorithm and launch parameters across workload shapes | Offline autotuning provides labels for a workload-based decision tree used for configuration choice | Still requires empirical data collection to build the decision tree; the abstract does not provide one transferable tuning-time number | Paged-attention workloads integrated into an inference server | NVIDIA and AMD GPUs | A simple rule or decision tree is a baseline, not the new contribution; the proposed prototype should rank a small top-k set on held-out workloads |

## Interpretation for Week 4

The literature exposes three families:

1. empirical search: Triton autotune, Ansor, and MetaSchedule;
2. learned ranking and transfer: TenSet and TLP;
3. low-search hardware-aware selection: ROLLER and tritonBLAS.

The Anatomy paper is especially important because it already applies a decision tree to attention configuration selection. Therefore, this project should evaluate simple rules and a small decision tree as baselines. The first prototype should instead rank a top-k set of configurations, benchmark only that set, and evaluate oracle-relative performance, coverage, and tuning work on held-out dense-prefill workloads.

This is a proposed direction, not a conclusion. The next expanded sweep must first establish that exhaustive tuning cost is meaningful and that configuration preferences remain workload-dependent.

## Week 3: Attention Context

| Paper | Problem | Main idea | Connection to this project |
|---|---|---|---|
| MAS-Attention, MLSys 2025 | Exact attention is difficult on resource-constrained edge accelerators | Memory-aware streaming, multi-tier tiling, and heterogeneous scheduling | Useful tiling context, but its hardware target differs from this RTX 3090/Triton prototype |
| FlashInfer, MLSys 2025 | LLM serving needs flexible attention across KV-cache layouts and dynamic requests | Customizable kernels, JIT specialization, and load-balanced scheduling | Motivates realistic workloads and eventual serving-level evaluation |
| LServe, MLSys 2025 | Long-context prefill and decode have different bottlenecks | Structured sparse attention and hierarchical KV-page pruning | Useful phase framing, but it changes the attention algorithm rather than selecting dense-kernel configs |
| FlexAttention, MLSys 2025 | Monolithic optimized kernels make attention variants difficult to implement | Compiler lowering from user-defined score and mask changes to fused kernels | Relevant compiler context; the current scope is fixed exact dense attention |
