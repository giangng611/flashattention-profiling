# Email to Dr. Wenwen Wang

**Subject:** Follow-up on Spring 2027 CURO research proposal

Dear Dr. Wang,

Near the beginning of this semester, I asked whether my ML systems research could potentially develop into a CURO project for Spring 2027, and you indicated that it could. I am writing to follow up and reconfirm whether this would still be possible.

For nearly two months, I have been working with Qingchen to develop the research direction and build a preliminary prototype. The project now studies whether a lightweight workload-aware selector can reduce the cost of tuning Triton attention kernels while retaining performance close to exhaustive search.

So far, I have:

- implemented and validated a forward Triton dense-prefill attention kernel;
- evaluated 48 configurations on 16 FP16/BF16 workloads using an RTX 3090;
- completed three exhaustive measurement runs and several configuration-selection baselines;
- tested robustness across 100 balanced development/test splits; and
- directly measured a top-4 selector that retained 96.97% of exhaustive-search performance while reducing warm-cache tuning time by 84.23% on six held-out workloads.

I have attached a first proposal draft for your review. I intentionally left the faculty-supervisor field blank until I could confirm your availability and whether this project is appropriate for the Spring 2027 CURO application or course process.

Would you be willing to review the draft and let me know whether you would still be comfortable supervising this CURO project? I would also appreciate any feedback on the scope, research question, and next experiments.

The repository is available at: <https://github.com/giangng611/flashattention-profiling>

Best,

Giang
