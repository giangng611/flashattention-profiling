# Week 3 Reproducibility Record

This record separates what is preserved locally from what must still be recovered from the CUDA server. Missing raw data is not reconstructed from the report.

## Preserved locally

- Experiment script: `src/sweep_triton_fused_attention_configs.py`
- Corrected-label snapshot of the six Week 3 shapes: `workloads/week3_config_sweep_workloads.csv`
- Narrative report and summary values: `docs/reports/FUSED_ATTENTION_CONFIG_SWEEP_REPORT.md`
- Seed: `0`
- Warmup iterations: `20`
- Timed trials: `100`
- Candidate space: `BLOCK_M={32,64}`, `BLOCK_N={32,64}`, `num_warps={4,8}`, `num_stages={2,3}`
- Environment: RTX 3090, CUDA 13.0 through PyTorch, PyTorch 2.14.0+cu130, Triton 3.8.0

## Expected raw artifacts

The Week 3 command named these outputs, but they were not present on the local machine during the Week 4 audit:

- `results/triton_fused_attention_config_sweep.csv`
- `results/triton_fused_attention_config_sweep_summary.csv`
- `results/triton_fused_attention_config_sweep_metadata.json`

If these files remain on the CUDA server, copy them into `results/` and add them to version control. The repository's ignore rules contain explicit exceptions for these three Week 3 artifacts.

## Reproduction command

```bash
python src/sweep_triton_fused_attention_configs.py \
  --workloads workloads/week3_config_sweep_workloads.csv \
  --block-ms 32 64 \
  --block-ns 32 64 \
  --num-warps-values 4 8 \
  --num-stages-values 2 3 \
  --warmup 20 \
  --trials 100 \
  --seed 0 \
  --output results/triton_fused_attention_config_sweep.csv \
  --summary-output results/triton_fused_attention_config_sweep_summary.csv \
  --metadata-output results/triton_fused_attention_config_sweep_metadata.json
```

The snapshot uses corrected labels for the dense Qwen-shaped proxies. Tensor shapes and configuration choices are unchanged from Week 3.
