# Qwen4Exp FP16 W4A16 MoE on MI100

Opt-in decode tuning for E=512, per-rank intermediate width 160, hidden width
2560, top-k 10, group size 32, and FP16 activations on four MI100 GPUs.
The file uses the current ROCm device name `AMD Instinct MI100`.

```bash
export VLLM_TUNED_CONFIG_FOLDER="$PWD/tools/mi100/moe-configs/qwen4_exp_fp16"
MI100_PUSH_AR=1 tools/mi100/serve_qwen4_exp.sh /path/to/checkpoint
```

Rows 1–5 use a 16x16x32 tile with one wave. `kpack=1` preserves the tested
native numerical results. K=32 divides the down-projection width 160.
Rows 6 and above reproduce the untuned WNA16 heuristics: the adjacent entries
at 20/21 and 40/41 preserve the exact block-M transition boundaries under
nearest-bucket lookup. The N/K heuristics still run for those entries.

This directory is deliberately opt-in and specific to the validated workload.
Remove `VLLM_TUNED_CONFIG_FOLDER` to return to normal config lookup.
Results and limitations are documented in
[FP16 push and decode tuning](../../../../docs/mi100_decode_opt/fp16_push_all_reduce.md).
