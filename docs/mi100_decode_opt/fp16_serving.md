# Qwen4Exp FP16 serving on MI100

`tools/mi100/serve_qwen4_exp.sh` defaults to `MI100_DTYPE=float16`.
It uses FP16 model parameters and activations, main KV caches, and automatic
raw/compressed QSA indexer caches. GPTQ expert weights retain their integer
packing. Normalization, softmax, attention accumulation, and GDN recurrent
state retain their required FP32 precision. This mode removes BF16 inference
paths; it does not force every intermediate operation to use FP16.

```bash
tools/mi100/serve_qwen4_exp.sh /path/to/checkpoint
```

To select the previous BF16 configuration:

```bash
MI100_DTYPE=bfloat16 tools/mi100/serve_qwen4_exp.sh /path/to/checkpoint
```

The launcher disables the BF16-only W8A16, W4A8, HyperConnection, push-all-reduce,
and fused GDN shortcuts in FP16 mode. QSA uses dtype-compatible Triton attention
and the unfused indexer preparation path. Disk-offloaded BF16 PLE checkpoint
rows are converted on the CPU before transfer into FP16 GPU staging buffers;
the checkpoint itself is unchanged. Loading BF16 checkpoint weights into FP16
parameters still produces conversion warnings; these describe checkpoint
conversion rather than BF16 runtime storage.

FP16 has less exponent range than BF16. Validate model outputs before relying
on a different checkpoint or larger workload. Changing dtype also changes
kernel dispatch, so native FP16 hardware throughput alone does not predict
end-to-end serving speed.

## Validation on 2026-11-10

With the local Qwen3.8-Flash-Next GPTQ-4bit checkpoint on four MI100 GPUs:

- Expanded QSA attention and disk PLE checks: 28 passed, including FP16 and BF16.
- QSA, configuration, and PLE regression selection: 186 passed, 3 skipped.
- API checks: text, image, video, 8192-token context retrieval, 12 arithmetic
  questions, and three seeded sampled-generation prompts passed.
- Model loading reported 20.84 GiB per GPU; graph capture completed on all ranks.
- A sampled explanation produced 89 tokens in 1.56 seconds, about 57 tokens/sec.
  The earlier BF16 request produced 91 tokens in 1.63 seconds, about 56 tokens/sec.
  These short requests are smoke measurements, not a controlled speedup claim.

The advertised 262144-token capacity was retained in the launcher, but this FP16
validation exercised context lengths only through 8192 tokens. Recovered ROCm
allocator warnings still occurred; no validation request failed.
