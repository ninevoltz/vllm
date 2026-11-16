# FP16 push all-reduce on MI100

The existing gfx908 sentinel push protocol now accepts contiguous FP16 and BF16
payloads. Both use FP32 accumulation in rank order and round once to the output
dtype. `consume_res` rounds the reduction before adding the residual, matching
native all-reduce followed by addition. The original exception for an all-rank
negative zero remains: push normalizes it to positive zero.

The buffers, polling instructions, sentinel, graph-site allocation, and peer
stores are shared with the BF16 path. Fused producer epilogues remain BF16-only.
FP16 support does not enable FP8, quantized collectives, or AITER custom AR.

## Enablement

For direct vLLM commands, set `VLLM_GFX908_PUSH_AR=1` before importing vLLM.
For the MI100 launcher, use `MI100_PUSH_AR=1`; FP16 keeps push off by default.
This path requires four gfx908 GPUs on the same node with supported peer access.
Messages outside its contiguous 16-bit shape/capacity contract use native AR.

## Reproduce

Run the real four-rank numerical and timing harness from a ROCm Python environment:

```bash
ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm \
GPU_ARCHS=gfx908 PYTORCH_ROCM_ARCH=gfx908 OMP_NUM_THREADS=1 MAX_JOBS=4 \
VLLM_GFX908_PUSH_AR=1 \
VLLM_GFX908_HIP_BUILD_DIR=/tmp/gfx908-push-build \
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 \
  benchmarks/kernels/benchmark_gfx908_push_ar.py \
  --dtype fp16 --replays 1000 --benchmark --output /tmp/push-fp16.json
```

Repeat with `--dtype bf16`. Use an idle GPU set for timing. The harness checks
native-result equality, sentinel values, subnormals, overflow, infinity, NaN,
residual consumers, alternating graphs, changing inputs, and queued replays.
It fails if FP16 silently falls back instead of using the push path.

## Primitive results, 2026-11-16

Four MI100s, PyTorch 2.14.0+rocm7.14. Five timing samples alternate native/push
order. Each sample times 200 replays of eight collectives; the table reports the
median of the slowest rank's per-collective time. The native path includes its
uncached staging copy, as serving does.

| FP16 shape | Native AR | Push AR |
| --- | ---: | ---: |
| 1 x 2560 | 9.14 us | 5.34 us |
| 4 x 2560 | 9.59 us | 5.64 us |
| 16 x 2560 | 15.56 us | 8.27 us |
| 48 x 2560 | 30.93 us | 14.84 us |

FP16 and BF16 each passed 10,103 numerical comparisons, including 1,000
input-changing graph replays plus 2,048 queued alternating-graph replays.
Both reported zero polling timeouts. The unmodified FP16 path failed the
harness's push-dispatch assertion, confirming the new coverage exercises the
new implementation.

These are collective microbenchmarks, not an end-to-end model speedup claim.
Full-model results and profiling artifacts are recorded separately during the
serving investigation.

## Checkpoint-specific MoE tuning

An optional WNA16 profile is in
`tools/mi100/moe-configs/qwen4_exp_fp16/`. It targets E=512, per-rank
intermediate width 160, hidden width 2560, top-k 10, group size 32, and FP16.
The current ROCm reports the device as `AMD Instinct MI100`; the older
`Arcturus GL-XL [Instinct MI100]` config files do not match that name or this
checkpoint's expert width.

```bash
export VLLM_TUNED_CONFIG_FOLDER="$PWD/tools/mi100/moe-configs/qwen4_exp_fp16"
MI100_PUSH_AR=1 tools/mi100/serve_qwen4_exp.sh /path/to/checkpoint
```

The selected decode tile is M=16, N=16, K=32, one wave, `kpack=1`. It produced
identical results to default dispatch for synthetic FP16 inputs at every row
count from 1 through 5. It keeps K divisible by the down projection width 160.
For rows 6 and above, explicit boundary entries preserve the untuned WNA16
heuristics, including the M=20/21 and M=40/41 transitions.

The single-GPU graph benchmark runs both GPTQ/AWQ GEMMs, activation, and expert
reduction with synthetic weights at the checkpoint's per-rank dimensions.
These timings use hot-cache repeated inputs and do not predict whole-model gains:

| Rows | Default pipeline | Selected pipeline |
| --- | ---: | ---: |
| 1 | 160.7 us | 115.1 us |
| 2 | 295.2 us | 200.4 us |
| 3 | 445.2 us | 257.5 us |
| 4 | 467.7 us | 342.1 us |
| 5 | 496.0 us | 405.4 us |

Reproduce the tile sweep with `VLLM_TUNED_CONFIG_FOLDER` unset:

```bash
.venv/bin/python benchmarks/kernels/benchmark_gfx908_wna16.py \
  --rows 1 2 3 4 5 --output /tmp/wna16-mi100.json
```

## Full-model decode and MTP depth

On the local FP16 Qwen4Exp GPTQ-4bit checkpoint, TP=4, one active request,
4 GiB KV cache per GPU, AITER normalization/RoPE/GEMM enabled, and the normal
idle-fixed ROCr runtime, three fixed tutorial prompts each generated 768 tokens.
Rates exclude time to first content. Temperature was zero, seed 123, and MTP
used probabilistic draft sampling. Generated text can vary across restarts.

| Configuration | Median decode |
| --- | ---: |
| Native custom AR, default MoE, MTP 3 | 76.91 tok/s |
| FP16 push, default MoE, MTP 3, warm repeat | 78.58 tok/s |
| FP16 push, tuned MoE, MTP 2 | 88.78 tok/s |
| FP16 push, tuned MoE, MTP 3 | 95.45 tok/s |
| FP16 push, tuned MoE, MTP 4 | 84.54 tok/s |

The tuned MTP-3 result is 24.1% above baseline for these prompts. Push alone
accounts for about 2%; most of the improvement comes from the MoE profile.
A final restart repeated MTP 3 at 98.49/95.35/95.90 tok/s (median 95.90,
24.7% above baseline). A fourth earlier tuned MTP-3 prompt measured 91.49 tok/s.
These are small local
single-request measurements, not a general model or throughput benchmark.
The launcher retains MTP 3 as its default; the checkpoint-specific MoE profile
and FP16 push remain opt-in.

The GPU profiler could not provide a usable full-model timeline in this
environment: Torch traces contained CPU events only, and external rocprofv3
crashed while importing vLLM. No GPU bottleneck percentages are inferred from
those traces. Timed serving runs used the normal runtime without profiling.

The final persistent MTP-3 setup passed 252 API requests: seven functional
checks (text, instructions, JSON, code, long retrieval, tool calling, image),
220 exact arithmetic answers, 22 interleaved long-context retrievals, and
three short decode benchmarks. The process did not restart during validation.
Retrieval prompts were around 9K tokens; 262K-token inference and concurrent
throughput were not validated in this investigation.
