# MI100 MTP and performance follow-up

Validated 2026-11-06 with the installed wheel on four 32 GiB MI100 GPUs.
This configuration replaces the eager serving baseline in
[the initial report](validation_current_vllm.md).

## Fix and configuration

The checkpoint stores BF16 MTP experts without declaring a GPTQ exclusion.
The previous draft loader therefore constructed GPTQ expert parameters and
failed with `has no parameter 'w13_weight'`. The port detects this layout and
applies the author's load-time W4 draft quantization. Standard vLLM rejection
sampling still verifies draft proposals; the earlier lossy rejection experiment
is not used.

The optimization source is btbtyler09/vllm-gfx908 branch
`qwen38-flash-next-mtp`, commit `bc5abbc72292747bec6a1efb858db1e2cfbf0211`.
Relevant kernels were adapted to current APIs rather than reverting current
NVIDIA, metadata, quantization, or model-runner implementations.
Code commits: `fe3b957b1f` and `12e32756f9`.

The launcher enables K=3 probabilistic MTP, full decode HIP graphs, W8 dense
projections, HIP expert kernels, fused HC/GDN paths, tiled QSA prefill, and
custom push reductions. Larger/ineligible reductions retain RCCL/PYNCCL;
AITER custom all-reduce stays off. Checkpoint-backed PLE rows are staged
before graph replay. The 95 GiB table remains unquantized on disk.

The target remains BF16 with the checkpoint's GPTQ experts. W8 dense/HC paths
use the author's approximations. The checks below are acceptance smoke tests;
the author's GSM8K/PPL results have not been reproduced on this port.

## Measured serving behavior

Three greedy streaming tutorials each generated exactly 512 tokens, with
thinking disabled, TP4, one request, and `max_model_len=262144` in both runs.
Decode rate excludes time to first streamed content.

| Prompt topic | MTP off, tokens/s | MTP K=3, tokens/s |
| --- | ---: | ---: |
| bicycle mechanics | 27.0 | 56.7 |
| the water cycle | 27.0 | 64.1 |
| how a compiler translates source code | 26.9 | 67.4 |

Mean rates: 27.0
and 62.7 tokens/s.
The MTP run accepted 989/1647 draft tokens
(60.0%), with mean acceptance length 2.80.
Time to first content was 0.13–0.24 seconds in the installed MTP run.

These results retain native context capacity, vision/video, and disk-backed
PLE. They are below the author's 167-token/s result with different context
and host-memory settings; no equivalent-performance claim is made.

| Installed-wheel check | Outcome | Seconds |
| --- | --- | ---: |
| text | Passed | 0.14 |
| vision | Passed | 0.67 |
| video | Passed | 0.30 |
| context 8192 | Passed | 3.57 |
| context 32768 | Passed | 10.48 |
| context 131072 | Passed | 53.16 |
| context 262144 | Passed | 138.30 |

The full request used 262112 prompt tokens plus 32 generated tokens and
correctly retrieved Paris from the beginning of the document. Context
cases use distinct facts and filler to avoid prefix reuse. Media checks use
an image with distinct shapes and an eight-frame video. Automatic tool choice
returned `lookup_inventory({"sku":"ABC123"})` with `finish_reason=tool_calls`.
All 12 arithmetic smoke questions passed.

No serving request failed and no worker crashed in the acceptance run.
The allocator logged recoverable allocation/retry messages during loading and
some first prefills; these are retained in the raw server log.
The 4 GiB static KV budget reports 283146 tokens of capacity. The extensible
cache's conservative ROCm reserve rejected 262144 with this MTP configuration,
so the measured static allocation is intentional.

## Build, checks, and operation

- Installed wheel: `0.33.0rc1.dev1072+g12e32756f.rocm714`. Its 62 changed runtime
  files match the repository; all 38 HIP source/header package-data files are
  present, and generated hipify copies are excluded.
- Native vLLM extensions reuse the previously built local ROCm wheel. The
  new HIP extensions were compiled locally with the installed ROCm toolchain.
- Focused W4A16/QSA/PLE checks: **39 passed**, including large row offsets,
  graph replay, tiled prefill, and unused long-context scorer capacity.
- MTP configuration checks: **10 passed**.
- Custom reduction validation: 40 graph replays × two reductions × three
  dtypes, exact integer-valued results; BF16 push is active, other dtypes
  exercise the native fallback.
- Pre-commit checks passed, including Ruff, mypy, clang-format, shellcheck,
  SPDX, and test wiring. The expanded tests caught an old int32 row-offset
  regression; the current upstream int64 guard was restored before packaging.

Start with `~/start_vllm.sh`, now a wrapper around the version-controlled
`tools/mi100/serve_qwen4_exp.sh`. Its previous contents are backed up at
`~/git-projects/vllm-mi100_support/start_vllm.before-mtp.sh`.
The wrapper retains `0.0.0.0:8001`, the automatic Qwen XML tool parser,
vision/video, and the 262144-token limit. Set `MI100_MTP=0` to compare without
speculation. The venv, HIP extensions, Triton cache, and other build caches
are under `~/git-projects/vllm-mi100_support/`.

Reproduce the serving acceptance checks:

```bash
~/git-projects/vllm-mi100_support/.venv/bin/python \
  tools/mi100/validate_multimodal_context.py <checkpoint> \
  --base http://127.0.0.1:8001
```

[Machine-readable results](validation_qwen38_mi100_mtp.json) record the wheel
hash, prompt topics, rates, acceptance counters, and checks. Full build logs,
benchmark text, failed attempts, and server logs remain under the environment
root. The branch has not been pushed.
