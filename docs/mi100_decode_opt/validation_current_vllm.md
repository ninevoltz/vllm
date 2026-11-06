# Current vLLM on MI100: Qwen3.8 validation

Validated 2026-11-06 on four 32 GiB AMD Instinct MI100 GPUs.

## Source and environment

- Upstream starting commit: `3cd3a7ad1c`.
- Merged source checkout HEAD: `745883c4197e5bacd438f68de2ec509f5a890f50`
  (`mi100-optimized`).
- Also ported the checkpoint-specific QSA correctness fix from
  `004b1ca779d5fcd09b785089c3af39e3e9846eda` on the source fork's
  `qwen38-flash-next` branch. That branch was preserved in the source backup.
- Checkpoint: `btbtyler09/Qwen3.8-Flash-Next-GPTQ-4bit`, snapshot
  `9ae8d0e1989ad3a4eaeea501d5f6c75f4eeda7e1`.
- Python 3.12.13, PyTorch `2.14.0+rocm7.14`, torchvision `0.29.0+rocm7.14`,
  Triton ROCm 3.8.0, ROCm 7.14 toolchain.
- Virtual environment and wheel artifacts:
  `~/git-projects/vllm-mi100_support/`.

The original source checkout was removed from its original path after merging.
A verified, complete Git bundle of all its refs remains at
`~/git-projects/vllm-mi100_support/gfx908-all-refs.bundle`.

## Serving configuration

Use `tools/mi100/serve_qwen4_exp.sh CHECKPOINT`.
It selects TP4, BF16, eager execution, a 262144-token context, one concurrent
request, and a 2048-token prefill batch budget. Both language and vision
attention use Triton. The launcher also enables automatic tool choice with
`qwen3_xml` tool parsing and `qwen3` reasoning parsing for Open WebUI.
The original BF16 n-gram table stays in its checkpoint
files; requested rows are staged without allocating the full 95 GiB table.

**MTP was off. All-reduce used RCCL through PYNCCL.** Both native custom
all-reduce and AITER custom all-reduce were disabled for these validation runs.
The imported custom reduction code remains in the branch, but these results
are not evidence for its accuracy or performance. MTP likewise needs its own
acceptance and memory validation before enabling it at this context length.

The launcher sets `HSA_ENABLE_SDMA=0` and `HSA_NO_SCRATCH_RECLAIM=1` by default.
One startup with the default DMA engines stalled inside `hipMemcpyWithStream`;
a saved native stack shows the host waiting for an HSA copy-completion signal.
Startup completed with DMA engines disabled. This is a local workaround,
not a claim that every ROCm copy issue is resolved.

Vision profiling is configured for one image at 1024x1024 and one video at
32 frames of 512x512; the processor can apply its own limits. The tested
image was 448x448 and the tested video contained eight 224x224 frames at 2 fps.
Other media sizes, batching, graph mode and speculative decoding were not
part of this acceptance run.

## Correctness issue found and fixed

Qwen4Exp at TP4 has six query heads per rank and one KV head, selecting an
8-row QSA head tile upstream. That tile miscompiles on gfx908. Before the
minimum-16-row fix, all three existing TP4 BF16 sparse-attention reference
cases failed; one case had 59.7% mismatched elements and maximum absolute
error 0.455078125. After the fix, all three passed with unchanged tolerances.

The defect produced fluent but incorrect answers, including apparent refusals,
a missed solid-color image, incorrect video ordering and failed retrieval.
Disabling skinny GEMMs did not fix those answers. After the QSA correction,
both the structured acceptance suite and the original solid-color/video/8K
code-retrieval inputs passed. Coherent generation alone was not a sufficient
accuracy check.

## End-to-end acceptance results

The test places a warehouse location at the beginning of a document, fills
its middle with distinct generated shipment records, and asks for that
location at the end. Each length uses a different target city, preventing
reuse of the preceding request's long prefix. The filler excludes those cities.
The script forces exactly 32 output tokens for boundary tests, so it exercises
the full combined token budget even after the short answer is emitted.

| Check | Observed answer | Prompt + output tokens | Seconds | Result |
| --- | --- | --- | --- | --- |
| Text arithmetic | 4 | 26 + 2 | 0.64 | Pass |
| Image: upper-left square | Red | 221 + 2 | 1.26 | Pass |
| Video: final square color | Blue | 252 + 2 | 0.52 | Pass |
| 8192-token retrieval | Oslo | 8160 + 32 | 6.53 | Pass |
| 32768-token retrieval | Kyoto | 32736 + 32 | 19.06 | Pass |
| 131072-token retrieval | Lima | 131040 + 32 | 72.66 | Pass |
| **262144-token retrieval** | **Paris** | **262112 + 32** | **164.00** | **Pass** |

No out-of-memory, indexing or worker errors occurred in the final run.
The engine reported a 557358-token KV-cache capacity, or 2.13 theoretical
requests at the configured maximum length; actual serving was limited to one
concurrent request. These checks establish the tested behavior, not universal
long-document accuracy or unrestricted media capacity.

Raw acceptance results: [validation_qwen38_mi100.json](validation_qwen38_mi100.json).
Full logs, including failed attempts and before/after kernel results, remain in
`~/git-projects/vllm-mi100_support/`.

## Focused checks

- GPTQ operators/dequantization, ROCm unquantized GEMMs and AMD disk rows:
  **51 passed, 2 existing architecture-specific skips**.
- Triton W4A16 numerical reference cases: **14 passed**.
- TP4 BF16 QSA reference cases: **3 failed before, 3 passed after**.
- AMD PLE compilation/disk-row checks: **4 passed**.
- Fresh-process vLLM configuration import regression: **1 passed**.
- Pre-commit checks passed for the follow-up fixes and validation tools.
  The initial imported historical benchmark scripts have pre-existing style,
  header and legacy API lint findings, retained in `pre-commit.log`.

The build also needed explicit CMake paths to the virtual environment's
`libtorch.so` and `libc10.so`, avoiding a conflicting system libtorch 2.9.
ROCm platform detection was corrected for this mixed AMD/NVIDIA host, and
Triton is loaded locally before promoting profiling symbols to avoid an LLVM
loader collision. The final wheel reuses the locally built gfx908 native
extensions and packages the validated Python sources.
