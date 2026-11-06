#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail

model_path="${1:?Usage: serve_qwen4_exp.sh CHECKPOINT [vllm arguments...]}"
shift
env_root="${MI100_ENV_ROOT:-${HOME}/git-projects/vllm-mi100_support}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1,2,3}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ROCM_USE_AITER=0
export VLLM_ROCM_USE_AITER_CUSTOM_AR=0
export TOKENIZERS_PARALLELISM=false
export PATH="${env_root}/.venv/bin:/opt/rocm/bin:${PATH}"
export ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm PYTORCH_ROCM_ARCH=gfx908
export MAX_JOBS=4 OMP_NUM_THREADS=1
export VLLM_GFX908_HIP_BUILD_DIR="${env_root}/extensions"
export TORCH_EXTENSIONS_DIR="${env_root}/extensions"
export TRITON_CACHE_DIR="${env_root}/triton-cache"
export TORCHINDUCTOR_CACHE_DIR="${env_root}/inductor-cache"
export VLLM_CACHE_ROOT="${env_root}/cache"
export VLLM_GFX908_MTP_QUANT=w4
export VLLM_GFX908_W8A16=1
export VLLM_GFX908_W4A8=1
export VLLM_GFX908_HC_W8=1 VLLM_GFX908_HC_W8_FREE=0
export VLLM_GFX908_SMALLM_HC_W8=1
export VLLM_GFX908_PUSH_AR=1
export VLLM_GFX908_GDN_FUSED=1 VLLM_GFX908_GDN_FUSED_SPEC=1
export VLLM_GFX908_W4_DEQUANT_LARGE_M=1
export VLLM_GFX908_QSA_PREFILL_TILED=1 VLLM_GFX908_QSA_NUM_COLUMNS=1
export HSA_ENABLE_SDMA="${HSA_ENABLE_SDMA:-0}"
export HSA_NO_SCRATCH_RECLAIM="${HSA_NO_SCRATCH_RECLAIM:-1}"

speculative_args=()
if [[ "${MI100_MTP:-1}" == "1" ]]; then
    speculative_args=(--speculative-config '{"method":"mtp","num_speculative_tokens":3,"draft_sample_method":"probabilistic"}')
fi

exec "${env_root}/.venv/bin/vllm" serve "${model_path}" \
    --host 127.0.0.1 --port 8001 \
    --tensor-parallel-size 4 --dtype bfloat16 \
    --max-model-len 262144 --max-num-seqs 1 --max-num-batched-tokens 8192 \
    --kv-cache-memory-bytes 4294967296 --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","decode_cudagraph_capture_request_sizes":[1],"prefill_cudagraph_capture_token_sizes":[1,2,4,8]}' \
    --engram-config '{"disk_offload":true}' \
    --safetensors-load-strategy lazy \
    --attention-backend TRITON_ATTN --mm-encoder-attn-backend TRITON_ATTN \
    --limit-mm-per-prompt '{"image":{"count":1,"width":1024,"height":1024},"video":{"count":1,"num_frames":32,"width":512,"height":512}}' \
    --enable-auto-tool-choice --tool-call-parser qwen3_xml \
    --reasoning-parser qwen3 \
    "${speculative_args[@]}" "$@"
