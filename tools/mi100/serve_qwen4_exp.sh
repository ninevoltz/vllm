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
export HSA_ENABLE_SDMA="${HSA_ENABLE_SDMA:-0}"
export HSA_NO_SCRATCH_RECLAIM="${HSA_NO_SCRATCH_RECLAIM:-1}"

exec "${env_root}/.venv/bin/vllm" serve "${model_path}" \
    --host 127.0.0.1 --port 8001 \
    --tensor-parallel-size 4 --dtype bfloat16 \
    --max-model-len 262144 --max-num-seqs 1 --max-num-batched-tokens 2048 \
    --gpu-memory-utilization 0.90 --enforce-eager \
    --engram-config '{"disk_offload":true}' \
    --safetensors-load-strategy lazy \
    --attention-backend TRITON_ATTN --mm-encoder-attn-backend TRITON_ATTN \
    --limit-mm-per-prompt '{"image":{"count":1,"width":1024,"height":1024},"video":{"count":1,"num_frames":32,"width":512,"height":512}}' \
    --enable-auto-tool-choice --tool-call-parser qwen3_xml \
    --reasoning-parser qwen3 \
    --disable-custom-all-reduce "$@"
