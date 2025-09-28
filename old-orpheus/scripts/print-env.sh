#!/usr/bin/env bash
set -euo pipefail
# Common helpers + env
source "scripts/lib/common.sh"
load_env_if_present
source_env_dir "scripts/env"

echo "HF_TOKEN set?        " $( [ -n "${HF_TOKEN:-}" ] && echo yes || echo no )
echo "MODEL_ID:            ${MODEL_ID:-}"
echo "HOST:PORT            ${HOST:-}:${PORT:-}"
echo "VLLM_MAX_SEQS:       ${VLLM_MAX_SEQS:-}"
echo "VLLM_MAX_MODEL_LEN:  ${VLLM_MAX_MODEL_LEN:-}"
echo "VLLM_GPU_UTIL:       ${VLLM_GPU_UTIL:-}"
echo "VLLM_DTYPE:          ${VLLM_DTYPE:-}"
echo "FIRST_CHUNK_WORDS:  ${FIRST_CHUNK_WORDS:-40}"
echo "NEXT_CHUNK_WORDS:   ${NEXT_CHUNK_WORDS:-140}"
echo "MIN_TAIL_WORDS:     ${MIN_TAIL_WORDS:-12}"
