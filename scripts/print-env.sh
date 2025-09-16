#!/usr/bin/env bash
set -euo pipefail
source ".env" || true
echo "HF_TOKEN set?        " $( [ -n "${HF_TOKEN:-}" ] && echo yes || echo no )
echo "MODEL_ID:            ${MODEL_ID:-}"
echo "HOST:PORT            ${HOST:-}:${PORT:-}"
echo "VLLM_MAX_SEQS:       ${VLLM_MAX_SEQS:-}"
echo "VLLM_MAX_MODEL_LEN:  ${VLLM_MAX_MODEL_LEN:-}"
echo "VLLM_GPU_UTIL:       ${VLLM_GPU_UTIL:-}"
echo "VLLM_DTYPE:          ${VLLM_DTYPE:-}"
echo "TRTLLM_ENABLE:       ${TRTLLM_ENABLE:-0}"


