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
echo "SNAC_PRIME_FRAMES:   ${SNAC_PRIME_FRAMES:-}"
echo "SNAC_DECODE_FRAMES:  ${SNAC_DECODE_FRAMES:-}"
echo "SNAC_STARTUP_SKIP:   ${SNAC_STARTUP_SKIP_SAMPLES:-}"
echo "WS_WORD_BUFFER_SIZE: ${WS_WORD_BUFFER_SIZE:-}"
echo "FLUSH_MS:            ${FLUSH_MS:-}"


