#!/usr/bin/env bash
set -euo pipefail
source ".env"

if [ "${TRTLLM_ENABLE}" != "1" ]; then
  echo "[trt-llm] Set TRTLLM_ENABLE=1 in .env to use this path."
  exit 1
fi

source "${VENV_DIR}/bin/activate"

# Install TRT-LLM from pip if available for your CUDA.
# (If unavailable, you’ll need NVIDIA's wheels or to build from source.)
pip install --upgrade tensorrt_llm

# Simple placeholder: run server.py the same, but the Orpheus package will
# use TRT-LLM backend if properly configured (this depends on orpheus-speech internals).
# For best results, adapt Orpheus to call TRT-LLM runner directly (see taresh18 repo).

export ORPHEUS_BACKEND="trtllm"   # if orpheus-speech supports toggling backend

exec uvicorn app.server:app --host "${HOST}" --port "${PORT}" --timeout-keep-alive 75 --log-level info


