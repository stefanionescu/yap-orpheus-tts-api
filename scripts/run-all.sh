#!/usr/bin/env bash
set -euo pipefail

# HF_TOKEN must be set by the deployment environment (no .env required)
if [ -z "${HF_TOKEN:-}" ]; then
  echo "[run-all] ERROR: HF_TOKEN not set. Export HF_TOKEN in the shell." >&2
  echo "           Example: export HF_TOKEN=\"hf_xxx\"" >&2
  exit 1
fi

echo "[run-all] 1/3 bootstrap"
bash scripts/00-bootstrap.sh

echo "[run-all] 2/3 install"
bash scripts/01-install.sh

# If BACKEND=trtllm, also install TRT-LLM dependencies
if [ "${BACKEND:-vllm}" = "trtllm" ]; then
  echo "[run-all] Installing TRT-LLM backend"
  bash scripts/01-install-trt.sh
  echo "[run-all] Building TRT-LLM engine (02-build-trt-engine.sh)"
  bash scripts/02-build-trt-engine.sh
  # If engine dir not provided, default to local models/orpheus-trt used by the builder
  : "${TRTLLM_ENGINE_DIR:=$PWD/models/orpheus-trt}"
  export TRTLLM_ENGINE_DIR
fi

echo "[run-all] 3/3 start server"
exec bash scripts/03-run-server.sh
