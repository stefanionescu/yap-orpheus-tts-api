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
# Ensure we use the project venv for install/build/run steps consistently
if [ -d .venv ]; then
  source .venv/bin/activate || true
fi
bash scripts/01-install.sh

BACKEND=${1:-${ORPHEUS_BACKEND:-trtllm}}
export ORPHEUS_BACKEND="$BACKEND"
if [ "$BACKEND" != "vllm" ]; then
  # Build engine if missing
  ENGINE_DIR=${ENGINE_DIR:-engine/orpheus_a100_fp16_kvint8}
  if [ ! -d "$ENGINE_DIR" ] || [ -z "$(ls -A "$ENGINE_DIR" 2>/dev/null)" ]; then
    echo "[run-all] Building TRT-LLM engine at $ENGINE_DIR"
    # Use the same Python/venv as install
    if [ -d .venv ]; then source .venv/bin/activate || true; fi
    # Ensure TF32 and arch list are propagated to any MPI workers
    export NVIDIA_TF32_OVERRIDE=${NVIDIA_TF32_OVERRIDE:-1}
    export TRTLLM_MPI_ENV_VARS="NVIDIA_TF32_OVERRIDE"
    # Source perf env (sets TORCH_CUDA_ARCH_LIST=8.0 by default)
    if [ -f scripts/env/perf.sh ]; then source scripts/env/perf.sh; fi
    python server/build_trtllm_engine.py || {
      echo "[run-all] ERROR: failed to build TRT-LLM engine" >&2; exit 1; }
  fi
fi

echo "[run-all] 3/3 start server"
exec bash scripts/02-run-server.sh
