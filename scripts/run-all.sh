#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/run-all.sh [--sync|--foreground] [--backend <backend>] [<backend>]
# By default, runs server in background with auto-tailing logs
# Use --sync or --foreground to run server in foreground (blocking)
# Backend options: trtllm (default), vllm

show_usage() {
  echo "Usage: $0 [OPTIONS] [BACKEND]"
  echo ""
  echo "Options:"
  echo "  --sync, --foreground    Run server in foreground (blocking) mode"
  echo "  --backend BACKEND       Specify backend (trtllm, vllm)"
  echo "  -h, --help             Show this help message"
  echo ""
  echo "Arguments:"
  echo "  BACKEND                Backend to use (trtllm, vllm). Default: trtllm"
  echo ""
  echo "Examples:"
  echo "  $0                     # Run with defaults (trtllm backend, background mode)"
  echo "  $0 --sync              # Run in foreground mode"
  echo "  $0 --sync vllm         # Run vllm backend in foreground"
  echo "  $0 --backend trtllm    # Explicitly specify trtllm backend"
  echo ""
  echo "Environment Variables:"
  echo "  HF_TOKEN               Required Hugging Face token"
  echo "  ORPHEUS_BACKEND        Default backend if not specified"
}

# Parse command line arguments
SYNC_MODE=false
BACKEND=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --sync|--foreground)
      SYNC_MODE=true
      shift
      ;;
    --backend)
      BACKEND="$2"
      shift 2
      ;;
    -h|--help)
      show_usage
      exit 0
      ;;
    *)
      # Assume it's the backend if no flag prefix
      if [[ "$1" != --* ]] && [ -z "$BACKEND" ]; then
        BACKEND="$1"
      fi
      shift
      ;;
  esac
done

# Export sync mode for child scripts
export ORPHEUS_SYNC_MODE="$SYNC_MODE"

# HF_TOKEN must be set by the deployment environment (no .env required)
if [ -z "${HF_TOKEN:-}" ]; then
  echo "[run-all] ERROR: HF_TOKEN not set. Export HF_TOKEN in the shell." >&2
  echo "           Example: export HF_TOKEN=\"hf_xxx\"" >&2
  exit 1
fi

if [ "$SYNC_MODE" = true ]; then
  echo "[run-all] Mode: synchronous (foreground)"
  echo "[run-all] 1/3 bootstrap"
  bash scripts/00-bootstrap.sh

  echo "[run-all] 2/3 install"
  # Ensure we use the project venv for install/build/run steps consistently
  if [ -d .venv ]; then
    source .venv/bin/activate || true
  fi
  bash scripts/01-install.sh
else
  echo "[run-all] Mode: background with auto-tail (default)"
  
  # Create log directories
  mkdir -p logs .run
  
  echo "[run-all] 1/3 bootstrap (running in background)"
  bash scripts/00-bootstrap.sh > logs/bootstrap.log 2>&1 &
  bootstrap_pid=$!
  echo $bootstrap_pid > .run/bootstrap.pid
  echo "[run-all] Bootstrap started in background (PID $bootstrap_pid)"
  echo "[run-all] Following bootstrap logs (Ctrl-C detaches, process keeps running)"
  tail -n +1 -f logs/bootstrap.log &
  tail_pid=$!
  
  # Wait for bootstrap to complete
  wait $bootstrap_pid
  kill $tail_pid 2>/dev/null || true
  echo "[run-all] Bootstrap completed"
  
  echo "[run-all] 2/3 install (running in background)"
  # Ensure we use the project venv for install/build/run steps consistently
  if [ -d .venv ]; then
    source .venv/bin/activate || true
  fi
  bash scripts/01-install.sh > logs/install.log 2>&1 &
  install_pid=$!
  echo $install_pid > .run/install.pid
  echo "[run-all] Install started in background (PID $install_pid)"
  echo "[run-all] Following install logs (Ctrl-C detaches, process keeps running)"
  tail -n +1 -f logs/install.log &
  tail_pid=$!
  
  # Wait for install to complete
  wait $install_pid
  kill $tail_pid 2>/dev/null || true
  echo "[run-all] Install completed"
fi

BACKEND=${BACKEND:-${ORPHEUS_BACKEND:-trtllm}}
export ORPHEUS_BACKEND="$BACKEND"

# Handle TRT-LLM engine building
if [ "$BACKEND" != "vllm" ]; then
  # Build engine if missing
  ENGINE_DIR=${ENGINE_DIR:-engine/orpheus_a100_fp16_kvint8}
  if [ ! -d "$ENGINE_DIR" ] || [ -z "$(ls -A "$ENGINE_DIR" 2>/dev/null)" ]; then
    if [ "$SYNC_MODE" = true ]; then
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
    else
      echo "[run-all] Building TRT-LLM engine at $ENGINE_DIR (running in background)"
      mkdir -p logs .run
      (
        # Use the same Python/venv as install
        if [ -d .venv ]; then source .venv/bin/activate || true; fi
        # Ensure TF32 and arch list are propagated to any MPI workers
        export NVIDIA_TF32_OVERRIDE=${NVIDIA_TF32_OVERRIDE:-1}
        export TRTLLM_MPI_ENV_VARS="NVIDIA_TF32_OVERRIDE"
        # Source perf env (sets TORCH_CUDA_ARCH_LIST=8.0 by default)
        if [ -f scripts/env/perf.sh ]; then source scripts/env/perf.sh; fi
        python server/build_trtllm_engine.py
      ) > logs/build_engine.log 2>&1 &
      
      engine_pid=$!
      echo $engine_pid > .run/engine.pid
      echo "[run-all] Engine build started in background (PID $engine_pid)"
      echo "[run-all] Following engine build logs (Ctrl-C detaches, process keeps running)"
      tail -n +1 -f logs/build_engine.log &
      tail_pid=$!
      
      # Wait for engine build to complete
      wait $engine_pid || {
        kill $tail_pid 2>/dev/null || true
        echo "[run-all] ERROR: failed to build TRT-LLM engine" >&2; exit 1; 
      }
      kill $tail_pid 2>/dev/null || true
      echo "[run-all] Engine build completed"
    fi
  fi
fi

echo "[run-all] 3/3 start server"
exec bash scripts/02-run-server.sh
