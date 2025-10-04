#!/usr/bin/env bash
set -euo pipefail

# Flags:
#  --clean-install  Remove venv and Python/HF/torch caches created by 01-install-trt.sh
#  --clean-system   Clean apt caches created by 00-bootstrap.sh (best-effort)
#  --clean-trt      Remove TensorRT-LLM build artefacts (engines, quantized checkpoints, repo clone, caches)

# Default paths - must match build scripts
: "${TRTLLM_REPO_DIR:=$PWD/.trtllm-repo}"
: "${MODELS_DIR:=$PWD/models}"

CLEAN_INSTALL=0
CLEAN_SYSTEM=0
CLEAN_TRT=0
for arg in "$@"; do
  case "$arg" in
    --clean-install) CLEAN_INSTALL=1 ;;
    --clean-system) CLEAN_SYSTEM=1 ;;
    --clean-trt) CLEAN_TRT=1 ;;
    *) ;;
  esac
done

echo "[stop] Stopping server and cleaning run artifacts"

echo "[stop] Stopping server if running"
if [ -f .run/server.pid ]; then
  PID=$(cat .run/server.pid || true)
  if [ -n "$PID" ]; then
    kill "$PID" || true
    sleep 1
  fi
  rm -f .run/server.pid || true
fi
pkill -f "uvicorn server.server:app" || true
sleep 1

if [ -f .run/run-all.pid ]; then
  echo "[stop] Stopping background pipeline"
  RA_PID=$(cat .run/run-all.pid || true)
  if [ -n "$RA_PID" ]; then
    kill "$RA_PID" 2>/dev/null || true
    sleep 1
    kill -9 "$RA_PID" 2>/dev/null || true
  fi
  rm -f .run/run-all.pid || true
fi
pkill -f "scripts/run-all.sh" 2>/dev/null || true

# Kill any lingering GPU compute processes (free VRAM)
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[stop] Killing GPU compute processes (if any)"
  PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' || true)
  if [ -n "$PIDS" ]; then
    echo "$PIDS" | xargs -r -n1 kill 2>/dev/null || true
    sleep 1
    # hard kill stragglers
    PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' || true)
    if [ -n "$PIDS" ]; then
      echo "$PIDS" | xargs -r -n1 kill -9 2>/dev/null || true
    fi
  fi
  
  # Kill any Python processes that might have CUDA contexts
  echo "[stop] Killing Python/TensorRT processes"
  pkill -9 -f "python.*tensorrt" 2>/dev/null || true
  pkill -9 -f "python.*trtllm" 2>/dev/null || true
  pkill -9 -f "mpirun" 2>/dev/null || true
  pkill -9 -f "mpi4py" 2>/dev/null || true
  sleep 1
  
  # As a fallback, kill any process holding /dev/nvidia* (container-local)
  if command -v fuser >/dev/null 2>&1; then
    echo "[stop] Killing processes using NVIDIA devices"
    for dev in /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools /dev/nvidia0 /dev/nvidia1; do
      if [ -e "$dev" ]; then
        fuser -k -9 "$dev" 2>/dev/null || true
      fi
    done
  fi
  
  # Clear CUDA IPC handles and shared memory
  echo "[stop] Clearing CUDA shared memory"
  rm -rf /dev/shm/nvidia* 2>/dev/null || true
  rm -rf /dev/shm/cuda* 2>/dev/null || true
  rm -rf /tmp/cuda* 2>/dev/null || true
  rm -rf /tmp/.X11-unix/* 2>/dev/null || true
  
  # Clear any lingering IPC semaphores/message queues
  if command -v ipcs >/dev/null 2>&1; then
    echo "[stop] Clearing IPC resources"
    # Get current user
    CURRENT_USER=$(whoami)
    # Clear semaphores
    ipcs -s | grep "$CURRENT_USER" | awk '{print $2}' | xargs -r -n1 ipcrm -s 2>/dev/null || true
    # Clear message queues
    ipcs -q | grep "$CURRENT_USER" | awk '{print $2}' | xargs -r -n1 ipcrm -q 2>/dev/null || true
    # Clear shared memory segments
    ipcs -m | grep "$CURRENT_USER" | awk '{print $2}' | xargs -r -n1 ipcrm -m 2>/dev/null || true
  fi
  
  sleep 2
  
  # Attempt GPU reset if no compute procs remain (may require privileges)
  LEFT=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' || true)
  if [ -z "$LEFT" ]; then
    echo "[stop] Resetting GPU"
    GPU_INDEX=${GPU_INDEX:-0}
    # Try multiple reset methods
    nvidia-smi --gpu-reset -i "$GPU_INDEX" 2>/dev/null || true
    # Alternative: reset compute mode
    nvidia-smi -i "$GPU_INDEX" -pm 0 2>/dev/null || true
    nvidia-smi -i "$GPU_INDEX" -pm 1 2>/dev/null || true
    # Clear persistence mode and re-enable
    nvidia-smi -i "$GPU_INDEX" -pm 0 2>/dev/null || true
    sleep 1
    nvidia-smi -i "$GPU_INDEX" -pm 1 2>/dev/null || true
  else
    echo "[stop] WARNING: GPU processes still running: $LEFT"
  fi
  
  # Final check
  echo "[stop] Final GPU memory status:"
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null || true
else
  echo "[stop] nvidia-smi not available; skipping GPU process kill"
fi

echo "[stop] Removing run-time artifacts (.run/, logs/)"
rm -rf .run || true
rm -rf logs || true

if [ "$CLEAN_INSTALL" = "1" ]; then
  echo "[stop] Removing venv and caches from install step"
  
  # Virtual environment
  echo "[stop] Removing venv: .venv"
  rm -rf .venv || true
  
  # Hugging Face caches (ALL locations)
  echo "[stop] Removing HuggingFace caches"
  rm -rf ~/.cache/huggingface || true
  rm -rf ~/.local/share/huggingface || true
  if [ -n "${HF_HOME:-}" ]; then rm -rf "${HF_HOME}" || true; fi
  if [ -n "${HF_HUB_CACHE:-}" ]; then rm -rf "${HF_HUB_CACHE}" || true; fi
  rm -rf /tmp/huggingface* 2>/dev/null || true
  
  # PyTorch caches
  echo "[stop] Removing PyTorch caches"
  rm -rf ~/.cache/torch || true
  rm -rf ~/.cache/torch_extensions || true
  rm -rf ~/.torch || true
  rm -rf /tmp/torch* 2>/dev/null || true
  rm -rf /tmp/.torch* 2>/dev/null || true
  rm -rf /dev/shm/torch* 2>/dev/null || true
  
  # Triton/compilation caches
  echo "[stop] Removing Triton/compilation caches"
  rm -rf ~/.triton || true
  rm -rf ~/.cache/triton || true
  rm -rf /tmp/triton* 2>/dev/null || true
  
  # pip cache
  echo "[stop] Removing pip cache"
  rm -rf ~/.cache/pip || true
  
  # Other ML framework caches
  echo "[stop] Removing misc ML caches"
  rm -rf ~/.cache/hf_transfer || true
  rm -rf ~/.cache/clip || true
  rm -rf ~/.cache/transformers || true
  rm -rf /tmp/.hf* 2>/dev/null || true
fi

# Remove TensorRT-LLM artefacts generated by build scripts
if [ "$CLEAN_TRT" = "1" ]; then
  echo "[stop] Removing TensorRT-LLM artefacts"
  
  # TensorRT-LLM repo clone (used by quantization scripts)
  echo "[stop] Removing TRT-LLM repo: ${TRTLLM_REPO_DIR}"
  rm -rf "${TRTLLM_REPO_DIR}" || true
  
  # Remove entire models directory (checkpoints + engines + HF downloads)
  if [ -d "${MODELS_DIR}" ]; then
    echo "[stop] Removing all models, checkpoints, and engines: ${MODELS_DIR}"
    rm -rf "${MODELS_DIR}" || true
  fi
  
  # Clear Python bytecode that might hold references
  echo "[stop] Removing Python cache"
  find "$PWD/server" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
  find "$PWD/tests" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
  rm -rf "$PWD/__pycache__" 2>/dev/null || true
  
  # Clear TensorRT compilation caches
  echo "[stop] Removing TensorRT compilation caches"
  rm -rf ~/.cache/tensorrt_llm || true
  rm -rf ~/.cache/tensorrt-llm || true
  rm -rf ~/.cache/nv_tensorrt || true
  rm -rf ~/.cache/nvidia || true
  rm -rf ~/.nv || true
  rm -rf /tmp/trt* 2>/dev/null || true
  rm -rf /tmp/tensorrt* 2>/dev/null || true
  rm -rf /dev/shm/trt* 2>/dev/null || true
  
  # ModelOpt/quantization caches
  echo "[stop] Removing quantization caches"
  rm -rf ~/.cache/modelopt || true
  rm -rf ~/.cache/model_optimizer || true
  rm -rf ~/.cache/nvidia_modelopt || true
  rm -rf /tmp/modelopt* 2>/dev/null || true
  rm -rf /tmp/quantiz* 2>/dev/null || true
  rm -rf /tmp/calib* 2>/dev/null || true
  rm -rf /dev/shm/quantiz* 2>/dev/null || true
fi

if [ "$CLEAN_SYSTEM" = "1" ]; then
  echo "[stop] Cleaning system caches"
  
  if command -v apt-get >/dev/null 2>&1; then
    echo "[stop] Cleaning apt caches"
    apt-get clean || true
    rm -rf /var/lib/apt/lists/* || true
  fi
  
  # Clean system-wide temp files
  echo "[stop] Cleaning system temp files"
  rm -rf /tmp/pip-* 2>/dev/null || true
  rm -rf /tmp/tmp* 2>/dev/null || true
  rm -rf /var/tmp/* 2>/dev/null || true
fi

# Always clean workspace temp files
echo "[stop] Cleaning workspace temp files"
rm -rf audio/*.wav 2>/dev/null || true  # Generated test audio files
rm -rf .pytest_cache 2>/dev/null || true
rm -rf .mypy_cache 2>/dev/null || true
rm -rf .ruff_cache 2>/dev/null || true
rm -rf *.egg-info 2>/dev/null || true

echo ""
echo "[stop] ============================================"
echo "[stop] Cleanup complete!"
if [ "$CLEAN_TRT" = "1" ]; then
  echo "[stop] Removed: TRT engines, checkpoints, models (~10-50GB)"
fi
if [ "$CLEAN_INSTALL" = "1" ]; then
  echo "[stop] Removed: venv, HF/Torch/pip caches (~20-100GB)"
fi
if [ "$CLEAN_SYSTEM" = "1" ]; then
  echo "[stop] Removed: system caches"
fi
echo "[stop] ============================================"
