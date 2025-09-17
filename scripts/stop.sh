#!/usr/bin/env bash
set -euo pipefail

# Flags:
#  --clean-install  Remove venv and Python/HF/torch caches
#  --clean-system   Clean apt caches

CLEAN_INSTALL=0
CLEAN_SYSTEM=0

for arg in "$@"; do
  case "$arg" in
    --clean-install) CLEAN_INSTALL=1 ;;
    --clean-system) CLEAN_SYSTEM=1 ;;
    *) ;;
  esac
done

# Helpers: collect PIDs using nvidia-smi, lsof and fuser
collect_gpu_pids() {
  local pids=""
  if command -v nvidia-smi >/dev/null 2>&1; then
    pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | tr '\n' ' ')"
  fi
  local holders=""
  if command -v lsof >/dev/null 2>&1; then
    holders="$(lsof -t /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia[0-9]* 2>/dev/null | tr -d ' ' | tr '\n' ' ')"
  fi
  local fusers=""
  if command -v fuser >/dev/null 2>&1; then
    for dev in /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia[0-9]*; do
      [ -e "$dev" ] || continue
      local out
      out="$(fuser "$dev" 2>/dev/null || true)"
      [ -n "$out" ] && fusers="$fusers $out"
    done
  fi
  printf "%s %s %s" "$pids" "$holders" "$fusers" | tr ' ' '\n' | grep -E '^[0-9]+$' | sort -u | tr '\n' ' '
}

# Kill a PID, its children, and its process group (best-effort)
kill_tree() {
  local pid="$1"
  [ -z "$pid" ] && return 0
  # Recurse into children first
  local children
  children="$(pgrep -P "$pid" 2>/dev/null || true)"
  for c in $children; do
    kill_tree "$c"
  done
  # Kill by process group
  local pgid
  pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ' || true)"
  if [ -n "$pgid" ]; then
    kill -TERM -"$pgid" 2>/dev/null || true
    sleep 0.5
    kill -KILL -"$pgid" 2>/dev/null || true
  fi
  # Finally the PID itself
  kill -TERM "$pid" 2>/dev/null || true
  sleep 0.5
  kill -KILL "$pid" 2>/dev/null || true
}

echo "[stop] Stopping server and cleaning run artifacts"

# 1) Kill uvicorn main proc (pidfile + pattern)
echo "[stop] Stopping uvicorn if running"
if [ -f .run/server.pid ]; then
  PID=$(cat .run/server.pid || true)
  if [ -n "${PID:-}" ] && kill -0 "$PID" 2>/dev/null; then
    kill "$PID" || true
    sleep 1
    kill -9 "$PID" 2>/dev/null || true
  fi
  rm -f .run/server.pid || true
fi
# Fallback: pattern kill
pkill -f "uvicorn server.server:app" 2>/dev/null || true
pkill -f "python.*server/server.py" 2>/dev/null || true
sleep 1

# 2) Kill vLLM core engine & related workers (best-effort)
echo "[stop] Killing vLLM/ray workers (best-effort)"
# Common vLLM patterns
pkill -f "python.*vllm" 2>/dev/null || true
pkill -f "vllm.v1" 2>/dev/null || true
pkill -f "vllm.entrypoints" 2>/dev/null || true
pkill -f "openai.api_server" 2>/dev/null || true
# Extra hard-kill for stubborn workers
pkill -9 -f "python.*vllm" 2>/dev/null || true
# Kill any lingering tail -F log followers launched by run script shells
pkill -f "tail -n \+1 -F logs/server.log" 2>/dev/null || true
# Ray (if any were spawned by other configs)
pkill -f "ray::" 2>/dev/null || true
pkill -f "raylet" 2>/dev/null || true
pkill -f "plasma_store" 2>/dev/null || true
pkill -f "gcs_server" 2>/dev/null || true
sleep 1

# 3) Last resort: kill any process with open CUDA handles, then reset GPU
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[stop] Checking for GPU compute processes"
  # Attempt several passes: process-tree kill, then SIGKILL, then driver-assisted kill
  for attempt in 1 2 3; do
    PIDS="$(collect_gpu_pids)"
    [ -z "$PIDS" ] && break
    echo "[stop] Attempt ${attempt}: kill-tree -> $PIDS"
    for p in $PIDS; do
      kill_tree "$p"
    done
    sleep 2
  done

  PIDS="$(collect_gpu_pids)"
  if [ -n "$PIDS" ]; then
    echo "[stop] Forcing SIGKILL (direct) -> $PIDS"
    echo "$PIDS" | xargs -r -n1 kill -9 2>/dev/null || true
    sleep 1
  fi

  # Best-effort driver kill of compute processes (may require admin privileges / supported drivers)
  PIDS="$(collect_gpu_pids)"
  if [ -n "$PIDS" ]; then
    echo "[stop] nvidia-smi --kill-compute-process for -> $PIDS"
    for p in $PIDS; do
      nvidia-smi --kill-compute-process "$p" >/dev/null 2>&1 || true
    done
    sleep 1
  fi

  # Stop CUDA MPS and persistence if present
  if command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
    echo "[stop] Stopping CUDA MPS daemon"
    echo quit | nvidia-cuda-mps-control >/dev/null 2>&1 || true
    pkill -f nvidia-cuda-mps-control 2>/dev/null || true
  fi
  nvidia-smi -pm 0 >/dev/null 2>&1 || true
  pkill -f nvidia-persistenced 2>/dev/null || true

  # Kill any process holding /dev/nvidia* (container-local)
  if command -v fuser >/dev/null 2>&1; then
    for dev in /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia[0-9]*; do
      [ -e "$dev" ] && fuser -k "$dev" 2>/dev/null || true
    done
  fi

  # Attempt GPU reset individually for idle GPUs (may require privileges)
  GPU_IDS=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr -d ' ' || true)
  for gid in $GPU_IDS; do
    LEFT_ON_GPU=$(nvidia-smi -i "$gid" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' || true)
    if [ -z "$LEFT_ON_GPU" ]; then
      echo "[stop] Attempting nvidia-smi --gpu-reset -i ${gid}"
      nvidia-smi --gpu-reset -i "$gid" 2>/dev/null || true
    else
      echo "[stop] GPU ${gid} still has processes: $LEFT_ON_GPU; skipping reset"
    fi
  done

  # Final check and diagnostic
  LEFT="$(collect_gpu_pids)"
  if [ -n "$LEFT" ]; then
    echo "[stop] WARNING: GPU processes still present after cleanup: $LEFT"
    ps -o pid,cmd --no-headers -p $LEFT 2>/dev/null || true
    echo "[stop] If processes persist, they may be zombie/driver-stuck. A node restart may be required."
  fi
else
  echo "[stop] nvidia-smi not available; skipping GPU process kill"
fi

# 4) Remove run-time artifacts and SHM leftovers
echo "[stop] Removing .run/, logs/, and shared-memory artifacts"
rm -rf .run || true
rm -rf logs || true
rm -rf /dev/shm/vllm* 2>/dev/null || true
rm -rf /tmp/vllm* 2>/dev/null || true
rm -rf /tmp/huggingface* 2>/dev/null || true
rm -rf /tmp/torch* 2>/dev/null || true

# 5) Optional deep clean of install caches
if [ "$CLEAN_INSTALL" = "1" ]; then
  echo "[stop] Removing venv and caches from install step"
  rm -rf .venv || true
  
  # HuggingFace caches (multiple possible locations)
  rm -rf ~/.cache/huggingface ~/.local/share/huggingface || true
  rm -rf ~/.huggingface || true
  [ -n "${HF_HOME:-}" ] && rm -rf "${HF_HOME}" || true
  [ -n "${HF_HUB_CACHE:-}" ] && rm -rf "${HF_HUB_CACHE}" || true
  [ -n "${HUGGINGFACE_HUB_CACHE:-}" ] && rm -rf "${HUGGINGFACE_HUB_CACHE}" || true
  [ -n "${TRANSFORMERS_CACHE:-}" ] && rm -rf "${TRANSFORMERS_CACHE}" || true
  
  # PyTorch and related caches
  rm -rf ~/.cache/torch ~/.cache/torch_extensions ~/.cache/torchvision || true
  rm -rf ~/.torch || true
  
  # vLLM specific caches (can be huge)
  rm -rf ~/.cache/vllm || true
  rm -rf /tmp/vllm_* /tmp/.vllm_* || true
  rm -rf /dev/shm/vllm_* || true
  
  # Other ML/AI caches
  rm -rf ~/.cache/triton ~/.nv || true
  rm -rf ~/.cache/pip ~/.cache/hf_transfer || true
  rm -rf ~/.cache/transformers || true
  
  # NVIDIA/CUDA caches
  rm -rf ~/.nv/ComputeCache || true
  rm -rf ~/.nvidia/compute-cache || true
  
  # Python package caches
  find /tmp -name "pip-*" -type d -exec rm -rf {} + 2>/dev/null || true
  find /tmp -name "tmppack*" -type d -exec rm -rf {} + 2>/dev/null || true
  
  echo "[stop] Checking remaining cache sizes..."
  du -sh ~/.cache 2>/dev/null || true
  du -sh /tmp/vllm* /tmp/huggingface* /tmp/torch* 2>/dev/null || true
fi

# 6) Optional Docker/container volume cleanup
if [ "$CLEAN_SYSTEM" = "1" ]; then
  if command -v docker >/dev/null 2>&1; then
    echo "[stop] Cleaning Docker volumes and images (if any)"
    # Clean up any orphaned volumes
    docker volume prune -f 2>/dev/null || true
    # Clean up any dangling images
    docker image prune -f 2>/dev/null || true
    # Clean up build cache
    docker builder prune -f 2>/dev/null || true
  fi
  
  # System package caches
  if command -v apt-get >/dev/null 2>&1; then
    echo "[stop] Cleaning apt caches"
    apt-get clean || true
    apt-get autoclean || true
    rm -rf /var/lib/apt/lists/* || true
    rm -rf /var/cache/apt/archives/* || true
  fi
  
  if command -v yum >/dev/null 2>&1; then
    echo "[stop] Cleaning yum caches"
    yum clean all || true
  fi
  
  if command -v dnf >/dev/null 2>&1; then
    echo "[stop] Cleaning dnf caches"
    dnf clean all || true
  fi
  
  # System-wide temp cleanup
  echo "[stop] Cleaning system temp directories"
  rm -rf /var/tmp/vllm* /var/tmp/huggingface* /var/tmp/torch* || true
  find /tmp -name "*vllm*" -type d -exec rm -rf {} + 2>/dev/null || true
  find /tmp -name "*huggingface*" -type d -exec rm -rf {} + 2>/dev/null || true
  find /tmp -name "*torch*" -type d -exec rm -rf {} + 2>/dev/null || true
  
  # Clean up any leftover NVIDIA persistence files
  rm -rf /var/run/nvidia-persistenced || true
  
  echo "[stop] Final disk usage check:"
  df -h / 2>/dev/null || true
fi

echo "[stop] Wipe complete."
