#!/usr/bin/env bash
set -euo pipefail
echo "[stop] Non-interactive wipe starting"

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

echo "[stop] Removing project artifacts"
rm -rf .venv || true
rm -rf .run || true
rm -rf logs || true

echo "[stop] Removing caches"
# Hugging Face caches (default and alternative locations)
rm -rf ~/.cache/huggingface || true
rm -rf ~/.local/share/huggingface || true
if [ -n "${HF_HOME:-}" ]; then rm -rf "${HF_HOME}" || true; fi
if [ -n "${HF_HUB_CACHE:-}" ]; then rm -rf "${HF_HUB_CACHE}" || true; fi

# Torch/pip caches
rm -rf ~/.cache/torch || true
rm -rf ~/.cache/torch_extensions || true
rm -rf ~/.cache/pip || true
rm -rf ~/.cache/hf_transfer || true
rm -rf ~/.cache/clip || true
rm -rf ~/.cache/vllm || true
rm -rf ~/.cache/triton || true
rm -rf ~/.nv || true

# Temp files that may accumulate
rm -rf /tmp/vllm* 2>/dev/null || true
rm -rf /tmp/huggingface* 2>/dev/null || true
rm -rf /tmp/torch* 2>/dev/null || true
rm -rf /dev/shm/vllm* 2>/dev/null || true

echo "[stop] Wipe complete."


