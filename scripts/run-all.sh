#!/usr/bin/env bash
set -euo pipefail

# HF_TOKEN must be set by the deployment environment (no .env required)
if [ -z "${HF_TOKEN:-}" ]; then
  echo "[run-all] ERROR: HF_TOKEN not set. Export HF_TOKEN in the shell." >&2
  echo "           Example: export HF_TOKEN=\"hf_xxx\"" >&2
  exit 1
fi

# Detach and run the entire pipeline in background
mkdir -p logs .run
CMD='\
  echo "[run-all] 1/3 bootstrap" && \
  bash scripts/00-bootstrap.sh && \
  echo "[run-all] 2/3 install" && \
  bash scripts/01-install.sh && \
  if [ "${BACKEND:-vllm}" = "trtllm" ]; then \
    echo "[run-all] Installing TRT-LLM backend" && \
    bash scripts/01-install-trt.sh && \
    echo "[run-all] Building TRT-LLM engine (INT8 SQ + KV INT8)" && \
    bash scripts/02-build-int8-sq-kvint8.sh && \
    : "${TRTLLM_ENGINE_DIR:=$PWD/models/orpheus-trt-int8sq}" && \
    export TRTLLM_ENGINE_DIR; \
  fi && \
  echo "[run-all] 3/3 start server" && \
  bash scripts/04-run-server.sh'

setsid nohup bash -lc "$CMD" </dev/null > logs/run-all.log 2>&1 &
bg_pid=$!
echo $bg_pid > .run/run-all.pid
echo "[run-all] Pipeline started in background (PID $bg_pid)"
echo "[run-all] Logs: logs/run-all.log (server logs: logs/server.log)"
echo "[run-all] To stop:   bash scripts/stop.sh"
echo "[run-all] Following logs (Ctrl-C detaches, pipeline continues)"
touch logs/run-all.log || true
exec tail -n +1 -F logs/run-all.log
