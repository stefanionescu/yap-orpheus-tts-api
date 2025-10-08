#!/usr/bin/env bash
set -euo pipefail

# Orchestrate: quantize → build engine → start server (server continues in background)

MODEL_ID=${MODEL_ID:-canopylabs/orpheus-3b-0.1-ft}
TRTLLM_ENGINE_DIR=${TRTLLM_ENGINE_DIR:-/opt/engines/orpheus-trt-int4-awq}

LOG_DIR=${LOG_DIR:-/var/log/orpheus}
PID_DIR=${PID_DIR:-/run/orpheus}
mkdir -p "$LOG_DIR" "$PID_DIR"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN is required" >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: GPU not available (nvidia-smi missing)" >&2
  exit 1
fi

echo "[orchestrator] Quantize and build engine..."
01-quantize-and-build.sh --model "$MODEL_ID" --engine-dir "$TRTLLM_ENGINE_DIR" ${@:-}

echo "[orchestrator] Starting server in background..."
setsid bash -lc "02-start-server.sh" </dev/null > "$LOG_DIR/server.log" 2>&1 &
srv_pid=$!
echo $srv_pid > "$PID_DIR/server.pid"
echo "[orchestrator] Server PID: $srv_pid, logs: $LOG_DIR/server.log"
echo "[orchestrator] Tailing logs (Ctrl-C to detach; server keeps running)"
touch "$LOG_DIR/server.log" || true
exec tail -n +1 -F "$LOG_DIR/server.log"


