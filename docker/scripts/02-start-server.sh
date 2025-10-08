#!/usr/bin/env bash
set -euo pipefail

# Start the FastAPI TTS server with preinstalled environment.

export HOST=${HOST:-0.0.0.0}
export PORT=${PORT:-8000}

if [[ -z "${TRTLLM_ENGINE_DIR:-}" ]]; then
  echo "ERROR: TRTLLM_ENGINE_DIR must point to a built engine directory" >&2
  exit 1
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN is required" >&2
  exit 1
fi

cd /app
# Validate engine presence
if [[ ! -f "$TRTLLM_ENGINE_DIR/rank0.engine" ]]; then
  echo "ERROR: Engine not found at $TRTLLM_ENGINE_DIR/rank0.engine" >&2
  exit 1
fi
exec uvicorn server.server:app --host "$HOST" --port "$PORT" --timeout-keep-alive 75 --log-level info


