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
  echo "WARNING: HF_TOKEN not set; some downloads may fail if required" >&2
fi

cd /app
exec uvicorn server.server:app --host "$HOST" --port "$PORT" --timeout-keep-alive 75 --log-level info


