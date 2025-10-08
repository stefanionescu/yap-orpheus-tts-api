#!/usr/bin/env bash
# =============================================================================
# Simple TTS Server Startup Script for Runpod/Cloud Containers
# =============================================================================
# This script starts the TTS server inside a running Docker container.
# Perfect for Runpod, Vast.ai, and similar cloud GPU services.
#
# Usage: bash start-server.sh [--background]
# =============================================================================

set -euo pipefail

echo "=== Orpheus 3B TTS Server Startup ==="

# Configuration
APP_DIR="/app"
VENV_DIR="$APP_DIR/.venv"
LOG_FILE="/tmp/tts-server.log"
HOST="0.0.0.0"
PORT="8000"

# Environment variables (matching main scripts)
export TRTLLM_ENGINE_DIR="/app/models/orpheus-trt-int4-awq"
export YAP_API_KEY="${YAP_API_KEY:-}"
export HF_TOKEN="${HF_TOKEN:-}"

# Parse arguments
BACKGROUND=false
for arg in "$@"; do
    case $arg in
        --background|-b)
            BACKGROUND=true
            ;;
        --help|-h)
            echo "Usage: $0 [--background]"
            echo ""
            echo "Options:"
            echo "  --background, -b    Run server in background"
            echo "  --help, -h          Show this help"
            exit 0
            ;;
    esac
done

# Validate environment (matching main scripts logic)
echo "[startup] Validating environment..."

# Check required directories
if [ ! -d "$APP_DIR" ]; then
    echo "ERROR: App directory not found at $APP_DIR" >&2
    exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "ERROR: Virtual environment not found at $VENV_DIR" >&2
    exit 1
fi

# Check TensorRT engine
if [ ! -f "$TRTLLM_ENGINE_DIR/rank0.engine" ]; then
    echo "ERROR: TensorRT engine not found at $TRTLLM_ENGINE_DIR/rank0.engine" >&2
    exit 1
fi

# Validate required environment variables (matching main scripts)
missing_vars=()

if [ -z "$YAP_API_KEY" ]; then
    missing_vars+=("YAP_API_KEY")
fi

if [ ${#missing_vars[@]} -gt 0 ]; then
    echo "ERROR: Missing required environment variables:" >&2
    printf "  - %s\n" "${missing_vars[@]}" >&2
    echo "" >&2
    echo "Please set these variables when running the container:" >&2
    echo "  docker run -e YAP_API_KEY=your_key ..." >&2
    exit 1
fi

echo "[startup] Environment validation passed"

# Change to app directory
cd "$APP_DIR"

# Activate virtual environment
echo "[startup] Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Check if server is already running
if pgrep -f "uvicorn server.server:app" >/dev/null; then
    echo "WARNING: TTS server appears to already be running"
    echo "To stop it: pkill -f 'uvicorn server.server:app'"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Build uvicorn command
UVICORN_CMD="python -m uvicorn server.server:app --host $HOST --port $PORT --timeout-keep-alive 75 --log-level info"

if [ "$BACKGROUND" = true ]; then
    echo "[startup] Starting TTS server in background..."
    echo "[startup] Logs will be written to: $LOG_FILE"
    
    # Start in background
    nohup $UVICORN_CMD > "$LOG_FILE" 2>&1 &
    SERVER_PID=$!
    
    echo "[startup] Server started with PID: $SERVER_PID"
    echo "[startup] Waiting for server to initialize..."
    
    # Wait for server to start
    for i in {1..30}; do
        if curl -f -s "http://localhost:$PORT/healthz" >/dev/null 2>&1; then
            echo "[startup] Server is ready!"
            break
        fi
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo "ERROR: Server process died" >&2
            echo "Last few log lines:" >&2
            tail -10 "$LOG_FILE" >&2
            exit 1
        fi
        echo "[startup] Waiting... ($i/30)"
        sleep 2
    done
    
    if ! curl -f -s "http://localhost:$PORT/healthz" >/dev/null 2>&1; then
        echo "WARNING: Server may not be fully ready yet" >&2
    fi
    
    echo ""
    echo "=== Server Started Successfully ==="
    echo "PID: $SERVER_PID"
    echo "URL: http://localhost:$PORT"
    echo "Health: http://localhost:$PORT/healthz"
    echo "WebSocket: ws://localhost:$PORT/ws/tts"
    echo ""
    echo "Commands:"
    echo "  View logs: tail -f $LOG_FILE"
    echo "  Stop server: pkill -f 'uvicorn server.server:app'"
    echo "  Check status: curl http://localhost:$PORT/healthz"
    
else
    echo "[startup] Starting TTS server in foreground..."
    echo "[startup] Press Ctrl+C to stop"
    echo ""
    
    # Start in foreground
    exec $UVICORN_CMD
fi
