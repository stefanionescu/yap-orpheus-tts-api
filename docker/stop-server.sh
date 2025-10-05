#!/usr/bin/env bash
# =============================================================================
# Stop TTS Server Script for Docker/Runpod
# =============================================================================
# Stops the TTS server process without affecting the Docker container or image.
# Perfect for Runpod where you want to stop/restart the server without
# destroying the container.
#
# Usage: bash /app/stop-server.sh
# =============================================================================

set -euo pipefail

echo "=== Stopping TTS Server ==="

# =============================================================================
# Stop Server Processes
# =============================================================================

echo "[stop] Stopping TTS server processes..."

# Kill uvicorn server processes
PIDS=$(pgrep -f "uvicorn server.server:app" 2>/dev/null || true)
if [ -n "$PIDS" ]; then
    echo "[stop] Found server processes: $PIDS"
    echo "$PIDS" | xargs -r kill 2>/dev/null || true
    sleep 2
    
    # Force kill if still running
    PIDS=$(pgrep -f "uvicorn server.server:app" 2>/dev/null || true)
    if [ -n "$PIDS" ]; then
        echo "[stop] Force killing remaining processes: $PIDS"
        echo "$PIDS" | xargs -r kill -9 2>/dev/null || true
    fi
else
    echo "[stop] No TTS server processes found"
fi

# Kill any Python processes running the server module
PYTHON_PIDS=$(pgrep -f "python.*server.server" 2>/dev/null || true)
if [ -n "$PYTHON_PIDS" ]; then
    echo "[stop] Stopping Python server processes: $PYTHON_PIDS"
    echo "$PYTHON_PIDS" | xargs -r kill 2>/dev/null || true
    sleep 1
fi

# =============================================================================
# Cleanup Runtime Files
# =============================================================================

echo "[stop] Cleaning up runtime files..."

# Remove log files
rm -f /tmp/tts-server.log /tmp/server.log server.log 2>/dev/null || true

# Remove any PID files
rm -f /tmp/server.pid /app/server.pid 2>/dev/null || true

# =============================================================================
# Verification
# =============================================================================

echo "[stop] Verifying server is stopped..."

# Check if any server processes are still running
REMAINING=$(pgrep -f "uvicorn server.server:app" 2>/dev/null || true)
if [ -n "$REMAINING" ]; then
    echo "WARNING: Some server processes may still be running: $REMAINING" >&2
else
    echo "[stop] ✓ All server processes stopped"
fi

# Check if port 8000 is still in use
if command -v netstat >/dev/null 2>&1; then
    PORT_CHECK=$(netstat -tlnp 2>/dev/null | grep ":8000 " || true)
    if [ -n "$PORT_CHECK" ]; then
        echo "WARNING: Port 8000 may still be in use:" >&2
        echo "$PORT_CHECK" >&2
    else
        echo "[stop] ✓ Port 8000 is free"
    fi
elif command -v ss >/dev/null 2>&1; then
    PORT_CHECK=$(ss -tlnp 2>/dev/null | grep ":8000 " || true)
    if [ -n "$PORT_CHECK" ]; then
        echo "WARNING: Port 8000 may still be in use:" >&2
        echo "$PORT_CHECK" >&2
    else
        echo "[stop] ✓ Port 8000 is free"
    fi
fi

echo "[stop] ✓ TTS server stopped successfully"
echo ""
echo "To restart the server:"
echo "  bash /app/start-server.sh"
echo "  bash /app/start-server.sh --background"
