#!/usr/bin/env bash
# Common helpers for scripts/

# Do not set -euo here; these functions are sourced by other scripts

# Load .env if present
load_env_if_present() {
  if [ -f ".env" ]; then
    # shellcheck disable=SC1091
    source ".env"
  fi
}

# Require an environment variable to be set (by name)
require_env() {
  local var_name="$1"
  # shellcheck disable=SC2016
  local val
  val=$(eval echo "\${$var_name:-}")
  if [ -z "$val" ]; then
    echo "[common] ERROR: $var_name not set. Export it in the shell or .env." >&2
    return 1
  fi
}

# Detect CUDA version via nvidia-smi (echoes X.Y or empty)
detect_cuda_version() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi | grep -o "CUDA Version: [0-9][0-9]*\.[0-9]*" | awk '{print $3}'
  else
    echo ""
  fi
}

# Map CUDA version to PyTorch index URL
map_torch_index_url() {
  local cuda_ver="$1"
  local cuda_minor
  if [ -z "$cuda_ver" ]; then
    echo "https://download.pytorch.org/whl/cu124"
    return 0
  fi
  cuda_minor=$(echo "$cuda_ver" | cut -d. -f1-2 | tr -d '.')
  case "$cuda_minor" in
    120|121) echo "https://download.pytorch.org/whl/cu121" ;;
    122|123|124) echo "https://download.pytorch.org/whl/cu124" ;;
    125|126) echo "https://download.pytorch.org/whl/cu126" ;;
    127|128|129) echo "https://download.pytorch.org/whl/cu128" ;;
    *) echo "https://download.pytorch.org/whl/cu124" ;;
  esac
}

# Choose a Python executable given $PYTHON_VERSION or best-effort
choose_python_exe() {
  local ver="${PYTHON_VERSION:-3.10}"
  if command -v python${ver} >/dev/null 2>&1; then
    echo "python${ver}"
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    echo "python3"
    return 0
  fi
  if command -v python >/dev/null 2>&1; then
    echo "python"
    return 0
  fi
  return 1
}

# Start a command in the background, write PID and tail logs
# Usage: start_background "<cmd>" [pid_file] [log_file]
start_background() {
  local cmd="$1"
  local pid_file="${2:-.run/server.pid}"
  local log_file="${3:-logs/server.log}"

  mkdir -p "$(dirname "$log_file")" "$(dirname "$pid_file")"
  # Fully detach from TTY so Ctrl-C on caller doesn't stop the server
  setsid bash -lc "$cmd" </dev/null > "$log_file" 2>&1 &
  local pid=$!
  echo $pid > "$pid_file"
  echo "[run] Server started in background (PID $pid)."
  echo "[run] Following logs (Ctrl-C detaches, server keeps running)"
  touch "$log_file" || true
  exec tail -n +1 -F "$log_file"
}

# Load centralized environment configuration
load_environment() {
  local env_file="${1:-scripts/environment.sh}"
  if [ -f "$env_file" ]; then
    # shellcheck disable=SC1090
    source "$env_file"
  else
    echo "[common] WARNING: Environment file $env_file not found" >&2
  fi
}

# Build a uvicorn command string for this server
build_uvicorn_cmd() {
  local host="${HOST:-0.0.0.0}"
  local port="${PORT:-8000}"
  echo "uvicorn server.server:app --host \"$host\" --port \"$port\" --timeout-keep-alive 75 --log-level info"
}


