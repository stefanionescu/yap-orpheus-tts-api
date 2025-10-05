#!/usr/bin/env bash
# =============================================================================
# System Bootstrap Script
# =============================================================================
# Installs system dependencies required for TensorRT-LLM and TTS server.
# This script handles:
# - CUDA driver verification
# - Python development libraries
# - OpenMPI runtime (for multi-GPU builds)
# - Basic system utilities
#
# Usage: bash scripts/setup/bootstrap.sh
# Environment: Requires HF_TOKEN to be set
# =============================================================================

set -euo pipefail

# Load common utilities and environment
source "scripts/lib/common.sh"
load_env_if_present
load_environment

echo "=== System Bootstrap ==="

# =============================================================================
# Helper Functions
# =============================================================================

_check_system_libraries() {
    echo "[bootstrap] Verifying system libraries..."
    
    # Check OpenMPI
    if ! ldconfig -p 2>/dev/null | grep -q "libmpi.so"; then
        echo "WARNING: libmpi.so not found in library path" >&2
        echo "  Add /usr/lib/x86_64-linux-gnu to LD_LIBRARY_PATH if needed" >&2
    fi
    
    # Check Python shared library
    if ! ldconfig -p 2>/dev/null | grep -q "libpython${PYTHON_VERSION}.so"; then
        echo "WARNING: libpython${PYTHON_VERSION}.so not found in library path" >&2
        echo "  Install python3-dev or add library path to LD_LIBRARY_PATH" >&2
    fi
}

# =============================================================================
# CUDA Environment Check
# =============================================================================

echo "[bootstrap] Checking CUDA environment..."
if ! command -v nvidia-smi >/dev/null; then
    echo "ERROR: nvidia-smi not found. Please install NVIDIA drivers first." >&2
    exit 1
fi

CUDA_VER=$(detect_cuda_version)
echo "[bootstrap] Detected CUDA version: ${CUDA_VER:-unknown}"

# =============================================================================
# Python Environment Check
# =============================================================================

echo "[bootstrap] Checking Python environment..."
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

if ! command -v "python${PYTHON_VERSION}" >/dev/null 2>&1; then
    echo "[bootstrap] python${PYTHON_VERSION} not found."
    echo "[bootstrap] This is expected on managed container images (Runpod, etc.)"
else
    echo "[bootstrap] Found python${PYTHON_VERSION}"
fi

# =============================================================================
# System Dependencies Installation
# =============================================================================

if command -v apt-get >/dev/null 2>&1; then
    if [ "${SKIP_APT:-0}" = "1" ]; then
        echo "[bootstrap] SKIP_APT=1 → skipping system package installation"
    else
        echo "[bootstrap] Installing system dependencies..."
        
        # Update package lists
        apt-get update -y || {
            echo "WARNING: apt-get update failed, continuing anyway" >&2
        }
        
        # Install core dependencies
        DEBIAN_FRONTEND=noninteractive apt-get install -y \
            git wget curl jq \
            python3-venv python3-dev python3.10-venv python3.10-dev \
            libopenmpi-dev openmpi-bin || {
            echo "WARNING: Some packages failed to install, continuing anyway" >&2
        }
        
        # Verify critical libraries are available
        _check_system_libraries
    fi
else
    echo "[bootstrap] apt-get not available, skipping system package installation"
    echo "[bootstrap] Ensure the following are installed manually:"
    echo "  - Python ${PYTHON_VERSION} with development headers"
    echo "  - OpenMPI runtime and development libraries"
    echo "  - Basic build tools (git, wget, curl)"
fi

# =============================================================================
# Environment Validation
# =============================================================================

echo "[bootstrap] Validating environment..."

# Check HF token
require_env HF_TOKEN

echo "[bootstrap] ✓ System bootstrap completed successfully"
