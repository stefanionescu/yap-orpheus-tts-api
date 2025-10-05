#!/usr/bin/env bash
# =============================================================================
# Python Dependencies Installation Script
# =============================================================================
# Creates Python virtual environment and installs all required dependencies:
# - PyTorch with appropriate CUDA support
# - TensorRT-LLM from NVIDIA PyPI
# - All Python package dependencies
# - Validates critical runtime libraries
#
# Usage: bash scripts/setup/install-dependencies.sh
# Environment: Requires HF_TOKEN, optionally PYTHON_VERSION, VENV_DIR
# =============================================================================

set -euo pipefail

# Load common utilities and environment
source "scripts/lib/common.sh"
load_env_if_present
load_environment

echo "=== Python Dependencies Installation ==="

# =============================================================================
# Helper Functions
# =============================================================================

_ensure_venv_support() {
    local py_exe="$1"
    local py_majmin="$2"
    
    if ! $py_exe -m ensurepip --version >/dev/null 2>&1; then
        if command -v apt-get >/dev/null 2>&1; then
            echo "[install] Installing Python venv support..."
            apt-get update -y || true
            DEBIAN_FRONTEND=noninteractive apt-get install -y \
                "python${py_majmin}-venv" || \
                DEBIAN_FRONTEND=noninteractive apt-get install -y python3-venv || true
        else
            echo "WARNING: ensurepip missing and apt-get unavailable. venv creation may fail." >&2
        fi
    fi
}

_bootstrap_pip() {
    # Robust pip bootstrap (handles broken vendor dependencies)
    python -m ensurepip --upgrade || true
    
    if ! python -m pip --version >/dev/null 2>&1; then
        python -m ensurepip --upgrade || true
    fi
    
    python -m pip install --upgrade --no-cache-dir pip setuptools wheel || {
        python -m ensurepip --upgrade || true
        python -m pip install --upgrade --no-cache-dir pip setuptools wheel
    }
}

_install_pytorch() {
    # Detect CUDA version and select appropriate PyTorch index
    local cuda_ver="${CUDA_VER:-$(detect_cuda_version)}"
    local torch_idx=$(map_torch_index_url "$cuda_ver")
    
    echo "[install] Installing PyTorch from: $torch_idx"
    pip install --index-url "$torch_idx" torch --only-binary=:all:
}

_install_tensorrt_llm() {
    # Set up CUDA library path for TensorRT-LLM
    if command -v ldconfig >/dev/null 2>&1; then
        local cuda_lib_dir
        cuda_lib_dir=$(ldconfig -p 2>/dev/null | awk '/libcuda\\.so/{print $NF; exit}' | xargs dirname 2>/dev/null || true)
        
        if [ -n "${cuda_lib_dir:-}" ]; then
            case ":${LD_LIBRARY_PATH:-}:" in
                *":${cuda_lib_dir}:"*) : ;;
                *) export LD_LIBRARY_PATH="${cuda_lib_dir}:${LD_LIBRARY_PATH:-}" ;;
            esac
        fi
    fi
    
    # Install TensorRT-LLM and related packages
    pip install --upgrade --extra-index-url https://pypi.nvidia.com \
        "${TRTLLM_WHEEL_URL}" \
        "tensorrt-cu12-bindings" \
        "tensorrt-cu12-libs"
}

_validate_python_libraries() {
    echo "[install] Checking Python shared library..."
    python - <<'EOF'
import ctypes
import ctypes.util
import sys

version = f"{sys.version_info.major}.{sys.version_info.minor}"
lib_name = ctypes.util.find_library(f"python{version}")

if not lib_name:
    raise SystemExit(
        "Unable to locate libpython shared library. "
        "Install python3-dev and ensure LD_LIBRARY_PATH includes its directory."
    )

try:
    ctypes.CDLL(lib_name)
except OSError as exc:
    raise SystemExit(f"Found {lib_name} but failed to load it: {exc}")

print("✓ Python shared library OK")
EOF
}

_validate_cuda_runtime() {
    echo "[install] Checking CUDA Python bindings..."
    local check_output
    check_output=$(python - <<'EOF'
import sys
from importlib.metadata import PackageNotFoundError, version

try:
    ver = version("cuda-python")
except PackageNotFoundError:
    print("MISSING: cuda-python not installed")
    sys.exit(1)

major = int(ver.split(".", 1)[0])
try:
    if major >= 13:
        from cuda.bindings import runtime as cudart
    else:
        from cuda import cudart
except Exception as exc:
    print(f"IMPORT_ERROR: {type(exc).__name__}: {exc}")
    sys.exit(1)

err, _ = cudart.cudaDriverGetVersion()
if err != 0:
    print(f"CUDART_ERROR: cudaDriverGetVersion -> {err}")
    sys.exit(1)

print("✓ CUDA runtime OK")
EOF
    ) || true
    
    if ! echo "$check_output" | grep -q "✓ CUDA runtime OK"; then
        echo "ERROR: CUDA Python bindings not working:" >&2
        echo "$check_output" >&2
        echo "Hint: Ensure cuda-python>=12.6,<13 and CUDA libraries are available" >&2
        exit 1
    fi
    
    echo "$check_output"
}

_validate_mpi_runtime() {
    local need_mpi="${NEED_MPI:-0}"
    
    if [ "$need_mpi" = "1" ]; then
        echo "[install] Checking MPI runtime..."
        python - <<'EOF'
import sys
try:
    from mpi4py import MPI
    MPI.Get_version()
    print("✓ MPI runtime OK")
except ImportError as exc:
    sys.exit(f"mpi4py not installed: {exc}")
except Exception as exc:
    sys.exit(f"MPI runtime error: {exc}")
EOF
    else
        echo "[install] Skipping MPI check (NEED_MPI=0)"
    fi
}

_validate_huggingface_auth() {
    echo "[install] Validating HuggingFace authentication..."
    python - <<'EOF'
import os
from huggingface_hub import login

token = os.environ.get("HF_TOKEN")
if not token:
    raise SystemExit("HF_TOKEN not set")

login(token=token, add_to_git_credential=False)
print("✓ HuggingFace authentication OK")
EOF
}

# =============================================================================
# Configuration
# =============================================================================

PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
VENV_DIR="${VENV_DIR:-$PWD/.venv}"
TRTLLM_WHEEL_URL="${TRTLLM_WHEEL_URL:-https://pypi.nvidia.com/tensorrt-llm/tensorrt_llm-1.0.0-cp310-cp310-linux_x86_64.whl}"

# Validate environment
require_env HF_TOKEN

# =============================================================================
# Platform Check
# =============================================================================

if [ "$(uname -s)" != "Linux" ]; then
    echo "[install] TensorRT-LLM requires Linux with NVIDIA GPUs. Skipping installation."
    exit 0
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "WARNING: nvidia-smi not detected. GPU functionality may not work." >&2
fi

# =============================================================================
# Python Virtual Environment Setup
# =============================================================================

echo "[install] Setting up Python virtual environment..."

# Find Python executable
PY_EXE=$(choose_python_exe) || {
    echo "ERROR: Python ${PYTHON_VERSION} not found. Please install it first." >&2
    exit 1
}

echo "[install] Using Python: $PY_EXE"
PY_MAJMIN=$($PY_EXE -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "[install] Python version: $PY_MAJMIN"

# Ensure venv module is available
_ensure_venv_support "$PY_EXE" "$PY_MAJMIN"

# Create virtual environment
echo "[install] Creating virtual environment at: $VENV_DIR"
$PY_EXE -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# =============================================================================
# Python Package Installation
# =============================================================================

echo "[install] Upgrading pip and core tools..."
_bootstrap_pip

echo "[install] Installing PyTorch with CUDA support..."
_install_pytorch

echo "[install] Installing application dependencies..."
pip install -r requirements.txt

echo "[install] Installing TensorRT-LLM..."
_install_tensorrt_llm

# =============================================================================
# Runtime Validation
# =============================================================================

echo "[install] Validating installation..."
_validate_python_libraries
_validate_cuda_runtime
_validate_mpi_runtime
_validate_huggingface_auth

echo "[install] Running dependency check..."
pip check || {
    echo "ERROR: Dependency conflicts detected" >&2
    exit 1
}

echo "[install] ✓ All dependencies installed successfully"
