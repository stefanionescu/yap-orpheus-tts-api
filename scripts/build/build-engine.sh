#!/usr/bin/env bash
# =============================================================================
# TensorRT-LLM Engine Build Script
# =============================================================================
# Builds optimized TensorRT-LLM engine for Orpheus 3B TTS model with:
# - INT4-AWQ weight quantization (4x compression)
# - INT8 KV cache (2x memory reduction)
# - Optimized for TTS workload (48 input, 1024 output tokens)
#
# Usage: bash scripts/build/build-engine.sh [--force] [--max-batch-size N]
# Environment: Requires HF_TOKEN, VENV_DIR, optionally TRTLLM_ENGINE_DIR
# =============================================================================

set -euo pipefail

# Load common utilities and environment
source "scripts/lib/common.sh"
load_env_if_present
load_environment

echo "=== TensorRT-LLM Engine Build ==="

# =============================================================================
# Helper Functions
# =============================================================================

_setup_huggingface_auth() {
    if [ -n "${HF_TOKEN:-}" ]; then
        export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-$HF_TOKEN}"
        export HF_HUB_TOKEN="${HF_HUB_TOKEN:-$HF_TOKEN}"
    else
        echo "ERROR: HF_TOKEN not set" >&2
        exit 1
    fi
}

_should_skip_build() {
    if [ "$FORCE_REBUILD" = "1" ]; then
        return 1  # Force rebuild
    fi
    
    if [ -f "$ENGINE_OUTPUT_DIR/rank0.engine" ] && [ -f "$ENGINE_OUTPUT_DIR/config.json" ]; then
        return 0  # Skip build
    fi
    
    return 1  # Need to build
}

_prepare_tensorrt_repo() {
    echo "[build] Preparing TensorRT-LLM repository..."
    
    # Remove existing repo if it exists (ensure clean state)
    if [ -d "$TRTLLM_REPO_DIR" ]; then
        echo "[build] Removing existing TensorRT-LLM repository for clean clone..."
        rm -rf "$TRTLLM_REPO_DIR"
    fi
    
    echo "[build] Cloning TensorRT-LLM repository..."
    git clone --depth 1 --branch v1.0.0 \
        https://github.com/NVIDIA/TensorRT-LLM.git "$TRTLLM_REPO_DIR"
    
    # Check for quantization examples directory
    local examples_dir=""
    if [ -d "$TRTLLM_REPO_DIR/examples/quantization" ]; then
        examples_dir="$TRTLLM_REPO_DIR/examples/quantization"
    elif [ -d "$TRTLLM_REPO_DIR/examples" ]; then
        echo "[build] Available examples:"
        ls -la "$TRTLLM_REPO_DIR/examples/"
        echo "ERROR: quantization examples not found. Available examples listed above." >&2
        exit 1
    else
        echo "ERROR: No examples directory found in TensorRT-LLM repository" >&2
        echo "Repository contents:" >&2
        ls -la "$TRTLLM_REPO_DIR/" >&2
        exit 1
    fi
    
    echo "[build] Using quantization examples directory: $examples_dir"
    export TRTLLM_EXAMPLES_DIR="$examples_dir"
}

_build_optimized_engine() {
    echo "[build] Building INT4-AWQ + INT8 KV cache engine (2-step process)..."
    
    # Build configuration
    local max_batch_size="${CUSTOM_MAX_BATCH_SIZE:-${TRTLLM_MAX_BATCH_SIZE}}"
    local max_input_len="${TRTLLM_MAX_INPUT_LEN}"
    local max_output_len="${TRTLLM_MAX_OUTPUT_LEN}"
    local dtype="${TRTLLM_DTYPE}"
    local checkpoint_dir="${ENGINE_OUTPUT_DIR}_checkpoint"
    local awq_block_size="${AWQ_BLOCK_SIZE:-128}"
    local calib_size="${CALIB_SIZE:-256}"
    
    echo "[build] Configuration:"
    echo "  Model: ${MODEL_ID}"
    echo "  Max batch size: $max_batch_size"
    echo "  Max input length: $max_input_len"
    echo "  Max output length: $max_output_len"
    echo "  Data type: $dtype"
    echo "  AWQ block size: $awq_block_size"
    echo "  Calibration size: $calib_size"
    echo "  Quantization: INT4-AWQ weights, INT8 KV cache"
    
    # Create output directories
    mkdir -p "$ENGINE_OUTPUT_DIR" "$checkpoint_dir"
    
    # Change to quantization examples directory
    cd "$TRTLLM_EXAMPLES_DIR"
    
    # Install quantization requirements if available
    if [ -f "requirements.txt" ]; then
        echo "[build] Installing quantization requirements..."
        pip install -r requirements.txt
    fi
    
    # Enable fast HF downloads
    export HF_HUB_ENABLE_HF_TRANSFER=1
    
    # Step 1: Quantize the model using quantize.py
    echo "[build] Step 1/2: Quantizing model to INT4-AWQ..."
    
    # Download model from HF if it's a model ID (not a local path)
    local model_dir_for_quant
    if [[ ! -d "${MODEL_ID}" ]]; then
        echo "[build] Downloading model from HuggingFace: ${MODEL_ID}"
        local local_model_dir="${PWD}/models/$(basename ${MODEL_ID})-hf"
        
        if [[ ! -d "${local_model_dir}" ]]; then
            mkdir -p "${local_model_dir}"
            python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='${MODEL_ID}',
    local_dir='${local_model_dir}',
    local_dir_use_symlinks=False
)
"
        else
            echo "[build] Using cached HF model at ${local_model_dir}"
        fi
        model_dir_for_quant="${local_model_dir}"
    else
        echo "[build] Using local model directory: ${MODEL_ID}"
        model_dir_for_quant="${MODEL_ID}"
    fi
    
    # Run quantization
    python quantize.py \
        --model_dir "${model_dir_for_quant}" \
        --output_dir "${checkpoint_dir}" \
        --dtype "${dtype}" \
        --qformat int4_awq \
        --awq_block_size "${awq_block_size}" \
        --calib_size "${calib_size}" \
        --kv_cache_dtype int8
    
    # Validate quantized checkpoint
    echo "[build] Validating quantized checkpoint..."
    if [ ! -f "${checkpoint_dir}/config.json" ]; then
        echo "ERROR: No config.json found in ${checkpoint_dir}" >&2
        exit 1
    fi
    if ! ls "${checkpoint_dir}"/rank*.safetensors >/dev/null 2>&1; then
        echo "ERROR: No rank*.safetensors found in ${checkpoint_dir}" >&2
        exit 1
    fi
    echo "[build] Quantized checkpoint validation passed"
    
    # Step 2: Build TensorRT engine from quantized checkpoint
    echo "[build] Step 2/2: Building TensorRT engine from quantized checkpoint..."
    
    trtllm-build \
        --checkpoint_dir "${checkpoint_dir}" \
        --output_dir "${ENGINE_OUTPUT_DIR}" \
        --gemm_plugin auto \
        --gpt_attention_plugin float16 \
        --context_fmha enable \
        --paged_kv_cache enable \
        --remove_input_padding enable \
        --max_input_len "${max_input_len}" \
        --max_seq_len $((max_input_len + max_output_len)) \
        --max_batch_size "${max_batch_size}" \
        --log_level info \
        --workers "$(nproc --all)"
    
    # Clean up intermediate checkpoint to save space
    echo "[build] Cleaning up intermediate checkpoint..."
    rm -rf "${checkpoint_dir}"
    
    # Return to original directory
    cd - >/dev/null
}

_validate_engine() {
    echo "[build] Validating built engine..."
    
    # Check required files
    local required_files=(
        "rank0.engine"
        "config.json"
    )
    
    for file in "${required_files[@]}"; do
        if [ ! -f "$ENGINE_OUTPUT_DIR/$file" ]; then
            echo "ERROR: Required file missing: $ENGINE_OUTPUT_DIR/$file" >&2
            exit 1
        fi
    done
    
    # Check engine file size (should be reasonable for INT4-AWQ)
    local engine_size
    engine_size=$(stat -f%z "$ENGINE_OUTPUT_DIR/rank0.engine" 2>/dev/null || stat -c%s "$ENGINE_OUTPUT_DIR/rank0.engine" 2>/dev/null || echo "0")
    
    if [ "$engine_size" -lt 1000000 ]; then  # Less than 1MB is suspicious
        echo "WARNING: Engine file seems unusually small ($engine_size bytes)" >&2
    fi
    
    echo "[build] ✓ Engine validation passed"
}

# =============================================================================
# Configuration and Argument Parsing
# =============================================================================

# Default paths and settings
VENV_DIR="${VENV_DIR:-$PWD/.venv}"
TRTLLM_REPO_DIR="${TRTLLM_REPO_DIR:-$PWD/.trtllm-repo}"
MODELS_DIR="${MODELS_DIR:-$PWD/models}"
ENGINE_OUTPUT_DIR="${TRTLLM_ENGINE_DIR:-$MODELS_DIR/orpheus-trt-int4-awq}"

# Parse command line arguments
FORCE_REBUILD=0
CUSTOM_MAX_BATCH_SIZE=""

for arg in "$@"; do
    case "$arg" in
        --force)
            FORCE_REBUILD=1
            ;;
        --max-batch-size)
            shift
            CUSTOM_MAX_BATCH_SIZE="$1"
            ;;
        --max-batch-size=*)
            CUSTOM_MAX_BATCH_SIZE="${arg#*=}"
            ;;
        *)
            echo "Unknown argument: $arg" >&2
            echo "Usage: $0 [--force] [--max-batch-size N]" >&2
            exit 1
            ;;
    esac
done

# =============================================================================
# Environment Validation
# =============================================================================

echo "[build] Validating build environment..."

# Check virtual environment
if [ ! -d "$VENV_DIR" ]; then
    echo "ERROR: Virtual environment not found at $VENV_DIR" >&2
    echo "Run scripts/setup/install-dependencies.sh first" >&2
    exit 1
fi

# Check GPU availability
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not detected. GPU required for engine build." >&2
    exit 1
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Set up HuggingFace authentication
_setup_huggingface_auth

# =============================================================================
# Engine Build Process
# =============================================================================

# Check if rebuild is needed
if _should_skip_build; then
    echo "[build] Engine already exists at: $ENGINE_OUTPUT_DIR"
    echo "[build] Use --force to rebuild"
    exit 0
fi

echo "[build] Building TensorRT-LLM engine..."
echo "[build] Output directory: $ENGINE_OUTPUT_DIR"

# Prepare TensorRT-LLM repository
_prepare_tensorrt_repo

# Build engine with optimized settings
_build_optimized_engine

# Validate built engine
_validate_engine

echo "[build] ✓ Engine build completed successfully"
echo "[build] Engine location: $ENGINE_OUTPUT_DIR"
