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
    
    if [ ! -d "$TRTLLM_REPO_DIR" ]; then
        echo "[build] Cloning TensorRT-LLM repository..."
        git clone --depth 1 --branch v1.0.0 \
            https://github.com/NVIDIA/TensorRT-LLM.git "$TRTLLM_REPO_DIR"
    else
        echo "[build] Using existing TensorRT-LLM repository"
    fi
    
    # Ensure examples directory exists
    if [ ! -d "$TRTLLM_REPO_DIR/examples/llama" ]; then
        echo "ERROR: TensorRT-LLM examples not found in $TRTLLM_REPO_DIR" >&2
        exit 1
    fi
}

_build_optimized_engine() {
    echo "[build] Building INT4-AWQ + INT8 KV cache engine..."
    
    # Create output directory
    mkdir -p "$ENGINE_OUTPUT_DIR"
    
    # Build configuration
    local max_batch_size="${CUSTOM_MAX_BATCH_SIZE:-${TRTLLM_MAX_BATCH_SIZE}}"
    local max_input_len="${TRTLLM_MAX_INPUT_LEN}"
    local max_output_len="${TRTLLM_MAX_OUTPUT_LEN}"
    local dtype="${TRTLLM_DTYPE}"
    
    echo "[build] Configuration:"
    echo "  Model: ${MODEL_ID}"
    echo "  Max batch size: $max_batch_size"
    echo "  Max input length: $max_input_len"
    echo "  Max output length: $max_output_len"
    echo "  Data type: $dtype"
    echo "  Quantization: INT4-AWQ weights, INT8 KV cache"
    
    # Change to TensorRT-LLM examples directory
    cd "$TRTLLM_REPO_DIR/examples/llama"
    
    # Build engine with optimized settings for TTS
    python build.py \
        --model_dir "${MODEL_ID}" \
        --output_dir "$ENGINE_OUTPUT_DIR" \
        --dtype "$dtype" \
        --use_gpt_attention_plugin "$dtype" \
        --use_gemm_plugin "$dtype" \
        --use_weight_only \
        --weight_only_precision int4_awq \
        --per_group \
        --enable_context_fmha \
        --enable_context_fmha_fp32_acc \
        --multi_block_mode \
        --use_paged_context_fmha \
        --use_fp8_context_fmha \
        --kv_cache_type int8 \
        --max_batch_size "$max_batch_size" \
        --max_input_len "$max_input_len" \
        --max_seq_len $((max_input_len + max_output_len)) \
        --max_num_tokens $((max_batch_size * max_input_len)) \
        --builder_opt 4 \
        --strongly_typed
    
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
