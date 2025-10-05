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
    if [ "$FORCE_REBUILD" = true ]; then
        return 1  # Force rebuild
    fi
    
    if [ -f "$ENGINE_OUTPUT_DIR/rank0.engine" ] && [ -f "$ENGINE_OUTPUT_DIR/config.json" ]; then
        return 0  # Skip build
    fi
    
    return 1  # Need to build
}

_prepare_tensorrt_repo() {
    echo "[build] Preparing TensorRT-LLM repository..."
    
    # If forcing rebuild, remove existing repo for clean state
    if [ "$FORCE_REBUILD" = true ] && [ -d "${TRTLLM_REPO_DIR}" ]; then
        echo "[build] --force specified: removing existing TensorRT-LLM repository..."
        rm -rf "${TRTLLM_REPO_DIR}"
    fi
    
    # Ensure TensorRT-LLM repo is available for quantization scripts
    if [ ! -d "${TRTLLM_REPO_DIR}" ]; then
        echo "[build] Cloning TensorRT-LLM repo for quantization scripts..."
        git clone "${TRTLLM_REPO_URL}" "${TRTLLM_REPO_DIR}"
    fi
    
    # Checkout specific commit from Yap-With-AI fork
    echo "[build] Checking out specific commit from Yap-With-AI fork..."
    echo "[build] Using commit: ${TRTLLM_COMMIT}"
    
    if ! git -C "${TRTLLM_REPO_DIR}" checkout "${TRTLLM_COMMIT}" 2>/dev/null; then
        echo "[build] ERROR: Could not checkout commit ${TRTLLM_COMMIT}" >&2
        echo "[build] Repository may need to be re-cloned. Try with --force flag." >&2
        exit 1
    fi
    
    # Verify quantization directory exists
    if [ ! -d "$TRTLLM_REPO_DIR/examples/quantization" ]; then
        echo "ERROR: quantization examples not found in $TRTLLM_REPO_DIR/examples/quantization" >&2
        echo "Available examples:" >&2
        ls -la "$TRTLLM_REPO_DIR/examples/" >&2
        exit 1
    fi
    
    export TRTLLM_EXAMPLES_DIR="$TRTLLM_REPO_DIR/examples/quantization"
}

_build_optimized_engine() {
    echo "[build] Building INT4-AWQ + INT8 KV cache engine (2-step process)..."
    
    # Build configuration (using variables that can be overridden by args)
    local max_batch_size="${TRTLLM_MAX_BATCH_SIZE}"
    local max_input_len="${TRTLLM_MAX_INPUT_LEN}"
    local max_output_len="${TRTLLM_MAX_OUTPUT_LEN}"
    local dtype="${TRTLLM_DTYPE}"
    local checkpoint_dir="${CHECKPOINT_DIR}"
    local awq_block_size="${AWQ_BLOCK_SIZE}"
    local calib_size="${CALIB_SIZE}"
    
    echo "[build] Configuration:"
    echo "  Model: ${MODEL_ID}"
    echo "  Max batch size: $max_batch_size"
    echo "  Max input length: $max_input_len"
    echo "  Max output length: $max_output_len"
    echo "  Data type: $dtype"
    echo "  AWQ block size: $awq_block_size"
    echo "  Calibration size: $calib_size"
    echo "  Quantization: INT4-AWQ weights, INT8 KV cache"
    
    echo "[build] ============================================"
    echo "[build] Step 1/2: Quantize to INT4-AWQ (weight-only)"
    echo "[build] ============================================"
    
    # Install quantization dependencies
    local quant_requirements="${TRTLLM_REPO_DIR}/examples/quantization/requirements.txt"
    if [ -f "${quant_requirements}" ]; then
        echo "[build] Installing quantization requirements..."
        pip install -r "${quant_requirements}"
    else
        echo "[build] WARNING: quantization requirements.txt not found, skipping..."
    fi
    
    # Enable fast HF downloads
    export HF_HUB_ENABLE_HF_TRANSFER=1
    
    # Skip quantization if directory exists and not forcing
    if [[ -d "${checkpoint_dir}" && "${FORCE_REBUILD}" != true ]]; then
        echo "[build] Reusing existing quantized checkpoint at ${checkpoint_dir}"
    else
        echo "[build] Quantizing to INT4-AWQ..."
        rm -rf "${checkpoint_dir}"
        mkdir -p "${checkpoint_dir}"
        
        # Download model from HF if it's a model ID (not a local path)
        local model_dir_for_quant
        if [[ ! -d "${MODEL_ID}" ]]; then
            echo "[build] Downloading model from HuggingFace: ${MODEL_ID}"
            local local_model_dir="${PWD}/models/$(basename ${MODEL_ID})-hf"
            
            if [[ ! -d "${local_model_dir}" ]]; then
                mkdir -p "${local_model_dir}"
                "${PYTHON_EXEC}" -c "
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
        
        # Quantize using ModelOpt
        local quant_script="${TRTLLM_REPO_DIR}/examples/quantization/quantize.py"
        
        if [ ! -f "${quant_script}" ]; then
            echo "[build] ERROR: quantize.py not found at ${quant_script}" >&2
            exit 1
        fi
        
        local quant_cmd=(
            "${PYTHON_EXEC}" "${quant_script}"
            --model_dir "${model_dir_for_quant}"
            --output_dir "${checkpoint_dir}"
            --dtype "${dtype}"
            --qformat int4_awq
            --awq_block_size "${awq_block_size}"
            --calib_size "${calib_size}"
            --kv_cache_dtype int8
        )
        echo "[build] Running: ${quant_cmd[*]}"
        "${quant_cmd[@]}"
    fi
    
    # Sanity check the quantized checkpoint before building
    echo "[build] Validating quantized checkpoint..."
    test -f "${checkpoint_dir}/config.json" || { echo "[build] ERROR: No config.json found in ${checkpoint_dir}"; exit 1; }
    ls "${checkpoint_dir}"/rank*.safetensors >/dev/null 2>&1 || { echo "[build] ERROR: No rank*.safetensors found in ${checkpoint_dir}"; exit 1; }
    echo "[build] Quantized checkpoint validation passed."
    
    echo ""
    echo "[build] ============================================"
    echo "[build] Step 2/2: Build TensorRT-LLM engine"
    echo "[build] ============================================"
    
    # Skip engine build if directory exists and not forcing
    if [[ -d "${ENGINE_OUTPUT_DIR}" && "${FORCE_REBUILD}" != true ]]; then
        echo "[build] Reusing existing engine at ${ENGINE_OUTPUT_DIR}"
    else
        echo "[build] Building TensorRT INT4-AWQ engine..."
        
        local build_cmd=(
            trtllm-build
            --checkpoint_dir "${checkpoint_dir}"
            --output_dir "${ENGINE_OUTPUT_DIR}"
            --gemm_plugin auto
            --gpt_attention_plugin float16
            --context_fmha enable
            --paged_kv_cache enable
            --remove_input_padding enable
            --max_input_len "${max_input_len}"
            --max_seq_len "$((max_input_len + max_output_len))"
            --max_batch_size "${max_batch_size}"
            --log_level info
            --workers "$(nproc --all)"
        )
        echo "[build] Running: ${build_cmd[*]}"
        "${build_cmd[@]}"
    fi
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

# Default configuration (matching original script)
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$PWD/models/orpheus-trtllm-ckpt-int4-awq}"
PYTHON_EXEC="${PYTHON_EXEC:-python}"

usage() {
    cat <<USAGE
Usage: $0 [--model ID_OR_PATH] [--checkpoint-dir DIR] [--engine-dir DIR] [--dtype float16|bfloat16] [--max-input-len N] [--max-output-len N] [--max-batch-size N] [--force]

End-to-end INT4-AWQ build (weight-only quantization):
  HF checkpoint → Quantized checkpoint → TRT engine

Defaults:
  --model              ${MODEL_ID}
  --checkpoint-dir     ${CHECKPOINT_DIR}
  --engine-dir         ${ENGINE_OUTPUT_DIR}
  --dtype              ${TRTLLM_DTYPE}
  --max-input-len      ${TRTLLM_MAX_INPUT_LEN}
  --max-output-len     ${TRTLLM_MAX_OUTPUT_LEN}
  --max-batch-size     ${TRTLLM_MAX_BATCH_SIZE}
  --awq-block-size     ${AWQ_BLOCK_SIZE}
  --calib-size         ${CALIB_SIZE}
USAGE
}

# Parse command line arguments (matching original)
ARGS=()
FORCE_REBUILD=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL_ID="$2"; shift 2 ;;
        --checkpoint-dir) CHECKPOINT_DIR="$2"; shift 2 ;;
        --engine-dir) ENGINE_OUTPUT_DIR="$2"; shift 2 ;;
        --dtype) TRTLLM_DTYPE="$2"; shift 2 ;;
        --max-input-len) TRTLLM_MAX_INPUT_LEN="$2"; shift 2 ;;
        --max-output-len) TRTLLM_MAX_OUTPUT_LEN="$2"; shift 2 ;;
        --max-batch-size) TRTLLM_MAX_BATCH_SIZE="$2"; shift 2 ;;
        --awq-block-size) AWQ_BLOCK_SIZE="$2"; shift 2 ;;
        --calib-size) CALIB_SIZE="$2"; shift 2 ;;
        --force) FORCE_REBUILD=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) ARGS+=("$1"); shift ;;
    esac
done
set -- "${ARGS[@]:-}"

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

echo ""
echo "[build] ============================================"
echo "[build] Done. Engine: ${ENGINE_OUTPUT_DIR}"
echo "[build] Configuration: INT4-AWQ weight-only"
echo "[build] Model weights: 6GB → ~1.5GB (≈4x smaller)"
echo "[build] To run server:"
echo "  export TRTLLM_ENGINE_DIR=\"${ENGINE_OUTPUT_DIR}\""
echo "  bash scripts/03-run-server.sh"
echo "[build] ============================================"
