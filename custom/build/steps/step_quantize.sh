#!/usr/bin/env bash
set -euo pipefail

source "custom/lib/common.sh"
load_env_if_present
load_environment
source "custom/build/helpers.sh"

echo "[step:quant] Quantization step..."

TRTLLM_REPO_DIR="${TRTLLM_REPO_DIR:-$PWD/.trtllm-repo}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$PWD/models/orpheus-trtllm-ckpt-int4-awq}"
PYTHON_EXEC="${PYTHON_EXEC:-python}"
TRTLLM_DTYPE="${TRTLLM_DTYPE:-float16}"
AWQ_BLOCK_SIZE="${AWQ_BLOCK_SIZE:-128}"
CALIB_SIZE="${CALIB_SIZE:-256}"

if [[ "${SKIP_QUANTIZATION:-false}" == true ]]; then
    echo "[step:quant] Skipping quantization (provided checkpoint)"
    _validate_downloaded_checkpoint "${CHECKPOINT_DIR}"
    exit 0
fi

echo "[step:quant] Installing quantization requirements (if present)..."
quant_requirements="${TRTLLM_REPO_DIR}/examples/quantization/requirements.txt"
if [ -f "${quant_requirements}" ]; then
    pip install -r "${quant_requirements}"
else
    echo "[step:quant] WARNING: quantization requirements.txt not found, continuing"
fi

export HF_HUB_ENABLE_HF_TRANSFER=1

if [[ -d "${CHECKPOINT_DIR}" && "${FORCE_REBUILD:-false}" != true ]]; then
    echo "[step:quant] Reusing existing quantized checkpoint at ${CHECKPOINT_DIR}"
else
    echo "[step:quant] Quantizing to INT4-AWQ..."
    rm -rf "${CHECKPOINT_DIR}"
    mkdir -p "${CHECKPOINT_DIR}"

    local_model_dir=""
    if [[ ! -d "${MODEL_ID}" ]]; then
        echo "[step:quant] Downloading model from HuggingFace: ${MODEL_ID}"
        local_model_dir="${PWD}/models/$(basename ${MODEL_ID})-hf"
        if [[ ! -d "${local_model_dir}" ]]; then
            mkdir -p "${local_model_dir}"
            "${PYTHON_EXEC}" -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='${MODEL_ID}', local_dir='${local_model_dir}', local_dir_use_symlinks=False)
print('✓ Downloaded complete model repository')
"
        else
            echo "[step:quant] Using cached HF model at ${local_model_dir}"
        fi
    else
        echo "[step:quant] Using local model directory: ${MODEL_ID}"
        local_model_dir="${MODEL_ID}"
    fi

    quant_script="${TRTLLM_REPO_DIR}/examples/quantization/quantize.py"
    if [ ! -f "${quant_script}" ]; then
        echo "[step:quant] ERROR: quantize.py not found at ${quant_script}" >&2
        exit 1
    fi

    quant_cmd=(
        "${PYTHON_EXEC}" "${quant_script}"
        --model_dir "${local_model_dir}"
        --output_dir "${CHECKPOINT_DIR}"
        --dtype "${TRTLLM_DTYPE}"
        --qformat int4_awq
        --awq_block_size "${AWQ_BLOCK_SIZE}"
        --calib_size "${CALIB_SIZE}"
        --kv_cache_dtype int8
    )
    echo "[step:quant] Running: ${quant_cmd[*]}"
    "${quant_cmd[@]}"
fi

echo "[step:quant] Validating quantized checkpoint..."
_validate_downloaded_checkpoint "${CHECKPOINT_DIR}"
echo "[step:quant] OK"


