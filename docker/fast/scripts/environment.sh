#!/usr/bin/env bash
# Fast image environment defaults (mirrors custom/environment.sh params)
# This file is automatically sourced in all bash shells

# =============================================================================
# SERVER CONFIGURATION
# =============================================================================
export HOST=${HOST:-0.0.0.0}
export PORT=${PORT:-8000}
export YAP_API_KEY=${YAP_API_KEY:-yap_api_key}

# Model and Hugging Face
export MODEL_ID=${MODEL_ID:-yapwithai/orpheus-3b-trt-int4-awq}
export HF_TOKEN=${HF_TOKEN:-}

# =============================================================================
# TENSORRT-LLM ENGINE CONFIGURATION
# =============================================================================
export TRTLLM_ENGINE_DIR=${TRTLLM_ENGINE_DIR:-}
export TRTLLM_MAX_INPUT_LEN=${TRTLLM_MAX_INPUT_LEN:-48}
export TRTLLM_MAX_OUTPUT_LEN=${TRTLLM_MAX_OUTPUT_LEN:-1024}
export TRTLLM_MAX_BATCH_SIZE=${TRTLLM_MAX_BATCH_SIZE:-16}
export KV_FREE_GPU_FRAC=${KV_FREE_GPU_FRAC:-0.92}
export KV_ENABLE_BLOCK_REUSE=${KV_ENABLE_BLOCK_REUSE:-0}

# =============================================================================
# TTS SYNTHESIS CONFIGURATION
# =============================================================================
export ORPHEUS_MAX_TOKENS=${ORPHEUS_MAX_TOKENS:-1024}
export DEFAULT_TEMPERATURE=${DEFAULT_TEMPERATURE:-0.40}
export DEFAULT_TOP_P=${DEFAULT_TOP_P:-0.9}
export DEFAULT_REPETITION_PENALTY=${DEFAULT_REPETITION_PENALTY:-1.30}
export SNAC_SR=${SNAC_SR:-24000}
export TTS_DECODE_WINDOW=${TTS_DECODE_WINDOW:-28}
export TTS_MAX_SEC=${TTS_MAX_SEC:-0}
export SNAC_TORCH_COMPILE=${SNAC_TORCH_COMPILE:-0}
export SNAC_MAX_BATCH=${SNAC_MAX_BATCH:-64}
export SNAC_BATCH_TIMEOUT_MS=${SNAC_BATCH_TIMEOUT_MS:-2}
export SNAC_GLOBAL_SYNC=${SNAC_GLOBAL_SYNC:-1}
export WS_END_SENTINEL=${WS_END_SENTINEL:-__END__}
export WS_CLOSE_BUSY_CODE=${WS_CLOSE_BUSY_CODE:-1013}
export WS_CLOSE_INTERNAL_CODE=${WS_CLOSE_INTERNAL_CODE:-1011}
export WS_QUEUE_MAXSIZE=${WS_QUEUE_MAXSIZE:-128}
export DEFAULT_VOICE=${DEFAULT_VOICE:-tara}
export YIELD_SLEEP_SECONDS=${YIELD_SLEEP_SECONDS:-0}
export STREAMING_DEFAULT_MAX_TOKENS=${STREAMING_DEFAULT_MAX_TOKENS:-1024}

# =============================================================================
# PERFORMANCE OPTIMIZATION
# =============================================================================
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-2}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.9,max_split_size_mb:512}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
export HF_TRANSFER=${HF_TRANSFER:-1}
export GPU_SM_ARCH=${GPU_SM_ARCH:-}

# =============================================================================
# HUGGING FACE REMOTE DEPLOY (PULL) - FAST IMAGE
# =============================================================================
export HF_DEPLOY_REPO_ID=${HF_DEPLOY_REPO_ID:-}
export HF_DEPLOY_USE=${HF_DEPLOY_USE:-auto}
export HF_DEPLOY_ENGINE_LABEL=${HF_DEPLOY_ENGINE_LABEL:-}
export HF_DEPLOY_SKIP_BUILD_IF_ENGINES=${HF_DEPLOY_SKIP_BUILD_IF_ENGINES:-1}
export HF_DEPLOY_STRICT_ENV_MATCH=${HF_DEPLOY_STRICT_ENV_MATCH:-1}
export HF_DEPLOY_WORKDIR=${HF_DEPLOY_WORKDIR:-/opt/models/_hf_download}
export HF_DEPLOY_VALIDATE=${HF_DEPLOY_VALIDATE:-1}

# =============================================================================
# DIRECTORIES
# =============================================================================
export MODELS_DIR=${MODELS_DIR:-/opt/models}
export ENGINES_DIR=${ENGINES_DIR:-/opt/engines}
