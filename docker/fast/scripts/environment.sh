#!/usr/bin/env bash
# Fast image environment defaults
# This file is automatically sourced in all bash shells

# Default API configuration
export YAP_API_KEY=${YAP_API_KEY:-"your_secret_key_here"}

# Default model configuration
export MODEL_ID=${MODEL_ID:-"canopylabs/orpheus-3b-0.1-ft"}
export MODELS_DIR=${MODELS_DIR:-"/opt/models"}
export ENGINES_DIR=${ENGINES_DIR:-"/opt/engines"}

# TensorRT-LLM configuration
export TRTLLM_ENGINE_DIR=${TRTLLM_ENGINE_DIR:-""}

# Hugging Face configuration
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1}
export HF_TRANSFER=${HF_TRANSFER:-1}

# Server configuration
export HOST=${HOST:-"0.0.0.0"}
export PORT=${PORT:-8000}

# Performance tuning
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}

# Default inference parameters (can be overridden)
export MAX_OUTPUT_LEN=${MAX_OUTPUT_LEN:-2048}
export TEMPERATURE=${TEMPERATURE:-0.7}
export TOP_P=${TOP_P:-0.9}
export TOP_K=${TOP_K:-50}
export REPETITION_PENALTY=${REPETITION_PENALTY:-1.0}
