# TRT-LLM backend configuration
export BACKEND=${BACKEND:-trtllm}

# Path to prebuilt TensorRT-LLM engine directory (recommended for production)
export TRTLLM_ENGINE_DIR=${TRTLLM_ENGINE_DIR:-}

# Logging and runtime knobs
export TLLM_LOG_LEVEL=${TLLM_LOG_LEVEL:-INFO}

# Sampling defaults used by server
export ORPHEUS_MAX_TOKENS=${ORPHEUS_MAX_TOKENS:-2048}

# Engine build/runtime defaults for A100
export TRTLLM_DTYPE=${TRTLLM_DTYPE:-float16}
export TRTLLM_MAX_INPUT_LEN=${TRTLLM_MAX_INPUT_LEN:-128}
export TRTLLM_MAX_OUTPUT_LEN=${TRTLLM_MAX_OUTPUT_LEN:-2048}
export TRTLLM_MAX_BATCH_SIZE=${TRTLLM_MAX_BATCH_SIZE:-16}

# KV-cache quantization defaults for A100 (use INT8)
export TRTLLM_KV_CACHE_DTYPE=${TRTLLM_KV_CACHE_DTYPE:-int8}

# Context attention fused kernels control (disable on A100)
export TRTLLM_CONTEXT_FMHA=${TRTLLM_CONTEXT_FMHA:-disable}


