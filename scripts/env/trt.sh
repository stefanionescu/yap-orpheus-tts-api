# TRT-LLM backend configuration
export BACKEND=${BACKEND:-trtllm}

# Path to prebuilt TensorRT-LLM engine directory (recommended for production)
export TRTLLM_ENGINE_DIR=${TRTLLM_ENGINE_DIR:-}

# Logging and runtime knobs
export TLLM_LOG_LEVEL=${TLLM_LOG_LEVEL:-INFO}

# Sampling defaults used by server
export ORPHEUS_MAX_TOKENS=${ORPHEUS_MAX_TOKENS:-1024}

# Engine build/runtime defaults - FP16 (no quantization)
export TRTLLM_DTYPE=${TRTLLM_DTYPE:-float16}
export TRTLLM_MAX_INPUT_LEN=${TRTLLM_MAX_INPUT_LEN:-128}
export TRTLLM_MAX_OUTPUT_LEN=${TRTLLM_MAX_OUTPUT_LEN:-1024}
export TRTLLM_MAX_BATCH_SIZE=${TRTLLM_MAX_BATCH_SIZE:-16}


