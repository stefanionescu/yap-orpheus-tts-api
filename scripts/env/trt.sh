# TRT-LLM backend configuration
export BACKEND=${BACKEND:-trtllm}

# Path to prebuilt TensorRT-LLM engine directory (recommended for production)
export TRTLLM_ENGINE_DIR=${TRTLLM_ENGINE_DIR:-}

# Logging and runtime knobs
export TLLM_LOG_LEVEL=${TLLM_LOG_LEVEL:-INFO}

# KV cache memory allocation (use 94% of free GPU memory for KV cache)
export KV_FREE_GPU_FRAC=${KV_FREE_GPU_FRAC:-0.94}

# Sampling defaults used by server
export ORPHEUS_MAX_TOKENS=${ORPHEUS_MAX_TOKENS:-1024}

# Engine build/runtime defaults - INT4-AWQ + INT8 KV cache
export TRTLLM_DTYPE=${TRTLLM_DTYPE:-float16}
export TRTLLM_MAX_INPUT_LEN=${TRTLLM_MAX_INPUT_LEN:-48} # Optimized for sentence-by-sentence TTS
export TRTLLM_MAX_OUTPUT_LEN=${TRTLLM_MAX_OUTPUT_LEN:-1024}
export TRTLLM_MAX_BATCH_SIZE=${TRTLLM_MAX_BATCH_SIZE:-24}  # Increased for INT4-AWQ


