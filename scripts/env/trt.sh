# TRT-LLM backend configuration
export BACKEND=${BACKEND:-vllm}                    # set to 'trtllm' to enable TensorRT-LLM backend

# Path to prebuilt TensorRT-LLM engine directory (recommended for production)
export TRTLLM_ENGINE_DIR=${TRTLLM_ENGINE_DIR:-}

# Logging and runtime knobs
export TENSORRT_LLM_LOG_LEVEL=${TENSORRT_LLM_LOG_LEVEL:-ERROR}

# Sampling defaults used by server
export ORPHEUS_MAX_TOKENS=${ORPHEUS_MAX_TOKENS:-2048}

# Engine build/runtime defaults (overridable)
export TRTLLM_DTYPE=${TRTLLM_DTYPE:-bfloat16}            # float16|bfloat16
export TRTLLM_MAX_INPUT_LEN=${TRTLLM_MAX_INPUT_LEN:-2048}
export TRTLLM_MAX_OUTPUT_LEN=${TRTLLM_MAX_OUTPUT_LEN:-2048}
export TRTLLM_MAX_BATCH_SIZE=${TRTLLM_MAX_BATCH_SIZE:-16}


