# TRT-LLM backend configuration
export BACKEND=${BACKEND:-vllm}                    # set to 'trtllm' to enable TensorRT-LLM backend

# Path to prebuilt TensorRT-LLM engine directory (recommended for production)
export TRTLLM_ENGINE_DIR=${TRTLLM_ENGINE_DIR:-}

# Logging and runtime knobs
export TENSORRT_LLM_LOG_LEVEL=${TENSORRT_LLM_LOG_LEVEL:-ERROR}

# Sampling defaults used by server
export ORPHEUS_MAX_TOKENS=${ORPHEUS_MAX_TOKENS:-2048}


