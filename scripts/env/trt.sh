# TRT-LLM backend configuration
export BACKEND=${BACKEND:-trtllm}

# Path to prebuilt TensorRT-LLM engine directory (recommended for production)
export TRTLLM_ENGINE_DIR=${TRTLLM_ENGINE_DIR:-}

# Logging and runtime knobs
export TLLM_LOG_LEVEL=${TLLM_LOG_LEVEL:-INFO}

# KV cache memory allocation (use 94% of free GPU memory for KV cache)
export KV_FREE_GPU_FRAC=${KV_FREE_GPU_FRAC:-0.92}

# Sampling defaults used by server
export ORPHEUS_MAX_TOKENS=${ORPHEUS_MAX_TOKENS:-1024}

# Engine build/runtime defaults - INT4-AWQ + INT8 KV cache
export TRTLLM_DTYPE=${TRTLLM_DTYPE:-float16}
export TRTLLM_MAX_INPUT_LEN=${TRTLLM_MAX_INPUT_LEN:-48} # Optimized for sentence-by-sentence TTS
export TRTLLM_MAX_OUTPUT_LEN=${TRTLLM_MAX_OUTPUT_LEN:-1024}
export TRTLLM_MAX_BATCH_SIZE=${TRTLLM_MAX_BATCH_SIZE:-20}  # Increased for INT4-AWQ

# Scheduler configuration for high-concurrency streaming
# max_num_tokens = max concurrent tokens in-flight across all requests at any moment
# Formula: MAX_BATCH_SIZE * (MAX_INPUT_LEN + avg_inflight_output_tokens)
#
# For streaming workloads, avg_inflight is typically 40-60% of MAX_OUTPUT_LEN because:
# - Requests start/finish at different times (not all at 1024 tokens simultaneously)
# - Average request is ~halfway through generation at any moment
#
# Conservative (all requests near max):  20 * (48 + 1024) = 21,440  (may waste memory)
# Balanced (streaming workload):        20 * (48 + 512)  = 11,200  (recommended)
# Aggressive (fast turnover):           20 * (48 + 256)  = 6,080   (may cause waits)
export TRTLLM_MAX_NUM_TOKENS=${TRTLLM_MAX_NUM_TOKENS:-12288}


