# TensorRT-LLM specific environment defaults
# These settings are sourced by scripts/02-build-trt-engine.sh.

# Target GPU architecture to avoid JITing for every SM variant.
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.0}

# Runtime defaults (aligned with vLLM 2048-token windows).
export TRTLLM_DTYPE=${TRTLLM_DTYPE:-float16}
export TRTLLM_MAX_INPUT_LEN=${TRTLLM_MAX_INPUT_LEN:-2048}
export TRTLLM_MAX_OUTPUT_LEN=${TRTLLM_MAX_OUTPUT_LEN:-2048}
export TRTLLM_MAX_BATCH_SIZE=${TRTLLM_MAX_BATCH_SIZE:-16}

# Quieter logs by default (INFO is the lowest level TRT-LLM accepts).
export TLLM_LOG_LEVEL=${TLLM_LOG_LEVEL:-INFO}

# Keep cuda-python pinned below 13 until upstream issues are resolved.
export CUDA_PYTHON_PIN="<13"
