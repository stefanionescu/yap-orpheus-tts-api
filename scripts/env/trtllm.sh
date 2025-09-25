# TensorRT-LLM defaults (A100 optimized for SINGLE-STREAM LATENCY)
export TRTLLM_MAX_BATCH=${TRTLLM_MAX_BATCH:-8}          # Lower batch for better TTFB
export TRTLLM_MAX_INPUT=${TRTLLM_MAX_INPUT:-192}        # Slightly higher for context
export TRTLLM_MAX_OUTPUT=${TRTLLM_MAX_OUTPUT:-2048}     # Keep for long audio
export TRTLLM_KV_FRACTION=${TRTLLM_KV_FRACTION:-0.90}   # Higher utilization for single stream

# Engine path and backend selection  
export ORPHEUS_BACKEND=${ORPHEUS_BACKEND:-trtllm}
export ENGINE_DIR=${ENGINE_DIR:-engine/orpheus_a100_fp16_kvint8}

# Model and tokenizer paths - ensure tokenizer matches engine build
export MODEL_LOCAL_DIR=${MODEL_LOCAL_DIR:-models/orpheus_hf}
export TOKENIZER_DIR=${TOKENIZER_DIR:-$MODEL_LOCAL_DIR}  # Very important for TRT

# Performance tuning for A100 single-stream latency (FP16 + INT8-KV)
export NVIDIA_TF32_OVERRIDE=${NVIDIA_TF32_OVERRIDE:-1}  # Enable TensorFloat-32 for A100
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}  # Single stream focus


