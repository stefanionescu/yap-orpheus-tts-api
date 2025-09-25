# TensorRT-LLM defaults (A100 tuned for optimal TTFB and throughput)
export TRTLLM_MAX_BATCH=${TRTLLM_MAX_BATCH:-24}
export TRTLLM_MAX_INPUT=${TRTLLM_MAX_INPUT:-160}
export TRTLLM_MAX_OUTPUT=${TRTLLM_MAX_OUTPUT:-2048}
export TRTLLM_KV_FRACTION=${TRTLLM_KV_FRACTION:-0.70}  # Leave more room for SNAC + CUDA context

# Engine path and backend selection  
export ORPHEUS_BACKEND=${ORPHEUS_BACKEND:-trtllm}
export ENGINE_DIR=${ENGINE_DIR:-engine/orpheus_a100_fp16_kvint8}

# Performance tuning for A100 (FP16/BF16 + INT8-KV, no FP8 on A100)
export NVIDIA_TF32_OVERRIDE=${NVIDIA_TF32_OVERRIDE:-1}  # Enable TensorFloat-32 for A100


