# Performance-related environment defaults (override via .env or shell)
# Increased from 1 to 4 for better concurrent kernel launch with TRT-LLM + SNAC batching
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-4}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.9,max_split_size_mb:512}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
export HF_TRANSFER=${HF_TRANSFER:-1}

# Optional dev toggles
export TORCH_COMPILE_DISABLE=${TORCH_COMPILE_DISABLE:-1}
export TRITON_DISABLE_COMPILATION=${TRITON_DISABLE_COMPILATION:-0}
