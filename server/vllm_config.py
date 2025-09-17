import os

def vllm_engine_kwargs() -> dict:
    """
    Conservative defaults for A100 + CUDA12.x.
    Tunable via env.
    """
    return dict(
        dtype=os.getenv("VLLM_DTYPE", "float16"), # 'float16'|'bfloat16'|'half'|'auto'
        max_model_len=int(os.getenv("VLLM_MAX_MODEL_LEN", "8192")),
        gpu_memory_utilization=float(os.getenv("VLLM_GPU_UTIL", "0.92")),
        max_num_seqs=int(os.getenv("VLLM_MAX_SEQS", "24")),
        trust_remote_code=True,
        disable_log_stats=True,
        tensor_parallel_size=int(os.getenv("TENSOR_PARALLEL_SIZE", "1")),
        enable_prefix_caching=bool(int(os.getenv("VLLM_PREFIX_CACHE", "1"))),
        # NOTE: do NOT include worker_use_ray; not valid on vLLM 0.8.x
    )


