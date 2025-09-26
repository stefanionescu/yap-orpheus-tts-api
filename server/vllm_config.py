import os

def vllm_engine_kwargs() -> dict:
    """
    Conservative defaults for A100 + CUDA12.x.
    Tunable via env.
    """
    return dict(
        dtype=os.getenv("VLLM_DTYPE", "bfloat16"), # 'float16'|'bfloat16'|'half'|'auto'
        max_model_len=int(os.getenv("VLLM_MAX_MODEL_LEN", "2048")),
        gpu_memory_utilization=float(os.getenv("VLLM_GPU_UTIL", "0.90")),
        max_num_seqs=int(os.getenv("VLLM_MAX_SEQS", "1")),
        max_num_batched_tokens=int(os.getenv("VLLM_MAX_BATCHED_TOKENS", "128")),
        enforce_eager=bool(int(os.getenv("VLLM_ENFORCE_EAGER", "1"))),
        swap_space=int(os.getenv("VLLM_SWAP_SPACE_GB", "4")),
        disable_custom_all_reduce=bool(int(os.getenv("VLLM_DISABLE_CUSTOM_ALL_REDUCE", "0"))),
        trust_remote_code=True,
        disable_log_stats=True,
        tensor_parallel_size=int(os.getenv("TENSOR_PARALLEL_SIZE", "1")),
        enable_prefix_caching=bool(int(os.getenv("VLLM_PREFIX_CACHE", "1"))),
        # NOTE: do NOT include worker_use_ray; not valid on vLLM 0.8.x
    )


