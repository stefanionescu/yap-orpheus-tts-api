import os

def vllm_engine_kwargs() -> dict:
    """
    Conservative defaults for A100 + CUDA12.x.
    You can tweak these via env without changing code.
    """
    max_num_seqs = int(os.getenv("VLLM_MAX_SEQS", "24"))
    gpu_mem_util = float(os.getenv("VLLM_GPU_UTIL", "0.92"))
    dtype = os.getenv("VLLM_DTYPE", "float16")
    max_model_len = int(os.getenv("VLLM_MAX_MODEL_LEN", "4096"))
    disable_log_stats = bool(int(os.getenv("VLLM_DISABLE_LOG_STATS", "1")))
    # We avoid FP8 KV cache: Ampere doesn’t fully benefit and may not be supported in kernels.
    # (KV FP8 helps on Hopper/Ada; on Ampere it’s unsupported or slower.)
    # Ref: vLLM docs + forum notes.
    return dict(
        dtype=dtype,                             # 'float16' or 'bfloat16'
        max_model_len=max_model_len,             # longer texts / context
        gpu_memory_utilization=gpu_mem_util,     # packing
        max_num_seqs=max_num_seqs,               # concurrency
        disable_log_stats=disable_log_stats,
        trust_remote_code=True,
    )


