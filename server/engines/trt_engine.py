import os


MODEL_ID = os.getenv("MODEL_ID", "canopylabs/orpheus-3b-0.1-ft")


class OrpheusTRTEngine:
    def __init__(self) -> None:
        engine_dir = os.getenv("TRTLLM_ENGINE_DIR", "").strip()
        dtype = os.getenv("TRTLLM_DTYPE", "float16")
        from tensorrt_llm.llmapi import LLM, KvCacheConfig  # type: ignore

        # KV precision is baked into the engine at build time; do not set here.
        kv_cfg = None
        free_frac = os.getenv("KV_FREE_GPU_FRAC")
        if free_frac is not None and free_frac != "":
            try:
                kv_cfg = KvCacheConfig(free_gpu_memory_fraction=float(free_frac))
            except ValueError:
                kv_cfg = KvCacheConfig()
        if os.getenv("KV_ENABLE_BLOCK_REUSE", "0") == "1":
            kv_cfg = kv_cfg or KvCacheConfig()
            kv_cfg.enable_block_reuse = True

        if engine_dir and os.path.isdir(engine_dir):
            self.engine = LLM(
                backend="trtllm",
                model=engine_dir,
                tokenizer=MODEL_ID,
                dtype=dtype,
                kv_cache_config=kv_cfg,
            )
        else:
            # Fallback: load by model id (e.g., vLLM/HF path). No KV config needed here.
            self.engine = LLM(model=MODEL_ID, dtype=dtype)

