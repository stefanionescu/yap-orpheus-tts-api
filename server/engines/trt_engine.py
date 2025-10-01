import os


MODEL_ID = os.getenv("MODEL_ID", "canopylabs/orpheus-3b-0.1-ft")


class OrpheusTRTEngine:
    def __init__(self) -> None:
        engine_dir = os.getenv("TRTLLM_ENGINE_DIR", "").strip()
        dtype = os.getenv("TRTLLM_DTYPE", "float16")
        from tensorrt_llm.llmapi import LLM, KvCacheConfig  # type: ignore

        if engine_dir and os.path.isdir(engine_dir):
            kv_dtype = os.getenv("TRTLLM_KV_CACHE_DTYPE", "int8").lower()
            kv_cfg = KvCacheConfig(dtype=kv_dtype) if kv_dtype in {"fp8", "int8"} else None
            self.engine = LLM(model=engine_dir, dtype=dtype, kv_cache_config=kv_cfg)
        else:
            kv_dtype = os.getenv("TRTLLM_KV_CACHE_DTYPE", "int8").lower()
            kv_cfg = KvCacheConfig(dtype=kv_dtype) if kv_dtype in {"fp8", "int8"} else None
            self.engine = LLM(model=MODEL_ID, dtype=dtype, kv_cache_config=kv_cfg)

