import os


MODEL_ID = os.getenv("MODEL_ID", "canopylabs/orpheus-3b-0.1-ft")


class OrpheusTRTEngine:
    def __init__(self) -> None:
        engine_dir = os.getenv("TRTLLM_ENGINE_DIR", "").strip()
        dtype = os.getenv("TRTLLM_DTYPE", "float16")
        from tensorrt_llm import LLM  # type: ignore

        # Require a valid TRT-LLM engine directory
        if not engine_dir or not os.path.isdir(engine_dir):
            raise RuntimeError(
                "TRTLLM_ENGINE_DIR must point to a valid TensorRT-LLM engine directory (e.g., contains rank0.engine)."
            )

        # Let TRT-LLM auto-detect the model path format (engine/ckpt/etc.)
        kwargs = {
            "model": engine_dir,
            "tokenizer": MODEL_ID,
            "dtype": dtype,
            "backend": "trtllm",
        }

        # Optional KV cache runtime tuning (memory/behavior, not precision)
        kv_cfg: dict = {}
        free_frac = os.getenv("KV_FREE_GPU_FRAC")
        if free_frac:
            try:
                kv_cfg["free_gpu_memory_fraction"] = float(free_frac)
            except ValueError:
                pass
        if os.getenv("KV_ENABLE_BLOCK_REUSE", "0") == "1":
            kv_cfg["enable_block_reuse"] = True

        # Only pass kv_cache_config when we actually have values
        if kv_cfg:
            kwargs["kv_cache_config"] = kv_cfg

        self.engine = LLM(**kwargs)

