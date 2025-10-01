import os


MODEL_ID = os.getenv("MODEL_ID", "canopylabs/orpheus-3b-0.1-ft")


class OrpheusTRTEngine:
    def __init__(self) -> None:
        engine_dir = os.getenv("TRTLLM_ENGINE_DIR", "").strip()
        dtype = os.getenv("TRTLLM_DTYPE", "float16")
        from tensorrt_llm.llmapi import LLM  # type: ignore

        if engine_dir and os.path.isdir(engine_dir):
            print(f"[trt] Loading TensorRT-LLM engine from: {engine_dir} (dtype={dtype})")
            kv_dtype = os.getenv("TRTLLM_KV_CACHE_DTYPE", "")
            wq = os.getenv("TRTLLM_WEIGHT_QUANT", "none")
            if os.path.isdir(os.path.join(engine_dir, "quantized-checkpoint")):
                print(
                    f"[trt] Detected quantized-checkpoint inside engine dir. "
                    f"KV-cache={kv_dtype or 'unknown'}, weight_quant={wq}"
                )
            self.engine = LLM(model=engine_dir, dtype=dtype)
        else:
            print(f"[trt] Loading model via HF: {MODEL_ID} (dtype={dtype})")
            self.engine = LLM(model=MODEL_ID, dtype=dtype)

