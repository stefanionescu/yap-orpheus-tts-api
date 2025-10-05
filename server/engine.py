import os
from ..config import settings


MODEL_ID = settings.model_id


class OrpheusTRTEngine:
    def __init__(self) -> None:
        engine_dir = settings.trtllm_engine_dir
        from tensorrt_llm._tensorrt_engine import LLM  # type: ignore

        # Require a valid TRT-LLM engine directory
        if not engine_dir or not os.path.isdir(engine_dir):
            raise RuntimeError(
                "TRTLLM_ENGINE_DIR must point to a valid TensorRT-LLM engine directory (e.g., contains rank0.engine)."
            )

        # Load a prebuilt TensorRT-LLM engine by directory (auto-detected format)
        kwargs = {
            "model": engine_dir,
            "tokenizer": MODEL_ID,
        }

        # Optional KV cache runtime tuning (memory/behavior, not precision)
        kv_cfg: dict = {}
        free_frac = settings.kv_free_gpu_frac
        if free_frac:
            try:
                kv_cfg["free_gpu_memory_fraction"] = float(free_frac)
            except ValueError:
                pass
        if settings.kv_enable_block_reuse:
            kv_cfg["enable_block_reuse"] = True

        # Only pass kv_cache_config when we actually have values
        if kv_cfg:
            kwargs["kv_cache_config"] = kv_cfg

        self.engine = LLM(**kwargs)
