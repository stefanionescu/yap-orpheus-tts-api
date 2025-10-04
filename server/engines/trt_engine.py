import os


MODEL_ID = os.getenv("MODEL_ID", "canopylabs/orpheus-3b-0.1-ft")


class OrpheusTRTEngine:
    def __init__(self) -> None:
        engine_dir = os.getenv("TRTLLM_ENGINE_DIR", "").strip()
        from tensorrt_llm._tensorrt_engine import LLM  # type: ignore
        from tensorrt_llm import SchedulerConfig, CapacitySchedulerPolicy  # type: ignore

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

        # Configure scheduler for high-concurrency streaming workloads
        # max_num_tokens = tokens actively being processed across ALL requests at any moment
        # NOT the same as max capacity (which would be batch_size * max_seq_len)
        # For streaming: requests are staggered, so use avg in-flight (40-60% of max_output_len)
        # Example: 20 concurrent * (48 input + ~512 avg output in-flight) = ~11,200 tokens
        max_num_tokens = int(os.getenv("TRTLLM_MAX_NUM_TOKENS", "12288"))
        scheduler_config = SchedulerConfig(
            capacity_scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION,
            max_num_tokens=max_num_tokens,
        )
        kwargs["scheduler_config"] = scheduler_config

        self.engine = LLM(**kwargs)

