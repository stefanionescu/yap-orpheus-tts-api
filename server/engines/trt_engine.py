import os


MODEL_ID = os.getenv("MODEL_ID", "canopylabs/orpheus-3b-0.1-ft")


class OrpheusTRTEngine:
    def __init__(self) -> None:
        engine_dir = os.getenv("TRTLLM_ENGINE_DIR", "").strip()
        tokenizer_path = os.getenv("HF_TOKENIZER_DIR", MODEL_ID)
        dtype = os.getenv("TRTLLM_DTYPE", "float16")
        
        from tensorrt_llm.llmapi import LLM, KvCacheConfig  # type: ignore

        if engine_dir and os.path.isdir(engine_dir):
            # Validate that engine files exist
            self._validate_engine_directory(engine_dir)
            
            # Configure KV cache with correct parameter name
            kv_dtype = os.getenv("TRTLLM_KV_CACHE_DTYPE", "int8").lower()
            kv_cfg = KvCacheConfig(precision=kv_dtype) if kv_dtype in {"fp8", "int8"} else None
            
            # Force TensorRT backend and provide explicit tokenizer
            self.engine = LLM(
                model=engine_dir,
                backend="tensorrt_llm",  # Force TRT backend, don't let it pick torch
                tokenizer=tokenizer_path,  # Explicit tokenizer path/model id
                dtype=dtype,
                kv_cache_config=kv_cfg
            )
        else:
            # Fallback to HF model (not recommended for production)
            kv_dtype = os.getenv("TRTLLM_KV_CACHE_DTYPE", "int8").lower()
            kv_cfg = KvCacheConfig(precision=kv_dtype) if kv_dtype in {"fp8", "int8"} else None
            
            self.engine = LLM(
                model=MODEL_ID,
                backend="tensorrt_llm",  # Force TRT backend
                tokenizer=tokenizer_path,
                dtype=dtype,
                kv_cache_config=kv_cfg
            )

    def _validate_engine_directory(self, engine_dir: str) -> None:
        """Validate that the engine directory contains required TRT engine files."""
        # Check for single-GPU engine file
        model_plan_path = os.path.join(engine_dir, "model.plan")
        
        # Check for multi-GPU engine files (rank0.engine, etc.)
        rank0_engine_path = os.path.join(engine_dir, "rank0.engine")
        
        has_engine_files = os.path.isfile(model_plan_path) or os.path.isfile(rank0_engine_path)
        
        if not has_engine_files:
            # Look for engine files in subdirectories (e.g., 1-gpu/, etc.)
            import glob
            engine_patterns = [
                os.path.join(engine_dir, "*", "model.plan"),
                os.path.join(engine_dir, "*", "rank*.engine"),
                os.path.join(engine_dir, "**", "model.plan"),
                os.path.join(engine_dir, "**", "rank*.engine")
            ]
            
            found_engines = []
            for pattern in engine_patterns:
                found_engines.extend(glob.glob(pattern, recursive=True))
            
            if found_engines:
                suggestion = found_engines[0]
                suggested_dir = os.path.dirname(suggestion)
                raise ValueError(
                    f"No TRT engine files found in {engine_dir}. "
                    f"Found engines in subdirectory: {suggested_dir}. "
                    f"Set TRTLLM_ENGINE_DIR to {suggested_dir}"
                )
            else:
                raise ValueError(
                    f"No TRT engine files (model.plan or rank*.engine) found in {engine_dir}. "
                    f"Please ensure the engine directory contains valid TensorRT-LLM engine files."
                )

