import os
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine

from .vllm_config import vllm_engine_kwargs

MODEL_ID = os.getenv("MODEL_ID", "canopylabs/orpheus-3b-0.1-ft")

class OrpheusTTSEngine:
    def __init__(self):
        ekw = vllm_engine_kwargs()
        # Async vLLM engine; server consumes .engine directly
        self.engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
            model=MODEL_ID,
            tokenizer=MODEL_ID,
            **ekw,
        ))
