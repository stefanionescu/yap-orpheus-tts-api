import os
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine

from .vllm_config import vllm_engine_kwargs
from .core.logging_config import get_logger

logger = get_logger(__name__)

MODEL_ID = os.getenv("MODEL_ID", "canopylabs/orpheus-3b-0.1-ft")

class OrpheusTTSEngine:
    def __init__(self):
        logger.info("Initializing vLLM engine...")
        logger.info(f"Model ID: {MODEL_ID}")
        
        ekw = vllm_engine_kwargs()
        logger.debug(f"vLLM engine kwargs: {ekw}")
        
        # Async vLLM engine; server consumes .engine directly
        logger.debug("Creating AsyncLLMEngine...")
        self.engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
            model=MODEL_ID,
            tokenizer=MODEL_ID,
            **ekw,
        ))
        logger.info("vLLM engine initialized successfully")
