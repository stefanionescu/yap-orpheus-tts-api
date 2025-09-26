import os


MODEL_ID = os.getenv("MODEL_ID", "canopylabs/orpheus-3b-0.1-ft")


class OrpheusTRTEngine:
    def __init__(self) -> None:
        engine_dir = os.getenv("TRTLLM_ENGINE_DIR", "").strip()
        from tensorrt_llm.llmapi import LLM  # type: ignore

        if engine_dir and os.path.isdir(engine_dir):
            self.engine = LLM(model=engine_dir)
        else:
            self.engine = LLM(model=MODEL_ID)


