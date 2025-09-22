import os
# Ensure TF32 policy is set BEFORE importing TRT-LLM runtime to avoid env capture issues
os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "1")
import threading
import asyncio
from typing import Any, Iterable, List

from transformers import AutoTokenizer
import numpy as np

from .core.logging_config import get_logger

logger = get_logger(__name__)


class _CompatOutput:
    def __init__(self, text: str) -> None:
        self.text = text


class _CompatResult:
    def __init__(self, outputs: List[_CompatOutput]) -> None:
        self.outputs = outputs


def _decode_full_text(
    tokenizer: Any,
    cumulative_text: str,
    step: Any,
) -> str:
    """
    Build a cumulative text string from a TRT-LLM streaming step.
    Handles multiple possible field names across TRT-LLM versions.
    """
    # Prefer direct text if provided
    step_text = getattr(step, "text", None)
    if isinstance(step_text, str) and step_text:
        # Heuristic: if it already starts with prior, treat as full text
        if step_text.startswith(cumulative_text):
            return step_text
        return cumulative_text + step_text

    # Try token id fields
    candidate_ids = None
    for attr in ("token_ids", "output_token_ids", "new_token_ids"):
        ids = getattr(step, attr, None)
        if ids is not None:
            candidate_ids = ids
            break

    if candidate_ids is None:
        return cumulative_text

    # Flatten batch dimension if present (we use batch size 1)
    if isinstance(candidate_ids, (list, tuple)) and candidate_ids and isinstance(candidate_ids[0], (list, tuple)):
        token_ids = list(candidate_ids[0])
    else:
        token_ids = list(candidate_ids)

    try:
        decoded = tokenizer.decode(token_ids, skip_special_tokens=False)
    except Exception:
        return cumulative_text

    if decoded.startswith(cumulative_text):
        return decoded
    return cumulative_text + decoded


class _VLLMLikeEngine:
    """
    Thin compatibility wrapper that exposes a vLLM-like async .generate(...) API
    backed by TensorRT-LLM ModelRunnerCpp streaming.
    """

    def __init__(self) -> None:
        logger.info("Initializing TensorRT-LLM engine...")
        # Lazy imports to avoid import errors on non-GPU systems
        from tensorrt_llm.runtime import ModelRunnerCpp  # type: ignore

        self.model_id = os.getenv("MODEL_ID", "canopylabs/orpheus-3b-0.1-ft")
        self.engine_dir = os.getenv("ENGINE_DIR", "engine/orpheus_a100_fp16_kvint8")
        self.max_batch_size = int(os.getenv("TRTLLM_MAX_BATCH", "24"))
        self.max_input_len = int(os.getenv("TRTLLM_MAX_INPUT", "160"))
        self.max_output_len = int(os.getenv("TRTLLM_MAX_OUTPUT", "2048"))
        self.kv_fraction = float(os.getenv("TRTLLM_KV_FRACTION", "0.90"))

        logger.info(f"TRT-LLM config: model_id={self.model_id}, engine_dir={self.engine_dir}, "
                   f"max_batch={self.max_batch_size}, max_input={self.max_input_len}, "
                   f"max_output={self.max_output_len}, kv_fraction={self.kv_fraction}")

        logger.debug("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        logger.debug("Tokenizer loaded successfully")

        # Ensure runtime TF32 policy matches build; we build with TF32 enabled
        os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "1")
        os.environ.setdefault("TRTLLM_MPI_ENV_VARS", "NVIDIA_TF32_OVERRIDE")

        logger.debug("Initializing ModelRunnerCpp...")
        self.runner = ModelRunnerCpp.from_dir(
            self.engine_dir,
            max_batch_size=self.max_batch_size,
            max_input_len=self.max_input_len,
            max_output_len=self.max_output_len,
        )
        logger.debug("ModelRunnerCpp initialized")
        
        try:
            self.runner.set_kv_cache_free_gpu_memory_fraction(self.kv_fraction)
            logger.debug(f"KV cache fraction set to {self.kv_fraction}")
        except Exception as e:
            logger.warning(f"Failed to set KV cache fraction: {e}")
        
        logger.info("TensorRT-LLM engine initialized successfully")

    async def generate(self, prompt: str, sp: Any, *_: Any) -> Iterable[_CompatResult]:
        """
        Async generator yielding vLLM-compatible results: result.outputs[0].text is the
        cumulative decoded text so far.
        """
        # Extract sampling params from vLLM's SamplingParams-like object
        temperature = float(getattr(sp, "temperature", 0.6) or 0.6)
        top_p = float(getattr(sp, "top_p", 0.8) or 0.8)
        repetition_penalty = float(getattr(sp, "repetition_penalty", 1.0) or 1.0)
        max_new_tokens = int(getattr(sp, "max_tokens", getattr(sp, "max_new_tokens", 2048)) or 2048)
        stop_ids = list(getattr(sp, "stop_token_ids", []) or [])
        
        logger.debug(f"TRT-LLM generation request: prompt_len={len(prompt)}, temp={temperature}, "
                    f"top_p={top_p}, rep_penalty={repetition_penalty}, max_tokens={max_new_tokens}, "
                    f"stop_ids={stop_ids}")

        # Tokenize prompt (batch size 1) and format as np.ndarray expected by TRT-LLM
        token_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        input_arr = np.asarray(token_ids, dtype=np.int32)
        batched_ids = [input_arr]

        # Prepare kwargs with multiple fallbacks for TRT-LLM API names
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stop_token_ids=stop_ids,
            streaming=True,
            enable_chunked_context=True,
        )

        q: asyncio.Queue[Any] = asyncio.Queue(maxsize=0)

        def _worker() -> None:
            # Try common positional and keyword forms for input ids
            try:
                logger.debug("Starting TRT-LLM generation worker...")
                generator = self.runner.generate(batched_ids, **gen_kwargs)
            except TypeError as e1:
                logger.debug(f"First generation call failed, trying input_ids: {e1}")
                try:
                    generator = self.runner.generate(input_ids=batched_ids, **gen_kwargs)
                except TypeError as e2:
                    logger.debug(f"Second generation call failed, trying input_token_ids: {e2}")
                    generator = self.runner.generate(input_token_ids=batched_ids, **gen_kwargs)

            cumulative_text = ""
            step_count = 0
            try:
                logger.debug("Starting TRT-LLM generation streaming...")
                for step in generator:
                    step_count += 1
                    cumulative_text = _decode_full_text(self.tokenizer, cumulative_text, step)
                    if not cumulative_text:
                        continue
                    q.put_nowait(_CompatResult([_CompatOutput(cumulative_text)]))
                logger.debug(f"TRT-LLM generation completed with {step_count} steps")
            except Exception as e:  # surface to async side
                logger.error(f"TRT-LLM generation error after {step_count} steps: {e}", exc_info=True)
                q.put_nowait(e)
            finally:
                q.put_nowait(None)

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

        while True:
            item = await q.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item


class OrpheusTTSEngine:
    def __init__(self) -> None:
        # server expects `.engine` to provide an async .generate(...)
        self.engine = _VLLMLikeEngine()


