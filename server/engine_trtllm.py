import os
# Ensure TF32 policy is set BEFORE importing TRT-LLM runtime to avoid env capture issues
os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "1")
import threading
import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional, Dict

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


@dataclass
class _Req:
    """Single websocket generation request for batch processing."""
    prompt: str
    sp: Any
    fut_q: asyncio.Queue  # carries _CompatResult or Exception or None sentinel
    req_id: int = -1      # assigned at batch build time
    tok_ids: Optional[np.ndarray] = None
    # streaming decode state
    cumulative_text: str = ""


class _BatchScheduler:
    """
    Single-runner micro-batching scheduler.
    - Gathers requests for a small window (batch_window_ms)
    - Executes one batched generate(streaming=True)
    - Routes per-seq outputs back to each request queue
    - Closes queues with None sentinel on completion/errors
    """

    def __init__(
        self,
        runner,
        tokenizer,
        max_batch_concurrency: int = 16,
        batch_window_ms: int = 10,
        max_input_len: int = 160,
        max_output_len: int = 2048,
    ) -> None:
        self.runner = runner
        self.tokenizer = tokenizer
        self.max_batch = max_batch_concurrency
        self.batch_window_ms = batch_window_ms
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len

        self._in_q: asyncio.Queue[_Req] = asyncio.Queue()
        self._bg_task: Optional[asyncio.Task] = None
        self._stop = False

        logger.debug(f"BatchScheduler initialized: max_batch={max_batch_concurrency}, "
                    f"batch_window_ms={batch_window_ms}")

    def start(self) -> None:
        if self._bg_task is None:
            loop = asyncio.get_running_loop()
            self._bg_task = loop.create_task(self._loop())
            logger.debug("BatchScheduler background loop started")

    async def stop(self) -> None:
        self._stop = True
        if self._bg_task:
            await self._bg_task
            logger.debug("BatchScheduler stopped")

    async def submit(self, prompt: str, sp: Any) -> asyncio.Queue:
        """Submit a request and get back a queue for streaming results."""
        fut_q: asyncio.Queue = asyncio.Queue()
        await self._in_q.put(_Req(prompt=prompt, sp=sp, fut_q=fut_q))
        return fut_q

    def _encode(self, s: str) -> np.ndarray:
        """Encode text to token IDs with length limiting."""
        ids = self.tokenizer.encode(s, add_special_tokens=True)
        if len(ids) > self.max_input_len:
            ids = ids[-self.max_input_len:]  # hard clip from left to respect engine input
        return np.asarray(ids, dtype=np.int32)

    async def _loop(self) -> None:
        """Main background loop that processes batches."""
        logger.debug("BatchScheduler loop started")
        
        while not self._stop:
            # 1) Block until at least one request arrives
            try:
                first = await asyncio.wait_for(self._in_q.get(), timeout=0.050)
            except asyncio.TimeoutError:
                continue

            batch: List[_Req] = [first]
            t0 = time.monotonic()

            # 2) Micro-batching window (gather up to max_batch)
            while (len(batch) < self.max_batch) and ((time.monotonic() - t0) * 1000.0 < self.batch_window_ms):
                try:
                    nxt = self._in_q.get_nowait()
                    batch.append(nxt)
                except asyncio.QueueEmpty:
                    await asyncio.sleep(0)  # yield and re-check until window expires

            logger.debug(f"Processing batch of {len(batch)} requests")

            # 3) Encode prompts and assign req indices
            batched_ids: List[np.ndarray] = []
            for i, req in enumerate(batch):
                req.req_id = i
                req.tok_ids = self._encode(req.prompt)
                batched_ids.append(req.tok_ids)

            # 4) Build kwargs (take from first req's sp; could fan-out per-req if needed)
            sp0 = batch[0].sp
            temperature = float(getattr(sp0, "temperature", 0.6) or 0.6)
            top_p = float(getattr(sp0, "top_p", 0.8) or 0.8)
            repetition_penalty = float(getattr(sp0, "repetition_penalty", 1.0) or 1.0)
            max_new_tokens = int(getattr(sp0, "max_tokens", getattr(sp0, "max_new_tokens", self.max_output_len)) or self.max_output_len)
            stop_ids = list(getattr(sp0, "stop_token_ids", []) or [])

            gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                stop_token_ids=stop_ids,
                streaming=True,
                enable_chunked_context=True,
            )

            logger.debug(f"Batch generation: temp={temperature}, top_p={top_p}, "
                        f"rep_penalty={repetition_penalty}, max_tokens={max_new_tokens}")

            # 5) Run ONE streaming generator covering the whole batch
            def _run():
                # Fall back across param names if needed
                gen = None
                try:
                    gen = self.runner.generate(batched_ids, **gen_kwargs)
                except TypeError:
                    try:
                        gen = self.runner.generate(input_ids=batched_ids, **gen_kwargs)
                    except TypeError:
                        gen = self.runner.generate(input_token_ids=batched_ids, **gen_kwargs)
                return gen

            # NOTE: keep the whole generator in one thread to avoid any races
            loop = asyncio.get_running_loop()
            q = asyncio.Queue()

            def _worker():
                try:
                    logger.debug("Starting batched generation worker...")
                    gen = _run()
                    step_count = 0
                    for step in gen:
                        step_count += 1
                        q.put_nowait(step)
                    logger.debug(f"Batched generation completed with {step_count} steps")
                except Exception as e:
                    logger.error(f"Batched generation error: {e}", exc_info=True)
                    q.put_nowait(e)
                finally:
                    q.put_nowait(None)

            t = threading.Thread(target=_worker, daemon=True)
            t.start()

            # 6) Demux steps back to each request
            try:
                while True:
                    step = await q.get()
                    if step is None:
                        break
                    if isinstance(step, Exception):
                        # fan-out error to all in batch
                        for req in batch:
                            if not req.fut_q.empty():
                                pass
                            await req.fut_q.put(step)
                            await req.fut_q.put(None)
                        raise step

                    # Extract per-sequence token ids using the most universal attribute
                    per_seq_ids = getattr(step, "output_token_ids", None)
                    if per_seq_ids is None:
                        per_seq_ids = getattr(step, "token_ids", None)
                    if per_seq_ids is None:
                        per_seq_ids = getattr(step, "new_token_ids", None)

                    if per_seq_ids is None:
                        # Last fallback: maybe step.text per-seq?
                        per_seq_texts = getattr(step, "texts", None) or getattr(step, "text", None)
                        if isinstance(per_seq_texts, list):
                            # emit per index
                            for i, req in enumerate(batch):
                                # cumulative text path
                                req.cumulative_text = _decode_full_text(self.tokenizer, req.cumulative_text, step)
                                await req.fut_q.put(_CompatResult([_CompatOutput(req.cumulative_text)]))
                        else:
                            # ignore; no visible delta
                            continue
                    else:
                        # per_seq_ids is [B, …] or flat+index; normalize to list per request
                        if isinstance(per_seq_ids, (list, tuple)) and per_seq_ids and isinstance(per_seq_ids[0], (list, tuple, np.ndarray)):
                            # shape [B, T]
                            for i, req in enumerate(batch):
                                # use our safe decoder (keeps cumulative)
                                req.cumulative_text = _decode_full_text(self.tokenizer, req.cumulative_text, type("S", (), {"token_ids": per_seq_ids[i]}))
                                await req.fut_q.put(_CompatResult([_CompatOutput(req.cumulative_text)]))
                        else:
                            # ambiguous; decode once and broadcast (rare)
                            for req in batch:
                                req.cumulative_text = _decode_full_text(self.tokenizer, req.cumulative_text, step)
                                await req.fut_q.put(_CompatResult([_CompatOutput(req.cumulative_text)]))
                # 7) close all queues for this batch
                for req in batch:
                    await req.fut_q.put(None)

            except Exception as e:
                # already fanned out error above
                logger.error(f"Error during batch demuxing: {e}", exc_info=True)
                pass


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
        
        # Create scheduler – batch up to 16 concurrent sessions, 10ms window
        max_concurrency = min(16, self.max_batch_size)
        logger.info(f"Initializing BatchScheduler with max_concurrency={max_concurrency}")
        self.scheduler = _BatchScheduler(
            runner=self.runner,
            tokenizer=self.tokenizer,
            max_batch_concurrency=max_concurrency,
            batch_window_ms=10,
            max_input_len=self.max_input_len,
            max_output_len=self.max_output_len,
        )
        
        # Start background loop
        try:
            # If __init__ runs inside an event loop, start immediately
            asyncio.get_running_loop()
            self.scheduler.start()
            self._sched_started = True
        except RuntimeError:
            # No event loop running, defer start to first generate() call
            self._sched_started = False
        
        logger.info("TensorRT-LLM engine initialized successfully")

    async def generate(self, prompt: str, sp: Any, *_: Any) -> Iterable[_CompatResult]:
        """
        Async generator yielding vLLM-compatible results: result.outputs[0].text is the
        cumulative decoded text so far.
        
        Now uses the batch scheduler for true concurrency without races.
        """
        if not self._sched_started:
            # Lazily start when first called inside an event loop
            self.scheduler.start()
            self._sched_started = True

        # Extract sampling params for logging
        temperature = float(getattr(sp, "temperature", 0.6) or 0.6)
        top_p = float(getattr(sp, "top_p", 0.8) or 0.8)
        repetition_penalty = float(getattr(sp, "repetition_penalty", 1.0) or 1.0)
        max_new_tokens = int(getattr(sp, "max_tokens", getattr(sp, "max_new_tokens", 2048)) or 2048)
        stop_ids = list(getattr(sp, "stop_token_ids", []) or [])
        
        logger.debug(f"TRT-LLM generation request: prompt_len={len(prompt)}, temp={temperature}, "
                    f"top_p={top_p}, rep_penalty={repetition_penalty}, max_tokens={max_new_tokens}, "
                    f"stop_ids={stop_ids}")

        # Submit to scheduler → get a queue back for streaming results
        q = await self.scheduler.submit(prompt, sp)

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


