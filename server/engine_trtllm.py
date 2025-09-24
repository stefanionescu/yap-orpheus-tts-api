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


# End-of-speech token id used by Orpheus models
EO_SPEECH_ID = 128258


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
    prompt_len: int,
) -> str:
    """
    Build a cumulative text string from a TRT-LLM streaming step.
    - If the step has raw text, append diffs against cumulative_text.
    - If the step has token ids, decode ONLY tokens after the prompt_len
      using tokenizer.decode(skip_special_tokens=False), then apply delta.
    """
    # Prefer direct text if provided
    step_text = getattr(step, "text", None)
    if isinstance(step_text, str) and step_text:
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

    # Skip prompt ids and decode the generation tail as text
    try:
        gen_ids = token_ids[prompt_len:]
        decoded_tail = tokenizer.decode(gen_ids, skip_special_tokens=False)
    except Exception:
        return cumulative_text

    if decoded_tail.startswith(cumulative_text):
        return decoded_tail
    return cumulative_text + decoded_tail


@dataclass
class _Req:
    """Single websocket generation request for batch processing."""
    prompt: Any
    sp: Any
    fut_q: asyncio.Queue  # carries _CompatResult or Exception or None sentinel
    req_id: int = -1      # assigned at batch build time
    tok_ids: Optional[np.ndarray] = None
    prompt_len: int = 0
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
        """Encode text to token IDs with length limiting, no auto specials, strip trailing EOS."""
        ids = self.tokenizer.encode(s, add_special_tokens=False)
        eos = int(self.tokenizer.eos_token_id) if (self.tokenizer.eos_token_id is not None) else None
        if eos is not None:
            while len(ids) > 0 and ids[-1] == eos:
                ids.pop()
        if len(ids) > self.max_input_len:
            ids = ids[-self.max_input_len:]  # hard clip from left to respect engine input
        try:
            logger.debug(f"prompt_ids_tail={ids[-12:] if len(ids) > 12 else ids}")
        except Exception:
            pass
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
                # Accept either pre-tokenized ids or raw text
                if isinstance(req.prompt, (list, tuple, np.ndarray)):
                    try:
                        # Convert to list[int] then ndarray
                        _lst = list(req.prompt)
                    except Exception:
                        _lst = []
                    req.tok_ids = np.asarray(_lst, dtype=np.int32)
                    try:
                        logger.debug(f"prompt_ids_tail[{i}]={_lst[-12:] if len(_lst) > 12 else _lst}")
                    except Exception:
                        pass
                else:
                    req.tok_ids = self._encode(str(req.prompt))
                # Record prompt length so we can skip it during streamed decode
                try:
                    req.prompt_len = int(len(req.tok_ids))
                except Exception:
                    req.prompt_len = 0
                if len(req.tok_ids) == 0:
                    # ensure at least one token; use pad_id
                    req.tok_ids = np.asarray([int(self.tokenizer.pad_token_id)], dtype=np.int32)
                    logger.warning(f"Request {i} had empty token sequence, using pad token for prompt: '{req.prompt}'")
                batched_ids.append(req.tok_ids)

            # Use NumPy int32 arrays for TRT-LLM (ModelRunnerCpp will .tolist() each)
            batch_input_ids = [
                (arr if isinstance(arr, np.ndarray) else np.asarray(arr, dtype=np.int32)).astype(np.int32, copy=False)
                for arr in batched_ids
            ]

            # 4) Build kwargs with explicit special IDs to avoid TRT-LLM ambiguity
            pad_id = int(self.tokenizer.pad_token_id)
            eos_id = int(self.tokenizer.eos_token_id) if (self.tokenizer.eos_token_id is not None) else pad_id
            # Never alias BOS to EOS; use -1 when BOS is missing or equals EOS
            bos_id = int(self.tokenizer.bos_token_id) if (self.tokenizer.bos_token_id is not None) else -1
            if bos_id == eos_id:
                bos_id = -1
            assert pad_id != eos_id, f"pad_id ({pad_id}) must not equal eos_id ({eos_id})"
            
            sp0 = batch[0].sp
            temperature = float(getattr(sp0, "temperature", 0.6) or 0.6)
            top_p = float(getattr(sp0, "top_p", 0.8) or 0.8)
            repetition_penalty = float(getattr(sp0, "repetition_penalty", 1.0) or 1.0)
            max_new_tokens = int(getattr(sp0, "max_tokens", getattr(sp0, "max_new_tokens", self.max_output_len)) or self.max_output_len)
            # Prefer stop_words (sequence(s) of ids) for TRT-LLM; default to END_OF_SPEECH (128258)
            stop_ids = list(getattr(sp0, "stop_token_ids", []) or [])
            stop_words = [[sid] for sid in stop_ids] if stop_ids else [[EO_SPEECH_ID]]
            # Defensive duration-based ceiling on tokens to avoid runaway generations
            try:
                def _tokens_for_seconds(seconds: float, frames_per_second: float = 100.0, tokens_per_frame: int = 7) -> int:
                    return int(seconds * frames_per_second * tokens_per_frame)
                _target_seconds = float(os.getenv("ORPHEUS_MAX_SECONDS", "20").strip() or 0)
                if _target_seconds > 0:
                    _ceiling = _tokens_for_seconds(_target_seconds)
                    if _ceiling > 0:
                        max_new_tokens = min(max_new_tokens, _ceiling)
            except Exception:
                pass

            gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                stop_words=stop_words,
                streaming=True,
                enable_chunked_context=True,
                # Tell TRT-LLM exactly what special IDs are to avoid ambiguity
                pad_id=pad_id,
                bos_id=bos_id,
                end_id=EO_SPEECH_ID,
            )

            # Support both TRT-LLM kw names for stop words (stop_words_list vs stop_words)
            try:
                import inspect as _inspect
                _sig = _inspect.signature(self.runner.generate)
                if ("stop_words_list" in _sig.parameters) and ("stop_words" in gen_kwargs):
                    gen_kwargs["stop_words_list"] = gen_kwargs.pop("stop_words")
                    logger.debug("Using stop_words_list kwarg for TRT-LLM")
                elif "stop_words" in _sig.parameters:
                    logger.debug("Using stop_words kwarg for TRT-LLM")
                else:
                    gen_kwargs.pop("stop_words", None)
                    logger.debug("TRT-LLM runner has no stop_words* parameter; skipping")
            except Exception:
                pass

            logger.debug(f"Batch generation: temp={temperature}, top_p={top_p}, "
                        f"rep_penalty={repetition_penalty}, max_tokens={max_new_tokens}")
            logger.debug(f"Special tokens: pad_id={pad_id}, bos_id={bos_id}, end_id={EO_SPEECH_ID}")
            logger.debug(f"Final stop_words passed to TRT: {gen_kwargs.get('stop_words') or gen_kwargs.get('stop_words_list')}")

            # 5) Run ONE streaming generator covering the whole batch
            def _run():
                logger.debug(
                    f"Attempting batched generation with {len(batch_input_ids)} sequences, lengths: {[len(seq) for seq in batch_input_ids]}"
                )
                try:
                    return self.runner.generate(batch_input_ids=batch_input_ids, **gen_kwargs)
                except TypeError as te:
                    # Fallbacks for TRT-LLM kwarg naming differences
                    _msg = str(te)
                    _local_kwargs = dict(gen_kwargs)
                    try:
                        if "stop_words_list" in _msg and ("stop_words" in _local_kwargs):
                            _local_kwargs.pop("stop_words", None)
                            logger.debug("Retrying generate() without stop_words due to TypeError: stop_words_list mismatch")
                            return self.runner.generate(batch_input_ids=batch_input_ids, **_local_kwargs)
                        if "stop_words" in _msg and ("stop_words" in _local_kwargs):
                            _sw = _local_kwargs.pop("stop_words", None)
                            _local_kwargs["stop_words_list"] = _sw
                            logger.debug("Retrying generate() with stop_words_list instead of stop_words")
                            return self.runner.generate(batch_input_ids=batch_input_ids, **_local_kwargs)
                    except Exception:
                        pass
                    # Last resort: remove any stop_words* and rely on end_id for stopping
                    try:
                        _local_kwargs.pop("stop_words", None)
                        _local_kwargs.pop("stop_words_list", None)
                        # Ensure end_id remains present
                        _local_kwargs.setdefault("end_id", EO_SPEECH_ID)
                        logger.debug("Retrying generate() without stop words; relying on end_id")
                        return self.runner.generate(batch_input_ids=batch_input_ids, **_local_kwargs)
                    except Exception:
                        raise te

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
                demux_step_idx = 0
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
                    demux_step_idx += 1
                    if demux_step_idx == 1:
                        try:
                            typ = type(step)
                            keys = list(step.keys()) if isinstance(step, dict) else dir(step)[:8]
                            logger.debug(f"stream step type={typ} keys={keys}")
                        except Exception:
                            pass

                    # --- DEMUX: normalize step to per-sequence texts, then emit ---
                    import numpy as _np
                    try:
                        import torch as _torch  # type: ignore
                    except Exception:
                        _torch = None  # type: ignore

                    def _to_list_1d(x):
                        # Convert single sequence of token ids to list[int]
                        if x is None:
                            return []
                        if (_torch is not None) and isinstance(x, _torch.Tensor):
                            return x.detach().cpu().tolist()
                        if isinstance(x, _np.ndarray):
                            return x.tolist()
                        if isinstance(x, (list, tuple)):
                            return list(x)
                        # last-resort: single int
                        try:
                            return [int(x)]
                        except Exception:
                            return []

                    def _extract(_step, key):
                        # Works for dict or attr objects
                        if isinstance(_step, dict):
                            return _step.get(key, None)
                        return getattr(_step, key, None)

                    def _get_token_matrix(_step):
                        # Try all known field names
                        for k in ("output_token_ids", "output_ids", "token_ids", "new_token_ids"):
                            v = _extract(_step, k)
                            if v is None:
                                continue
                            # torch / np / list cases
                            if (_torch is not None) and isinstance(v, _torch.Tensor):
                                # Ensure integer dtype for decoder compatibility
                                try:
                                    v = v.to(dtype=_torch.int64)
                                except Exception:
                                    pass
                                if v.ndim == 2:
                                    return [v[i].detach().cpu().tolist() for i in range(v.shape[0])]
                                return [v.detach().cpu().tolist()]
                            if isinstance(v, _np.ndarray):
                                if v.ndim == 2:
                                    return [v[i].tolist() for i in range(v.shape[0])]
                                return [v.tolist()]
                            if isinstance(v, (list, tuple)):
                                # Could be [B, T] or [T]
                                if (
                                    (len(v) > 0 and isinstance(v[0], (list, tuple, _np.ndarray)))
                                    or ((_torch is not None) and len(v) > 0 and isinstance(v[0], _torch.Tensor))
                                ):
                                    out = []
                                    for row in v:
                                        out.append(_to_list_1d(row))
                                    return out
                                # Flat -> broadcast later
                                return [_to_list_1d(v)]
                            # Unknown type → try to coerce
                            return [_to_list_1d(v)]
                        return None  # no token field found

                    def _get_texts(_step):
                        v = _extract(_step, "texts")
                        if isinstance(v, list) and all(isinstance(t, str) for t in v):
                            return v
                        v = _extract(_step, "text")
                        if isinstance(v, str):
                            return [v]
                        return None

                    # NEW: handle bare tensor steps from TRT-LLM as token matrix directly
                    tok_matrix = None
                    if (_torch is not None) and isinstance(step, _torch.Tensor):
                        # Force integer dtype to avoid decode exceptions
                        try:
                            step = step.to(dtype=_torch.int64)
                        except Exception:
                            pass
                        if step.ndim == 2:
                            tok_matrix = [step[i].detach().cpu().tolist() for i in range(step.shape[0])]
                        else:
                            tok_matrix = [step.detach().cpu().tolist()]
                    else:
                        tok_matrix = _get_token_matrix(step)
                    # Squeeze leading singleton dims (e.g., [1, B, T] -> [B, T])
                    if tok_matrix is not None:
                        try:
                            while (
                                isinstance(tok_matrix, list)
                                and len(tok_matrix) == 1
                                and isinstance(tok_matrix[0], list)
                                and (len(tok_matrix[0]) > 0)
                                and isinstance(tok_matrix[0][0], (list, tuple, _np.ndarray))
                            ):
                                tok_matrix = tok_matrix[0]
                        except Exception:
                            pass
                    if tok_matrix is not None:
                        # Early debug: show a sample of token ids for the first few steps
                        if demux_step_idx <= 3:
                            try:
                                logger.debug(f"tok_matrix[0][:10]={tok_matrix[0][:10] if tok_matrix and tok_matrix[0] else tok_matrix}")
                            except Exception:
                                pass
                        # Have tokens → decode per sequence and emit cumulative text
                        for i, req in enumerate(batch):
                            toks = tok_matrix[i] if i < len(tok_matrix) else []
                            req.cumulative_text = _decode_full_text(
                                self.tokenizer,
                                req.cumulative_text,
                                type("S", (), {"token_ids": toks}),
                                prompt_len=req.prompt_len,
                            )
                            await req.fut_q.put(_CompatResult([_CompatOutput(req.cumulative_text)]))
                    else:
                        # Fall back to texts/text (string) and broadcast if needed
                        tlist = _get_texts(step)
                        if tlist is not None and len(tlist) > 0:
                            for i, req in enumerate(batch):
                                txt = tlist[i] if i < len(tlist) else tlist[0]
                                st = type("S", (), {"text": txt})
                                req.cumulative_text = _decode_full_text(self.tokenizer, req.cumulative_text, st, prompt_len=req.prompt_len)
                                await req.fut_q.put(_CompatResult([_CompatOutput(req.cumulative_text)]))
                        else:
                            # Nothing usable in this step; skip
                            continue
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
        
        try:
            logger.info(
                f"Tokenizer specials: bos={self.tokenizer.bos_token_id}, "
                f"eos={self.tokenizer.eos_token_id}, pad={self.tokenizer.pad_token_id}, "
                f"unk={getattr(self.tokenizer, 'unk_token_id', None)}"
            )
        except Exception:
            pass

        # Ensure pad_token_id exists and is NOT equal to EOS
        if (self.tokenizer.pad_token_id is None) or (self.tokenizer.pad_token_id == self.tokenizer.eos_token_id):
            if getattr(self.tokenizer, "unk_token_id", None) is not None:
                self.tokenizer.pad_token = self.tokenizer.unk_token
                logger.debug(f"Set pad_token to unk_token: {self.tokenizer.unk_token_id}")
            else:
                # Last resort: use id 0 if safe in this vocab
                try:
                    self.tokenizer.pad_token_id = 0  # type: ignore[attr-defined]
                    logger.debug("Set pad_token_id to 0 as fallback")
                except Exception:
                    logger.warning("Failed to set pad_token_id; pad may equal EOS which can break streaming")
        else:
            logger.debug(f"Using existing pad_token_id: {self.tokenizer.pad_token_id}")
            
        try:
            from .core.custom_tokens import get_audio_token_regex
            rx = get_audio_token_regex(self.tokenizer)
            logger.info(f"Audio token regex: {rx.pattern}")
            # show a small sample for sanity and check that audio tokens exist
            try:
                added = getattr(self.tokenizer, "get_added_vocab", lambda: {})()
                added_items = list(added.keys()) if isinstance(added, dict) else []
                audio_like = [k for k in added_items if rx.search(k)]
                logger.info(f"added_tokens={len(added_items)} audio_like_count={len(audio_like)}")
                logger.info(f"audio_token_samples={audio_like[:5]}")
                if len(audio_like) == 0:
                    logger.warning(
                        "Tokenizer has no audio-like <custom_token_*> entries. "
                        "Streaming text will not surface audio tokens; ensure tokenizer is "
                        "loaded from the same repo/dir as the TRT engine and that added_tokens.json exists."
                    )
            except Exception:
                pass
        except Exception:
            pass

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


