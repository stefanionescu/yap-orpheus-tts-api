import os
import threading
import queue
import asyncio
from typing import Generator, Optional, Dict, Any, AsyncGenerator, List

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm import SamplingParams
from vllm.utils import random_uuid

from .vllm_config import vllm_engine_kwargs
from .text_chunker import chunk_text
from .prompts import build_prompt, resolve_voice
from .snac_stream import SnacDecoder

MODEL_ID = os.getenv("MODEL_ID", "canopylabs/orpheus-3b-0.1-ft")

# Finetune audio code ID layout (7 streams of 4096 codes each)
FINETUNE_BASE = 128266      # start of audio code space
FINETUNE_SPAN = 4096
FINETUNE_LAYERS = 7
FINETUNE_EOT = 128258       # stop id
FINETUNE_START_AUDIO = 128257

class FTStreamNormalizer:
    """Consumes token ids and yields aligned 7-code frames once streaming starts."""
    def __init__(self):
        # Prompt already contains FINETUNE_START_AUDIO, so generated tokens are codes.
        # Start immediately to avoid waiting for 128257 in the generated stream.
        self.started = True
        self.pos = 0
        self.buf = []
        self.drop_count = 0

    def _decode_code(self, tid: int):
        rel = tid - FINETUNE_BASE
        if 0 <= rel < FINETUNE_LAYERS * FINETUNE_SPAN:
            k = rel // FINETUNE_SPAN
            v = rel - k * FINETUNE_SPAN
            return (k, v)
        return None

    def push_id(self, tid: int):
        # Start-of-audio was in the prompt; proceed directly with codes.
        # ignore EOT in stream
        if tid == FINETUNE_EOT:
            return None

        dv = self._decode_code(tid)
        if dv is None:
            self.drop_count += 1
            return None
        k, v = dv
        if k == self.pos:
            self.buf.append(v)
            self.pos += 1
            if self.pos == FINETUNE_LAYERS:
                frame = self.buf
                self.buf = []
                self.pos = 0
                return frame
            return None
        # resync only on layer 0
        if k == 0:
            self.buf = [v]
            self.pos = 1
        # else drop until we hit a layer-0 again
        return None

DEFAULT_PARAMS: Dict[str, Any] = dict(
    temperature=0.6,
    top_p=0.9,
    repetition_penalty=1.2,  # Orpheus requires >=1.1 for stability
    num_predict=6144,         # default aligned with Baseten example max_tokens
)

# Simple voice aliases for backward compatibility with docs/tests
ALIASES: Dict[str, str] = {
    "female": "tara",
    "male": "zac",
}

class OrpheusTTSEngine:
    def __init__(self):
        ekw = vllm_engine_kwargs()
        # Async engine
        self.engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
            model=MODEL_ID,
            tokenizer=MODEL_ID,            # <-- Force use of model's tokenizer
            **ekw,
        ))
        self.decode_frames = int(os.getenv("SNAC_DECODE_FRAMES", "2"))  # 2–3 for smoother cadence

    # -------- internal: async text→frames extractor --------
    async def _async_extract_frames(
        self,
        text: str,
        voice: str,
        params: Dict[str, Any],
    ) -> AsyncGenerator[List[int], None]:
        """Yield 7-code frames by processing token IDs from the finetuned model."""
        sp = SamplingParams(
            temperature=float(params.get("temperature", 0.8)),
            top_p=float(params.get("top_p", 0.9)),
            repetition_penalty=float(params.get("repetition_penalty", 1.2)),
            max_tokens=int(params.get("num_predict", DEFAULT_PARAMS["num_predict"])),
            detokenize=False,                 # <-- text not needed
            skip_special_tokens=False,        # <-- keep specials; we need ids
            ignore_eos=False,
            stop_token_ids=[128258],          # <-- finetune EOT
        )

        req_id = random_uuid()
        normalizer = FTStreamNormalizer()
        prev_n = 0

        DEBUG = bool(int(os.getenv("SNAC_DEBUG", "0")))
        dbg_frames = 0

        try:
            # With prefix caching enabled in vLLM, the shared prompt prefix
            # (SOH + voice label + colon/space) benefits across calls.
            async for out in self.engine.generate(build_prompt(text, resolve_voice(voice)), sp, req_id):
                outs = out.outputs or []
                if not outs:
                    continue
                toks = outs[0].token_ids or []
                # take only the newly generated tokens
                for tid in toks[prev_n:]:
                    # Optional: early break on explicit EOT
                    if tid == 128258:
                        break
                    frame = normalizer.push_id(tid)
                    if frame is not None:
                        if DEBUG:
                            dbg_frames += 1
                        yield frame
                prev_n = len(toks)
        except Exception:
            # Only abort on error/cancellation; avoid noisy "Aborted request" logs on normal completion
            try:
                await self.engine.abort(req_id)
            except Exception:
                pass
            raise

        if DEBUG:
            print(f"[snac] frames={dbg_frames} drops={normalizer.drop_count}", flush=True)

    # -------- internal: async producer → PCM (built on frames extractor) --------
    async def _async_pcm_stream(self, text: str, voice: str, params: Dict[str, Any]):
        snac = SnacDecoder(sample_rate=24000)
        frames_batch: List[List[int]] = []

        DEBUG = bool(int(os.getenv("SNAC_DEBUG", "0")))
        dbg_bytes = 0

        PRIME_FRAMES = int(os.getenv("SNAC_PRIME_FRAMES", "2"))
        primed = False

        async for frame in self._async_extract_frames(text, voice, params):
            frames_batch.append(frame)

            # Hold back initial PCM until we have a short prime window of frames
            if not primed and len(frames_batch) < PRIME_FRAMES:
                continue

            if not primed:
                # Prime immediately with what we have, then continue at normal cadence
                snac.add_frames(frames_batch)
                frames_batch.clear()
                pcm = snac.take_new_pcm16()
                if pcm:
                    if DEBUG:
                        dbg_bytes += len(pcm)
                    yield pcm
                primed = True
                continue

            if len(frames_batch) >= self.decode_frames:
                snac.add_frames(frames_batch)
                frames_batch.clear()
                pcm = snac.take_new_pcm16()
                if pcm:
                    if DEBUG:
                        dbg_bytes += len(pcm)
                    yield pcm

        # flush tail
        if frames_batch:
            snac.add_frames(frames_batch)
        pcm = snac.take_new_pcm16()
        if pcm:
            if DEBUG:
                dbg_bytes += len(pcm)
            yield pcm
        if DEBUG:
            print(f"[snac] bytes={dbg_bytes}", flush=True)

    # -------- public: sync generator for FastAPI StreamingResponse --------
    def _piece_pcm_gen(self, text: str, voice: str, params: Dict[str, Any]) -> Generator[bytes, None, None]:
        q: "queue.Queue[Optional[bytes]]" = queue.Queue(maxsize=32)

        async def runner():
            try:
                async for pcm in self._async_pcm_stream(text, voice, params):
                    q.put(pcm)
            finally:
                q.put(None)

        def bg():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(runner())
            loop.close()

        t = threading.Thread(target=bg, daemon=True)
        t.start()

        while True:
            item = q.get()
            if item is None:
                break
            yield item
        t.join()

    def stream_pcm_chunks(
        self,
        text: str,
        voice: str = "tara",
        chunk_chars: int = 450,
        *,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        **kwargs,
    ) -> Generator[bytes, None, None]:
        params = DEFAULT_PARAMS.copy()
        if temperature is not None: params["temperature"] = float(temperature)
        if top_p is not None: params["top_p"] = float(top_p)
        if repetition_penalty is not None: params["repetition_penalty"] = float(repetition_penalty)

        resolved_voice = ALIASES.get(str(voice).lower(), voice)
        # Stream per chunk for predictable latency on very long inputs
        for piece in chunk_text(text, target_chars=chunk_chars):
            for pcm in self._piece_pcm_gen(piece, resolved_voice, params):
                yield pcm

    # -------- public: async frames for a single text piece (for WS continuity) --------
    async def aiter_frames(
        self,
        text: str,
        voice: str = "tara",
        *,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        **kwargs,
    ) -> AsyncGenerator[List[int], None]:
        params = DEFAULT_PARAMS.copy()
        if temperature is not None: params["temperature"] = float(temperature)
        if top_p is not None: params["top_p"] = float(top_p)
        if repetition_penalty is not None: params["repetition_penalty"] = float(repetition_penalty)
        # Support pass-through overrides (Baseten-style max_tokens/num_predict)
        if "num_predict" in kwargs and kwargs["num_predict"] is not None:
            try:
                params["num_predict"] = int(kwargs["num_predict"])  # noqa: C401
            except Exception:
                pass
        if "max_tokens" in kwargs and kwargs["max_tokens"] is not None:
            try:
                params["num_predict"] = int(kwargs["max_tokens"])  # alias
            except Exception:
                pass

        resolved_voice = ALIASES.get(str(voice).lower(), voice)
        async for frame in self._async_extract_frames(text, resolved_voice, params):
            yield frame

    def synthesize_wav_bytes(
        self,
        text: str,
        voice: str = "tara",
        chunk_chars: int = 450,
        **gen_kwargs,
    ) -> bytes:
        import io, wave
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            for chunk in self.stream_pcm_chunks(text, voice, chunk_chars, **gen_kwargs):
                if chunk:
                    wf.writeframes(chunk)
        return buf.getvalue()
