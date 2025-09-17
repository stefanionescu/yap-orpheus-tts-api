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
from .snac_stream import StreamNormalizer, SnacDecoder, AUDIO_TOKEN_RE

MODEL_ID = os.getenv("MODEL_ID", "canopylabs/orpheus-3b-0.1-ft")

DEFAULT_PARAMS: Dict[str, Any] = dict(
    temperature=0.8,
    top_p=0.9,
    repetition_penalty=1.2,   # Orpheus requires >=1.1 for stability
    num_predict=8192,         # sane default; override per call if you need
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
            **ekw,
        ))
        self.decode_frames = int(os.getenv("SNAC_DECODE_FRAMES", "5"))  # 5 ≈ ~100ms; 10 ≈ ~200ms

    # -------- internal: async text→frames extractor --------
    async def _async_extract_frames(
        self,
        text: str,
        voice: str,
        params: Dict[str, Any],
    ) -> AsyncGenerator[List[int], None]:
        """Yield 7-code frames by parsing streamed TEXT deltas for <custom_token_####>."""
        sp = SamplingParams(
            temperature=float(params.get("temperature", 0.8)),
            top_p=float(params.get("top_p", 0.9)),
            repetition_penalty=float(params.get("repetition_penalty", 1.2)),
            max_tokens=int(params.get("num_predict", 8192)),
            detokenize=True,                 # stream TEXT deltas
            skip_special_tokens=False,       # keep <custom_token_####>
            ignore_eos=False,                # stop on EOS tokens
            # Stop on either special terminator the tokenizer may emit:
            stop=["<|eot_id|>", "<|end_of_text|>"],
        )

        req_id = random_uuid()
        prev_text = ""
        carry = ""

        normalizer = StreamNormalizer()

        DEBUG = bool(int(os.getenv("SNAC_DEBUG", "0")))
        dbg_frames = 0

        try:
            async for out in self.engine.generate(build_prompt(text, resolve_voice(voice)), sp, req_id):
                outs = out.outputs or []
                if not outs:
                    continue
                full_txt = outs[0].text or ""
                if not full_txt:
                    continue

                # delta since last step
                delta = full_txt[len(prev_text):]
                prev_text = full_txt
                if not delta:
                    continue

                carry += delta

                # extract numbers from the rolling carry
                last_end = 0
                for m in AUDIO_TOKEN_RE.finditer(carry):
                    last_end = m.end()
                    n = int(m.group(1))
                    frame = normalizer.push_number(n)
                    if frame is not None:
                        if DEBUG:
                            dbg_frames += 1
                        yield frame

                # trim processed part
                if last_end > 0:
                    carry = carry[last_end:]
        finally:
            # Ensure pending generation is cancelled if we broke out early.
            try:
                self.engine.abort(req_id)
            except Exception:
                pass

        # finalize
        normalizer.close()
        if DEBUG:
            print(f"[snac] frames={dbg_frames}", flush=True)

    # -------- internal: async producer → PCM (built on frames extractor) --------
    async def _async_pcm_stream(self, text: str, voice: str, params: Dict[str, Any]):
        snac = SnacDecoder(sample_rate=24000)
        frames_batch: List[List[int]] = []

        DEBUG = bool(int(os.getenv("SNAC_DEBUG", "0")))
        dbg_bytes = 0

        async for frame in self._async_extract_frames(text, voice, params):
            frames_batch.append(frame)
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
