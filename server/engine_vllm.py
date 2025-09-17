import os
import threading
import queue
import asyncio
from typing import Generator, Optional, Dict, Any

from transformers import AutoTokenizer
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm import SamplingParams
from vllm.utils import random_uuid

from .vllm_config import vllm_engine_kwargs
from .text_chunker import chunk_text
from .prompts import build_prompt
from .snac_stream import StreamNormalizer, SnacDecoder, extract_token_numbers

MODEL_ID = os.getenv("MODEL_ID", "canopylabs/orpheus-3b-0.1-ft")

DEFAULT_PARAMS: Dict[str, Any] = dict(
    temperature=0.8,
    top_p=0.9,
    repetition_penalty=1.2,
    num_predict=49152,
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
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
        self.decode_frames = int(os.getenv("SNAC_DECODE_FRAMES", "10"))  # 10 ≈ ~200ms; 5 ≈ ~100ms

    # -------- internal: async producer → PCM --------
    async def _async_pcm_stream(self, text: str, voice: str, params: Dict[str, Any]):
        sp = SamplingParams(
            temperature=float(params.get("temperature", 0.8)),
            top_p=float(params.get("top_p", 0.9)),
            repetition_penalty=float(params.get("repetition_penalty", 1.2)),
            max_tokens=int(params.get("num_predict", 49152)),
            detokenize=False,                # IMPORTANT: stream token_ids, not text
            skip_special_tokens=False,
        )

        req_id = random_uuid()
        prev_len = 0
        normalizer = StreamNormalizer()
        snac = SnacDecoder(sample_rate=24000)
        frames_batch = []

        eot_seen = False
        N_EXTRA_AFTER_EOT = 8192
        extra_budget = N_EXTRA_AFTER_EOT

        async for out in self.engine.generate(build_prompt(text, voice), sp, req_id):
            outs = out.outputs or []
            if not outs:
                continue
            tids = outs[0].token_ids or []
            if not tids:
                continue

            new_ids = tids[prev_len:]
            prev_len = len(tids)

            for tid in new_ids:
                tok_str = self.tokenizer.decode(
                    [tid],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )

                # Capture audio tokens
                for n in extract_token_numbers(tok_str):
                    frame = normalizer.push_number(n)
                    if frame is not None:
                        frames_batch.append(frame)
                        if eot_seen:
                            extra_budget -= 7
                            if extra_budget <= 0:
                                break

                if (not eot_seen) and "<|eot_id|>" in tok_str:
                    eot_seen = True
                    extra_budget = N_EXTRA_AFTER_EOT

            # decode every N frames for ~100–200ms cadence
            if len(frames_batch) >= self.decode_frames:
                snac.add_frames(frames_batch)
                frames_batch.clear()
                pcm = snac.take_new_pcm16()
                if pcm:
                    yield pcm

            if eot_seen and extra_budget <= 0:
                break

        # flush tail
        if frames_batch:
            snac.add_frames(frames_batch)
        normalizer.close()
        pcm = snac.take_new_pcm16()
        if pcm:
            yield pcm

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
