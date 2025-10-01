import asyncio
import os
import numpy as np
import torch

from ..prompts import build_prompt, resolve_voice
from ..core.snac_batcher import get_snac_batched, SNAC_DEVICE


_CODE_OFFSET = 128266   # first audio code id
_CODE_SIZE = 4096       # codes per sub-stream
_FRAME = 7              # sub-streams per frame
_WINDOW_TOKENS_RAW = max(int(os.getenv("TTS_DECODE_WINDOW", "28")), 28)
_WINDOW_TOKENS_ADJ = _WINDOW_TOKENS_RAW - (_WINDOW_TOKENS_RAW % _FRAME)
if _WINDOW_TOKENS_ADJ < (_FRAME * 4):
    _WINDOW_TOKENS_ADJ = _FRAME * 4
WINDOW_TOKENS = _WINDOW_TOKENS_ADJ
_SAMPLE_RATE = int(os.getenv("SNAC_SR", "24000"))
_MAX_SEC = float(os.getenv("TTS_MAX_SEC", "0"))
# Optional wall-clock guard; zero disables the cap.
_MAX_SAMPLES = int(_MAX_SEC * _SAMPLE_RATE) if _MAX_SEC > 0 else 0


async def aiter_pcm_from_custom_tokens(engine, prompt: str, voice: str, sp) -> bytes:
    """
    TRT-LLM streaming: read token_ids deltas, not detokenized text.
    Map Orpheus audio token ids → 7-stream RVQ codes → SNAC decode.
    """
    from tensorrt_llm.llmapi import SamplingParams as _SP  # type: ignore

    # ---- sampling params that work with TRT-LLM streaming
    sp = _SP(
        temperature=float(getattr(sp, "temperature", 0.6)) if hasattr(sp, "temperature") else 0.6,
        top_p=float(getattr(sp, "top_p", 0.8)) if hasattr(sp, "top_p") else 0.8,
        repetition_penalty=float(getattr(sp, "repetition_penalty", 1.1)) if hasattr(sp, "repetition_penalty") else 1.1,
        max_tokens=int(getattr(sp, "max_tokens", 2048)) if hasattr(sp, "max_tokens") else 2048,
        # Important bits for Orpheus:
        stop_token_ids=[128258, 128262, 128009],  # EOS(speech), EOA, EOT
        detokenize=False,              # <- we don't want LLM-side detok; we read token_ids
        skip_special_tokens=False,     # keep specials in token_ids stream
        add_special_tokens=False,      # don't inject extras that break the layout
        ignore_eos=False,
    )

    # Build the Orpheus prompt
    formatted = build_prompt(prompt, resolve_voice(voice))

    # SNAC decoder (+ async batcher)
    snacx = get_snac_batched()

    # Rolling buffer of raw per-frame code values (0..4095)
    codes_buf: list[int] = []

    # How many **audio** tokens we've consumed so far (mod 7 selects the sub-codec stream)
    audio_tok_idx = 0

    # Track emitted frames and samples
    frames_emitted = 0
    total_samples = 0

    WINDOW = max(WINDOW_TOKENS, _FRAME * 4)
    FRAME = _FRAME
    frames_per_chunk = max(WINDOW // FRAME, 1)

    # decode scheduling (lets SNAC work overlap with TRT token generation)
    decode_queue: asyncio.Queue[tuple[int | None, np.ndarray | Exception | None]] = asyncio.Queue()
    pcm_queue: asyncio.Queue[bytes | Exception | None] = asyncio.Queue()
    decode_tasks: list[asyncio.Task] = []
    frames_scheduled = 0
    next_chunk_idx = 1

    prev_len = 0  # previous length of token_ids

    async def _decode_window(window_codes: list[int]) -> np.ndarray:
        arr = np.asarray(window_codes, dtype=np.int32).reshape(-1, FRAME)
        if arr.size == 0:
            return np.empty(0, dtype=np.int16)

        # channel regrouping: [0], [1,4], [2,3,5,6] (SNAC's layout)
        codes_0 = torch.from_numpy(arr[:, 0]).unsqueeze(0).to(SNAC_DEVICE)
        codes_1 = torch.from_numpy(arr[:, [1, 4]].reshape(-1)).unsqueeze(0).to(SNAC_DEVICE)
        codes_2 = torch.from_numpy(arr[:, [2, 3, 5, 6]].reshape(-1)).unsqueeze(0).to(SNAC_DEVICE)

        audio = await snacx.decode_codes([codes_0, codes_1, codes_2])  # [-1, 1]
        wav = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
        return wav.reshape(-1)

    async def _decode_and_queue(idx: int, window_codes: list[int]) -> None:
        try:
            pcm = await _decode_window(window_codes)
        except Exception as exc:  # propagate errors to main loop
            await decode_queue.put((-1, exc))
            return
        await decode_queue.put((idx, pcm))

    def _schedule_decode(frames_ready: int) -> None:
        nonlocal frames_scheduled
        if len(codes_buf) < WINDOW:
            return
        while (frames_ready - frames_scheduled) >= frames_per_chunk:
            chunk_idx = (frames_scheduled // frames_per_chunk) + 1
            snapshot = list(codes_buf[-WINDOW:])
            task = asyncio.create_task(_decode_and_queue(chunk_idx, snapshot))
            decode_tasks.append(task)
            frames_scheduled += frames_per_chunk

    async def _consume_decode_queue() -> None:
        nonlocal next_chunk_idx, frames_emitted, total_samples
        pending: dict[int, np.ndarray] = {}
        finished_producer = False
        stop_emission = False

        while True:
            if next_chunk_idx in pending:
                pcm = pending.pop(next_chunk_idx)
                frames_emitted = next_chunk_idx * frames_per_chunk
                next_chunk_idx += 1

                if pcm.size == 0 or stop_emission:
                    continue

                emit_pcm = pcm
                emit_samples = emit_pcm.size
                if _MAX_SAMPLES:
                    remaining = _MAX_SAMPLES - total_samples
                    if remaining <= 0:
                        stop_emission = True
                        continue
                    if emit_samples > remaining:
                        emit_pcm = emit_pcm[-remaining:]
                        emit_samples = emit_pcm.size

                if emit_samples == 0:
                    continue

                total_samples += emit_samples
                if _MAX_SAMPLES and total_samples >= _MAX_SAMPLES:
                    stop_emission = True

                await asyncio.sleep(0)
                await pcm_queue.put(emit_pcm.tobytes())
                continue

            if finished_producer and not pending:
                break

            idx, payload = await decode_queue.get()
            if idx == -1:
                assert isinstance(payload, Exception)
                await pcm_queue.put(payload)
                finished_producer = True
                pending.clear()
                break
            if idx is None:
                finished_producer = True
                continue
            assert isinstance(payload, np.ndarray)
            pending[idx] = payload

        await pcm_queue.put(None)

    async def _produce_tokens() -> None:
        nonlocal prev_len, audio_tok_idx
        try:
            async for chunk in engine.generate_async(formatted, sp, streaming=True):
                if not getattr(chunk, "outputs", None):
                    continue

                out = chunk.outputs[0]
                tids = getattr(out, "token_ids", None)
                if not tids:
                    tids = getattr(out, "output_token_ids", None)
                if not tids:
                    continue

                new = tids[prev_len:]
                if not new:
                    continue
                prev_len = len(tids)

                for tid in new:
                    chan = audio_tok_idx % FRAME
                    code = tid - _CODE_OFFSET - (chan * _CODE_SIZE)

                    if 0 <= code < _CODE_SIZE:
                        codes_buf.append(int(code))
                        audio_tok_idx += 1
                        if (audio_tok_idx % FRAME) == 0:
                            frames_ready = audio_tok_idx // FRAME
                            _schedule_decode(frames_ready)
                    else:
                        continue

            if audio_tok_idx >= FRAME:
                _schedule_decode(audio_tok_idx // FRAME)
        except Exception as exc:
            await decode_queue.put((-1, exc))
            raise
        finally:
            if decode_tasks:
                await asyncio.gather(*decode_tasks, return_exceptions=False)
            await decode_queue.put((None, None))

    producer_task = asyncio.create_task(_produce_tokens())
    consumer_task = asyncio.create_task(_consume_decode_queue())

    try:
        while True:
            item = await pcm_queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        await producer_task
        await consumer_task
