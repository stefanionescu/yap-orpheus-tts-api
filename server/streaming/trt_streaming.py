import asyncio
import os
import numpy as np
import torch

from ..prompts import build_prompt, resolve_voice
from ..core.snac_batcher import get_snac_batched, SNAC_DEVICE


_CODE_OFFSET = 128266   # first audio code id
_CODE_SIZE = 4096       # codes per sub-stream
_FRAME = 7              # sub-streams per frame
_WINDOW_TOKENS_RAW = max(int(os.getenv("TTS_DECODE_WINDOW", "14")), 14)
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
    from tensorrt_llm import SamplingParams as _SP  # type: ignore

    # ---- sampling params that work with TRT-LLM streaming
    sp = _SP(
        temperature=float(getattr(sp, "temperature", 0.6)) if hasattr(sp, "temperature") else 0.6,
        top_p=float(getattr(sp, "top_p", 0.8)) if hasattr(sp, "top_p") else 0.8,
        repetition_penalty=float(getattr(sp, "repetition_penalty", 1.1)) if hasattr(sp, "repetition_penalty") else 1.1,
        max_tokens=int(getattr(sp, "max_tokens", 1024)) if hasattr(sp, "max_tokens") else 1024,
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

    async def _emit_window(frames_ready: int) -> bytes | None:
        nonlocal frames_emitted, total_samples
        if len(codes_buf) < WINDOW:
            return None
        if frames_ready <= frames_emitted:
            return None

        # Decode the most recent full window and emit it entirely (constant-size chunk)
        window_codes = codes_buf[-WINDOW:]
        pcm = await _decode_window(window_codes)
        if pcm.size == 0:
            frames_emitted = frames_ready
            return None

        emit_pcm = pcm
        emit_samples = emit_pcm.size

        if _MAX_SAMPLES:
            remaining = _MAX_SAMPLES - total_samples
            if remaining <= 0:
                return None
            if emit_samples > remaining:
                emit_pcm = emit_pcm[-remaining:]
                emit_samples = emit_pcm.size

        total_samples += emit_samples
        frames_emitted = frames_ready

        await asyncio.sleep(0)
        return emit_pcm.tobytes()

    async for chunk in engine.generate_async(formatted, sp, streaming=True):
        if not getattr(chunk, "outputs", None):
            continue

        out = chunk.outputs[0]
        tids = getattr(out, "token_ids", None)
        if not tids:
            # Some TRT builds deliver `output_token_ids` instead
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
                    chunk_bytes = await _emit_window(frames_ready)
                    if chunk_bytes:
                        yield chunk_bytes
            else:
                continue

    # natural termination on EOS / EOA / EOT
    frames_ready = audio_tok_idx // FRAME
    if frames_ready > frames_emitted:
        # Emit a final window sized to the remaining frames (may be < WINDOW)
        leftover_frames = frames_ready - frames_emitted
        window_frames = min(frames_ready, WINDOW // FRAME)
        window_tokens = window_frames * FRAME
        if window_tokens <= len(codes_buf):
            window_codes = codes_buf[-window_tokens:]
            pcm = await _decode_window(window_codes)
            if pcm.size > 0:
                emit_pcm = pcm
                emit_samples = emit_pcm.size
                if _MAX_SAMPLES:
                    remaining = _MAX_SAMPLES - total_samples
                    if remaining > 0 and emit_samples > remaining:
                        emit_pcm = emit_pcm[-remaining:]
                        emit_samples = emit_pcm.size
                if emit_samples > 0:
                    total_samples += emit_samples
                    frames_emitted = frames_ready
                    await asyncio.sleep(0)
                    yield emit_pcm.tobytes()