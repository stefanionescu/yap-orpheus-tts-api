import asyncio
import os
import numpy as np
import torch

from ..prompts import build_prompt, resolve_voice
from ..core.snac_batcher import get_snac_batched, SNAC_DEVICE


_CODE_OFFSET = 128266   # first audio code id
_CODE_SIZE = 4096       # codes per sub-stream
_FRAME = 7              # sub-streams per frame
_WINDOW_TOKENS = max(int(os.getenv("TTS_DECODE_WINDOW", "28")), 28)
if _WINDOW_TOKENS % _FRAME != 0:
    _WINDOW_TOKENS -= _WINDOW_TOKENS % _FRAME
if _WINDOW_TOKENS < (_FRAME * 4):  # ensure at least 4 frames of context
    _WINDOW_TOKENS = _FRAME * 4
_WINDOW = _WINDOW_TOKENS
_FRAMES_PER_CHUNK = max(_WINDOW // _FRAME, 1)
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

    # Track frames and emitted PCM length so we don't resend the same chunk
    frames_emitted = 0
    total_samples = 0

    # We decode in windows of 28 tokens (4 frames * 7)
    WINDOW = max(_WINDOW, 28)
    FRAME = _FRAME  # 7

    prev_len = 0  # previous length of token_ids

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

        # Process only the delta since last step
        new = tids[prev_len:]
        if not new:
            continue
        prev_len = len(tids)

        for tid in new:
            # Map token id → RVQ code id if it's an audio token
            # The 7 interleaved audio streams occupy 7*4096 ids starting at _CODE_OFFSET.
            # Channel is determined by current audio_tok_idx % 7.
            chan = audio_tok_idx % FRAME
            code = tid - _CODE_OFFSET - (chan * _CODE_SIZE)

            if 0 <= code < _CODE_SIZE:
                codes_buf.append(int(code))
                audio_tok_idx += 1

                # Emit once per completed frame once we have enough context for SNAC (4 frames by default)
                if (audio_tok_idx % FRAME) == 0 and len(codes_buf) >= WINDOW:
                    frames_ready = audio_tok_idx // FRAME
                    if frames_ready <= frames_emitted:
                        continue

                    new_frames = frames_ready - frames_emitted
                    if new_frames < _FRAMES_PER_CHUNK:
                        continue

                    arr = np.asarray(codes_buf[-WINDOW:], dtype=np.int32).reshape(-1, FRAME)
                    frames_in_window = arr.shape[0]
                    if frames_in_window == 0:
                        continue

                    emit_frames = min((new_frames // _FRAMES_PER_CHUNK) * _FRAMES_PER_CHUNK, frames_in_window)
                    if emit_frames <= 0:
                        continue

                    # channel regrouping: [0], [1,4], [2,3,5,6] (SNAC's layout)
                    codes_0 = torch.from_numpy(arr[:, 0]).unsqueeze(0).to(SNAC_DEVICE)
                    codes_1 = torch.from_numpy(arr[:, [1, 4]].reshape(-1)).unsqueeze(0).to(SNAC_DEVICE)
                    codes_2 = torch.from_numpy(arr[:, [2, 3, 5, 6]].reshape(-1)).unsqueeze(0).to(SNAC_DEVICE)

                    audio = await snacx.decode_codes([codes_0, codes_1, codes_2])  # [-1,1]
                    wav = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
                    if wav.size == 0:
                        frames_emitted = frames_ready
                        continue

                    flat = wav.reshape(-1)
                    samples_total = flat.size
                    samples_per_frame = samples_total // frames_in_window
                    if samples_per_frame == 0:
                        frames_emitted = frames_ready
                        continue

                    emit_samples = emit_frames * samples_per_frame
                    if emit_samples > samples_total:
                        emit_samples = samples_total

                    start = samples_total - emit_samples
                    pcm_chunk = flat[start:]
                    if pcm_chunk.size == 0:
                        frames_emitted = frames_ready
                        continue

                    if _MAX_SAMPLES:
                        remaining = _MAX_SAMPLES - total_samples
                        if remaining <= 0:
                            return
                        if emit_samples > remaining:
                            start = samples_total - remaining
                            pcm_chunk = flat[start:]
                            emit_samples = remaining

                    total_samples += emit_samples

                    yield pcm_chunk.tobytes()
                    frames_emitted += emit_frames
                    await asyncio.sleep(0)

                    if _MAX_SAMPLES and total_samples >= _MAX_SAMPLES:
                        return
            else:
                # Not an audio token; ignore without advancing chan
                continue

    # natural termination on EOS / EOA / EOT
    leftover_frames = (audio_tok_idx // FRAME) - frames_emitted
    if leftover_frames > 0 and len(codes_buf) >= FRAME:
        window_tokens = min(len(codes_buf), WINDOW)
        window_tokens -= window_tokens % FRAME
        if window_tokens < FRAME:
            return
        arr = np.asarray(codes_buf[-window_tokens:], dtype=np.int32).reshape(-1, FRAME)
        frames_in_window = arr.shape[0]
        if frames_in_window > 0:
            emit_frames = min(leftover_frames, frames_in_window)

            codes_0 = torch.from_numpy(arr[:, 0]).unsqueeze(0).to(SNAC_DEVICE)
            codes_1 = torch.from_numpy(arr[:, [1, 4]].reshape(-1)).unsqueeze(0).to(SNAC_DEVICE)
            codes_2 = torch.from_numpy(arr[:, [2, 3, 5, 6]].reshape(-1)).unsqueeze(0).to(SNAC_DEVICE)

            audio = await snacx.decode_codes([codes_0, codes_1, codes_2])
            wav = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
            flat = wav.reshape(-1)
            samples_total = flat.size
            samples_per_frame = samples_total // frames_in_window
            if samples_per_frame > 0:
                emit_samples = emit_frames * samples_per_frame
                if emit_samples > samples_total:
                    emit_samples = samples_total

                start = samples_total - emit_samples
                pcm_chunk = flat[start:]
                if pcm_chunk.size > 0:
                    if _MAX_SAMPLES:
                        remaining = _MAX_SAMPLES - total_samples
                        if remaining <= 0:
                            return
                        if emit_samples > remaining:
                            start = samples_total - remaining
                            pcm_chunk = flat[start:]
                            emit_samples = remaining
                    total_samples += emit_samples
                    yield pcm_chunk.tobytes()
