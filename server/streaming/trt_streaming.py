import asyncio
import os
import numpy as np
import torch

from ..prompts import build_prompt, resolve_voice
from ..core.snac_batcher import get_snac_batched, SNAC_DEVICE
from ..core.custom_tokens import split_custom_tokens, turn_token_into_id


_CODE_START = 128257
_CODE_OFFSET = 128266
_CODE_SIZE = 4096
_AUDIO_MIN = _CODE_OFFSET
_AUDIO_MAX = _CODE_OFFSET + _CODE_SIZE - 1
_FILTER_OUT_ID = 128258
_FRAME = 7
_WINDOW = int(os.getenv("TTS_DECODE_WINDOW", "7"))
_SAMPLE_RATE = int(os.getenv("SNAC_SR", "24000"))
_MAX_SEC = float(os.getenv("TTS_MAX_SEC", "0"))
# Optional wall-clock guard; zero disables the cap.
_MAX_SAMPLES = int(_MAX_SEC * _SAMPLE_RATE) if _MAX_SEC > 0 else 0


async def aiter_pcm_from_custom_tokens(engine, prompt: str, voice: str, sp) -> bytes:
    from tensorrt_llm.llmapi import SamplingParams  # type: ignore
    from tensorrt_llm.llmapi import SamplingParams as _SP  # type: ignore

    # Always ensure TRT returns detokenized text with specials preserved, and stop on 128258
    temperature = float(getattr(sp, "temperature", 0.6)) if hasattr(sp, "temperature") else 0.6
    top_p = float(getattr(sp, "top_p", 0.8)) if hasattr(sp, "top_p") else 0.8
    repetition_penalty = float(getattr(sp, "repetition_penalty", 1.1)) if hasattr(sp, "repetition_penalty") else 1.1
    max_tokens = int(getattr(sp, "max_tokens", 2048)) if hasattr(sp, "max_tokens") else 2048
    sp = _SP(
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        max_tokens=max_tokens,
        stop_token_ids=[128258],
        detokenize=True,
        skip_special_tokens=False,
        ignore_eos=False,
    )

    formatted = build_prompt(prompt, resolve_voice(voice))
    snacx = get_snac_batched()

    buf: list[int] = []
    emitted_samples = 0
    tok_index = 0
    prev_len = 0

    async for chunk in engine.generate_async(formatted, sp, streaming=True):
        if not chunk.outputs:
            continue

        outs = chunk.outputs or []
        if not outs:
            continue
        piece = getattr(outs[0], "text", "") or ""
        if len(piece) <= prev_len:
            continue
        delta = piece[prev_len:]
        prev_len = len(piece)

        for token_number in split_custom_tokens(delta):
            audio_id = turn_token_into_id(token_number, tok_index)
            tok_index += 1
            if 0 <= audio_id < 4096:
                buf.append(int(audio_id))

            # Whenever we have >= 7 tokens, decode and emit only the delta
            if len(buf) >= _WINDOW:
                arr = np.asarray(buf[-_WINDOW:], dtype=np.int32).reshape(-1, _FRAME)
                codes_0 = torch.from_numpy(arr[:, 0]).unsqueeze(0).to(SNAC_DEVICE)
                codes_1 = torch.from_numpy(arr[:, [1, 4]].reshape(-1)).unsqueeze(0).to(SNAC_DEVICE)
                codes_2 = torch.from_numpy(arr[:, [2, 3, 5, 6]].reshape(-1)).unsqueeze(0).to(SNAC_DEVICE)
                
                # Decode the full window to get complete audio
                audio = await snacx.decode_codes([codes_0, codes_1, codes_2])  # [-1,1], full
                
                # Convert to int16 PCM
                wav = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
                
                # Emit only what we haven't sent yet
                if emitted_samples < wav.shape[-1]:
                    new_pcm = wav[emitted_samples:].tobytes()
                    emitted_samples = wav.shape[-1]
                    if new_pcm:
                        yield new_pcm
                        if _MAX_SAMPLES and emitted_samples >= _MAX_SAMPLES:
                            return
                await asyncio.sleep(0)

    # Natural termination when stop_token_ids trigger or generator completes
