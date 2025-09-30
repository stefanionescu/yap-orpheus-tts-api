import asyncio
import os
import numpy as np
import torch

from ..prompts import build_prompt, resolve_voice
from ..core.snac_batcher import get_snac_batched, SNAC_DEVICE


_AUDIO_START_ID = 128266
_AUDIO_STOP_ID = 156938
_FILTER_OUT_ID = 128258
_FRAME = 7
_WINDOW = 28
_SAMPLE_RATE = int(os.getenv("SNAC_SR", "24000"))
_MAX_SEC = float(os.getenv("TTS_MAX_SEC", "0"))
# Optional wall-clock guard; zero disables the cap.
_MAX_SAMPLES = int(_MAX_SEC * _SAMPLE_RATE) if _MAX_SEC > 0 else 0


async def aiter_pcm_from_custom_tokens(engine, prompt: str, voice: str, sp) -> bytes:
    from tensorrt_llm.llmapi import SamplingParams  # type: ignore

    if not isinstance(sp, SamplingParams):
        from tensorrt_llm.llmapi import SamplingParams as _SP  # type: ignore

        raw_stops = getattr(sp, "stop_token_ids", None)
        if raw_stops is None:
            stop_token_ids = [128009]
        else:
            # 128257 is the start-of-audio sentinel; never treat it as a stop token.
            stop_token_ids = [int(token) for token in raw_stops if int(token) != 128257]
            if not stop_token_ids:
                stop_token_ids = [128009]

        sp = _SP(
            temperature=float(getattr(sp, "temperature", 0.6)),
            top_p=float(getattr(sp, "top_p", 0.9)),
            repetition_penalty=float(getattr(sp, "repetition_penalty", 1.05)),
            max_tokens=int(getattr(sp, "max_tokens", 2048)),
            stop_token_ids=stop_token_ids,
        )

    formatted = build_prompt(prompt, resolve_voice(voice))
    snacx = get_snac_batched()

    loop = asyncio.get_running_loop()

    def _run_blocking():
        return engine.generate([formatted], sp)

    outputs = await loop.run_in_executor(None, _run_blocking)
    if not outputs:
        return

    raw_token_ids = outputs[0].outputs[0].token_ids
    if not raw_token_ids:
        return

    audio_tokens: list[int] = []
    started = False
    for token in raw_token_ids:
        if _AUDIO_START_ID <= token <= _AUDIO_STOP_ID:
            started = True
            if token != _FILTER_OUT_ID:
                audio_tokens.append(token)
        elif started:
            break

    if not audio_tokens:
        return

    audio_ids = torch.tensor(audio_tokens, dtype=torch.int64)
    codes = ((audio_ids - _AUDIO_START_ID) % 4096).tolist()

    buf: list[int] = []
    emitted_samples = 0
    for c in codes:
        buf.append(int(c))
        if len(buf) >= _WINDOW:
            window = buf[-_WINDOW:]
            arr = np.asarray(window, dtype=np.int32).reshape(-1, _FRAME)

            codes_0 = torch.from_numpy(arr[:, 0]).unsqueeze(0).to(SNAC_DEVICE)
            codes_1 = torch.from_numpy(arr[:, [1, 4]].reshape(-1)).unsqueeze(0).to(SNAC_DEVICE)
            codes_2 = torch.from_numpy(arr[:, [2, 3, 5, 6]].reshape(-1)).unsqueeze(0).to(SNAC_DEVICE)

            audio = await snacx.decode_codes([codes_0, codes_1, codes_2])
            pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
            if pcm:
                yield pcm
                emitted_samples += len(pcm) // 2  # int16 mono samples
                if _MAX_SAMPLES and emitted_samples >= _MAX_SAMPLES:
                    return
                await asyncio.sleep(0)
