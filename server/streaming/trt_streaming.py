import asyncio
import os
import numpy as np
import torch

from ..prompts import build_prompt, resolve_voice
from ..core.snac_batcher import get_snac_batched, SNAC_DEVICE


_CODE_START = 128257
_CODE_OFFSET = 128266
_CODE_SIZE = 4096
_AUDIO_MIN = _CODE_OFFSET
_AUDIO_MAX = _CODE_OFFSET + _CODE_SIZE - 1
_FILTER_OUT_ID = 128258
_FRAME = 7
_WINDOW = int(os.getenv("TTS_DECODE_WINDOW", "14"))
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
            stop_token_ids = [128009, 128260]
        else:
            # Never allow start-of-audio (128257) or any id inside the audio code range.
            filtered: list[int] = []
            for token in raw_stops:
                t = int(token)
                if (t == _CODE_START) or (_AUDIO_MIN <= t <= _AUDIO_MAX):
                    continue
                filtered.append(t)
            stop_token_ids = filtered if filtered else [128009, 128260]

        sp = _SP(
            temperature=float(getattr(sp, "temperature", 0.6)),
            top_p=float(getattr(sp, "top_p", 0.9)),
            repetition_penalty=float(getattr(sp, "repetition_penalty", 1.12)),
            max_tokens=int(getattr(sp, "max_tokens", 2048)),
            stop_token_ids=stop_token_ids,
        )

    formatted = build_prompt(prompt, resolve_voice(voice))
    snacx = get_snac_batched()

    collected: list[int] = []
    buf: list[int] = []
    emitted_samples = 0
    started = False

    async for chunk in engine.generate_async(formatted, sp, streaming=True):
        if not chunk.outputs:
            continue

        seq_out = chunk.outputs[0]
        token_ids = seq_out.token_ids
        if not token_ids:
            continue

        new_tokens = token_ids[len(collected) :]
        if os.getenv("TTS_DEBUG"):
            try:
                print("TOKENS:", new_tokens[:64])
            except Exception:
                pass
        if not new_tokens:
            continue

        collected.extend(new_tokens)

        stop_stream = False
        for token in new_tokens:
            # Look for explicit start-of-audio marker, then consume only in-range codes
            if token == _CODE_START:
                # Start a new audio segment; do not treat the marker as a code
                started = True
                continue

            if started and (_AUDIO_MIN <= token <= _AUDIO_MAX):
                code = int(token - _CODE_OFFSET)  # 0..4095
                buf.append(code)
            elif started and (token == _FILTER_OUT_ID):
                # Ignorable marker inside audio segment
                continue
            elif started:
                # First non-audio token after codes -> end of segment
                stop_stream = True
                break

            if len(buf) >= _WINDOW:
                window = buf[-_WINDOW:]
                arr = np.asarray(window, dtype=np.int32).reshape(-1, _FRAME)

                codes_0 = torch.from_numpy(arr[:, 0]).unsqueeze(0).to(SNAC_DEVICE)
                codes_1 = (
                    torch.from_numpy(arr[:, [1, 4]].reshape(-1)).unsqueeze(0).to(SNAC_DEVICE)
                )
                codes_2 = (
                    torch.from_numpy(arr[:, [2, 3, 5, 6]].reshape(-1)).unsqueeze(0).to(SNAC_DEVICE)
                )

                audio = await snacx.decode_codes([codes_0, codes_1, codes_2])
                pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
                if pcm:
                    yield pcm
                    emitted_samples += len(pcm) // 2  # int16 mono samples
                    if _MAX_SAMPLES and emitted_samples >= _MAX_SAMPLES:
                        return
                    await asyncio.sleep(0)

        if stop_stream:
            return
