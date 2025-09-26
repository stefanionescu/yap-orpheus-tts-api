import asyncio
import numpy as np
import torch
from vllm import SamplingParams
from vllm.utils import random_uuid

from ..prompts import build_prompt, resolve_voice
from ..core.custom_tokens import split_custom_tokens, turn_token_into_id
from ..core.snac_batcher import get_snac_batched, SNAC_DEVICE


async def aiter_pcm_from_custom_tokens(engine, prompt: str, voice: str, sp: SamplingParams):
    tok_index = 0
    buf_ids: list[int] = []
    snacx = get_snac_batched()

    prev_len = 0
    async for out in engine.generate(build_prompt(prompt, resolve_voice(voice)), sp, random_uuid()):
        outs = out.outputs or []
        if not outs:
            continue

        piece = outs[0].text or ""
        if len(piece) <= prev_len:
            continue
        delta = piece[prev_len:]
        prev_len = len(piece)

        for n in split_custom_tokens(delta):
            tid = turn_token_into_id(n, tok_index)
            tok_index += 1
            buf_ids.append(tid)

            if (tok_index % 7 == 0) and len(buf_ids) >= 28:
                window = buf_ids[-28:]
                arr = np.asarray(window, dtype=np.int32).reshape(-1, 7)
                codes_0 = torch.from_numpy(arr[:, 0]).unsqueeze(0).to(SNAC_DEVICE)
                codes_1 = torch.from_numpy(arr[:, [1, 4]].reshape(-1)).unsqueeze(0).to(SNAC_DEVICE)
                codes_2 = torch.from_numpy(arr[:, [2, 3, 5, 6]].reshape(-1)).unsqueeze(0).to(SNAC_DEVICE)

                audio = await snacx.decode_codes([codes_0, codes_1, codes_2])
                pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
                if pcm:
                    yield pcm


