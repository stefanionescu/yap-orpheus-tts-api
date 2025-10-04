import os
import asyncio
from typing import Optional

import numpy as np
import torch
from snac import SNAC


SNAC_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SnacBatched:
    def __init__(self) -> None:
        self.dtype_decoder = torch.float32
        m = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(SNAC_DEVICE)
        m.decoder = m.decoder.to(self.dtype_decoder)
        if bool(int(os.getenv("SNAC_TORCH_COMPILE", "0"))):
            m.decoder = torch.compile(m.decoder, dynamic=True)
            m.quantizer = torch.compile(m.quantizer, dynamic=True)
        self.m = m
        self.stream = torch.cuda.Stream(device=torch.device(SNAC_DEVICE)) if torch.cuda.is_available() else None
        # Match legacy pacing: optionally force global CUDA sync after decode
        self.global_sync = bool(int(os.getenv("SNAC_GLOBAL_SYNC", "1")))

        # Dynamic batching controls
        self.max_batch = int(os.getenv("SNAC_MAX_BATCH", "64"))
        self.batch_timeout_ms = int(os.getenv("SNAC_BATCH_TIMEOUT_MS", "2"))
        self._req_q: Optional[asyncio.Queue] = None
        self._worker_started = False

    async def _ensure_worker(self) -> None:
        if self._req_q is None:
            self._req_q = asyncio.Queue()
        if not self._worker_started:
            asyncio.create_task(self._batch_worker())
            self._worker_started = True

    async def _batch_worker(self) -> None:
        assert self._req_q is not None
        while True:
            first = await self._req_q.get()
            items = [first]
            try:
                await asyncio.sleep(max(0.0, self.batch_timeout_ms / 1000.0))
            except Exception:
                pass
            while len(items) < self.max_batch:
                try:
                    items.append(self._req_q.get_nowait())
                except Exception:
                    break

            codes_list = [it[0] for it in items]
            futs = [it[1] for it in items]

            try:
                with torch.inference_mode():
                    if self.stream is not None:
                        with torch.cuda.stream(self.stream):
                            shapes = [(c[0].shape, c[1].shape, c[2].shape) for c in codes_list]
                            can_cat = len(set(shapes)) == 1
                            if can_cat:
                                c0 = torch.cat([c[0] for c in codes_list], dim=0)
                                c1 = torch.cat([c[1] for c in codes_list], dim=0)
                                c2 = torch.cat([c[2] for c in codes_list], dim=0)
                                z_q = self.m.quantizer.from_codes([c0, c1, c2])
                                audio_hat = self.m.decoder(z_q.to(self.dtype_decoder))[:, :, 2048:4096]
                                outs = list(audio_hat.split(1, dim=0))
                            else:
                                outs = []
                                for c0, c1, c2 in codes_list:
                                    z_q = self.m.quantizer.from_codes([c0, c1, c2])
                                    outs.append(self.m.decoder(z_q.to(self.dtype_decoder))[:, :, 2048:4096])
                            # Synchronization strategy:
                            # - Legacy (default): global synchronize to pace GPU like old code
                            # - Alternative: synchronize only the dedicated stream to reduce stalls
                            if torch.cuda.is_available():
                                if self.global_sync:
                                    torch.cuda.synchronize()
                                elif self.stream is not None:
                                    self.stream.synchronize()
                    else:
                        shapes = [(c[0].shape, c[1].shape, c[2].shape) for c in codes_list]
                        can_cat = len(set(shapes)) == 1
                        if can_cat:
                            c0 = torch.cat([c[0] for c in codes_list], dim=0)
                            c1 = torch.cat([c[1] for c in codes_list], dim=0)
                            c2 = torch.cat([c[2] for c in codes_list], dim=0)
                            z_q = self.m.quantizer.from_codes([c0, c1, c2])
                            audio_hat = self.m.decoder(z_q.to(self.dtype_decoder))[:, :, 2048:4096]
                            outs = list(audio_hat.split(1, dim=0))
                        else:
                            outs = []
                            for c0, c1, c2 in codes_list:
                                z_q = self.m.quantizer.from_codes([c0, c1, c2])
                                outs.append(self.m.decoder(z_q.to(self.dtype_decoder))[:, :, 2048:4096])
                for fut, out in zip(futs, outs):
                    if not fut.done():
                        fut.set_result(out[0].detach().cpu().numpy())
            except Exception as e:
                for fut in futs:
                    if not fut.done():
                        fut.set_exception(e)

    async def decode_codes(self, codes_triplet: list[torch.Tensor]) -> np.ndarray:
        await self._ensure_worker()
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        assert self._req_q is not None
        await self._req_q.put((codes_triplet, fut))
        return await fut


_BATCHER: Optional[SnacBatched] = None


def get_snac_batched() -> SnacBatched:
    global _BATCHER
    if _BATCHER is None:
        _BATCHER = SnacBatched()
    return _BATCHER

