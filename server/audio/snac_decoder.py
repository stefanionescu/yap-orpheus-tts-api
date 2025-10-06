"""Refactored SNAC batching logic with reduced duplication."""

import asyncio
from typing import Optional

import numpy as np
import torch
from snac import SNAC

from server.config import settings


SNAC_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SnacProcessor:
    """Handles SNAC model operations with batching optimization."""
    
    def __init__(self):
        self.dtype_decoder = torch.float32
        self.model = self._initialize_model()
        self.stream = torch.cuda.Stream(device=torch.device(SNAC_DEVICE)) if torch.cuda.is_available() else None
    
    def _initialize_model(self) -> SNAC:
        """Initialize and configure SNAC model."""
        model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(SNAC_DEVICE)
        model.decoder = model.decoder.to(self.dtype_decoder)
        
        if settings.snac_torch_compile:
            model.decoder = torch.compile(model.decoder, dynamic=True)
            model.quantizer = torch.compile(model.quantizer, dynamic=True)
        
        return model
    
    def _process_batch_cuda(self, codes_list: list) -> list:
        """Process batch on CUDA with stream optimization."""
        with torch.cuda.stream(self.stream):
            return self._decode_codes_batch(codes_list)
    
    def _process_batch_cpu(self, codes_list: list) -> list:
        """Process batch on CPU."""
        return self._decode_codes_batch(codes_list)
    
    def _decode_codes_batch(self, codes_list: list) -> list:
        """Decode a batch of codes, with optional concatenation optimization."""
        shapes = [(c[0].shape, c[1].shape, c[2].shape) for c in codes_list]
        can_concatenate = len(set(shapes)) == 1
        
        if can_concatenate and len(codes_list) > 1:
            # Batch processing via concatenation
            c0 = torch.cat([c[0] for c in codes_list], dim=0)
            c1 = torch.cat([c[1] for c in codes_list], dim=0)
            c2 = torch.cat([c[2] for c in codes_list], dim=0)
            z_q = self.model.quantizer.from_codes([c0, c1, c2])
            audio_hat = self.model.decoder(z_q.to(self.dtype_decoder))[:, :, 2048:4096]
            return list(audio_hat.split(1, dim=0))
        else:
            # Individual processing
            outputs = []
            for c0, c1, c2 in codes_list:
                z_q = self.model.quantizer.from_codes([c0, c1, c2])
                outputs.append(self.model.decoder(z_q.to(self.dtype_decoder))[:, :, 2048:4096])
            return outputs
    
    def process_batch(self, codes_list: list) -> list:
        """Process a batch of codes with device-appropriate method."""
        with torch.inference_mode():
            if self.stream is not None:
                outputs = self._process_batch_cuda(codes_list)
                torch.cuda.synchronize()
                return outputs
            else:
                return self._process_batch_cpu(codes_list)


class SnacBatched:
    """Async batching wrapper for SNAC processing."""
    
    def __init__(self):
        self.processor = SnacProcessor()
        self.max_batch = settings.snac_max_batch
        self.batch_timeout_ms = settings.snac_batch_timeout_ms
        self._req_q: Optional[asyncio.Queue] = None
        self._worker_started = False
    
    async def _ensure_worker(self) -> None:
        """Ensure batch worker is running."""
        if self._req_q is None:
            self._req_q = asyncio.Queue()
        if not self._worker_started:
            asyncio.create_task(self._batch_worker())
            self._worker_started = True
    
    async def _batch_worker(self) -> None:
        """Main batching worker loop."""
        assert self._req_q is not None
        
        while True:
            # Collect batch
            first_item = await self._req_q.get()
            batch_items = [first_item]
            
            # Wait for more items with timeout
            try:
                await asyncio.sleep(max(0.0, self.batch_timeout_ms / 1000.0))
            except Exception:
                pass
            
            # Collect additional items up to max batch size
            while len(batch_items) < self.max_batch:
                try:
                    batch_items.append(self._req_q.get_nowait())
                except Exception:
                    break
            
            # Process batch
            codes_list = [item[0] for item in batch_items]
            futures = [item[1] for item in batch_items]
            
            try:
                outputs = self.processor.process_batch(codes_list)
                
                # Set results
                for future, output in zip(futures, outputs):
                    if not future.done():
                        future.set_result(output[0].detach().cpu().numpy())
                        
            except Exception as e:
                # Set exceptions
                for future in futures:
                    if not future.done():
                        future.set_exception(e)
    
    async def decode_codes(self, codes_triplet: list[torch.Tensor]) -> np.ndarray:
        """Decode codes asynchronously via batching."""
        await self._ensure_worker()
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        
        assert self._req_q is not None
        await self._req_q.put((codes_triplet, future))
        
        return await future


# Global instance
_BATCHER: Optional[SnacBatched] = None


def get_snac_batched() -> SnacBatched:
    """Get global SNAC batcher instance."""
    global _BATCHER
    if _BATCHER is None:
        _BATCHER = SnacBatched()
    return _BATCHER
