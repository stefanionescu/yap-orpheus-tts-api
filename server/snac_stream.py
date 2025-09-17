import os
import numpy as np
import torch
from typing import List
from snac import SNAC

_SNAC_SINGLETON = None  # module-level cache

class SnacDecoder:
    """Incremental SNAC decode; emits only new PCM16 since last call."""
    def __init__(self, device: str = None, sample_rate: int = 24000):
        # Keep on CUDA by default for performance, but allow override via env
        default_dev = os.environ.get("SNAC_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
        self.device = device or default_dev
        self.sample_rate = sample_rate
        global _SNAC_SINGLETON
        if _SNAC_SINGLETON is None:
            _SNAC_SINGLETON = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(self.device)
        self.snac = _SNAC_SINGLETON
        self.codes: List[int] = []
        self._decoded_samples = 0

    def add_frames(self, frames_7: List[List[int]]) -> None:
        for f in frames_7:
            self.codes.extend(f)

    def _decode_all_f32(self) -> np.ndarray:
        if len(self.codes) < 7:
            return np.zeros(0, dtype=np.float32)
        ids = torch.tensor(self.codes, dtype=torch.int32, device=self.device).reshape(-1, 7)
        c0 = ids[:, 0].unsqueeze(0)
        c1 = torch.stack((ids[:, 1], ids[:, 4])).T.flatten().unsqueeze(0)
        c2 = torch.stack((ids[:, 2], ids[:, 3], ids[:, 5], ids[:, 6])).T.flatten().unsqueeze(0)
        with torch.inference_mode():
            audio_hat = self.snac.decode([c0, c1, c2])[0]
        audio = audio_hat.squeeze().detach().float().cpu().numpy()
        return np.clip(audio, -1.0, 1.0)

    def take_new_pcm16(self) -> bytes:
        audio = self._decode_all_f32()
        if audio.size == 0:
            return b""
        total = audio.shape[-1]
        if total <= self._decoded_samples:
            return b""
        new = audio[self._decoded_samples:]
        self._decoded_samples = total
        return (new * 32767.0).astype(np.int16).tobytes()
