import os
import numpy as np
import torch
from typing import List
from snac import SNAC

_SNAC_SINGLETON = None  # module-level cache

class SnacDecoder:
    """Incremental SNAC decode; emits only new PCM16 since last call. Matches Baseten approach."""
    def __init__(self, device: str = None, sample_rate: int = 24000):
        # Keep on CUDA by default for performance, but allow override via env
        default_dev = os.environ.get("SNAC_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
        self.device = device or default_dev
        self.sample_rate = sample_rate
        self.dtype_decoder = torch.float32
        global _SNAC_SINGLETON
        if _SNAC_SINGLETON is None:
            model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(self.device)
            # Match Baseten: ensure decoder is in float32
            model.decoder = model.decoder.to(self.dtype_decoder)
            
            # Optional torch.compile for performance (like Baseten)
            use_compile = bool(int(os.getenv("SNAC_TORCH_COMPILE", "0")))
            if use_compile:
                print("[snac] Compiling decoder and quantizer with torch.compile...")
                model.decoder = torch.compile(model.decoder, dynamic=True)
                model.quantizer = torch.compile(model.quantizer, dynamic=True)
                print("[snac] torch.compile complete")
                
            _SNAC_SINGLETON = model
        self.snac = _SNAC_SINGLETON
        self.codes: List[int] = []
        self._decoded_samples = 0

    def add_frames(self, frames_7: List[List[int]]) -> None:
        for f in frames_7:
            self.codes.extend(f)

    def _decode_all_f32(self) -> np.ndarray:
        if len(self.codes) < 7:
            return np.zeros(0, dtype=np.float32)
        
        # Reshape codes into frames (n_frames, 7)
        n_frames = len(self.codes) // 7
        if n_frames == 0:
            return np.zeros(0, dtype=np.float32)
            
        # Take only complete 7-code frames
        frame_codes = self.codes[:n_frames * 7]
        ids = torch.tensor(frame_codes, dtype=torch.int32, device=self.device).reshape(n_frames, 7)
        
        # Baseten's exact code mapping
        codes_0 = ids[:, 0].unsqueeze(0)  # shape: (1, n_frames)
        codes_1 = ids[:, [1, 4]].reshape(-1).unsqueeze(0)  # shape: (1, n_frames*2)
        codes_2 = ids[:, [2, 3, 5, 6]].reshape(-1).unsqueeze(0)  # shape: (1, n_frames*4)
        
        codes = [codes_0, codes_1, codes_2]
        
        with torch.inference_mode():
            # Use quantizer.from_codes + decoder like Baseten
            z_q = self.snac.quantizer.from_codes(codes)
            # CRITICAL: Apply the same audio slicing as Baseten [:, :, 2048:4096]
            audio_hat = self.snac.decoder(z_q.to(self.dtype_decoder))[:, :, 2048:4096]
            
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
