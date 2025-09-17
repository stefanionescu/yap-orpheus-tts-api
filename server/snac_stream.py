import re
import numpy as np
import torch
from typing import List, Optional
from snac import SNAC

# 7-code frame layout; each code in [0..4095]
POS_OFFSETS = [0, 4096, 8192, 12288, 16384, 20480, 24576]
AUDIO_TOKEN_RE = re.compile(r"<custom_token_(\d+)>")


class StreamNormalizer:
    """Takes raw <custom_token_N> integers and yields aligned 7-code frames."""
    def __init__(self):
        self.pos = 0
        self.buf: List[int] = []
        self.dropped = 0

    def _fits(self, val: int, k: int) -> bool:
        lo = 10 + POS_OFFSETS[k]
        return lo <= val < lo + 4096

    def push_number(self, t: int) -> Optional[List[int]]:
        if self._fits(t, self.pos):
            self.buf.append(t - 10 - POS_OFFSETS[self.pos])
            self.pos += 1
            if self.pos == 7:
                frame = self.buf
                self.buf = []
                self.pos = 0
                return frame
            return None
        # try resync at frame start
        if self._fits(t, 0):
            if self.buf:
                self.dropped += len(self.buf)
            self.buf = [t - 10 - POS_OFFSETS[0]]
            self.pos = 1
        else:
            self.dropped += 1
        return None

    def close(self) -> int:
        if self.buf:
            self.dropped += len(self.buf)
            self.buf = []
            self.pos = 0
        return self.dropped


class SnacDecoder:
    """Incremental SNAC decode; emits only new PCM16 since last call."""
    def __init__(self, device: str = None, sample_rate: int = 24000):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sample_rate = sample_rate
        self.snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(self.device)
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


def extract_token_numbers(token_text: str) -> List[int]:
    """Extract any <custom_token_####> numbers from a *single-token* string."""
    return [int(m.group(1)) for m in AUDIO_TOKEN_RE.finditer(token_text)]


