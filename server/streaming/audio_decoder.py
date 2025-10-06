"""Clean audio decoding logic extracted from streaming.py."""

import asyncio
import numpy as np
import torch

from server.audio.snac_decoder import get_snac_batched, SNAC_DEVICE
from server.config import settings


class AudioDecoder:
    """Handles conversion of audio token codes to PCM audio via SNAC."""
    
    def __init__(self):
        self.snac = get_snac_batched()
        self.frame_size = settings.frame_substreams
        self.code_size = settings.code_size
        self.code_offset = settings.code_offset
        self.sample_rate = settings.snac_sr
        self.max_samples = int(settings.tts_max_sec * self.sample_rate) if settings.tts_max_sec > 0 else 0
        
        # Calculate window parameters
        window_raw = settings.tts_decode_window
        window_adj = window_raw - (window_raw % self.frame_size)
        if window_adj < (self.frame_size * settings.min_window_frames):
            window_adj = self.frame_size * settings.min_window_frames
        self.window_tokens = max(window_adj, self.frame_size * settings.min_window_frames)
    
    async def decode_window(self, window_codes: list[int]) -> np.ndarray:
        """Decode a window of audio codes to PCM."""
        if not window_codes:
            return np.empty(0, dtype=np.int16)
        
        # Reshape codes into frame matrix
        arr = np.asarray(window_codes, dtype=np.int32).reshape(-1, self.frame_size)
        if arr.size == 0:
            return np.empty(0, dtype=np.int16)
        
        # Group channels according to SNAC layout
        grp0, grp1, grp2 = settings.snac_groups
        codes_0 = torch.from_numpy(arr[:, grp0].reshape(-1)).unsqueeze(0).to(SNAC_DEVICE)
        codes_1 = torch.from_numpy(arr[:, grp1].reshape(-1)).unsqueeze(0).to(SNAC_DEVICE)
        codes_2 = torch.from_numpy(arr[:, grp2].reshape(-1)).unsqueeze(0).to(SNAC_DEVICE)
        
        # Decode to audio
        audio = await self.snac.decode_codes([codes_0, codes_1, codes_2])  # [-1, 1]
        wav = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
        return wav.reshape(-1)


class TokenProcessor:
    """Processes streaming tokens and manages audio code buffer."""
    
    def __init__(self, decoder: AudioDecoder):
        self.decoder = decoder
        self.codes_buffer: list[int] = []
        self.audio_token_idx = 0
        self.frames_emitted = 0
        self.total_samples = 0
    
    def process_token(self, token_id: int) -> Optional[int]:
        """Process a single token and return audio code if valid."""
        channel = self.audio_token_idx % self.decoder.frame_size
        code = token_id - self.decoder.code_offset - (channel * self.decoder.code_size)
        
        if 0 <= code < self.decoder.code_size:
            self.codes_buffer.append(int(code))
            self.audio_token_idx += 1
            
            # Return frame count when frame is complete
            if (self.audio_token_idx % self.decoder.frame_size) == 0:
                return self.audio_token_idx // self.decoder.frame_size
        
        return None
    
    async def emit_window(self, frames_ready: int) -> Optional[bytes]:
        """Emit PCM audio for completed frames."""
        if len(self.codes_buffer) < self.decoder.window_tokens:
            return None
        if frames_ready <= self.frames_emitted:
            return None
        
        # Decode the most recent full window
        window_codes = self.codes_buffer[-self.decoder.window_tokens:]
        pcm = await self.decoder.decode_window(window_codes)
        
        if pcm.size == 0:
            self.frames_emitted = frames_ready
            return None
        
        # Apply sample limit if configured
        emit_pcm = pcm
        if self.decoder.max_samples:
            remaining = self.decoder.max_samples - self.total_samples
            if remaining <= 0:
                return None
            if pcm.size > remaining:
                emit_pcm = pcm[-remaining:]
        
        self.total_samples += emit_pcm.size
        self.frames_emitted = frames_ready
        
        await asyncio.sleep(settings.yield_sleep_seconds)
        return emit_pcm.tobytes()
    
    async def emit_final_window(self) -> Optional[bytes]:
        """Emit final window for remaining frames at end of stream."""
        frames_ready = self.audio_token_idx // self.decoder.frame_size
        if frames_ready <= self.frames_emitted:
            return None
        
        # Calculate final window size
        window_frames = min(frames_ready, self.decoder.window_tokens // self.decoder.frame_size)
        window_tokens = window_frames * self.decoder.frame_size
        
        if window_tokens <= len(self.codes_buffer):
            window_codes = self.codes_buffer[-window_tokens:]
            pcm = await self.decoder.decode_window(window_codes)
            
            if pcm.size > 0:
                emit_pcm = pcm
                if self.decoder.max_samples:
                    remaining = self.decoder.max_samples - self.total_samples
                    if remaining > 0 and pcm.size > remaining:
                        emit_pcm = pcm[-remaining:]
                
                if emit_pcm.size > 0:
                    self.total_samples += emit_pcm.size
                    self.frames_emitted = frames_ready
                    await asyncio.sleep(settings.yield_sleep_seconds)
                    return emit_pcm.tobytes()
        
        return None
