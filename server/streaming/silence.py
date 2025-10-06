"""Leading silence trimming utilities for streaming audio chunks."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SilenceTrimConfig:
    sample_rate: int
    enabled: bool
    rms_threshold: float
    activation_ms: float
    prepad_ms: float
    max_leading_sec: float


class SilenceTrimmer:
    """Drops leading silence based on amplitude/RMS heuristics.

    The trimmer inspects each PCM chunk (int16) until it detects sustained
    energy above `rms_threshold`. Once speech is detected, the remaining audio
    is passed through untouched.
    """

    def __init__(self, cfg: SilenceTrimConfig):
        self.cfg = cfg
        self._started = not cfg.enabled
        self._trimmed_samples = 0
        self._amplitude_threshold = int(cfg.rms_threshold * 32768.0)
        if self._amplitude_threshold <= 0:
            self._amplitude_threshold = 1
        self._activation_samples = max(1, int(cfg.sample_rate * (cfg.activation_ms / 1000.0)))
        self._max_leading_samples = (
            int(cfg.sample_rate * cfg.max_leading_sec) if cfg.max_leading_sec > 0 else None
        )
        self._prepad_samples = max(0, int(cfg.sample_rate * (cfg.prepad_ms / 1000.0)))
        self._prepad_bytes = self._prepad_samples * 2
        self._buffer = bytearray()

    def push(self, audio_bytes: bytes) -> bytes:
        """Process a PCM chunk and return trimmed bytes if leading silence remains."""
        if not audio_bytes:
            return b""
        if self._started:
            return audio_bytes

        frame = np.frombuffer(audio_bytes, dtype=np.int16)
        if frame.size == 0:
            return b""

        float_frame = frame.astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(np.square(float_frame))))

        activation_index = self._detect_activation(frame)

        if activation_index is None and rms < self.cfg.rms_threshold:
            self._trimmed_samples += frame.size
            self._append_silence(audio_bytes)
            if self._should_force_start():
                self._started = True
                logger.debug(
                    "Silence trim forced after %.3fs", self._trimmed_samples / self.cfg.sample_rate
                )
                return self._emit_with_prepad(audio_bytes)
            return b""

        if activation_index is None:
            activation_index = 0

        if activation_index > 0:
            pre_bytes = frame[:activation_index].tobytes()
            self._append_silence(pre_bytes)
            self._trimmed_samples += activation_index

        trimmed = frame[activation_index:].tobytes()
        self._started = True

        if self._trimmed_samples > 0:
            trimmed_sec = self._trimmed_samples / self.cfg.sample_rate
            logger.debug("Trimmed %.3fs of leading silence", trimmed_sec)

        return self._emit_with_prepad(trimmed)

    def flush(self) -> bytes:
        """Ensure downstream consumers know we won't emit more audio."""
        self._started = True
        return b""

    def _detect_activation(self, frame: np.ndarray) -> int | None:
        mask = np.abs(frame) >= self._amplitude_threshold
        if not mask.any():
            return None

        if self._activation_samples <= 1 or mask.size <= self._activation_samples:
            return int(np.argmax(mask))

        window = np.ones(self._activation_samples, dtype=np.int32)
        energy = np.convolve(mask.astype(np.int32), window, mode="valid")
        indices = np.where(energy >= self._activation_samples)[0]
        if indices.size == 0:
            return None

        return int(indices[0])

    def _should_force_start(self) -> bool:
        if self._max_leading_samples is None:
            return False
        return self._trimmed_samples >= self._max_leading_samples

    def _append_silence(self, data: bytes) -> None:
        if self._prepad_bytes == 0 or not data:
            return
        self._buffer.extend(data)
        if len(self._buffer) > self._prepad_bytes:
            del self._buffer[:-self._prepad_bytes]

    def _emit_with_prepad(self, active_bytes: bytes) -> bytes:
        if self._prepad_bytes and self._buffer:
            prepad = bytes(self._buffer[-self._prepad_bytes:])
        else:
            prepad = b""
        self._buffer.clear()
        return prepad + active_bytes
