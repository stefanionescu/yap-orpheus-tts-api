"""Synthesis pipeline logic extracted from server.py for clean separation."""

import asyncio
from typing import Optional

from tensorrt_llm import SamplingParams

from server.text.chunking import chunk_by_sentences
from server.prompts import resolve_voice
from server.tts_streaming import aiter_pcm_from_custom_tokens
from server.config import settings


class SynthesisPipeline:
    """Handles text-to-speech synthesis with pipelining for multi-sentence text."""
    
    def __init__(self, engine):
        self.engine = engine
    
    async def synthesize_text(self, text: str, voice: str, sampling_kwargs: dict, ws) -> None:
        """Synthesize text and stream PCM audio to WebSocket."""
        # Build sampling params
        sp = SamplingParams(**sampling_kwargs)
        
        # Resolve voice and chunk text
        resolved_voice = resolve_voice(voice) or settings.default_voice
        chunks = chunk_by_sentences(text)
        
        if not chunks:
            return
        
        if len(chunks) == 1:
            # Single chunk: stream directly
            await self._stream_single_chunk(chunks[0], resolved_voice, sp, ws)
        else:
            # Multiple chunks: use pipelining
            await self._stream_pipelined_chunks(chunks, resolved_voice, sp, ws)
    
    async def _stream_single_chunk(self, text: str, voice: str, sp: SamplingParams, ws) -> None:
        """Stream a single text chunk directly to WebSocket."""
        async for pcm in aiter_pcm_from_custom_tokens(self.engine.engine, text, voice, sp):
            await ws.send_bytes(pcm)
            await asyncio.sleep(settings.yield_sleep_seconds)
    
    async def _stream_pipelined_chunks(self, chunks: list[str], voice: str, sp: SamplingParams, ws) -> None:
        """Stream multiple chunks with pipelining for reduced latency."""
        # Start with chunk 0 streaming directly and chunk 1 pre-generating
        queued_q = asyncio.Queue(maxsize=settings.ws_queue_maxsize)
        queued_task = asyncio.create_task(self._produce_to_queue(chunks[1], voice, sp, queued_q))
        
        # Stream chunk 0 directly
        async for pcm in aiter_pcm_from_custom_tokens(self.engine.engine, chunks[0], voice, sp):
            await ws.send_bytes(pcm)
            await asyncio.sleep(settings.yield_sleep_seconds)
        
        # Process remaining chunks with pipelining
        for idx in range(1, len(chunks)):
            # Start next chunk generation before draining current
            next_q = None
            next_task = None
            if (idx + 1) < len(chunks):
                next_q = asyncio.Queue(maxsize=settings.ws_queue_maxsize)
                next_task = asyncio.create_task(self._produce_to_queue(chunks[idx + 1], voice, sp, next_q))
            
            # Drain current queued chunk
            await self._drain_queue_to_websocket(queued_q, ws)
            
            # Move to next
            queued_q = next_q if next_q is not None else asyncio.Queue()
            queued_task = next_task
    
    async def _produce_to_queue(self, text: str, voice: str, sp: SamplingParams, queue: asyncio.Queue) -> None:
        """Generate PCM for text and put results in queue."""
        async for pcm in aiter_pcm_from_custom_tokens(self.engine.engine, text, voice, sp):
            await queue.put(pcm)
        await queue.put(None)  # End sentinel
    
    async def _drain_queue_to_websocket(self, queue: asyncio.Queue, ws) -> None:
        """Drain all PCM data from queue to WebSocket."""
        while True:
            pcm = await queue.get()
            if pcm is None:
                break
            await ws.send_bytes(pcm)
            await asyncio.sleep(settings.yield_sleep_seconds)
