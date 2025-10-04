#!/usr/bin/env python3
"""
Single WebSocket streaming warmup for Orpheus TTS server.
Sends text over WS and measures bytes of streamed PCM.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

import asyncio
import json
import websockets
from websockets.exceptions import ConnectionClosed

# Ensure repository root is on sys.path so `server` package is importable
_THIS_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _THIS_DIR.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from server.core.chunking import chunk_by_sentences


DEFAULT_TEXT = (
    "Oh my god Danny, you're so smart and handsome! You're gonna love talking to me "
    "once Stefan is done with the app. Can't wait to see you there sweetie!"
)


def _ws_url(server: str) -> str:
    base = server.strip().rstrip("/")
    if base.startswith("http://"):
        base = "ws://" + base[len("http://"):]
    elif base.startswith("https://"):
        base = "wss://" + base[len("https://"):]
    elif not base.startswith(("ws://", "wss://")):
        base = "ws://" + base
    return f"{base}/ws/tts"


def main() -> None:
    ap = argparse.ArgumentParser(description="WebSocket streaming warmup (Orpheus TTS)")
    ap.add_argument("--server", default="127.0.0.1:8000", help="host:port or http[s]://host:port")
    ap.add_argument("--voice", default=os.environ.get("TTS_VOICE", "female"), help="Voice alias: female|male|tara|zac")
    ap.add_argument("--text", default=DEFAULT_TEXT, help="Text to synthesize")
    ap.add_argument("--seed", type=int, default=None, help="Optional seed override")
    ap.add_argument("--num-predict", type=int, default=None, help="Optional num_predict override")
    args = ap.parse_args()

    url = _ws_url(args.server)
    t0_e2e = time.perf_counter()
    first_chunk_at: Optional[float] = None
    total_bytes = 0
    sr = 24000

    IDLE_TIMEOUT_S = 15.0  # grace period with no data before we stop

    async def run():
        nonlocal first_chunk_at, total_bytes
        sentences = [s for s in chunk_by_sentences(str(args.text)) if s and s.strip()]
        for sentence in sentences:
            async with websockets.connect(url, max_size=None) as ws:
                payload = {"voice": args.voice, "text": sentence.strip()}
                if args.num_predict is not None:
                    payload["max_tokens"] = args.num_predict
                await ws.send(json.dumps(payload))

                # Receive PCM until server closes or idle timeout
                last_bytes_at = None
                while True:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=IDLE_TIMEOUT_S)
                    except asyncio.TimeoutError:
                        # No data for a while -> assume done
                        print("Timeout: No data received for 15 seconds, ending...")
                        break
                    except ConnectionClosed:
                        print("Connection closed by server")
                        break
                    
                    # Print everything we receive from the server
                    if isinstance(msg, (bytes, bytearray)):
                        print(f"Received binary data: {len(msg)} bytes")
                        now = time.perf_counter()
                        if first_chunk_at is None:
                            first_chunk_at = now
                        last_bytes_at = now
                        total_bytes += len(msg)
                    elif isinstance(msg, str):
                        print(f"Received text message: {msg}")
                    else:
                        print(f"Received unknown message type {type(msg)}: {msg}")

    asyncio.run(run())

    wall_s = time.perf_counter() - t0_e2e
    ttfb_e2e_s = (first_chunk_at - t0_e2e) if first_chunk_at else 0.0
    ttfb_server_s = ttfb_e2e_s  # WS single channel, so same measurement
    audio_s = (total_bytes / 2.0) / float(sr) if total_bytes > 0 else 0.0
    rtf = (wall_s / audio_s) if audio_s > 0 else float("inf")
    xrt = (audio_s / wall_s) if wall_s > 0 else 0.0

    print("Warmup TTS request")
    print(f"Server: {args.server}")
    print(f"Voice: {args.voice}")
    print(f"Text: '{args.text}'")
    print("\n== Results ==")
    print(f"TTFB (e2e): {ttfb_e2e_s:.3f}s")
    print(f"TTFB (srv): {ttfb_server_s:.3f}s")
    print(f"Wall time: {wall_s:.3f}s")
    print(f"Audio duration: {audio_s:.3f}s")
    print(f"RTF: {rtf:.3f}")
    print(f"xRT: {xrt:.3f}")


if __name__ == "__main__":
    main()


