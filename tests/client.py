#!/usr/bin/env python3
"""
Orpheus TTS WebSocket client.

- Connects to a remote Orpheus TTS server (RunPod or local)
- Sends a single JSON payload with full text (Baseten Mode A) and receives streaming PCM chunks
- Aggregates all audio and saves a WAV file under ROOT/audio/
- Tracks metrics similar to other test files (TTFB, connect, handshake)
- Supports env vars: RUNPOD_TCP_HOST, RUNPOD_TCP_PORT, RUNPOD_API_KEY
"""


from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import wave
from pathlib import Path
from typing import List, Optional

import websockets
from websockets.exceptions import ConnectionClosed
from dotenv import load_dotenv

# Ensure repository root is on sys.path so `server` package is importable
_THIS_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _THIS_DIR.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from server.text.chunking import chunk_by_sentences


ROOT_DIR = Path(__file__).resolve().parent.parent
AUDIO_DIR = ROOT_DIR / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# Load environment variables from .env file
load_dotenv(ROOT_DIR / ".env")

DEFAULT_TEXT = (
    "Oh my god Danny, you're so smart and handsome! You're gonna love talking to me "
    "once Stefan is done with the app. Can't wait to see you there sweetie!"
)


def _looks_like_runpod_proxy(host: str) -> bool:
    h = (host or "").lower()
    return ("proxy.runpod.net" in h) or h.endswith("runpod.net")


def _ws_url(server: str) -> str:
    """Build WebSocket URL for the Orpheus TTS streaming endpoint."""
    if server.startswith(("ws://", "wss://")):
        base = server.rstrip("/")
    else:
        # Auto-detect TLS for RunPod proxy hosts
        use_tls = _looks_like_runpod_proxy(server)
        scheme = "wss" if use_tls else "ws"
        base = f"{scheme}://{server.strip().rstrip('/')}"
    return f"{base}/ws/tts"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Orpheus TTS WebSocket client")
    ap.add_argument(
        "--server",
        default=os.getenv("YAP_TTS_SERVER", ""),
        help="Full server URL or host:port (overrides --host/--port)",
    )
    ap.add_argument(
        "--host",
        default=os.getenv("RUNPOD_TCP_HOST", "127.0.0.1"),
        help="RunPod public host (defaults to RUNPOD_TCP_HOST or 127.0.0.1)",
    )
    ap.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("RUNPOD_TCP_PORT", "8000")),
        help="RunPod public port (defaults to RUNPOD_TCP_PORT or 8000)",
    )
    ap.add_argument(
        "--secure",
        action="store_true",
        help="Use wss:// (TLS); auto-enabled for RunPod proxy hosts",
    )
    ap.add_argument(
        "--voice",
        default=os.getenv("TTS_VOICE", "female"),
        help="Voice alias: female|male",
    )
    ap.add_argument(
        "--text",
        action="append",
        default=None,
        help="Text to synthesize (repeat flag for multiple sentences)",
    )
    ap.add_argument(
        "--api-key",
        default=os.getenv("YAP_API_KEY", "YAP_API_KEY"),
        help="API key for server Authorization (Bearer)",
    )
    ap.add_argument(
        "--outfile",
        default=None,
        help="Output WAV filename (default: tts_<timestamp>.wav under ROOT/audio)",
    )
    return ap.parse_args()


def _compose_server_from_host_port(host: str, port: int, secure: bool) -> str:
    host = (host or "").strip().strip("/")
    if not host:
        host = "127.0.0.1"
    # Auto-secure for runpod proxy
    use_tls = secure or _looks_like_runpod_proxy(host)
    scheme = "wss" if use_tls else "ws"
    # If host includes :port already, don't duplicate
    netloc = host if (":" in host) else f"{host}:{port}"
    return f"{scheme}://{netloc}"


async def tts_client(
    server: str,
    voice: str,
    texts: List[str],
    api_key: Optional[str],
    out_path: Path,
) -> dict:
    url = _ws_url(server)
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    ws_options = {
        "extra_headers": headers if headers else None,
        "max_size": None,
        "ping_interval": 30,
        "ping_timeout": 30,
        "open_timeout": 30,
        "close_timeout": 0.5,
    }

    # Metrics
    t0_e2e = time.perf_counter()
    time_to_first_audio_e2e: Optional[float] = None
    time_to_first_audio_server: Optional[float] = None
    final_time: Optional[float] = None

    pcm_chunks: List[bytes] = []
    sample_rate = 24000

    # Sticky WS: open once, send all sentences, then send __END__ sentinel
    full_text = " ".join(texts).strip()
    sentences = [s for s in chunk_by_sentences(full_text) if s and s.strip()]
    connect_ms = 0.0
    connect_start = time.perf_counter()
    async with websockets.connect(url, **ws_options) as ws:
        connect_ms += (time.perf_counter() - connect_start) * 1000.0

        # Send meta first (voice only)
        meta = {"voice": voice}
        await ws.send(json.dumps(meta))

        # Start server TTFB timer when we send the first text
        t0_server = None

        # Task to receive and buffer audio until server closes
        async def _recv_loop():
            nonlocal time_to_first_audio_e2e, time_to_first_audio_server
            while True:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=30.0)
                except asyncio.TimeoutError:
                    print("Timeout: No data received for 30 seconds, ending...")
                    break
                except ConnectionClosed:
                    break
                if isinstance(msg, (bytes, bytearray)):
                    if time_to_first_audio_e2e is None:
                        time_to_first_audio_e2e = time.perf_counter() - t0_e2e
                    if (t0_server is not None) and (time_to_first_audio_server is None):
                        time_to_first_audio_server = time.perf_counter() - t0_server
                    pcm_chunks.append(msg)
                elif isinstance(msg, str):
                    # Optional textual messages
                    pass

        recv_task = asyncio.create_task(_recv_loop())

        for idx, sentence in enumerate(sentences):
            payload = {"text": sentence.strip()}
            await ws.send(json.dumps(payload))
            if t0_server is None:
                t0_server = time.perf_counter()

        # Signal end of stream
        await ws.send("__END__")
        await recv_task

    final_time = time.perf_counter()

    wall_s = final_time - t0_e2e if final_time else time.perf_counter() - t0_e2e

    if pcm_chunks:
        # Concatenate all PCM chunks and write WAV
        pcm_data = b''.join(pcm_chunks)
        
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(out_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_data)
        
        audio_s = len(pcm_data) / 2.0 / float(sample_rate)  # 16-bit = 2 bytes per sample
    else:
        audio_s = 0.0

    metrics = {
        "server": server,
        "voice": voice,
        "outfile": str(out_path),
        "wall_s": float(wall_s),
        "audio_s": float(audio_s),
        "ttfb_e2e_s": float(time_to_first_audio_e2e or 0.0),
        "ttfb_server_s": float(time_to_first_audio_server or 0.0),
        "connect_ms": float(connect_ms),
        "rtf": float(wall_s / audio_s) if audio_s > 0 else float("inf"),
        "xrt": float(audio_s / wall_s) if wall_s > 0 else 0.0,
    }

    return metrics


def main() -> None:
    args = parse_args()
    texts = [t for t in (args.text or [DEFAULT_TEXT]) if t and t.strip()]

    # Build server string from args/env
    server_str = (args.server or "").strip()
    if not server_str:
        host = args.host
        port = args.port
        server_str = _compose_server_from_host_port(host, port, args.secure)

    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = Path(args.outfile) if args.outfile else (AUDIO_DIR / f"tts_{ts}.wav")

    print(f"Server: {server_str}")
    print(f"Voice:  {args.voice}")
    print(f"Out:    {out}")
    print(f"Text(s): {len(texts)}")
    # Buffer size is ignored in Mode A; kept for CLI compatibility

    res = asyncio.run(
        tts_client(
            server_str,
            args.voice,
            texts,
            args.api_key,
            out,
        )
    )
    
    print("\n== Result ==")
    print(f"Saved: {res['outfile']}")
    print(f"TTFB (e2e): {res['ttfb_e2e_s']:.3f}s")
    print(f"TTFB (srv): {res['ttfb_server_s']:.3f}s")
    print(f"Wall:  {res['wall_s']:.3f}s")
    print(f"Audio: {res['audio_s']:.3f}s")
    print(f"RTF: {res['rtf']:.3f}")
    print(f"xRT: {res['xrt']:.3f}")
    print(f"Connect: {res['connect_ms']:.1f}ms")


if __name__ == "__main__":
    main()
