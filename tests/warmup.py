#!/usr/bin/env python3
"""
Single HTTP streaming warmup request for Orpheus TTS FastAPI server.
Does not save WAV; only measures timings and bytes to infer audio duration.
"""
from __future__ import annotations

import argparse
import os
import time
from typing import Optional

import httpx


DEFAULT_TEXT = (
    "Oh my god Danny, you're so smart and handsome! You're gonna love talking to me "
    "once Stefan is done with the app. Can't wait to see you there sweetie!"
)


def _http_url(server: str) -> str:
    if server.startswith(("http://", "https://")):
        base = server.rstrip("/")
    else:
        base = f"http://{server.strip().rstrip('/')}"
    return f"{base}/tts"


def main() -> None:
    ap = argparse.ArgumentParser(description="HTTP streaming warmup (Orpheus TTS)")
    ap.add_argument("--server", default="127.0.0.1:8000", help="host:port or http[s]://host:port")
    ap.add_argument("--voice", default=os.environ.get("TTS_VOICE", "female"), help="Voice alias: female|male|tara|zac")
    ap.add_argument("--text", default=DEFAULT_TEXT, help="Text to synthesize")
    ap.add_argument("--seed", type=int, default=None, help="Optional seed override")
    ap.add_argument("--num-predict", type=int, default=None, help="Optional num_predict override")
    args = ap.parse_args()

    url = _http_url(args.server)
    payload = {
        "text": args.text,
        "voice": args.voice,
        "stream": True,
    }
    if args.seed is not None:
        payload["seed"] = int(args.seed)
    if args.num_predict is not None:
        payload["num_predict"] = int(args.num_predict)

    timeout = httpx.Timeout(connect=30.0, read=1200.0, write=30.0, pool=30.0)
    with httpx.Client(timeout=timeout) as client:
        t0_e2e = time.perf_counter()
        with client.stream("POST", url, json=payload) as resp:
            resp.raise_for_status()
            t0_server = time.perf_counter()
            sr = 24000
            try:
                sr = int(resp.headers.get("X-Audio-Sample-Rate", "24000"))
            except Exception:
                pass
            first_chunk_at: Optional[float] = None
            total_bytes = 0
            for chunk in resp.iter_bytes():
                if not chunk:
                    continue
                if first_chunk_at is None:
                    first_chunk_at = time.perf_counter()
                total_bytes += len(chunk)

        wall_s = time.perf_counter() - t0_e2e
        ttfb_e2e_s = (first_chunk_at - t0_e2e) if first_chunk_at else 0.0
        ttfb_server_s = (first_chunk_at - t0_server) if first_chunk_at else 0.0
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


