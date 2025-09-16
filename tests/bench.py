#!/usr/bin/env python3
"""
HTTP streaming benchmark for Orpheus TTS FastAPI server.

Sends concurrent POST /tts requests with stream=true and consumes PCM16 bytes.
Reports wall, audio duration, TTFB (e2e and post-headers), RTF, xRT, throughput.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import statistics as stats
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import httpx


ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "tests" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


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


async def _tts_one(
    client: httpx.AsyncClient,
    server: str,
    text: str,
    voice: str,
    *,
    seed: Optional[int] = None,
    num_predict: Optional[int] = None,
) -> Dict[str, float]:
    url = _http_url(server)
    payload = {
        "text": text,
        "voice": voice,
        "stream": True,
    }
    if seed is not None:
        payload["seed"] = int(seed)
    if num_predict is not None:
        payload["num_predict"] = int(num_predict)

    t0_e2e = time.perf_counter()
    # stream context yields after response headers are received
    async with client.stream("POST", url, json=payload) as resp:
        resp.raise_for_status()
        t0_server = time.perf_counter()
        sr = 24000
        with contextlib.suppress(Exception):
            sr = int(resp.headers.get("X-Audio-Sample-Rate", "24000"))

        first_chunk_at: Optional[float] = None
        total_bytes = 0
        async for chunk in resp.aiter_bytes():
            if not chunk:
                continue
            if first_chunk_at is None:
                first_chunk_at = time.perf_counter()
            total_bytes += len(chunk)

    wall_s = time.perf_counter() - t0_e2e
    ttfb_e2e_s = (first_chunk_at - t0_e2e) if first_chunk_at else 0.0
    ttfb_server_s = (first_chunk_at - t0_server) if first_chunk_at else 0.0

    # PCM16 mono @ sr Hz → 2 bytes/sample
    audio_s = (total_bytes / 2.0) / float(sr) if total_bytes > 0 else 0.0
    rtf = (wall_s / audio_s) if audio_s > 0 else float("inf")
    xrt = (audio_s / wall_s) if wall_s > 0 else 0.0

    return {
        "wall_s": float(wall_s),
        "audio_s": float(audio_s),
        "ttfb_e2e_s": float(ttfb_e2e_s),
        "ttfb_server_s": float(ttfb_server_s),
        "rtf": float(rtf),
        "xrt": float(xrt),
        "throughput_min_per_min": float(xrt),
    }


def _summarize(title: str, results: List[Dict[str, float]]) -> None:
    if not results:
        print(f"{title}: no results")
        return

    def p(v: List[float], q: float) -> float:
        if not v:
            return 0.0
        k = max(0, min(len(v) - 1, int(round(q * (len(v) - 1)))))
        return sorted(v)[k]

    wall = [r.get("wall_s", 0.0) for r in results]
    audio = [r.get("audio_s", 0.0) for r in results]
    rtf = [r.get("rtf", 0.0) for r in results]
    xrt = [r.get("xrt", 0.0) for r in results]
    ttfb_e2e = [r.get("ttfb_e2e_s", 0.0) for r in results if r.get("ttfb_e2e_s", 0.0) > 0]
    ttfb_srv = [r.get("ttfb_server_s", 0.0) for r in results if r.get("ttfb_server_s", 0.0) > 0]

    print(f"\n== {title} ==")
    print(f"n={len(results)}")
    print(f"Wall s      | avg={stats.mean(wall):.4f}  p50={stats.median(wall):.4f}  p95={p(wall,0.95):.4f}")
    if ttfb_e2e:
        print(f"TTFB (e2e)  | avg={stats.mean(ttfb_e2e):.4f}  p50={stats.median(ttfb_e2e):.4f}  p95={p(ttfb_e2e,0.95):.4f}")
    if ttfb_srv:
        print(f"TTFB (srv)  | avg={stats.mean(ttfb_srv):.4f}  p50={stats.median(ttfb_srv):.4f}  p95={p(ttfb_srv,0.95):.4f}")
    print(f"Audio s     | avg={stats.mean(audio):.4f}")
    print(f"RTF         | avg={stats.mean(rtf):.4f}  p50={stats.median(rtf):.4f}  p95={p(rtf,0.95):.4f}")
    print(f"xRT         | avg={stats.mean(xrt):.4f}")
    print(f"Throughput  | avg={stats.mean([r.get('throughput_min_per_min',0.0) for r in results]):.2f} min/min")


def _load_texts(inline_texts: Optional[List[str]]) -> List[str]:
    if inline_texts:
        return [t for t in inline_texts if t and t.strip()]
    return [DEFAULT_TEXT]


async def bench_http(
    server: str,
    total_reqs: int,
    concurrency: int,
    voice: str,
    texts: List[str],
    seed: Optional[int],
    num_predict: Optional[int],
) -> Tuple[List[Dict[str, float]], int]:
    sem = asyncio.Semaphore(max(1, concurrency))
    results: List[Dict[str, float]] = []
    errors_total = 0

    limits = httpx.Limits(max_keepalive_connections=concurrency, max_connections=max(10, concurrency * 2))
    timeout = httpx.Timeout(connect=30.0, read=1200.0, write=30.0, pool=30.0)
    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        async def worker(req_idx: int):
            nonlocal errors_total
            text = texts[req_idx % len(texts)]
            async with sem:
                try:
                    r = await _tts_one(client, server, text, voice, seed=seed, num_predict=num_predict)
                    results.append(r)
                except Exception as e:
                    errors_total += 1
                    err_path = RESULTS_DIR / "bench_errors.txt"
                    with contextlib.suppress(Exception):
                        with open(err_path, "a", encoding="utf-8") as ef:
                            ef.write(f"{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} idx={req_idx} err={e}\n")

        tasks = [asyncio.create_task(worker(i)) for i in range(total_reqs)]
        await asyncio.gather(*tasks, return_exceptions=True)

    return results[:total_reqs], errors_total


def main() -> None:
    ap = argparse.ArgumentParser(description="HTTP streaming benchmark (Orpheus TTS)")
    ap.add_argument("--server", default="127.0.0.1:8000", help="host:port or http[s]://host:port")
    ap.add_argument("--n", type=int, default=10, help="Total requests")
    ap.add_argument("--concurrency", type=int, default=10, help="Max concurrent sessions")
    ap.add_argument("--voice", type=str, default=os.environ.get("TTS_VOICE", "female"), help="Voice alias: female|male|tara|zac")
    ap.add_argument("--text", action="append", default=None, help="Inline text prompt (repeat for multiple)")
    ap.add_argument("--seed", type=int, default=None, help="Optional seed override")
    ap.add_argument("--num-predict", type=int, default=None, help="Optional num_predict override")
    args = ap.parse_args()

    texts = _load_texts(args.text)

    print(f"Benchmark → HTTP stream | n={args.n} | concurrency={args.concurrency} | server={args.server}")
    print(f"Voice: {args.voice}")
    print(f"Texts: {len(texts)}")

    t0 = time.time()
    results, errors = asyncio.run(bench_http(args.server, args.n, args.concurrency, args.voice, texts, args.seed, args.num_predict))
    elapsed = time.time() - t0

    _summarize("TTS Streaming", results)
    print(f"Errors: {errors}")
    print(f"Total elapsed: {elapsed:.4f}s")
    if results:
        total_audio = sum(r.get("audio_s", 0.0) for r in results)
        print(f"Total audio synthesized: {total_audio:.2f}s")
        print(f"Overall throughput: {total_audio/elapsed:.2f} min/min")

    # per-session JSONL
    try:
        metrics_path = RESULTS_DIR / "bench_metrics.jsonl"
        with open(metrics_path, "w", encoding="utf-8") as f:
            for rec in results:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Saved per-session metrics to {metrics_path}")
    except Exception as e:
        print(f"Warning: could not write metrics JSONL: {e}")


if __name__ == "__main__":
    main()


