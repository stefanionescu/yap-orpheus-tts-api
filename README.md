# Yap Orpheus TTS API

Run Orpheus 3B TTS behind a FastAPI server with TensorRT-LLM (default) or vLLM. Optimized for A100 GPU instances (Runpod/AWS) using a plain Python virtualenv (no Docker).

- **Server**: `server/`
- **Scripts**: `scripts/`
- **Tests**: `tests/`

### Prerequisites

- NVIDIA GPU with CUDA 12.8 drivers (A100 recommended)
- Ubuntu-based image with `nvidia-smi`
- Python 3.10
- Hugging Face token (`HF_TOKEN`) with access to the model

### Quickstart

```bash
# 1) Set required token (deployment step)
export HF_TOKEN="hf_xxx"

# 2) Bootstrap → install → run (auto-installs TRT-LLM and auto-builds engine on first run)
bash scripts/run-all.sh

# 3) Health check
curl -s http://127.0.0.1:8000/healthz
```

### Environment

- Required: `HF_TOKEN`
- **Critical for correct audio**: `MODEL_LOCAL_DIR` and `TOKENIZER_DIR` must point to the **exact HF snapshot used during engine build**
- Optional knobs (aggressively optimized for TTFB in `scripts/env/`):
  - Text chunking: `FIRST_CHUNK_WORDS` (16), `NEXT_CHUNK_WORDS` (120), `MIN_TAIL_WORDS` (12)
  - Audio streaming: `MIN_TOKENS_FIRST` (7, 1 frame!), `MIN_TOKENS_SUBSEQ` (28), `TOKENS_EVERY` (7)  
  - SNAC: `SNAC_TORCH_COMPILE` (1), `SNAC_MAX_BATCH` (64), `SNAC_BATCH_TIMEOUT_MS` (2)
  - SNAC lane order: `ORPHEUS_LANE_ORDER` ("0|1,2|3,4,5,6") - **change this if you hear gunshot/static**
  - vLLM: see `server/vllm_config.py` and `scripts/env/vllm.sh`
  - TensorRT-LLM: `scripts/env/trtllm.sh` — A100 single-stream: `TRTLLM_MAX_BATCH` (8), `TRTLLM_KV_FRACTION` (0.90)
- Inspect current values:
```bash
bash scripts/print-env.sh
```

### Logging & Monitoring

The server provides comprehensive logging for debugging, monitoring, and performance analysis:

**View real-time logs:**
```bash
# Follow live server logs (default location)
tail -f logs/orpheus-tts.log

# View recent logs with timestamps
tail -100 logs/orpheus-tts.log

# Monitor logs with automatic refresh
watch -n 2 'tail -20 logs/orpheus-tts.log'
```

**Configure logging:**
```bash
# Set log level (DEBUG is default for detailed info, INFO for normal, WARNING for errors only)
export LOG_LEVEL="INFO"  # Override default DEBUG level if you want less verbose logging

# Custom log file location
export LOG_FILE="/path/to/custom/server.log"

# Log rotation settings
export LOG_MAX_BYTES=20971520    # 20MB before rotation
export LOG_BACKUP_COUNT=10       # Keep 10 backup files
```

**What gets logged:**
- Server startup/shutdown and component initialization
- WebSocket connections and client interactions
- TTS generation requests with parameters and performance metrics
- Audio processing pipeline (tokenization → SNAC → PCM streaming)
- Engine operations (TRT-LLM/vLLM generation, batching)
- All errors with full context and stack traces
- Resource usage and processing statistics

**Log analysis examples:**
```bash
# Find all errors
grep "ERROR" logs/orpheus-tts.log

# Monitor WebSocket connections
grep "WebSocket" logs/orpheus-tts.log

# Track TTS generation performance
grep "TTS generation completed" logs/orpheus-tts.log

# View SNAC batch processing stats
grep "SNAC batch" logs/orpheus-tts.log
```

### Stop and cleanup

```bash
# Stop server and remove .run/ and logs/
bash scripts/stop.sh

# Also remove venv and caches from install step
bash scripts/stop.sh --clean-install

# Also clean system apt caches from bootstrap step
bash scripts/stop.sh --clean-system

# Remove everything
bash scripts/stop.sh --clean-system --clean-install
```

### Tests (optional)

```bash
# Activate venv if not already active
source .venv/bin/activate

# Warmup (single WS stream)
python tests/warmup.py

# Benchmark (concurrent WS sessions)
python tests/bench.py
```