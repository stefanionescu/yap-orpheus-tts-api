### Orpheus TTS Fast Image

Lean, production-focused image. Pulls a pre-quantized Orpheus checkpoint or prebuilt engines from Hugging Face (or mounts a local engine), validates, optionally builds the engine with `trtllm-build`, then runs the API server.

---

#### What's Included

- CUDA 12.1 runtime + Python 3.10 venv
- PyTorch with matching CUDA wheels
- TensorRT-LLM runtime wheel (no repo clone)
- App code (`/app/server/`) and tests (`/app/tests/`, minus `client.py`)
- Runtime scripts: `start-server.sh` and `environment.sh`

Not included: engines and models (fetched/mounted at runtime).

---

#### Build

```bash
cd /path/to/yap-orpheus-tts-api
bash docker/fast/build.sh

# Optional: push
bash docker/fast/build.sh --push
```

---

#### Run (Recommended: Pull from Hugging Face)

Pull engines or a quantized checkpoint from a HF repo created with our tooling. If engines match your GPU/driver, the container skips build; if only checkpoints exist, it builds the engine locally.

```bash
docker run --gpus all --rm -p 8000:8000 \
  -e HF_TOKEN=$HF_TOKEN \
  -e YAP_API_KEY=$YAP_API_KEY \
  -e HF_DEPLOY_REPO_ID=your-org/orpheus-trtllm \
  -e HF_DEPLOY_ENGINE_LABEL=sm80_trt-llm-1.0.0_cuda12.4 \
  IMAGE:TAG
```

Alternate: mount a prebuilt engine directory (no HF pull):

```bash
docker run --gpus all --rm -p 8000:8000 \
  -e HF_TOKEN=$HF_TOKEN \
  -e YAP_API_KEY=$YAP_API_KEY \
  -e TRTLLM_ENGINE_DIR=/opt/engines/orpheus-trt-int4-awq \
  -v /path/to/engine:/opt/engines/orpheus-trt-int4-awq:ro \
  IMAGE:TAG
```

---

#### Environment Variables (Fast Image)

- `HF_TOKEN` (required): HF token for repo access
- `YAP_API_KEY` (optional): API auth for server
- `HF_DEPLOY_REPO_ID` (optional): HF repo to pull from (engines/checkpoints layout)
- `HF_DEPLOY_ENGINE_LABEL` (optional): engines/<label> selector for prebuilt engines
- `HF_DEPLOY_USE` (default: `auto`): `auto|engines|checkpoints`
- `TRTLLM_ENGINE_DIR` (optional): engine dir if mounting locally
- `MODEL_ID` (optional): tokenizer source; used if not provided by HF repo
- `MODELS_DIR` (default: `/opt/models`)
- `ENGINES_DIR` (default: `/opt/engines`)
- `HOST` (default: `0.0.0.0`), `PORT` (default: `8000`)

Advanced knobs (used by server/runtime):
- `TRTLLM_MAX_INPUT_LEN`, `TRTLLM_MAX_OUTPUT_LEN`, `TRTLLM_MAX_BATCH_SIZE`
- `KV_FREE_GPU_FRAC`, `KV_ENABLE_BLOCK_REUSE`
- `ORPHEUS_MAX_TOKENS`, `DEFAULT_TEMPERATURE`, `DEFAULT_TOP_P`, `DEFAULT_REPETITION_PENALTY`
- `SNAC_SR`, `SNAC_MAX_BATCH`, `SNAC_BATCH_TIMEOUT_MS`, `TTS_DECODE_WINDOW`, `TTS_MAX_SEC`
- `WS_END_SENTINEL`, `WS_CLOSE_BUSY_CODE`, `WS_CLOSE_INTERNAL_CODE`, `WS_QUEUE_MAXSIZE`, `DEFAULT_VOICE`

---

#### Testing Inside the Container

```bash
# Warmup
docker exec -it <container> python /app/tests/warmup.py --server 127.0.0.1:8000 --voice female

# Benchmark
docker exec -it <container> python /app/tests/bench.py --n 10 --concurrency 4
```
