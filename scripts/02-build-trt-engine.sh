#!/usr/bin/env bash
set -euo pipefail

# Common helpers and env
source "scripts/lib/common.sh"
load_env_if_present

# Defaults
: "${VENV_DIR:=$PWD/.venv}"
: "${MODEL_ID:=canopylabs/orpheus-3b-0.1-ft}"
: "${TRTLLM_ENGINE_DIR:=$PWD/models/orpheus-trt}"
: "${TRTLLM_DTYPE:=bfloat16}"
: "${TRTLLM_MAX_INPUT_LEN:=128}"
: "${TRTLLM_MAX_OUTPUT_LEN:=2048}"
: "${TRTLLM_MAX_BATCH_SIZE:=16}"
: "${TRTLLM_KV_CACHE_DTYPE:=int8}"
: "${TRTLLM_QUANTIZED_DIR:=}"
: "${TRTLLM_WEIGHT_QUANT:=none}"  # none|w8a8|auto8
: "${TRTLLM_AUTO_QUANTIZE_BITS:=8.0}"
: "${PYTHON_EXEC:=python}"

usage() {
  cat <<USAGE
Usage: $0 [--model ID] [--output DIR] [--dtype float16|bfloat16] [--max-input-len N] [--max-output-len N] [--max-batch-size N] [--kv-cache-dtype fp8|int8] [--quantized-dir DIR] [--minimal] [--cli]

Build a TensorRT-LLM engine directory for Orpheus.

Options:
  --minimal        Build tiny config (128/128/1) to validate toolchain
  --cli            Use trtllm-build CLI instead of Python builder

Defaults:
  --model           ${MODEL_ID}
  --output          ${TRTLLM_ENGINE_DIR}
  --dtype           ${TRTLLM_DTYPE}
  --max-input-len   ${TRTLLM_MAX_INPUT_LEN}
  --max-output-len  ${TRTLLM_MAX_OUTPUT_LEN}
  --max-batch-size  ${TRTLLM_MAX_BATCH_SIZE}
  --kv-cache-dtype  ${TRTLLM_KV_CACHE_DTYPE:-<unset>}
  --quantized-dir   ${TRTLLM_QUANTIZED_DIR:-<unset>}
USAGE
}

ARGS=()
MINIMAL=false
USE_CLI=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL_ID="$2"; shift 2 ;;
    --output) TRTLLM_ENGINE_DIR="$2"; shift 2 ;;
    --dtype) TRTLLM_DTYPE="$2"; shift 2 ;;
    --max-input-len) TRTLLM_MAX_INPUT_LEN="$2"; shift 2 ;;
    --max-output-len) TRTLLM_MAX_OUTPUT_LEN="$2"; shift 2 ;;
    --max-batch-size) TRTLLM_MAX_BATCH_SIZE="$2"; shift 2 ;;
    --kv-cache-dtype) TRTLLM_KV_CACHE_DTYPE="$2"; shift 2 ;;
    --quantized-dir) TRTLLM_QUANTIZED_DIR="$2"; shift 2 ;;
    --minimal) MINIMAL=true; shift ;;
    --cli) USE_CLI=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
set -- "${ARGS[@]:-}"

[ -d "${VENV_DIR}" ] || { echo "[build-trt] venv missing. Run scripts/01-install.sh first." >&2; exit 1; }
source "${VENV_DIR}/bin/activate"

if [ "$(uname -s)" != "Linux" ]; then
  echo "[build-trt] ERROR: TensorRT-LLM builds require Linux with NVIDIA GPUs." >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[build-trt] ERROR: nvidia-smi not detected. Ensure the GPU is visible to this runtime." >&2
  exit 1
fi

# Source modular env snippets (may override defaults)
source_env_dir "scripts/env"

if [ -n "${HF_TOKEN:-}" ]; then
  export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-$HF_TOKEN}"
  export HF_HUB_TOKEN="${HF_HUB_TOKEN:-$HF_TOKEN}"
fi

MODE="python"
if [ "$USE_CLI" = true ]; then
  MODE="cli"
fi

echo "[build-trt] Model: ${MODEL_ID}"
echo "[build-trt] Output: ${TRTLLM_ENGINE_DIR}"
echo "[build-trt] Mode: ${MODE}${MINIMAL:+ (minimal)}"

mkdir -p "${TRTLLM_ENGINE_DIR}"

if [ "$USE_CLI" = true ]; then
  echo "[build-trt] Pre-downloading weights for ${MODEL_ID}..."
  "${PYTHON_EXEC}" - <<'PY'
import os
from huggingface_hub import snapshot_download
repo_id = os.environ["MODEL_ID"]
ckpt_dir = os.path.join(os.getcwd(), ".hf", repo_id.replace("/", "-"))
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
path = snapshot_download(
    repo_id=repo_id,
    local_dir=ckpt_dir,
    local_dir_use_symlinks=False,
    allow_patterns=[
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "*.safetensors",
        "model.safetensors*",
    ],
    resume_download=True,
    token=os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN"),
)
print(f"[build-trt] Weights cached at: {path}")
PY

  CKPT_DIR="${PWD}/.hf/${MODEL_ID//\//-}"
  OUT_DIR="${TRTLLM_ENGINE_DIR}"

  # Optional quantize-and-export for KV cache
  if [ -n "${TRTLLM_KV_CACHE_DTYPE}" ]; then
    echo "[build-trt] Exporting quantized checkpoint (kv_cache_dtype=${TRTLLM_KV_CACHE_DTYPE})..."
    QUANT_DIR="${TRTLLM_QUANTIZED_DIR:-${OUT_DIR}/quantized-checkpoint}"
    export CKPT_DIR QUANT_DIR TRTLLM_KV_CACHE_DTYPE
    mkdir -p "${QUANT_DIR}"
    "${PYTHON_EXEC}" - <<'PY'
import os
from pathlib import Path
from tensorrt_llm.quantization import quantize_and_export
import inspect
from transformers import AutoTokenizer
src = os.environ["CKPT_DIR"]
dst = os.environ["QUANT_DIR"]
kv = os.environ["TRTLLM_KV_CACHE_DTYPE"].lower()
Path(dst).mkdir(parents=True, exist_ok=True)
if any(Path(dst).iterdir()):
    print(f"[build-trt] Using existing quantized checkpoint at: {dst}")
else:
    sig = inspect.signature(quantize_and_export)
    params = sig.parameters
    if "kv_cache_dtype" not in params:
        print("[build-trt] WARN: quantize_and_export does not support kv_cache_dtype on this version; skipping export.")
    else:
        # Some versions require full calibration kwargs. If we detect those parameters, skip.
        required_kwonly = {
            "device",
            "calib_dataset",
            "dtype",
            "qformat",
            "calib_size",
            "batch_size",
            "calib_max_seq_length",
            "awq_block_size",
            "tp_size",
            "pp_size",
            "cp_size",
            "seed",
            "tokenizer_max_seq_length",
        }
        needs_calib = any(name in params for name in required_kwonly)
        if needs_calib:
            # Build a tiny synthetic calibration dataset
            try:
                tokenizer = AutoTokenizer.from_pretrained(src)
            except Exception:
                tokenizer = AutoTokenizer.from_pretrained(os.environ.get("MODEL_ID", "canopylabs/orpheus-3b-0.1-ft"))
            texts = [
                "Hello, this is a short sentence for calibration.",
                "The quick brown fox jumps over the lazy dog.",
                "Orpheus TTS uses KV cache during decoding.",
                "Please quantize the KV cache to reduce memory.",
            ]
            desired_batch = int(os.environ.get("TRTLLM_MAX_BATCH_SIZE", "16"))
            desired_calib_size = int(os.environ.get("TRTLLM_CALIB_SIZE", "$((TRTLLM_MAX_BATCH_SIZE*4))"))
            if len(texts) < desired_calib_size:
                reps = (desired_calib_size + len(texts) - 1) // len(texts)
                texts = (texts * reps)[:desired_calib_size]

            tokenizer_limit = getattr(tokenizer, "model_max_length", 2048) or 2048
            seq_hint = int(os.environ.get("TRTLLM_MAX_INPUT_LEN", "128")) + int(os.environ.get("TRTLLM_MAX_OUTPUT_LEN", "2048"))
            max_len = min(seq_hint, tokenizer_limit)
            enc = tokenizer(texts, padding=False, truncation=True, max_length=max_len, return_tensors=None)
            class _Calib:
                def __init__(self, e): self._ids = e["input_ids"] if isinstance(e, dict) else e
                def __len__(self): return len(self._ids)
                def __iter__(self):
                    for ids in self._ids:
                        yield {"input_ids": ids}
            calib = _Calib(enc)
            try:
                from tensorrt_llm.quantization import QuantMode
                qformat = getattr(QuantMode, "PTQ", "ptq")
            except Exception:
                qformat = "ptq"
            print("[build-trt] Performing PTQ export for KV-cache aligned to build config...")
            quantize_and_export(
                model=src if "model" in params else None,
                model_dir=src if "model_dir" in params else None,
                checkpoint_dir=src if "checkpoint_dir" in params else None,
                export_dir=dst if "export_dir" in params else None,
                output_dir=dst if "output_dir" in params else None,
                kv_cache_dtype=kv,
                device="cuda",
                calib_dataset=calib,
                dtype=os.environ.get("TRTLLM_DTYPE", "bfloat16"),
                qformat=qformat,
                calib_size=len(calib),
                batch_size=max(1, min(desired_batch, len(calib))),
                calib_max_seq_length=int(max_len),
                awq_block_size=128,
                tp_size=int(os.environ.get("TP_SIZE", "1")),
                pp_size=int(os.environ.get("PP_SIZE", "1")),
                cp_size=int(os.environ.get("CP_SIZE", "1")),
                seed=17,
                tokenizer_max_seq_length=int(max_len),
            )
        else:
            print(f"[build-trt] quantize_and_export(model={src}, export_dir={dst}, kv_cache_dtype={kv})")
            quantize_and_export(model=src, export_dir=dst, kv_cache_dtype=kv)
print(f"[build-trt] Quantized checkpoint ready: {dst}")
PY
    CKPT_DIR="${QUANT_DIR}"
  fi

  CONTEXT_FMHA_ARGS=""
  if [ "${TRTLLM_KV_CACHE_DTYPE}" = "fp8" ]; then
    CONTEXT_FMHA_ARGS=" --use_fp8_context_fmha enable"
  fi

  if [ "$MINIMAL" = true ]; then
    echo "[build-trt] Running minimal CLI build..."
    trtllm-build \
      --checkpoint_dir "$CKPT_DIR" \
      --output_dir "$OUT_DIR" \
      --max_input_len 128 \
      --max_seq_len 256 \
      --max_batch_size 1 \
      --remove_input_padding enable \
      --log_level info${CONTEXT_FMHA_ARGS}
  else
    echo "[build-trt] Running full CLI build..."
    trtllm-build \
      --checkpoint_dir "$CKPT_DIR" \
      --output_dir "$OUT_DIR" \
      --max_input_len "${TRTLLM_MAX_INPUT_LEN}" \
      --max_seq_len "$((TRTLLM_MAX_INPUT_LEN + TRTLLM_MAX_OUTPUT_LEN))" \
      --max_batch_size "${TRTLLM_MAX_BATCH_SIZE}" \
      --remove_input_padding enable \
      --log_level info${CONTEXT_FMHA_ARGS}
  fi
else
  CMD=(
    "${PYTHON_EXEC}" server/build/build-trt-engine.py
    --model "${MODEL_ID}"
    --output "${TRTLLM_ENGINE_DIR}"
    --dtype "${TRTLLM_DTYPE}"
    --max_input_len "${TRTLLM_MAX_INPUT_LEN}"
    --max_output_len "${TRTLLM_MAX_OUTPUT_LEN}"
    --max_batch_size "${TRTLLM_MAX_BATCH_SIZE}"
  )
  if [ -n "${TRTLLM_KV_CACHE_DTYPE}" ]; then CMD+=(--kv_cache_dtype "${TRTLLM_KV_CACHE_DTYPE}"); fi
  if [ -n "${TRTLLM_QUANTIZED_DIR}" ]; then CMD+=(--quantized_dir "${TRTLLM_QUANTIZED_DIR}"); fi
  if [ -n "${TRTLLM_WEIGHT_QUANT}" ]; then CMD+=(--weight_quant "${TRTLLM_WEIGHT_QUANT}"); fi
  if [ "${TRTLLM_WEIGHT_QUANT}" = "auto8" ]; then CMD+=(--auto_quantize_bits "${TRTLLM_AUTO_QUANTIZE_BITS}"); fi
  if [ "$MINIMAL" = true ]; then CMD+=(--minimal); fi
  echo "[build-trt] Running: ${CMD[*]}"
  "${CMD[@]}"
fi

echo "[build-trt] Engine directory ready: ${TRTLLM_ENGINE_DIR}"
echo "[build-trt] To use it: export BACKEND=trtllm; export TRTLLM_ENGINE_DIR=\"${TRTLLM_ENGINE_DIR}\"; bash scripts/03-run-server.sh"
