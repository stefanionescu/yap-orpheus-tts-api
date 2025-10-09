#!/usr/bin/env bash
set -euo pipefail

source "custom/lib/common.sh"
load_env_if_present
load_environment
source "custom/build/helpers.sh"

echo "[step:remote] Checking remote deploy configuration..."

SKIP_QUANTIZATION="${SKIP_QUANTIZATION:-false}"

if [ -z "${HF_DEPLOY_REPO_ID:-}" ]; then
    echo "[step:remote] HF_DEPLOY_REPO_ID not set → no remote deploy"
    exit 0
fi

export HF_HUB_ENABLE_HF_TRANSFER=1

echo "[step:remote] Querying Hugging Face repo: ${HF_DEPLOY_REPO_ID}"

py_out=$(python - <<'PY'
import os
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

repo_id=os.environ.get('HF_DEPLOY_REPO_ID')
use=os.environ.get('HF_DEPLOY_USE','auto').strip().lower()
engine_label=os.environ.get('HF_DEPLOY_ENGINE_LABEL','').strip()
workdir=os.environ.get('HF_DEPLOY_WORKDIR','') or str(Path.cwd()/ 'models' / '_hf_download')
gpu_sm=os.environ.get('GPU_SM_ARCH','').strip()

api=HfApi()
try:
    files=api.list_repo_files(repo_id=repo_id, repo_type='model')
except Exception as exc:
    print(f"MODE=error MSG={type(exc).__name__}:{exc}")
    raise SystemExit(0)

engine_labels=set()
for f in files:
    if f.startswith('trt-llm/engines/'):
        parts=f.split('/')
        if len(parts)>=4:
            engine_labels.add(parts[3])
has_ckpt=any(f.startswith('trt-llm/checkpoints/') for f in files)

selected=''
if use in ('engines','auto') and engine_labels:
    if engine_label and engine_label in engine_labels:
        selected=engine_label
    elif len(engine_labels)==1:
        selected=next(iter(engine_labels))
    elif gpu_sm:
        matches=[lab for lab in sorted(engine_labels) if lab.startswith(gpu_sm)]
        if len(matches)==1:
            selected=matches[0]

if selected:
    path=snapshot_download(repo_id=repo_id, local_dir=workdir, local_dir_use_symlinks=False,
                           allow_patterns=[f"trt-llm/engines/{selected}/**", "trt-llm/engines/**/build_metadata.json"])
    eng_dir=str(Path(path)/'trt-llm'/'engines'/selected)
    print(f"MODE=engines")
    print(f"ENGINE_DIR={eng_dir}")
    print(f"ENGINE_LABEL={selected}")
    raise SystemExit(0)

if use in ('checkpoints','auto') and has_ckpt:
    path=snapshot_download(repo_id=repo_id, local_dir=workdir, local_dir_use_symlinks=False,
                           allow_patterns=["trt-llm/checkpoints/**"]) 
    ckpt_dir=str(Path(path)/'trt-llm'/'checkpoints')
    print(f"MODE=checkpoints")
    print(f"CHECKPOINT_DIR={ckpt_dir}")
    raise SystemExit(0)

print("MODE=none")
PY
)

mode=$(echo "$py_out" | awk -F= '/^MODE=/{print $2; exit}' | tr -d '\r' | tr -d '\n')
case "$mode" in
  engines)
    eng_dir=$(echo "$py_out" | awk -F= '/^ENGINE_DIR=/{print $2; exit}')
    _validate_downloaded_engine "$eng_dir"
    if _engine_env_compatible "$eng_dir"; then
        export TRTLLM_ENGINE_DIR="$eng_dir"
        _record_engine_dir_env "$TRTLLM_ENGINE_DIR"
        echo "[step:remote] Using prebuilt engine: $TRTLLM_ENGINE_DIR"
        echo "REMOTE_RESULT=10" > .run/remote_result.env
    else
        echo "[step:remote] Downloaded engines incompatible; will try checkpoints if available"
        # fallthrough to possibly checkpoints info below
        :
    fi
    ;;
  checkpoints)
    ckpt_dir=$(echo "$py_out" | awk -F= '/^CHECKPOINT_DIR=/{print $2; exit}')
    if [ -n "$ckpt_dir" ]; then
        _validate_downloaded_checkpoint "$ckpt_dir"
        export CHECKPOINT_DIR="$ckpt_dir"
        echo "[step:remote] Using downloaded checkpoint: $CHECKPOINT_DIR"
        echo "REMOTE_RESULT=11" > .run/remote_result.env
    fi
    ;;
  error)
    echo "[step:remote] WARNING: HF repo query failed; continuing with local path" >&2
    ;;
  none|*)
    echo "[step:remote] No usable remote artifacts; continuing with local quantization/build"
    ;;
esac

echo "[step:remote] OK"


