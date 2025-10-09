#!/usr/bin/env bash
# =============================================================================
# Build Helpers for TensorRT-LLM Engine
# =============================================================================
# Sourced by step scripts and orchestrator. Do NOT `set -euo` here.

# Assumes the following are already sourced by orchestrator:
#   source "custom/lib/common.sh"
#   load_env_if_present
#   load_environment

_setup_huggingface_auth() {
    if [ -n "${HF_TOKEN:-}" ]; then
        export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-$HF_TOKEN}"
        export HF_HUB_TOKEN="${HF_HUB_TOKEN:-$HF_TOKEN}"
    else
        echo "ERROR: HF_TOKEN not set" >&2
        return 1
    fi
}

_should_skip_build() {
    if [ "${FORCE_REBUILD:-false}" = true ]; then
        return 1  # Force rebuild
    fi
    if [ -f "${ENGINE_OUTPUT_DIR}/rank0.engine" ] && [ -f "${ENGINE_OUTPUT_DIR}/config.json" ]; then
        return 0  # Skip build
    fi
    return 1  # Need to build
}

_detect_gpu_sm_arch() {
    # Echoes an smXX string if detectable, otherwise empty
    local sm_env="${GPU_SM_ARCH:-}"
    if [ -n "$sm_env" ]; then
        echo "$sm_env"
        return 0
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
        local cap
        cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '\r' | tr -d '\n')
        if [ -n "$cap" ]; then
            local major minor
            major=$(echo "$cap" | cut -d. -f1)
            minor=$(echo "$cap" | cut -d. -f2)
            echo "sm${major}${minor}"
            return 0
        fi
    fi
    echo ""
}

_validate_downloaded_checkpoint() {
    local ckpt_dir="$1"
    echo "[build] Validating downloaded checkpoint at $ckpt_dir"
    test -f "$ckpt_dir/config.json" || { echo "[build] ERROR: Missing config.json in $ckpt_dir" >&2; return 1; }
    ls "$ckpt_dir"/rank*.safetensors >/dev/null 2>&1 || { echo "[build] ERROR: No rank*.safetensors found in $ckpt_dir" >&2; return 1; }
    echo "[build] ✓ Checkpoint validation passed"
}

_validate_downloaded_engine() {
    local eng_dir="$1"
    echo "[build] Validating downloaded engine at $eng_dir"
    test -f "$eng_dir/rank0.engine" || { echo "[build] ERROR: Missing rank0.engine in $eng_dir" >&2; return 1; }
    test -f "$eng_dir/config.json" || { echo "[build] ERROR: Missing config.json in $eng_dir" >&2; return 1; }
    local eng_size
    eng_size=$(stat -f%z "$eng_dir/rank0.engine" 2>/dev/null || stat -c%s "$eng_dir/rank0.engine" 2>/dev/null || echo "0")
    if [ "$eng_size" -lt 1000000 ]; then
        echo "WARNING: Engine file seems unusually small ($eng_size bytes)" >&2
    fi
    echo "[build] ✓ Engine files present"
}

_engine_env_compatible() {
    # Returns 0 if compatible or non-strict; 1 if strict mismatch
    local eng_dir="$1"
    local strict="${HF_DEPLOY_STRICT_ENV_MATCH:-1}"
    if [ "$strict" != "1" ]; then
        return 0
    fi
    local want_sm
    want_sm=$(_detect_gpu_sm_arch)
    if [ -z "$want_sm" ]; then
        return 0
    fi
    local meta_file="$eng_dir/build_metadata.json"
    if [ -f "$meta_file" ]; then
        local sm_in_meta
        sm_in_meta=$(python - <<'PY'
import json,sys
p=sys.argv[1]
try:
    with open(p,'r') as f:
        d=json.load(f)
    print((d.get('sm_arch') or '').strip())
except Exception:
    print('')
PY
 "$meta_file" | tr -d '\r' | tr -d '\n')
        if [ -n "$sm_in_meta" ] && [ "$sm_in_meta" != "$want_sm" ]; then
            echo "[build] Engine SM mismatch: repo=$sm_in_meta local=$want_sm" >&2
            return 1
        fi
    else
        local base
        base=$(basename "$eng_dir")
        case "$base" in
            ${want_sm}_*) : ;;
            *) echo "[build] Engine label likely incompatible with local SM ($want_sm): $base" >&2; return 1 ;;
        esac
    fi
    return 0
}

_record_engine_dir_env() {
    local eng_dir="$1"
    mkdir -p .run
    echo "export TRTLLM_ENGINE_DIR=\"$eng_dir\"" > .run/engine_dir.env
}

_validate_engine() {
    echo "[build] Validating built engine..."
    local required_files=("rank0.engine" "config.json")
    for file in "${required_files[@]}"; do
        if [ ! -f "$ENGINE_OUTPUT_DIR/$file" ]; then
            echo "ERROR: Required file missing: $ENGINE_OUTPUT_DIR/$file" >&2
            return 1
        fi
    done
    local engine_size
    engine_size=$(stat -f%z "$ENGINE_OUTPUT_DIR/rank0.engine" 2>/dev/null || stat -c%s "$ENGINE_OUTPUT_DIR/rank0.engine" 2>/dev/null || echo "0")
    if [ "$engine_size" -lt 1000000 ]; then
        echo "WARNING: Engine file seems unusually small ($engine_size bytes)" >&2
    fi
    echo "[build] ✓ Engine validation passed"
}


