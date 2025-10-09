#!/usr/bin/env python
import argparse
import json
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_folder
import subprocess
import platform
from datetime import datetime, timezone

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # noqa: N816


def detect_engine_label(engine_dir: Path, default_label: str | None) -> str:
    # Prefer explicit label
    if default_label:
        return default_label
    # Try to infer from build metadata
    meta = engine_dir / "build_metadata.json"
    if meta.is_file():
        try:
            data = json.loads(meta.read_text())
            sm = data.get("sm_arch") or "smxx"
            trtllm = data.get("tensorrt_llm_version") or ""
            cuda = (data.get("cuda_toolkit") or "").replace(".", "")
            # Example: sm80_trt-llm-1.0.0_cuda12.4
            if trtllm and cuda:
                return f"{sm}_trt-llm-{trtllm}_cuda{data.get('cuda_toolkit')}"
            if sm:
                return sm
        except Exception:
            pass
    # Fallback
    return "engine"


def build_staging_tree(
    repo_root: Path,
    tokenizer_src: Path | None,
    checkpoint_src: Path | None,
    engine_src: Path | None,
    engine_label: str,
):
    # Root layout
    (repo_root / "trt-llm").mkdir(parents=True, exist_ok=True)
    (repo_root / "trt-llm" / "checkpoints").mkdir(parents=True, exist_ok=True)
    (repo_root / "trt-llm" / "engines" / engine_label).mkdir(parents=True, exist_ok=True)

    # Tokenizer files (best-effort)
    if tokenizer_src and tokenizer_src.is_dir():
        for name in ["tokenizer.json", "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json"]:
            src = tokenizer_src / name
            if src.exists():
                dst = repo_root / name
                dst.write_bytes(src.read_bytes())

    # Checkpoints
    if checkpoint_src and checkpoint_src.is_dir():
        dst_ckpt = repo_root / "trt-llm" / "checkpoints"
        for p in checkpoint_src.rglob("*"):
            if p.is_file():
                rel = p.relative_to(checkpoint_src)
                dst = dst_ckpt / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(p.read_bytes())

    # Engines
    if engine_src and engine_src.is_dir():
        dst_eng = repo_root / "trt-llm" / "engines" / engine_label
        for p in engine_src.rglob("*"):
            if p.is_file():
                rel = p.relative_to(engine_src)
                dst = dst_eng / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(p.read_bytes())

    # .gitattributes for LFS
    (repo_root / ".gitattributes").write_text(
        """*.engine filter=lfs diff=lfs merge=lfs -text
*.plan   filter=lfs diff=lfs merge=lfs -text
*.safetensors filter=lfs diff=lfs merge=lfs -text
*.bin    filter=lfs diff=lfs merge=lfs -text
"""
    )


def read_text_safe(path: Path) -> str:
    try:
        return path.read_text()
    except Exception:
        return ""


def collect_env_metadata(engine_dir: Path) -> dict:
    meta_path = engine_dir / "build_metadata.json"
    data: dict = {}
    if meta_path.is_file():
        try:
            data = json.loads(meta_path.read_text())
        except Exception:
            data = {}
    # Add runtime info
    data.setdefault("platform", platform.platform())
    build_image = os.environ.get("BUILD_IMAGE")
    if build_image:
        data.setdefault("build_image", build_image)
    trtllm_repo = os.environ.get("TRTLLM_REPO_URL")
    if trtllm_repo:
        data.setdefault("tensorrt_llm_repo", trtllm_repo)
    if torch is not None:
        data.setdefault("torch_version", getattr(torch, "__version__", ""))
        try:
            data.setdefault("torch_cuda", getattr(torch.version, "cuda", ""))
        except Exception:
            pass
    try:
        smi = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,compute_cap",
                "--format=csv,noheader",
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        line = (smi.stdout or "").splitlines()[0].strip() if smi.stdout else ""
        if line:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                data.setdefault("gpu_name", parts[0])
                data.setdefault("nvidia_driver", parts[1])
                data.setdefault("compute_capability", parts[2])
    except Exception:
        pass
    return data


def write_readme(repo_root: Path, engine_label: str, meta: dict, what: str, repo_id: str):
    """Write README.md into the staging folder. Prefer a rich template if present."""
    # Attempt to load rich template from repository
    try:
        repo_src_root = Path(__file__).resolve().parents[2]
    except Exception:
        repo_src_root = Path.cwd()

    template_path = repo_src_root / "server" / "hf" / "orpheus-readme.md"

    def _safe_get(dct: dict, *keys, default=None):
        cur = dct or {}
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur.get(k)
        return cur if cur is not None else default

    def _source_model_from_env_or_meta() -> str:
        env_model = os.environ.get("MODEL_ID") or ""
        if env_model:
            return env_model
        m = _safe_get(meta, "model_id", default="") or _safe_get(meta, "base_model", default="")
        return str(m) if m else "canopylabs/orpheus-3b-0.1-ft"

    def _to_link(model_id: str) -> str:
        model_id = (model_id or "").strip()
        if not model_id:
            return "Orpheus 3B"
        return f"[{model_id}](https://huggingface.co/{model_id})"

    def _render(template: str, mapping: dict) -> str:
        rendered = template
        for k, v in mapping.items():
            rendered = rendered.replace("{{" + k + "}}", str(v))
        return rendered

    # Build mapping for template
    base_model = _source_model_from_env_or_meta()
    awq_block_size = (
        os.environ.get("AWQ_BLOCK_SIZE")
        or str(_safe_get(meta, "quantization", "awq_block_size", default="") or "")
        or "128"
    )
    calib_size = (
        os.environ.get("CALIB_SIZE")
        or str(_safe_get(meta, "quantization", "calib_size", default="") or "")
        or "256"
    )
    dtype = os.environ.get("TRTLLM_DTYPE") or str(meta.get("dtype") or "float16")
    max_input_len = os.environ.get("TRTLLM_MAX_INPUT_LEN") or str(meta.get("max_input_len") or meta.get("max_input_len_tokens") or "48")
    max_output_len = os.environ.get("TRTLLM_MAX_OUTPUT_LEN") or str(meta.get("max_output_len") or meta.get("max_output_len_tokens") or "1024")
    max_batch_size = os.environ.get("TRTLLM_MAX_BATCH_SIZE") or str(meta.get("max_batch_size") or "16")
    trtllm_ver = meta.get("tensorrt_llm_version") or meta.get("tensorrt_version") or ""

    # Estimated sizes for Orpheus 3B
    original_size_gb = str(_safe_get(meta, "original_size_gb", default=6.0))
    quantized_size_gb = str(_safe_get(meta, "quantized_size_gb", default=1.6))

    quant_summary = {
        "quantization": {
            "weights_precision": "int4_awq",
            "kv_cache_dtype": _safe_get(meta, "quantization", "kv_cache", default="int8") or "int8",
            "awq_block_size": int(awq_block_size) if str(awq_block_size).isdigit() else awq_block_size,
            "calib_size": int(calib_size) if str(calib_size).isdigit() else calib_size,
        },
        "build": {
            "dtype": dtype,
            "max_input_len": int(max_input_len) if str(max_input_len).isdigit() else max_input_len,
            "max_output_len": int(max_output_len) if str(max_output_len).isdigit() else max_output_len,
            "max_batch_size": int(max_batch_size) if str(max_batch_size).isdigit() else max_batch_size,
            "engine_label": engine_label,
            "tensorrt_llm_version": trtllm_ver,
        },
        "environment": {
            "sm_arch": meta.get("sm_arch", ""),
            "gpu_name": meta.get("gpu_name", ""),
            "cuda_toolkit": meta.get("cuda_toolkit", ""),
            "nvidia_driver": meta.get("nvidia_driver", ""),
        },
    }

    mapping = {
        "license": "apache-2.0",
        "base_model": base_model,
        "model_name": "Orpheus 3B",
        "source_model_link": _to_link(base_model),
        "w_bit": "4",
        "q_group_size": awq_block_size,
        "awq_version": trtllm_ver,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "original_size_gb": original_size_gb,
        "quantized_size_gb": quantized_size_gb,
        "quant_summary": json.dumps(quant_summary, ensure_ascii=False, indent=2),
        "calib_section": (
            f"- Method: Activation-aware weight quantization (AWQ)\n"
            f"- Calibration size: {calib_size}\n"
            f"- AWQ block/group size: {awq_block_size}\n"
            f"- DType for build: {dtype}\n"
        ),
        "repo_name": repo_id,
        "engine_label": engine_label,
    }

    if template_path.is_file():
        try:
            template_text = template_path.read_text()
            rendered = _render(template_text, mapping)
            (repo_root / "README.md").write_text(rendered)
            return
        except Exception:
            pass

    # Fallback: generic README if template missing or failed to render
    title = "TRT-LLM Artifacts"
    env_lines = []
    for k in [
        "tensorrt_llm_version",
        "tensorrt_version",
        "cuda_toolkit",
        "sm_arch",
        "gpu_name",
        "nvidia_driver",
        "platform",
        "build_image",
        "torch_version",
        "torch_cuda",
    ]:
        v = meta.get(k)
        if v:
            env_lines.append(f"- {k.replace('_', ' ').title()}: {v}")

    build_knobs = []
    for k in [
        "dtype",
        "max_batch_size",
        "max_input_len",
        "max_output_len",
    ]:
        v = meta.get(k)
        if v is not None and v != "":
            build_knobs.append(f"- {k.replace('_', ' ').title()}: {v}")
    q = meta.get("quantization", {})
    if q:
        build_knobs.append(
            f"- Quantization: weights={q.get('weights')}, kv_cache={q.get('kv_cache')}, awq_block_size={q.get('awq_block_size')}, calib_size={q.get('calib_size')}"
        )

    build_cmd = meta.get("build_command", "")

    lines = []
    lines.append(f"# {title}\n")
    lines.append(
        "This repo contains TensorRT-LLM artifacts for Orpheus 3B (TTS). Engines are hardware/driver specific; checkpoints are portable."
    )
    lines.append("")
    lines.append("## Contents")
    lines.append("```")
    lines.append("trt-llm/")
    if what in ("engines", "both"):
        lines.append(f"  engines/{engine_label}/")
        lines.append("    rank*.engine")
        lines.append("    build_command.sh")
        lines.append("    build_metadata.json")
    if what in ("checkpoints", "both"):
        lines.append("  checkpoints/")
        lines.append("    rank*.safetensors")
        lines.append("    config.json")
    lines.append("```")

    lines.append("")
    lines.append("## Environment & Build Info")
    if env_lines:
        lines.extend(env_lines)
    if build_knobs:
        lines.append("")
        lines.append("### Build knobs")
        lines.extend(build_knobs)
    if build_cmd:
        lines.append("")
        lines.append("### Build command")
        lines.append("```")
        lines.append(build_cmd)
        lines.append("```")

    lines.append("")
    lines.append("## Portability Notes")
    lines.append(
        "- Engines (.engine/.plan) are NOT portable across GPU SM or TRT/CUDA versions. Use the exact environment above or rebuild from checkpoints."
    )
    lines.append("- Checkpoints (post-convert, pre-engine) are portable and recommended for sharing.")

    lines.append("")
    lines.append("## Download Examples (Python)")
    lines.append("```")
    lines.append("from huggingface_hub import snapshot_download")
    lines.append("# Download only engines for this arch")
    lines.append(
        f"path = snapshot_download(repo_id=\"{repo_id}\", allow_patterns=[\"trt-llm/engines/{engine_label}/**\"])"
    )
    lines.append("")
    lines.append("# Or download checkpoints only")
    lines.append(
        f"path = snapshot_download(repo_id=\"{repo_id}\", allow_patterns=[\"trt-llm/checkpoints/**\"])"
    )
    lines.append("```)\n")

    (repo_root / "README.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Upload TRT-LLM artifacts to Hugging Face")
    parser.add_argument("--repo-id", required=True, help="HF repo id, e.g. your-org/my-model-trtllm")
    parser.add_argument("--private", action="store_true", help="Create as private repo")
    parser.add_argument(
        "--what",
        choices=["engines", "checkpoints", "both"],
        default="both",
        help="What to upload",
    )
    parser.add_argument("--checkpoint-dir", default=os.getenv("CHECKPOINT_DIR"))
    parser.add_argument("--engine-dir", default=os.getenv("TRTLLM_ENGINE_DIR"))
    parser.add_argument("--tokenizer-dir", default=os.getenv("MODEL_ID"))
    parser.add_argument("--engine-label", default=os.getenv("HF_PUSH_ENGINE_LABEL", ""))
    parser.add_argument("--workdir", default=".hf_upload_staging")
    parser.add_argument("--no-readme", action="store_true", help="Do not generate README.md in repo")
    parser.add_argument("--prune", action="store_true", help="Delete matching remote files before upload")
    args = parser.parse_args()

    # Resolve paths
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else None
    engine_dir = Path(args.engine_dir) if args.engine_dir else None
    tokenizer_dir = Path(args.tokenizer_dir) if args.tokenizer_dir and os.path.isdir(args.tokenizer_dir) else None

    if args.what in ("checkpoints", "both") and not (checkpoint_dir and checkpoint_dir.is_dir()):
        print(f"[push] ERROR: checkpoint dir not found: {checkpoint_dir}", file=sys.stderr)
        sys.exit(1)
    if args.what in ("engines", "both") and not (engine_dir and engine_dir.is_dir()):
        print(f"[push] ERROR: engine dir not found: {engine_dir}", file=sys.stderr)
        sys.exit(1)

    engine_label = detect_engine_label(engine_dir or Path("."), args.engine_label or None)

    # Build staging tree
    staging = Path(args.workdir)
    if staging.exists():
        for p in sorted(staging.rglob("*"), reverse=True):
            try:
                if p.is_file():
                    p.unlink()
                else:
                    p.rmdir()
            except Exception:
                pass
        try:
            staging.rmdir()
        except Exception:
            pass
    staging.mkdir(parents=True, exist_ok=True)

    build_staging_tree(
        repo_root=staging,
        tokenizer_src=tokenizer_dir,
        checkpoint_src=checkpoint_dir if args.what in ("checkpoints", "both") else None,
        engine_src=engine_dir if args.what in ("engines", "both") else None,
        engine_label=engine_label,
    )

    # Create repo and upload
    api = HfApi()
    create_repo(args.repo_id, repo_type="model", exist_ok=True, private=args.private)

    if not args.no_readme:
        meta = collect_env_metadata(engine_dir or Path("."))
        write_readme(staging, engine_label, meta, args.what, args.repo_id)

    commit_msg = f"Upload TRT-LLM {args.what} (engine_label={engine_label})"
    delete_patterns = None
    if args.prune:
        pats = []
        if args.what in ("engines", "both"):
            pats.append(f"trt-llm/engines/{engine_label}/**")
        if args.what in ("checkpoints", "both"):
            pats.append("trt-llm/checkpoints/**")
        delete_patterns = pats if pats else None

    print(f"[push] Uploading folder {staging} to {args.repo_id}...")
    upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=str(staging),
        allow_patterns=["*"],
        delete_patterns=delete_patterns,
        commit_message=commit_msg,
    )
    print("[push] Upload complete.")


if __name__ == "__main__":
    main()


