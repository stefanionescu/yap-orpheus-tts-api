#!/usr/bin/env python
import argparse
import json
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_folder
import subprocess
import platform

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


