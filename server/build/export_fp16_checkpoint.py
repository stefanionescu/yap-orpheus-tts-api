#!/usr/bin/env python3
"""Export a Hugging Face checkpoint to FP16 for TensorRT builds."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

import torch

SENTINEL_NAME = ".fp16-export.json"
DEFAULT_MAX_SHARD_SIZE = "2GB"


class ExportError(RuntimeError):
    """Custom error for export workflow."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        help="Hugging Face repo id or local path to the source checkpoint",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Directory where the FP16 checkpoint will be stored",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-export even if the sentinel indicates a valid FP16 snapshot",
    )
    return parser.parse_args()


def env_token() -> Optional[str]:
    return (
        os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HF_HUB_TOKEN")
    )


def snapshot_dir_for(model_id: str) -> Path:
    safe_name = model_id.replace("/", "-")
    base = Path(os.environ.get("HF_SNAPSHOT_BASE", Path.cwd() / ".hf"))
    return base / safe_name


def ensure_snapshot(model_id: str, token: Optional[str]) -> Path:
    from huggingface_hub import snapshot_download

    local_dir = snapshot_dir_for(model_id)
    local_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    path = snapshot_download(
        repo_id=model_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
        token=token,
    )
    return Path(path)


def clear_directory(path: Path) -> None:
    if not path.exists():
        return
    for item in path.iterdir():
        if item.is_file() or item.is_symlink():
            item.unlink()
        else:
            shutil.rmtree(item)


def load_json(path: Path) -> Optional[dict]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)


def copy_optional_files(source: Path, dest: Path) -> None:
    patterns = {
        "README.md",
        "license.txt",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "tokenizer.model",
        "spiece.model",
        "vocab.json",
        "vocab.txt",
        "merges.txt",
        "tokenizer_special_tokens_map.json",
    }
    for pattern in patterns:
        src = source / pattern
        dst = dest / pattern
        if src.exists() and not dst.exists():
            if src.is_file():
                shutil.copy2(src, dst)


def ensure_generation_config(source: Path, dest: Path) -> None:
    from transformers import GenerationConfig

    try:
        gen_config = GenerationConfig.from_pretrained(str(source))
    except OSError:
        return
    gen_config.save_pretrained(dest)


def convert_to_fp16(
    source: Path,
    dest: Path,
    *,
    source_id: str,
    force: bool,
) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not source.exists():
        raise ExportError(f"Source checkpoint not found: {source}")

    sentinel_path = dest / SENTINEL_NAME
    sentinel = load_json(sentinel_path)
    if sentinel and not force:
        if sentinel.get("dtype") == "float16" and sentinel.get("source") == str(source):
            print(
                f"[export-fp16] Existing FP16 snapshot found at {dest}, skipping conversion",
                file=sys.stderr,
            )
            return sentinel

    same_location = source.resolve() == dest.resolve()
    if same_location and (force or not sentinel):
        raise ExportError(
            "Cannot re-export in place; choose a distinct --output directory or set FP16_MODEL_DIR."
        )

    if dest.exists():
        if force:
            clear_directory(dest)
        else:
            sentinel_msg = " (sentinel missing or invalid)" if not sentinel else ""
            print(f"[export-fp16] Clearing existing directory{sentinel_msg}: {dest}")
            clear_directory(dest)
    else:
        dest.mkdir(parents=True, exist_ok=True)

    print(f"[export-fp16] Loading source checkpoint from {source}")
    target_dtype = torch.float16
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(source),
            torch_dtype=target_dtype,
            device_map="cpu",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    except (TypeError, ValueError) as exc:
        if "low_cpu_mem_usage" not in str(exc):
            raise
        print(
            "[export-fp16] Falling back to standard load (low_cpu_mem_usage unavailable)",
            file=sys.stderr,
        )
        model = AutoModelForCausalLM.from_pretrained(
            str(source),
            torch_dtype=target_dtype,
            device_map="cpu",
            trust_remote_code=True,
        )
    model.to(dtype=target_dtype)
    if hasattr(model, "config"):
        model.config.torch_dtype = target_dtype

    print(f"[export-fp16] Saving FP16 weights to {dest}")
    model.save_pretrained(
        dest,
        safe_serialization=True,
        max_shard_size=DEFAULT_MAX_SHARD_SIZE,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(source),
        use_fast=True,
        trust_remote_code=True,
    )
    tokenizer.save_pretrained(dest)

    ensure_generation_config(source, dest)
    copy_optional_files(source, dest)

    metadata = {
        "source": str(source),
        "source_id": source_id,
        "dtype": "float16",
        "safe_serialization": True,
        "max_shard_size": DEFAULT_MAX_SHARD_SIZE,
        "trust_remote_code": True,
    }
    write_json(sentinel_path, metadata)
    return metadata


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output).resolve()

    model_arg_path = Path(args.model).expanduser()
    if model_arg_path.exists():
        source = model_arg_path.resolve()
        source_id = str(source)
    else:
        token = env_token()
        try:
            source = ensure_snapshot(args.model, token)
        except Exception as exc:  # pragma: no cover - runtime download
            raise ExportError(f"Failed to download checkpoint {args.model}: {exc}") from exc
        source_id = args.model

    metadata = convert_to_fp16(
        source,
        output_dir,
        source_id=source_id,
        force=args.force,
    )
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ExportError as exc:
        print(f"[export-fp16] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
