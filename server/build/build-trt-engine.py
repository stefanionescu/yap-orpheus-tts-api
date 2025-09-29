#!/usr/bin/env python3
"""TensorRT-LLM engine builder for Orpheus."""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Iterable, Optional


ALLOW_PATTERNS: tuple[str, ...] = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "*.safetensors",
    "model.safetensors*",
)
ENGINE_SENTINELS: tuple[str, ...] = (
    "model.plan",
    "rank0.engine",
)


class BuildError(RuntimeError):
    """Raised when the engine build fails."""


def ensure_linux_gpu() -> None:
    if platform.system() != "Linux":
        raise BuildError(
            "TensorRT-LLM build requires Linux with NVIDIA drivers; "
            f"detected {platform.system()}."
        )
    if shutil.which("nvidia-smi") is None:
        raise BuildError("nvidia-smi not found. Ensure NVIDIA drivers are available.")


def ensure_mpi_runtime() -> None:
    try:
        from mpi4py import MPI  # type: ignore  # noqa: WPS433
        _ = MPI.Get_version()
    except ImportError as exc:  # pragma: no cover - runtime dependency only
        raise BuildError(
            "mpi4py is not installed inside the active virtualenv. "
            "Re-run scripts/01-install-trt.sh."
        ) from exc
    except AttributeError as exc:  # pragma: no cover - runtime dependency only
        raise BuildError(
            "mpi4py is installed but its MPI extension failed to load. "
            "Install or expose a working MPI runtime (libmpi.so) and reinstall mpi4py."
        ) from exc
    except RuntimeError as exc:  # pragma: no cover - runtime dependency only
        raise BuildError(
            "MPI runtime libraries (libmpi.so) are missing. "
            "Install OpenMPI (e.g. apt-get install libopenmpi-dev openmpi-bin) "
            "or expose the correct libmpi.so directory via LD_LIBRARY_PATH."
        ) from exc


def ensure_token_env() -> Optional[str]:
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HF_HUB_TOKEN")
    )
    if token:
        os.environ.setdefault("HF_TOKEN", token)
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)
        os.environ.setdefault("HF_HUB_TOKEN", token)
    return token


def resolve_model_source(model: str, token: Optional[str]) -> Path:
    path = Path(model)
    if path.exists():
        print(f"[build-trt] Using local checkpoint: {path}")
        return path

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - dependency missing at runtime only
        raise BuildError(
            "huggingface_hub is required to download weights. Install requirements-trt.txt"
        ) from exc

    cache_root = Path(os.environ.get("TRTLLM_CACHE_DIR", Path.cwd() / ".hf"))
    cache_root.mkdir(parents=True, exist_ok=True)
    local_dir = cache_root / model.replace("/", "-")

    print(f"[build-trt] Downloading {model} → {local_dir}")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    path = Path(
        snapshot_download(
            repo_id=model,
            token=token,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            allow_patterns=ALLOW_PATTERNS,
            resume_download=True,
        )
    )
    print(f"[build-trt] Snapshot ready: {path}")
    return path


def build_engine(args: argparse.Namespace) -> Path:
    ensure_linux_gpu()
    ensure_mpi_runtime()
    token = ensure_token_env()

    try:
        from tensorrt_llm.llmapi import BuildConfig, LLM  # type: ignore  # noqa: WPS433
    except ImportError as exc:  # pragma: no cover - runtime dependency only
        message = str(exc).lower()
        if "libpython" in message:
            raise BuildError(
                "TensorRT-LLM failed to import because libpython shared libraries are missing. "
                "Install python3-dev/python3.10-dev and ensure libpython3.10.so is on LD_LIBRARY_PATH."
            ) from exc
        if "cuda" in message:
            raise BuildError(
                "TensorRT-LLM requires the cuda-python package and access to CUDA runtime libraries. "
                "Install cuda-python>=12.4 and ensure libcuda/libcudart are visible (LD_LIBRARY_PATH)."
            ) from exc
        raise

    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    max_in = 128 if args.minimal else int(args.max_input_len)
    max_out = 128 if args.minimal else int(args.max_output_len)
    max_bsz = 1 if args.minimal else int(args.max_batch_size)

    build_cfg = BuildConfig()
    build_cfg.max_input_len = max_in
    build_cfg.max_seq_len = max_in + max_out
    build_cfg.max_batch_size = max_bsz
    build_cfg.precision = args.dtype
    try:
        build_cfg.profiling_verbosity = "layer_names_only"  # type: ignore[attr-defined]
        build_cfg.force_num_profiles = 1  # type: ignore[attr-defined]
        build_cfg.monitor_memory = True  # type: ignore[attr-defined]
    except Exception:
        pass

    print(
        "[build-trt] Build configuration: "
        f"in={build_cfg.max_input_len} out={max_out} seq={build_cfg.max_seq_len} "
        f"bsz={build_cfg.max_batch_size} dtype={build_cfg.precision}"
    )

    model_path = resolve_model_source(args.model, token)

    print("[build-trt] Initializing LLM API (this can take a few minutes)...")
    llm = LLM(model=str(model_path), build_config=build_cfg)

    print("[build-trt] Saving engine artefacts...")
    llm.save(str(output_dir))

    ensure_engine_files(output_dir)
    return output_dir


def ensure_engine_files(output_dir: Path) -> None:
    if any((output_dir / sentinel).exists() for sentinel in ENGINE_SENTINELS):
        return

    candidate = next(iter(output_dir.rglob("*.engine")), None)
    if candidate:
        return

    raise BuildError(
        f"No TensorRT engine artefacts found under {output_dir}. "
        "Check builder logs for errors."
    )


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build TensorRT-LLM engine for Orpheus")
    parser.add_argument("--model", default=os.getenv("MODEL_ID", "canopylabs/orpheus-3b-0.1-ft"))
    parser.add_argument("--output", default=os.getenv("TRTLLM_ENGINE_DIR", "./models/orpheus-trt"))
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--max_input_len", type=int, default=1024)
    parser.add_argument("--max_output_len", type=int, default=1024)
    parser.add_argument("--max_batch_size", type=int, default=1)
    parser.add_argument(
        "--minimal",
        action="store_true",
        help="Build a tiny engine (128/128/1) to validate toolchain",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)

    if os.environ.get("TENSORRT_LLM_LOG_LEVEL") and not os.environ.get("TLLM_LOG_LEVEL"):
        os.environ["TLLM_LOG_LEVEL"] = os.environ["TENSORRT_LLM_LOG_LEVEL"]
    os.environ.setdefault("TLLM_LOG_LEVEL", "INFO")

    print(f"[build-trt] Model: {args.model}")
    print(f"[build-trt] Output directory: {Path(args.output).resolve()}")

    try:
        engine_dir = build_engine(args)
    except BuildError as exc:
        print(f"[build-trt] ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"[build-trt] Engine ready at: {engine_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
