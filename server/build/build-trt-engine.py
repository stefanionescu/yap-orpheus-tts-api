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


def ensure_quantization_available() -> None:
    try:
        import tensorrt_llm.quantization  # type: ignore # noqa: F401
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise BuildError(
            "TensorRT-LLM quantization utilities are unavailable. "
            "Install the full tensorrt-llm package with quantization extras."
        ) from exc


def export_quantized_checkpoint(
    source: Path,
    export_dir: Path,
    kv_cache_dtype: str,
    calib_max_seq_len: Optional[int] = None,
    calib_batch_size: Optional[int] = None,
    tokenizer_max_seq_length_hint: Optional[int] = None,
    dtype_str: Optional[str] = None,
    weight_quant: str = "none",
    auto_quantize_bits: Optional[float] = None,
) -> Path:
    from tensorrt_llm.quantization import quantize_and_export  # type: ignore
    import inspect
    import os

    export_dir.mkdir(parents=True, exist_ok=True)
    if any(export_dir.iterdir()):
        print(f"[build-trt] Using existing quantized checkpoint at: {export_dir}")
        return export_dir

    print(
        "[build-trt] Exporting quantized checkpoint (kv_cache_dtype="
        f"{kv_cache_dtype}) → {export_dir}"
    )
    sig = inspect.signature(quantize_and_export)
    params = sig.parameters

    kwargs: dict[str, str] = {}
    # Newer releases (0.20+) use keyword-only `model_dir`
    if "model_dir" in params:
        kwargs["model_dir"] = str(source)
    elif "checkpoint_dir" in params:
        kwargs["checkpoint_dir"] = str(source)
    elif "model" in params:
        kwargs["model"] = str(source)
    else:
        raise BuildError(
            "Unsupported quantize_and_export signature: expected one of model_dir/checkpoint_dir/model"
        )

    if "export_dir" in params:
        kwargs["export_dir"] = str(export_dir)
    elif "output_dir" in params:
        kwargs["output_dir"] = str(export_dir)
    else:
        raise BuildError(
            "Unsupported quantize_and_export signature: missing export/output dir parameter"
        )

    # If the function does not accept kv_cache_dtype at all, abort early
    if "kv_cache_dtype" not in params:
        raise BuildError("quantize_and_export does not accept kv_cache_dtype on this version")

    # Some TRT-LLM versions (e.g., 0.20.0) require a large set of keyword-only
    # calibration arguments for PTQ. Provide minimal viable defaults and a
    # lightweight synthetic calibration dataset to enable KV-cache INT8 export.
    required_kwonly = {
        "device",
        "calib_dataset",
        "dtype",
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
    # Treat qformat as required if present in signature; provide a default below
    if "qformat" in params:
        required_kwonly = set(required_kwonly) | {"qformat"}
    missing_required = [name for name in required_kwonly if name in params and name not in kwargs]

    if missing_required:
        # Build a tiny synthetic calibration dataset from short prompts.
        try:
            from transformers import AutoTokenizer  # type: ignore
        except Exception as exc:  # pragma: no cover - runtime dependency only
            print(
                f"[build-trt] WARN: transformers unavailable ({type(exc).__name__}: {exc}); "
                "cannot prepare calibration dataset. Skipping export."
            )
            return source

        model_id_for_tokenizer = str(source)
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id_for_tokenizer)
        except Exception:
            # Fallback to env-provided MODEL_ID if local dir is not a HF repo
            model_id_for_tokenizer = os.environ.get("MODEL_ID", "canopylabs/orpheus-3b-0.1-ft")
            tokenizer = AutoTokenizer.from_pretrained(model_id_for_tokenizer)

        # Use shared calibration samples module (varied, short), load by path to
        # work whether this script is executed as a module or as a file
        sample_texts: list[str] = []
        try:
            import importlib.util
            cal_path = Path(__file__).with_name("calibration_samples.py")
            spec = importlib.util.spec_from_file_location("calib_samples", str(cal_path))
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)  # type: ignore[attr-defined]
                sample_texts = list(getattr(mod, "SAMPLES", []))
        except Exception:
            sample_texts = []
        if not sample_texts:
            # Minimal safe fallback if samples module is unavailable
            sample_texts = [
                "Hello there!",
                "Please read this sentence clearly and naturally.",
                "Good morning, how can I help you today?",
                "Welcome to our service. Your order has shipped.",
            ]
        desired_batch = calib_batch_size or int(os.environ.get("TRTLLM_MAX_BATCH_SIZE", "16"))
        desired_calib_size = int(os.environ.get("TRTLLM_CALIB_SIZE", str(max(32, desired_batch * 4))))
        if len(sample_texts) < desired_calib_size:
            reps = (desired_calib_size + len(sample_texts) - 1) // len(sample_texts)
            sample_texts = (sample_texts * reps)[:desired_calib_size]

        tokenizer_limit = getattr(tokenizer, "model_max_length", 2048) or 2048
        token_max_seq = min(
            tokenizer_limit,
            tokenizer_max_seq_length_hint or calib_max_seq_len or 128,
        )
        max_len_guess = min(token_max_seq, 128)
        encodings = tokenizer(
            sample_texts,
            padding=False,
            truncation=True,
            max_length=max_len_guess,
            return_tensors=None,
        )

        class _CalibDataset:
            def __init__(self, enc):
                self._ids = enc["input_ids"] if isinstance(enc, dict) else enc

            def __len__(self):  # noqa: D401
                return len(self._ids)

            def __iter__(self):
                for ids in self._ids:
                    yield {"input_ids": ids}

        calib_dataset = _CalibDataset(encodings)

        # Fill defaults for common required params.
        dataset_len = len(_CalibDataset(encodings))
        eff_batch = max(1, min(desired_batch, dataset_len))

        defaults: dict[str, object] = {
            "device": "cuda",
            "calib_dataset": calib_dataset,
            "dtype": dtype_str or os.environ.get("TRTLLM_DTYPE", "bfloat16"),
            "calib_size": dataset_len,
            "batch_size": eff_batch,
            "calib_max_seq_length": int(min(max_len_guess, calib_max_seq_len or max_len_guess)),
            "awq_block_size": 128,
            "tp_size": int(os.environ.get("TP_SIZE", "1")),
            "pp_size": int(os.environ.get("PP_SIZE", "1")),
            "cp_size": int(os.environ.get("CP_SIZE", "1")),
            "seed": 17,
            "tokenizer_max_seq_length": int(token_max_seq),
        }

        # Quantization format: prefer enum if available, else a known string.
        qformat_value: Optional[object] = None
        try:
            from tensorrt_llm.quantization import QuantMode  # type: ignore
            # Use SmoothQuant by default even for KV-only, since TRT-LLM 0.20 expects a supported qformat
            qformat_value = getattr(QuantMode, "SMOOTHQUANT", None)
        except Exception:
            qformat_value = None
        defaults["qformat"] = qformat_value or "smoothquant"

        for name in list(defaults.keys()):
            if name in params and name not in kwargs:
                kwargs[name] = defaults[name]  # type: ignore[assignment]

    # Add optional auto quantization controls when available and requested
    if weight_quant == "auto8" and "auto_quantize_bits" in params:
        kwargs.setdefault("auto_quantize_bits", auto_quantize_bits or 8.0)
    # Try to quantize LM head as well when supported (optional)
    if "quantize_lm_head" in params and weight_quant != "none":
        kwargs.setdefault("quantize_lm_head", True)

    # Attempt the export; if it fails due to unsupported qformat, try fallbacks.
    def _try_export(qfmt: Optional[object]) -> None:
        if qfmt is None and "qformat" in kwargs:
            kwargs.pop("qformat", None)
        elif qfmt is not None:
            kwargs["qformat"] = qfmt
        quantize_and_export(kv_cache_dtype=kv_cache_dtype, **kwargs)

    try:
        _try_export(kwargs.get("qformat"))
    except ValueError as exc:
        msg = str(exc).lower()
        if "unsupported quantization format" in msg:
            # Build fallback candidates
            candidates: list[object] = []
            candidates.extend(["smoothquant", "sq", "awq", "woq_int8"])
            for cand in candidates:
                try:
                    _try_export(cand)
                    break
                except ValueError:
                    continue
            else:
                print(f"[build-trt] ERROR: quantize_and_export failed: {exc}")
                raise BuildError("quantize_and_export rejected provided qformat values.") from exc
        else:
            raise
    except TypeError as exc:
        print(f"[build-trt] ERROR: quantize_and_export failed: {exc}")
        raise BuildError(
            "quantize_and_export requires additional arguments that could not be inferred. "
            "Provide calibration controls or adjust TRT-LLM version."
        ) from exc
    return export_dir


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

    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    max_in = int(args.max_input_len)
    max_out = int(args.max_output_len)
    max_bsz = int(args.max_batch_size)

    print(
        "[build-trt] Build configuration: "
        f"in={max_in} out={max_out} seq={max_in + max_out} "
        f"bsz={max_bsz} dtype={args.dtype}"
    )

    model_path = resolve_model_source(args.model, token)

    # Convert HuggingFace checkpoint to TensorRT-LLM checkpoint format
    trtllm_ckpt_dir = output_dir.parent / f"{output_dir.name}-ckpt"
    trtllm_ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if conversion is needed (look for TensorRT-LLM checkpoint marker)
    needs_conversion = not (trtllm_ckpt_dir / "config.json").exists() or \
                       not any(trtllm_ckpt_dir.glob("rank*.safetensors"))
    
    if needs_conversion:
        print(f"[build-trt] Converting HuggingFace checkpoint to TensorRT-LLM format...")
        print(f"[build-trt] Source: {model_path}")
        print(f"[build-trt] Target: {trtllm_ckpt_dir}")
        
        import subprocess
        import json
        
        # Read the HF config to determine model type
        hf_config_path = model_path / "config.json"
        with open(hf_config_path) as f:
            hf_config = json.load(f)
        
        model_type = hf_config.get("model_type", "llama")
        
        # Use convert_checkpoint command for Llama models
        convert_cmd = [
            "python", "-m", "tensorrt_llm.commands.convert_checkpoint",
            "--model_dir", str(model_path),
            "--output_dir", str(trtllm_ckpt_dir),
            "--dtype", args.dtype,
        ]
        
        print(f"[build-trt] Running: {' '.join(convert_cmd)}")
        result = subprocess.run(convert_cmd, check=False)
        if result.returncode != 0:
            raise BuildError(f"Checkpoint conversion failed with exit code {result.returncode}")
        
        print(f"[build-trt] Conversion complete")
    else:
        print(f"[build-trt] Using existing TensorRT-LLM checkpoint at {trtllm_ckpt_dir}")
    
    model_for_build = trtllm_ckpt_dir

    # Use trtllm-build CLI tool to build the engine
    print("[build-trt] Building TensorRT engine using trtllm-build CLI...")
    
    import subprocess
    
    max_seq = max_in + max_out
    
    cmd = [
        "trtllm-build",
        "--checkpoint_dir", str(model_for_build),
        "--output_dir", str(output_dir),
        "--max_input_len", str(max_in),
        "--max_seq_len", str(max_seq),
        "--max_batch_size", str(max_bsz),
        "--remove_input_padding", "enable",
        "--log_level", "info",
    ]
    
    # Add context FMHA settings
    if hasattr(args, 'context_fmha') and args.context_fmha:
        if args.context_fmha != "auto":
            cmd.extend(["--context_fmha", args.context_fmha])
    
    # Add KV cache dtype if specified
    if hasattr(args, 'kv_cache_dtype') and args.kv_cache_dtype:
        if args.kv_cache_dtype == "fp8":
            cmd.extend(["--use_fp8_context_fmha", "enable"])
    
    print(f"[build-trt] Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise BuildError(f"trtllm-build command failed with exit code {result.returncode}")

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
    kv_env = os.getenv("TRTLLM_KV_CACHE_DTYPE")
    if kv_env:
        kv_env = kv_env.lower()
        if kv_env not in {"fp8", "int8"}:
            kv_env = None
    parser.add_argument("--model", default=os.getenv("MODEL_ID", "canopylabs/orpheus-3b-0.1-ft"))
    parser.add_argument("--output", default=os.getenv("TRTLLM_ENGINE_DIR", "./models/orpheus-trt"))
    parser.add_argument("--dtype", default=os.getenv("TRTLLM_DTYPE", "float16"), choices=["float16", "bfloat16"])
    parser.add_argument("--max_input_len", type=int, default=128)
    parser.add_argument("--max_output_len", type=int, default=2048)
    parser.add_argument("--max_batch_size", type=int, default=1)
    parser.add_argument("--kv_cache_dtype", choices=["fp8", "int8"], default=kv_env)
    parser.add_argument(
        "--context_fmha",
        choices=["enable", "disable", "auto"],
        default=os.getenv("TRTLLM_CONTEXT_FMHA", "disable"),
        help="Control context-phase fused attention; disable on A100 per docs",
    )
    args = parser.parse_args(argv)
    if not args.kv_cache_dtype:
        args.kv_cache_dtype = None
    return args


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
