#!/usr/bin/env python3
import argparse
import os
from pathlib import Path


def main():
    # Ensure TensorRT-LLM logs are visible unless explicitly overridden
    os.environ.setdefault("TLLM_LOG_LEVEL", os.environ.get("TENSORRT_LLM_LOG_LEVEL", "INFO"))

    parser = argparse.ArgumentParser(description="Build TensorRT-LLM engine for Orpheus")
    parser.add_argument("--model", default=os.getenv("MODEL_ID", "canopylabs/orpheus-3b-0.1-ft"))
    parser.add_argument("--output", default=os.getenv("TRTLLM_ENGINE_DIR", "./models/orpheus-trt"))
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])  # float16 compiles fastest typically
    parser.add_argument("--max_input_len", type=int, default=1024)
    parser.add_argument("--max_output_len", type=int, default=1024)
    parser.add_argument("--max_batch_size", type=int, default=1)
    parser.add_argument("--minimal", action="store_true", help="Build a tiny engine (128/128/1) to validate toolchain")
    args = parser.parse_args()

    from tensorrt_llm.llmapi import LLM, BuildConfig  # type: ignore

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    max_in = 128 if args.minimal else int(args.max_input_len)
    max_out = 128 if args.minimal else int(args.max_output_len)
    max_bsz = 1 if args.minimal else int(args.max_batch_size)

    cfg = BuildConfig()
    cfg.max_input_len = max_in
    cfg.max_seq_len = int(max_in + max_out)
    cfg.max_batch_size = max_bsz
    cfg.precision = args.dtype
    # Conservative first-pass knobs to reduce tactic search space if available
    try:
        cfg.profiling_verbosity = "layer_names_only"  # type: ignore[attr-defined]
        cfg.force_num_profiles = 1  # type: ignore[attr-defined]
        cfg.monitor_memory = True  # type: ignore[attr-defined]
    except Exception:
        pass

    print(f"[build-trt] TLLM_LOG_LEVEL={os.environ.get('TLLM_LOG_LEVEL')}")
    print(f"[build-trt] Model: {args.model}")
    print(f"[build-trt] Output: {out}")
    print(f"[build-trt] max_in={cfg.max_input_len} max_out={max_out} max_seq={cfg.max_seq_len} bsz={cfg.max_batch_size} dtype={cfg.precision}")

    print("[build-trt] Loading model (may download weights if not cached)...")
    llm = LLM(model=args.model, build_config=cfg)
    print("[build-trt] Building and saving engine (this can take a while)...")
    llm.save(str(out))
    print(f"[build-trt] Done: {out}")


if __name__ == "__main__":
    main()


