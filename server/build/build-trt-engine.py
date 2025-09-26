#!/usr/bin/env python3
import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Build TensorRT-LLM engine for Orpheus")
    parser.add_argument("--model", default=os.getenv("MODEL_ID", "canopylabs/orpheus-3b-0.1-ft"))
    parser.add_argument("--output", default=os.getenv("TRTLLM_ENGINE_DIR", "./models/orpheus-trt"))
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--max_input_len", type=int, default=512)
    parser.add_argument("--max_output_len", type=int, default=1024)
    parser.add_argument("--max_batch_size", type=int, default=1)
    args = parser.parse_args()

    from tensorrt_llm.llmapi import LLM, BuildConfig  # type: ignore

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    cfg = BuildConfig()
    cfg.max_input_len = int(args.max_input_len)
    cfg.max_seq_len = int(args.max_input_len + args.max_output_len)
    cfg.max_batch_size = int(args.max_batch_size)
    cfg.precision = args.dtype
    cfg.use_paged_kv_cache = False
    cfg.remove_input_padding = False

    print(f"[build-trt] Loading model: {args.model}")
    llm = LLM(model=args.model, build_config=cfg)
    print("[build-trt] Saving engine...")
    llm.save(str(out))
    print(f"[build-trt] Done: {out}")


if __name__ == "__main__":
    main()


