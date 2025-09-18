import os


def _select_backend() -> str:
    b = os.getenv("ORPHEUS_BACKEND", "trtllm").strip().lower()
    if b in {"trt", "trt-llm", "tensorrt", "tensorrt-llm"}:
        return "trtllm"
    if b in {"vllm"}:
        return "vllm"
    return "trtllm"


_BACKEND = _select_backend()

if _BACKEND == "trtllm":
    from .engine_trtllm import OrpheusTTSEngine  # noqa: F401
else:
    from .engine_vllm import OrpheusTTSEngine  # noqa: F401


