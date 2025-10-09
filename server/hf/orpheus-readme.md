---
license: {{license}}
base_model: {{base_model}}
tags:
- awq
- quantized
- {{w_bit}}-bit
- tensorrt-llm
- orpheus-tts
language:
- en
pipeline_tag: text-to-speech
---

<div align="center">
  <h1 style="font-size: 48px; color: #2E86AB; font-weight: bold;">
    {{model_name}} — INT{{w_bit}} AWQ Quantized
  </h1>
  <p style="font-size: 18px; color: #666;">
    Streaming optimized Orpheus 3B quant. Meant to run on an A100 with TensorRT 
  </p>
</div>

---

<div align="center">
  <img src="https://img.shields.io/badge/Quantization-AWQ-blue?style=for-the-badge" alt="AWQ">
  <img src="https://img.shields.io/badge/Precision-INT{{w_bit}}-green?style=for-the-badge" alt="INT{{w_bit}}">
  <img src="https://img.shields.io/badge/Framework-TensorRT--LLM-red?style=for-the-badge" alt="TensorRT-LLM">
  <img src="https://img.shields.io/badge/License-Apache%202.0-yellow?style=for-the-badge" alt="License">
</div>

---

## Model Overview

This is a streaming optimized INT4 AWQ quantized version of [canopylabs/orpheus-3b-0.1-ft](https://huggingface.co/canopylabs/orpheus-3b-0.1-ft), meant to run with TensorRT-LLM.

**Key Features:**
- **Optimized for Production**: Built for high-throughput, low-latency TTS serving
- **Faster Inference**: Up to ~3x faster than FP16 with minimal perceived quality loss
- **Memory Efficient**: ≈4x smaller weights vs. FP16 (INT4)
- **Ready for Streaming**: Designed for real-time streaming TTS backends
- **Calibrated**: Calibrated for 48 tokens of input and up to 1024 tokens of output (roughly 12 seconds worth of audio per text chunk)

---

## Technical Specifications

| Specification | Details |
|---------------|---------|
| **Source Model** | [canopylabs/orpheus-3b-0.1-ft](https://huggingface.co/canopylabs/orpheus-3b-0.1-ft) |
| **Quantization Method** | AWQ (Activation-aware Weight Quantization) |
| **Precision** | INT4 weights, INT8 KV cache |
| **AWQ Group/Block Size** | 128 |
| **TensorRT-LLM Version** | `1.0.0` |
| **Generated** | 2025-10-09 |
| **Pipeline** | TensorRT-LLM AWQ Quantization |

### Artifact Layout

```
trt-llm/
  checkpoints/            # Quantized TRT-LLM checkpoints (portable)
    *.safetensors
    config.json
  engines/sm80_trt-llm-1.0.0_cuda12.4/     # Built TensorRT-LLM engines (hardware-specific)
    rank*.engine
    build_metadata.json
    build_command.sh
```

## Quick Start

### Download Artifacts (Python)

```python
from huggingface_hub import snapshot_download

# Download quantized checkpoints (portable)
ckpt_path = snapshot_download(
    repo_id="yapwithai/orpheus-3b-trt-int4-awq",
    allow_patterns=["trt-llm/checkpoints/**"],
)

# Or download TensorRT-LLM engines for a specific build label
eng_path = snapshot_download(
    repo_id="yapwithai/orpheus-3b-trt-int4-awq",
    allow_patterns=["trt-llm/engines/sm80_trt-llm-1.0.0_cuda12.4/**"],
)
print("checkpoints:", ckpt_path)
print("engines:", eng_path)
```

### Run a Streaming TTS Server (TensorRT-LLM)

```bash
# Point your server to the downloaded engines
export TRTLLM_ENGINE_DIR=/path/to/trt-llm/engines/sm80_trt-llm-1.0.0_cuda12.4

# Start your TTS server (example: FastAPI + WebSocket)
python -m server.server
```

## Quantization Details

- Method: Activation-aware weight quantization (AWQ)
- Calibration size: 256
- AWQ block/group size: 128
- DType for build: float16


### Configuration Summary

```json
{
  "quantization": {
    "weights_precision": "int4_awq",
    "kv_cache_dtype": "int8",
    "awq_block_size": 128,
    "calib_size": 256
  },
  "build": {
    "dtype": "float16",
    "max_input_len": 48,
    "max_output_len": 1024,
    "max_batch_size": 16,
    "engine_label": "sm80_trt-llm-1.0.0_cuda12.4",
    "tensorrt_llm_version": "1.0.0"
  },
  "environment": {
    "sm_arch": "sm80",
    "gpu_name": "NVIDIA A100 80GB PCIe",
    "cuda_toolkit": "12.4",
    "nvidia_driver": "550.127.05"
  }
}
```

---

## Use Cases

- **Realtime Voice**: assistants, product demos, interactive agents
- **High-throughput Serving**: batch TTS pipelines, APIs
- **Edge & Cost-sensitive**: limited VRAM environments

---

## Advanced Configuration (Build-time)

- Max input length: tune `--max_input_len`
- Max output length: tune `--max_seq_len`
- Batch size: tune `--max_batch_size`
- Plugins: `--gpt_attention_plugin`, `--context_fmha`, `--paged_kv_cache`

---

## Requirements & Compatibility

### System Requirements
- **GPU**: NVIDIA, Compute Capability ≥ 8.0 (A100/RTX 40/H100 class recommended)
- **VRAM**: ≥ 1.6 GB for INT4 engines (per GPU)
- **CUDA**: 12.x recommended
- **Python**: 3.10+

### Framework Compatibility
- **TensorRT-LLM** (engines), version `1.0.0`
- **TRT-LLM Checkpoints** are portable across systems; engines are not

### Installation

```bash
pip install huggingface_hub
# Install TensorRT-LLM per NVIDIA docs
# https://nvidia.github.io/TensorRT-LLM/
```

---

## Troubleshooting

<details>
<summary><b>Engine not portable</b></summary>
Engines are specific to GPU SM and TRT/CUDA versions. Rebuild on the target
system or download a matching `engines/sm80_trt-llm-1.0.0_cuda12.4` variant if provided.
</details>

<details>
<summary><b>OOM or Slow Loading</b></summary>
Reduce `max_seq_len`, lower `max_batch_size`, and ensure `gpu_memory_utilization`
on your server is tuned to your GPU.
</details>

---

## Additional Resources

- **TensorRT-LLM Docs**: https://nvidia.github.io/TensorRT-LLM/
- **Activation-aware Weight Quantization (AWQ)**: https://github.com/mit-han-lab/llm-awq

---

## License

This quantized model inherits the license from the original model: **Apache 2.0**