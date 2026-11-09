# AMD Hardware & ROCm Configuration Guide

> **VeriFind AI — AMD AI League Match 4**

---

## ⚠️ Honest Development Status

> [!IMPORTANT]
> ROCm execution has **NOT** been tested on actual AMD hardware during this development session.
>
> The current development machine is a **MacBook Air M2** running **macOS** (Apple Silicon, Darwin 27.0, Python 3.14.6).  
> No AMD GPU, `rocm-smi`, or any ROCm runtime is present on this machine.
>
> The AMD ROCm provider architecture (`backend/rocm_provider.py`) is verified **only** via unit tests that mock the OpenAI-compatible HTTP API contract. All AMD-specific behaviour is documented below as the **target configuration** for deployment on AMD hardware or AMD Developer Cloud.

---

## Development Environment (Current)

| Property | Value |
|---|---|
| Machine | MacBook Air M2 |
| OS | macOS (Darwin 27.0) |
| CPU | Apple M2 (ARM64) |
| GPU | Apple M2 integrated GPU (Metal) |
| Python | 3.14.6 (Homebrew) |
| ROCm | **Not present / not applicable** |
| AMD GPU | **None** |
| Testing method | Mocked HTTP client (unit tests) |

---

## Target AMD Hardware Configuration

VeriFind AI is designed to connect to an AMD GPU inference backend via an **OpenAI-compatible REST API** (`/v1/chat/completions`). The recommended target hardware configurations are:

| Tier | Hardware | VRAM | Recommended Model |
|---|---|---|---|
| Development | AMD Radeon RX 7900 XTX | 24 GB GDDR6 | Qwen2.5-7B-Instruct (4-bit GPTQ) |
| Production | AMD Instinct MI250X | 128 GB HBM2e | Qwen2.5-14B-Instruct |
| High Performance | AMD Instinct MI300X | 192 GB HBM3 | Qwen2.5-72B-Instruct |

---

## ROCm Version Requirements

| Component | Version |
|---|---|
| ROCm | 6.1 or later |
| PyTorch (ROCm build) | 2.4+ (`torch-rocm`) |
| vLLM | 0.5.0+ (with ROCm support) |
| HIP | Included with ROCm 6.x |

---

## Target Model

**Primary:** `Qwen/Qwen2.5-7B-Instruct`  
**Alternative:** `Qwen/Qwen2.5-14B-Instruct` (for MI300X / MI250X)

These models are Apache 2.0-licensed, open-source, and optimised for instruction-following and multi-step reasoning tasks — well-suited to the VeriFind AI evidence extraction workload.

---

## Launching vLLM on AMD ROCm Hardware

### 1. Pull the official ROCm-enabled vLLM Docker image

```bash
docker pull rocm/vllm:latest
# or pin a specific ROCm version:
docker pull rocm/vllm:rocm6.1_mi300_ubuntu22.04_py3.11_vllm_0.5.0
```

### 2. Start the vLLM OpenAI-compatible server

```bash
docker run -d \
  --name verifind-vllm \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add video \
  --ipc=host \
  --shm-size 32g \
  -p 8000:8000 \
  -e HSA_OVERRIDE_GFX_VERSION=11.0.0 \
  rocm/vllm:latest \
  python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dtype auto \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --port 8000
```

> [!NOTE]
> `HSA_OVERRIDE_GFX_VERSION` may differ based on your GPU. For MI300X: `9.4.2`. For RX 7900 XTX: `11.0.0`.

### 3. Verify the server is running

```bash
curl http://localhost:8000/v1/models
# Expected: {"data": [{"id": "Qwen/Qwen2.5-7B-Instruct", ...}]}
```

---

## AMD Developer Cloud Setup

If you do not have local AMD hardware, use [AMD Developer Cloud](https://developer.amd.com/amd-developer-cloud/):

1. Create a free AMD Developer Cloud account.
2. Launch a MI300X instance with the **ROCm vLLM** pre-installed image.
3. SSH-tunnel the vLLM port to your local machine:
   ```bash
   ssh -L 8000:localhost:8000 user@<amd-cloud-instance-ip>
   ```
4. Set the VeriFind AI environment variables (see below).

---

## VeriFind AI Environment Variables for AMD ROCm

Add these to your `.env` file or export them before starting VeriFind AI:

```bash
# Select the AMD ROCm inference provider
MODEL_PROVIDER=amd_rocm

# URL of the vLLM OpenAI-compatible server
ROCM_INFERENCE_ENDPOINT=http://localhost:8000/v1

# Model identifier (must match what vLLM was started with)
ROCM_MODEL_NAME=Qwen/Qwen2.5-7B-Instruct

# Optional API key (set if the server uses authentication)
ROCM_API_KEY=

# Max tokens for generation
ROCM_MAX_TOKENS=1024

# Temperature (0.0 = deterministic, recommended for factual retrieval)
ROCM_TEMPERATURE=0.1
```

For generic OpenAI-compatible servers (non-ROCm vLLM or TGI):

```bash
MODEL_PROVIDER=openai_compatible
OPENAI_COMPATIBLE_ENDPOINT=http://your-server:8000/v1
OPENAI_MODEL_NAME=your-model-name
```

---

## PyTorch ROCm Installation (native, non-Docker)

```bash
# Install ROCm 6.1 system packages first (Ubuntu 22.04 / RHEL 9):
# https://rocm.docs.amd.com/projects/install-on-linux/

# Then install PyTorch ROCm build:
pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/rocm6.1

# Verify GPU is detected:
python -c "import torch; print(torch.cuda.is_available(), torch.version.hip)"
```

> [!WARNING]
> PyTorch ROCm wheels use the same `torch.cuda.*` API as NVIDIA CUDA. `torch.cuda.is_available()` returns `True` on a healthy ROCm setup even though the backend is HIP, not CUDA.

---

## Health Check & Benchmarking

VeriFind AI exposes API endpoints for provider health and benchmarking:

```bash
# Check all provider health statuses
curl http://localhost:8080/api/v1/providers/health

# Benchmark inference latency for the AMD ROCm provider
curl -X POST http://localhost:8080/api/v1/providers/benchmark \
  -H "Content-Type: application/json" \
  -d '{"provider": "amd_rocm"}'
```

Expected benchmark output:

```json
{
  "provider_type": "AMD_ROCM",
  "model_name": "Qwen/Qwen2.5-7B-Instruct",
  "total_latency_ms": 850.0,
  "time_to_first_token_ms": 220.0,
  "tokens_generated": 48,
  "tokens_per_second": 56.5
}
```

> [!NOTE]
> Actual throughput depends on GPU model, memory bandwidth, quantisation, and batch size. The figures above are illustrative targets for a MI250X.

---

## Logging & Observability

The `AmdROCmProvider` logs all inference calls via Python's standard logging module at `INFO` level:

```
INFO backend.rocm_provider - [AMD ROCm] Sending request: model=Qwen/Qwen2.5-7B-Instruct prompt_chars=342
INFO backend.rocm_provider - [AMD ROCm] Response received in 850.3ms (48 tokens)
```

Enable debug logging:

```bash
export LOG_LEVEL=DEBUG
```

---

## Architecture Diagram

```
┌──────────────────────────────────────────────────┐
│               VeriFind AI (macOS / Linux)         │
│                                                  │
│  Streamlit UI  ──►  AnswerEngine                 │
│                         │                        │
│                 ModelProviderFactory              │
│                    /           \                 │
│          GeminiAIEngine    AmdROCmProvider        │
│           (dev / cloud)    (OpenAI-compat HTTP)   │
└──────────────────────────────────────────────────┘
                              │
                     HTTP /v1/chat/completions
                              │
         ┌────────────────────▼─────────────────────┐
         │   AMD GPU Server (ROCm + vLLM)           │
         │                                          │
         │   Hardware: MI300X / MI250X / RX7900 XTX │
         │   ROCm:     6.1+                         │
         │   vLLM:     0.5.0+                       │
         │   Model:    Qwen2.5-7B-Instruct          │
         └──────────────────────────────────────────┘
```

---

## References

- [ROCm Documentation](https://rocm.docs.amd.com/)
- [AMD Developer Cloud](https://developer.amd.com/amd-developer-cloud/)
- [vLLM ROCm Support](https://docs.vllm.ai/en/latest/getting_started/amd-installation.html)
- [Qwen2.5 on Hugging Face](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- [PyTorch ROCm Wheels](https://download.pytorch.org/whl/rocm6.1)

---

*Last updated: 2026-11-09 | Development environment: MacBook Air M2, macOS, Python 3.14.6*  
*ROCm actual execution status: **NOT YET TESTED** — target hardware required.*
