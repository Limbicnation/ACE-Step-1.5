# ACE-Step 1.5 - AI Agent Guide

This document provides essential information for AI coding agents working on the ACE-Step 1.5 project.

## Project Overview

ACE-Step 1.5 is an open-source music generation foundation model developed by ACE Studio and StepFun. It uses a hybrid architecture combining:

- **5Hz Language Model (LM)**: An omni-capable planner based on Qwen3 that generates song blueprints, metadata, lyrics, and audio semantic codes via Chain-of-Thought reasoning
- **DiT (Diffusion Transformer)**: Generates audio waveforms from the LM's semantic codes
- **VAE (AutoencoderOobleck)**: Encodes/decodes audio to/from latent representations

### Key Features
- Text-to-music generation (10s to 10 minutes)
- Audio continuation, repainting, and cover generation
- Multi-track generation and track extraction
- LoRA fine-tuning for style personalization
- Support for 50+ languages

## Technology Stack

- **Python**: 3.11+ (strictly required)
- **Deep Learning**: PyTorch 2.7+ with CUDA 12.8 support
- **Core Dependencies**:
  - `transformers` (4.51.0 - 4.58.0): HuggingFace model support
  - `diffusers`: Diffusion model components
  - `gradio` (6.2.0): Web UI framework
  - `fastapi` + `uvicorn`: REST API server
  - `accelerate`: Multi-device training support
  - `peft` + `lightning`: LoRA training
  - `soundfile`, `torchaudio`: Audio I/O
  - `einops`, `torchcodec`, `torchao`: Performance optimizations
- **Package Manager**: `uv` (Astral's Python package manager)
- **Build System**: `hatchling`

## Project Structure

```
acestep/
├── acestep_v15_pipeline.py    # Main Gradio UI entry point
├── api_server.py              # FastAPI REST server
├── handler.py                 # AceStepHandler - core business logic wrapper
├── inference.py               # GenerationParams, GenerationConfig dataclasses
├── llm_inference.py           # LLMHandler - 5Hz LM management
├── constants.py               # Centralized constants (languages, task types, etc.)
├── gpu_config.py              # GPU memory detection and tier configuration
├── model_downloader.py        # HuggingFace/ModelScope model download
├── audio_utils.py             # Audio processing utilities
├── training/                  # LoRA training implementation
│   ├── trainer.py             # Lightning Fabric-based LoRA trainer
│   ├── configs.py             # LoRAConfig, TrainingConfig
│   ├── lora_utils.py          # LoRA injection and weight management
│   └── dataset_builder_modules/  # Dataset preprocessing pipeline
├── gradio_ui/                 # Gradio web interface
│   ├── interfaces/            # UI component definitions
│   ├── events/                # Event handlers (generation, training, results)
│   └── api_routes.py          # API routes for Gradio integration
├── core/                      # Core business logic
│   ├── generation/            # Generation handler mixins
│   ├── lora/                  # LoRA service and registry
│   └── scoring/               # Quality scoring
├── mlx_dit/                   # Apple Silicon MLX DiT acceleration
├── mlx_vae/                   # Apple Silicon MLX VAE acceleration
└── third_parts/nano-vllm/     # Custom vLLM implementation for LM inference
```

## Build and Run Commands

### Setup (Required First Time)
```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/ACE-Step/ACE-Step-1.5.git
cd ACE-Step-1.5
uv sync
```

### Launch Applications
```bash
# Gradio Web UI (default port 7860)
uv run acestep

# REST API Server (default port 8001)
uv run acestep-api

# Model downloader utility
uv run acestep-download
```

### Platform-Specific Launch Scripts
```bash
# Windows
start_gradio_ui.bat
start_api_server.bat

# Windows (ROCm/AMD)
start_gradio_ui_rocm.bat
start_api_server_rocm.bat

# Linux
./start_gradio_ui.sh
./start_api_server.sh

# macOS (Apple Silicon)
./start_gradio_ui_macos.sh
./start_api_server_macos.sh
```

### Command-Line Options
```bash
# Gradio UI with options
uv run acestep --port 7860 --server-name 127.0.0.1 --language en
uv run acestep --init_service true --config_path acestep-v15-turbo --lm_model_path acestep-5Hz-lm-1.7B
uv run acestep --service_mode true  # Preset configurations for production

# API Server with options
uv run acestep-api --port 8001 --host 127.0.0.1
```

## Configuration

### Environment Variables (.env file)
```bash
# Model Settings
ACESTEP_CONFIG_PATH=acestep-v15-turbo           # DiT model path
ACESTEP_LM_MODEL_PATH=acestep-5Hz-lm-1.7B       # LM model path
ACESTEP_DEVICE=auto                             # auto, cuda, cpu, xpu
ACESTEP_LM_BACKEND=vllm                         # vllm, pt, mlx
ACESTEP_INIT_LLM=auto                           # auto, true, false

# Download Settings
ACESTEP_DOWNLOAD_SOURCE=auto                    # auto, huggingface, modelscope

# API Settings
ACESTEP_API_KEY=sk-your-secret-key

# GPU Debug (simulate different GPU memory)
MAX_CUDA_VRAM=8                                 # Simulate 8GB GPU
MAX_MPS_VRAM=16                                 # Simulate 16GB MPS
```

### GPU Tier Configuration
The project automatically detects GPU memory and configures:
- **tier1 (≤4GB)**: Offload mode, INT8 quantization, no LM
- **tier2 (4-6GB)**: Offload mode, INT8 quantization
- **tier3 (6-8GB)**: 0.6B LM, offload enabled
- **tier4 (8-12GB)**: 0.6B LM, batch size 2-4
- **tier5 (12-16GB)**: 1.7B LM, batch size 4
- **tier6a (16-20GB)**: 1.7B LM, batch size 4-8
- **tier6b (20-24GB)**: Up to 4B LM, batch size 8
- **unlimited (≥24GB)**: All models, no quantization needed

## Code Style Guidelines

### Import Organization
```python
# Standard library imports
import os
import sys
from typing import Optional, Dict, Any

# Third-party imports
import torch
from loguru import logger
from transformers import AutoTokenizer

# Project imports
from acestep.constants import TASK_INSTRUCTIONS
from acestep.gpu_config import get_gpu_config
```

### Key Coding Patterns
1. **GPU Compatibility**: Always check for CUDA/MPS/XPU/CPU availability
2. **Type Hints**: Use `Optional`, `Dict`, `Any`, `Tuple` from typing module
3. **Error Handling**: Use try-except with specific exceptions; use `logger` for logging
4. **Environment Variables**: Check `os.environ.get()` for configuration options
5. **Device Handling**: Use `.to(device)` pattern; support `cuda`, `mps`, `xpu`, `cpu`

### Platform-Specific Considerations
- **MPS (Apple Silicon)**: Disable `torch.compile`, disable quantization, use `mlx` backend
- **ROCm (AMD)**: May need `HSA_OVERRIDE_GFX_VERSION` environment variable
- **CPU**: Expect slower performance; disable flash attention

## Testing Strategy

### Existing Test Files
- `acestep/core/generation/handler/init_service_test.py`
- `acestep/core/generation/handler/lora_integration_test.py`
- `acestep/core/lora/service_test.py`

### Manual Testing Checklist
When making changes, verify:
1. **Target platform behavior** - Does the fix work on intended hardware?
2. **Non-target platforms unchanged** - No regressions on CUDA/CPU/MPS/XPU
3. **Error handling** - Graceful degradation when models unavailable
4. **GPU memory** - No OOM on tier3+ GPUs with default settings

### GPU Testing with Simulated Memory
```bash
# Test with 8GB GPU simulation
MAX_CUDA_VRAM=8 uv run acestep

# Test with 16GB GPU simulation  
MAX_CUDA_VRAM=16 uv run acestep
```

## Security Considerations

1. **Path Traversal**: Use `os.path.realpath()` and `os.path.commonpath()` for path validation
2. **API Authentication**: Check `ACESTEP_API_KEY` environment variable for API endpoints
3. **Temporary Files**: Use `tempfile.mkstemp()` for uploaded audio files
4. **Proxy Settings**: Clear `http_proxy`/`https_proxy` before Gradio launch to avoid conflicts

## Key Modules for Common Tasks

### Adding a New Generation Task Type
1. Add task type to `acestep/constants.py` (`TASK_TYPES`, `TASK_INSTRUCTIONS`)
2. Update task validation in `acestep/inference.py`
3. Add UI support in `acestep/gradio_ui/interfaces/generation.py`

### Modifying LoRA Training
1. Core logic: `acestep/training/trainer.py`
2. Configuration: `acestep/training/configs.py`
3. UI integration: `acestep/gradio_ui/interfaces/training.py`

### Adding New GPU Support
1. Detection: `acestep/gpu_config.py` (`get_gpu_memory_gb()`)
2. Tier config: `GPU_TIER_CONFIGS` dictionary
3. Platform checks: `is_mps_platform()`, `is_rocm_platform()` equivalents

### Adding API Endpoints
1. Define models in `acestep/api_server.py` (Pydantic BaseModel)
2. Implement endpoint in `create_app()` function
3. Add to Gradio routes in `acestep/gradio_ui/api_routes.py` if needed

## Documentation References

- User-facing docs: `docs/en/` (English), `docs/zh/` (Chinese), `docs/ja/` (Japanese), `docs/ko/` (Korean)
- Installation: `docs/en/INSTALL.md`
- API Guide: `docs/en/API.md`
- GPU Compatibility: `docs/en/GPU_COMPATIBILITY.md`
- Contributing Guidelines: `CONTRIBUTING.md`

## Important Notes for AI Agents

1. **Minimal Changes**: Make only the changes necessary to fix the issue. Avoid "drive-by" improvements.
2. **Platform Preservation**: When fixing CUDA issues, don't change MPS/CPU/XPU paths unless required.
3. **GPU Memory Awareness**: Changes affecting model loading or inference must consider 4GB-24GB+ GPU range.
4. **Backward Compatibility**: Maintain compatibility with existing checkpoints and LoRA weights.
5. **Testing Scope**: Verify your change works on target platform and doesn't break others.
6. **Constants**: Add new constants to `acestep/constants.py` rather than hardcoding values.
7. **Logging**: Use `from loguru import logger` instead of print statements.

## License

MIT License - See `LICENSE` file for details.
