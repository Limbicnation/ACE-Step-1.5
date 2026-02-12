# ACE-Step 1.5

## Project Overview

ACE-Step 1.5 is a high-performance, open-source music foundation model designed for music generation. It utilizes a hybrid architecture combining a Language Model (LM) for high-level planning (lyrics, structure, metadata) and a Diffusion Transformer (DiT) for high-fidelity audio synthesis.

**Key Features:**

* **Text-to-Music:** Generate full songs from text descriptions.
* **Hybrid Architecture:** LM planner + DiT synthesizer.
* **Performance:** Optimized for consumer hardware (runs on RTX 3090, <4GB VRAM capable).
* **Versatility:** Supports cover generation, vocal-to-BGM, repainting, and track separation.
* **Fine-tuning:** Supports lightweight personalization via LoRA.

## Building and Running

This project uses `uv` for package management and task execution.

### Prerequisites

* Python 3.11
* CUDA-enabled GPU (recommended)
* `uv` package manager

### Installation

1. Clone the repository.
2. Install dependencies using `uv`:

    ```bash
    # We use a specific stable Torch version for RTX 4090/Ubuntu 24.04 stability
    uv sync
    ```

### Troubleshooting & Known Issues

#### CUDA / PyTorch Stability

* **Issue:** `CUBLAS_STATUS_INVALID_VALUE` when using `bf16`.
* **Root Cause:** Nightly Torch builds (e.g., 2.10.0+cu128) have broken GEMM implementations for some kernels.
* **Fix:** Ensure `pyproject.toml` uses a stable release (e.g., `2.6.0+cu124`).

#### Flash-Attention Conflicts

* **Issue:** `undefined symbol` errors during `diffusers` import.
* **Fix:** We have removed the hard `flash-attn` dependency from `nano-vllm`. The system will automatically fall back to Torch's optimized **SDPA (Scaled Dot Product Attention)**.

#### Memory (VRAM) Management

* **Issue:** OOM during Language Model (LM) generation.
* **Fix:** For the 0.6B LM, ensure at least **10GB** of VRAM headroom is available. Use `MAX_CUDA_VRAM=10` to simulate limits if testing on smaller GPUs or multi-process environments.

#### Port Conflicts

* **Issue:** Port 7860 already in use.
* **Fix:** Launch on a different port using `--port 7865`.

### Running the Application

**Gradio Web UI (Recommended):**
To start the interactive web interface:

```bash
uv run acestep
```

Access at: `http://localhost:7860`

**REST API Server:**
To start the API server:

```bash
uv run acestep-api
```

Access at: `http://localhost:8001`

**Model Management:**
Models are downloaded automatically on the first run. To manually download:

```bash
uv run acestep-download
```

## Development Conventions

* **Package Management:** `uv` is the standard for dependency management and running scripts.
* **Code Structure:**
  * `acestep/`: Core source code package.
  * `acestep/gradio_ui/`: Web interface implementation.
  * `acestep/training/`: Training and fine-tuning logic.
  * `examples/`: JSON examples for generation prompts.
* **Documentation:** Extensive documentation is available in the `docs/` directory, supporting English, Chinese, and Japanese.

## Key Files

* `README.md`: Primary entry point and documentation.
* `pyproject.toml`: Project metadata and dependencies.
* `acestep/acestep_v15_pipeline.py`: Main generation pipeline logic.
* `acestep/inference.py`: Inference execution handling.
* `acestep/api_server.py`: REST API server implementation.
* `acestep/gradio_ui/__init__.py`: Entry point for the Gradio UI.
