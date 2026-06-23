# Boogu-Image-0.1 Turbo Demo

A demo for running the [Boogu-Image-0.1-Turbo-fp8](https://huggingface.co/Boogu/Boogu-Image-0.1-Turbo-fp8) model locally on Mac with Apple Silicon GPU acceleration.

## Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.11
- [uv](https://docs.astral.sh/uv/) package manager
- ~21GB disk space for model weights

## Setup

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Run setup script
./setup.sh
source .venv/bin/activate
```

## Download Model Weights

```bash
python download_model.py
```

This downloads ~21GB of model weights to `models/Boogu-Image-0.1-Turbo-fp8/`.

## Usage

Basic generation:
```bash
python demo.py
```

With custom prompt:
```bash
python demo.py --prompt "A cyberpunk city at night with neon lights"
```

With all options:
```bash
python demo.py \
  --prompt "Your prompt here" \
  --output my_image.png \
  --steps 4 \
  --width 1024 \
  --height 1024 \
  --device mps \
  --seed 42
```

## Options

- `--prompt`: Text prompt for image generation
- `--output`: Output filename (default: `boogu_output.png`)
- `--steps`: Number of inference steps, 3-4 recommended for Turbo (default: 4)
- `--width`: Image width (default: 1024)
- `--height`: Image height (default: 1024)
- `--device`: Device to use: `mps`, `cuda:0`, or `cpu` (default: auto-detect)
- `--seed`: Random seed for reproducibility (default: 42)

## Performance

| Device | 1024x1024, 4 steps |
|--------|-------------------|
| MPS (Apple Silicon) | ~45s |
| CPU | ~210s |

## Known Limitations

- **MPS dtype**: MPS backend does not support bfloat16 matmul. The demo automatically uses float16 on MPS, which may cause minor quality differences compared to bfloat16 on CUDA.
- **FP8 dequantization**: The fp8 model weights are dequantized to float16 on load, which takes extra memory.
- **No image editing**: This demo only supports text-to-image generation. Image editing requires the separate Edit model.
- **Model download size**: The full model is ~21GB. The `mllm` component (Qwen3-VL) alone is ~10GB.

## Project Structure

```
imageGen/
├── demo.py              # Main inference script
├── download_model.py    # Model download script
├── requirements.txt     # Python dependencies
├── setup.sh             # Environment setup
├── README.md            # This file
├── boogu_pkg/           # Boogu Python package (with MPS patches)
└── models/              # Model weights (git-ignored)
```
