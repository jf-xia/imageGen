# Boogu-Image-0.1 Turbo Demo

A demo for running the [Boogu-Image-0.1-Turbo-fp8](https://huggingface.co/Boogu/Boogu-Image-0.1-Turbo-fp8) model locally on Mac with Apple Silicon GPU acceleration.

## Setup

```bash
./setup.sh
source .venv/bin/activate
```

Or manually:
```bash
uv venv -p 3.11
uv pip install -r requirements.txt
source .venv/bin/activate
```

## Usage

Basic generation:
```bash
python demo.py
```

With custom prompt:
```bash
python demo.py --prompt "A beautiful landscape painting of mountains at sunset"
```

With all options:
```bash
python demo.py \
  --prompt "Your prompt here" \
  --output my_image.png \
  --steps 4 \
  --width 1024 \
  --height 1024 \
  --guidance 1.0
```

## Options

- `--prompt`: Text prompt for image generation
- `--output`: Output filename (default: `boogu_output.png`)
- `--steps`: Number of inference steps, 3-4 recommended for Turbo (default: 4)
- `--width`: Image width (default: 1024)
- `--height`: Image height (default: 1024)
- `--guidance`: Guidance scale, 1.0 for Turbo (default: 1.0)
