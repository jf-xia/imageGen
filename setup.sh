#!/bin/bash
set -e

echo "Setting up Boogu-Image-0.1 Turbo Demo..."

if ! command -v uv &> /dev/null; then
    echo "Error: uv not found. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "Creating virtual environment..."
uv venv -p 3.11

echo "Installing dependencies..."
uv pip install -r requirements.txt

echo "Cloning Boogu-Image repository..."
if [ ! -d "Boogu-Image" ]; then
    git clone https://github.com/boogu-project/Boogu-Image.git
fi

echo "Installing Boogu package..."
uv pip install -e ./Boogu-Image

echo ""
echo "Setup complete! Run the demo with:"
echo "  source .venv/bin/activate"
echo "  python demo.py"
