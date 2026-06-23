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

echo "Installing boogu package from local boogu_pkg/..."
uv pip install -e ./boogu_pkg

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Download model weights (if not already present):"
echo "     python download_model.py"
echo "  2. Run the demo:"
echo "     source .venv/bin/activate"
echo "     python demo.py"
