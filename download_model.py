#!/usr/bin/env python3
"""Download Boogu-Image-0.1-Turbo-fp8 model weights from HuggingFace."""

import os
import sys
import subprocess


MODEL_ID = "Boogu/Boogu-Image-0.1-Turbo-fp8"
LOCAL_DIR = os.path.join(os.path.dirname(__file__), "models", "Boogu-Image-0.1-Turbo-fp8")


def main():
    if os.path.exists(LOCAL_DIR) and os.path.isfile(os.path.join(LOCAL_DIR, "model_index.json")):
        print(f"Model already exists at {LOCAL_DIR}")
        print("Delete it and re-run to re-download.")
        return

    print(f"Downloading {MODEL_ID} to {LOCAL_DIR}...")
    print("This will download ~21GB of model weights.")
    print()

    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            MODEL_ID,
            local_dir=LOCAL_DIR,
        )
        print(f"\nDownload complete! Model saved to: {LOCAL_DIR}")
    except ImportError:
        print("huggingface_hub not installed. Falling back to huggingface-cli...")
        subprocess.run(
            ["huggingface-cli", "download", MODEL_ID, "--local-dir", LOCAL_DIR],
            check=True,
        )
    except Exception as e:
        print(f"Download failed: {e}")
        print()
        print("You can also download manually:")
        print(f"  huggingface-cli download {MODEL_ID} --local-dir {LOCAL_DIR}")
        sys.exit(1)


if __name__ == "__main__":
    main()
