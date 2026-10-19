"""
Download the GTSRB dataset from Kaggle into the data/ folder.

Requirements:
    - The `kaggle` package (in requirements.txt)
    - Your Kaggle API token saved as ~/.kaggle/kaggle.json

Run from the project root:
    python scripts/download_data.py
"""

import subprocess
import sys
from pathlib import Path

# Make "src" importable when running this script directly
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src import config


def download_gtsrb():
    """Download and unzip GTSRB into config.DATA_DIR using the Kaggle CLI."""
    if config.TRAIN_DIR.exists():
        print(f"Dataset already present at {config.TRAIN_DIR}, skipping download.")
        return

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {config.KAGGLE_DATASET} ... (about 600 MB)")

    # Same as running: kaggle datasets download -d <slug> -p data --unzip
    command = [
        sys.executable, "-m", "kaggle", "datasets", "download",
        "-d", config.KAGGLE_DATASET,
        "-p", str(config.DATA_DIR),
        "--unzip",
    ]
    subprocess.run(command, check=True)
    print("Download finished.")


if __name__ == "__main__":
    download_gtsrb()
