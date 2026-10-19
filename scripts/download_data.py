"""
Download the GTSRB dataset from Kaggle into the data/ folder.

Requirements:
    - The `kaggle` package (in requirements.txt; already installed on Colab)
    - Kaggle credentials: either ~/.kaggle/kaggle.json, or the environment
      variables KAGGLE_USERNAME and KAGGLE_KEY (used in the Colab notebook)

Run from the project root:
    python scripts/download_data.py
"""

import sys
from pathlib import Path

# Make "src" importable when running this script directly
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src import config


def download_gtsrb():
    """Download and unzip GTSRB into config.DATA_DIR using the Kaggle Python API."""
    if config.TRAIN_DIR.exists():
        print(f"Dataset already present at {config.TRAIN_DIR}, skipping download.")
        return

    # Imported here because importing kaggle immediately looks for credentials
    from kaggle.api.kaggle_api_extended import KaggleApi

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()   # reads kaggle.json or KAGGLE_USERNAME / KAGGLE_KEY

    print(f"Downloading {config.KAGGLE_DATASET} ... (about 600 MB)")
    api.dataset_download_files(config.KAGGLE_DATASET, path=str(config.DATA_DIR),
                               unzip=True, quiet=False)
    print("Download finished.")


if __name__ == "__main__":
    download_gtsrb()
