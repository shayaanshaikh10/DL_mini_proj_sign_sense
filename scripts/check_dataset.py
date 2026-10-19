"""
Check that the downloaded GTSRB dataset has the structure we expect.

This script only READS files. It prints a short report.

Run from the project root:
    python scripts/check_dataset.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src import config

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".ppm"}


def count_images(folder):
    """Count image files directly inside a folder."""
    return sum(1 for f in folder.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS)


def check_train_folder():
    """Check Train/ has 43 numbered class folders and count images per class."""
    print("\n--- Train folder ---")
    if not config.TRAIN_DIR.exists():
        print(f"MISSING: {config.TRAIN_DIR}")
        return

    class_folders = [f for f in config.TRAIN_DIR.iterdir() if f.is_dir()]
    # Sort numerically (0, 1, 2, ... 42), not alphabetically (0, 1, 10, 11, ...)
    class_folders.sort(key=lambda f: int(f.name))
    print(f"Class folders found: {len(class_folders)} (expected {config.NUM_CLASSES})")

    counts = {int(f.name): count_images(f) for f in class_folders}
    print(f"Total training images: {sum(counts.values())}")
    print(f"Smallest class: {min(counts, key=counts.get)} ({min(counts.values())} images)")
    print(f"Largest class:  {max(counts, key=counts.get)} ({max(counts.values())} images)")


def check_test_data():
    """Check Test.csv columns and that the image paths in it exist."""
    print("\n--- Test data ---")
    if not config.TEST_CSV.exists():
        print(f"MISSING: {config.TEST_CSV}")
        return

    test_df = pd.read_csv(config.TEST_CSV)
    print(f"Test.csv columns: {list(test_df.columns)}")
    print(f"Rows in Test.csv: {len(test_df)}")
    print(f"Example Path value: {test_df['Path'].iloc[0]}")

    # Paths in Test.csv are relative to the data/ folder (e.g. "Test/00000.png")
    first_image = config.DATA_DIR / test_df["Path"].iloc[0]
    print(f"First test image exists (data/ + Path): {first_image.exists()}")
    if config.TEST_DIR.exists():
        print(f"Images in Test/: {count_images(config.TEST_DIR)}")


def check_other_items():
    """Report Meta/, Train.csv and any duplicate lowercase folders."""
    print("\n--- Other items in data/ ---")
    print(f"Meta/ exists: {config.META_DIR.exists()}")
    print(f"Train.csv exists: {(config.DATA_DIR / 'Train.csv').exists()}")
    print(f"Everything in data/: {sorted(p.name for p in config.DATA_DIR.iterdir())}")


if __name__ == "__main__":
    check_train_folder()
    check_test_data()
    check_other_items()
