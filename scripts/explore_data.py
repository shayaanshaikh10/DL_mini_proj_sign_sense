"""
Phase 2: load the data, run preprocessing, and save the exploration plots.

Run from the project root:
    python scripts/explore_data.py

Saves to results/:
    sample_images.png, class_distribution.png, augmentation_examples.png
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # save plots to files without opening windows

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src import config
from src.data_loader import (load_sign_names, load_train_data, plot_class_distribution,
                             plot_sample_images, set_seeds)
from src.preprocess import normalize_images, plot_augmented_examples, prepare_data


def main():
    """Create the three exploration plots and print the data shapes."""
    set_seeds()
    config.RESULTS_DIR.mkdir(exist_ok=True)
    sign_names = load_sign_names()

    # 1. Raw data: sample images and class counts
    X, y = load_train_data()
    print(f"Raw training data: X {X.shape}, y {y.shape}")
    plot_sample_images(X, y, sign_names, save_path=config.RESULTS_DIR / "sample_images.png")
    plot_class_distribution(y, save_path=config.RESULTS_DIR / "class_distribution.png")

    # 2. Augmentation preview on one "Turn left ahead" sign (class 34):
    #    a directional sign shows why we never flip images
    turn_left_images = X[y == 34]
    example_image = normalize_images(turn_left_images[:1])[0]
    plot_augmented_examples(example_image,
                            save_path=config.RESULTS_DIR / "augmentation_examples.png")

    # 3. Full preprocessing pipeline: check shapes and value ranges
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data()
    print(f"X_train {X_train.shape}  y_train {y_train.shape}")
    print(f"X_val   {X_val.shape}  y_val   {y_val.shape}")
    print(f"X_test  {X_test.shape}  y_test  {y_test.shape}")
    print(f"Pixel range: {X_train.min():.2f} to {X_train.max():.2f}, dtype {X_train.dtype}")
    print(f"Plots saved to {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
