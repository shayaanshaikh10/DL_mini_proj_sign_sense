"""
Load GTSRB images and labels from disk.

Main functions:
    load_train_data()  -> X, y from data/Train/<class_id>/*.png
    load_test_data()   -> X, y from data/Test + Test.csv
    load_sign_names()  -> {class_id: "sign name"}

Loading ~52,000 images takes a minute, so the result is cached in a
.npz file inside data/. The next load reads the cache in a few seconds.
"""

import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from src import config


def set_seeds(seed=config.RANDOM_SEED):
    """Fix random seeds so results are the same every run."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_sign_names():
    """Return a dictionary mapping class id (int) to sign name (str)."""
    names_df = pd.read_csv(config.SIGNNAMES_CSV)
    return dict(zip(names_df["ClassId"], names_df["SignName"]))


def load_image(image_path, img_size=config.IMG_SIZE):
    """Read one image, convert BGR -> RGB, and resize to img_size x img_size."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    # OpenCV loads colours as BGR; the rest of our code (and Matplotlib) expects RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_AREA)
    return image


def _read_train_folders(img_size):
    """Read every image in Train/<class_id>/ and return (X, y) arrays."""
    images = []
    labels = []
    # Sort numerically: 0, 1, 2, ... 42 (alphabetical would give 0, 1, 10, 11, ...)
    for class_id in range(config.NUM_CLASSES):
        class_folder = config.TRAIN_DIR / str(class_id)
        for image_path in sorted(class_folder.glob("*.png")):
            images.append(load_image(image_path, img_size))
            labels.append(class_id)
        print(f"  class {class_id:2d}: done", end="\r")
    print()
    return np.array(images, dtype=np.uint8), np.array(labels, dtype=np.int64)


def _read_test_csv(img_size):
    """Read every test image listed in Test.csv and return (X, y) arrays."""
    test_df = pd.read_csv(config.TEST_CSV)
    images = []
    # Paths in Test.csv look like "Test/00000.png", relative to data/
    for relative_path in test_df["Path"]:
        images.append(load_image(config.DATA_DIR / relative_path, img_size))
    labels = test_df["ClassId"].to_numpy(dtype=np.int64)
    return np.array(images, dtype=np.uint8), labels


def _load_with_cache(split_name, read_function, img_size):
    """Load (X, y) from a .npz cache if it exists, otherwise read images and save the cache."""
    cache_path = config.DATA_DIR / f"cache_{split_name}_{img_size}.npz"
    if cache_path.exists():
        print(f"Loading {split_name} data from cache: {cache_path.name}")
        cached = np.load(cache_path)
        return cached["X"], cached["y"]

    print(f"Reading {split_name} images from disk (first time only)...")
    X, y = read_function(img_size)
    np.savez_compressed(cache_path, X=X, y=y)
    return X, y


def load_train_data(img_size=config.IMG_SIZE):
    """Return training images X (N, img_size, img_size, 3) uint8 and labels y (N,)."""
    return _load_with_cache("train", _read_train_folders, img_size)


def load_test_data(img_size=config.IMG_SIZE):
    """Return test images X (N, img_size, img_size, 3) uint8 and labels y (N,)."""
    return _load_with_cache("test", _read_test_csv, img_size)


def plot_sample_images(X, y, sign_names, num_samples=10, save_path=None):
    """Show a grid of random training images with their sign names."""
    random_indices = np.random.choice(len(X), size=num_samples, replace=False)
    fig, axes = plt.subplots(2, num_samples // 2, figsize=(14, 7))

    for ax, index in zip(axes.flat, random_indices):
        ax.imshow(X[index])
        ax.set_title(f"{y[index]}: {sign_names[y[index]]}", fontsize=8, wrap=True)
        ax.axis("off")

    fig.suptitle("Random training samples")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_class_distribution(y, save_path=None):
    """Bar chart of how many training images each class has (shows the imbalance)."""
    counts = np.bincount(y, minlength=config.NUM_CLASSES)
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.bar(range(config.NUM_CLASSES), counts, color="#3b6fb6", width=0.8)
    ax.axhline(counts.mean(), color="#555555", linestyle="--", linewidth=1)
    ax.text(config.NUM_CLASSES - 0.5, counts.mean(), f" mean = {counts.mean():.0f}",
            va="bottom", ha="right", fontsize=9, color="#333333")

    ax.set_title(f"Images per class: {counts.min()} to {counts.max()} (imbalanced)")
    ax.set_xlabel("Class ID")
    ax.set_ylabel("Number of images")
    ax.set_xticks(range(config.NUM_CLASSES))
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="y", color="#e0e0e0", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
