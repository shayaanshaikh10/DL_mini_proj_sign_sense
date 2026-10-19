"""
Preprocessing: normalisation, one-hot labels, train/validation split
and data augmentation.

Main function:
    prepare_data() -> X_train, X_val, X_test, y_train, y_val, y_test
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

from src import config
from src.data_loader import load_test_data, load_train_data


def normalize_images(X):
    """Convert pixel values from 0-255 integers to 0-1 floats."""
    return X.astype("float32") / 255.0


def one_hot_labels(y):
    """Turn class ids (e.g. 14) into one-hot vectors of length 43."""
    return to_categorical(y, config.NUM_CLASSES)


def split_train_val(X, y):
    """Split into 80% train / 20% validation, keeping class proportions equal (stratify)."""
    return train_test_split(
        X, y,
        test_size=config.VAL_SPLIT,
        stratify=y,
        random_state=config.RANDOM_SEED,
    )


def prepare_data(img_size=config.IMG_SIZE, normalize=True):
    """
    Load, split and one-hot encode the data (and normalise it for the custom CNN).

    Returns X_train, X_val, X_test (images) and y_train, y_val, y_test (one-hot labels).
    With normalize=False the images stay uint8 (0-255), which MobileNetV2 needs
    and which uses 4x less memory at 96x96.
    """
    X, y = load_train_data(img_size)
    X_test, y_test = load_test_data(img_size)

    # Split BEFORE anything else, so validation images never leak into training
    X_train, X_val, y_train, y_val = split_train_val(X, y)

    if normalize:
        X_train = normalize_images(X_train)
        X_val = normalize_images(X_val)
        X_test = normalize_images(X_test)

    y_train = one_hot_labels(y_train)
    y_val = one_hot_labels(y_val)
    y_test = one_hot_labels(y_test)

    return X_train, X_val, X_test, y_train, y_val, y_test


def create_augmenter():
    """
    Build an ImageDataGenerator that randomly rotates, zooms, shifts and shears images.

    No horizontal/vertical flips: flipping "turn left" would make it look
    like "turn right", which would teach the model wrong labels.
    """
    return ImageDataGenerator(
        rotation_range=config.ROTATION_RANGE,
        zoom_range=config.ZOOM_RANGE,
        width_shift_range=config.WIDTH_SHIFT_RANGE,
        height_shift_range=config.HEIGHT_SHIFT_RANGE,
        shear_range=config.SHEAR_RANGE,
        horizontal_flip=False,
        vertical_flip=False,
        fill_mode="nearest",
    )


def plot_augmented_examples(image, num_versions=8, save_path=None):
    """Show one original image next to several randomly augmented versions of it."""
    augmenter = create_augmenter()
    fig, axes = plt.subplots(1, num_versions + 1, figsize=(2 * (num_versions + 1), 2.4))

    axes[0].imshow(image)
    axes[0].set_title("Original", fontsize=9)

    for ax in axes[1:]:
        # random_transform applies one random augmentation to a single image
        augmented = augmenter.random_transform(image)
        ax.imshow(np.clip(augmented, 0, 1))
        ax.set_title("Augmented", fontsize=9)

    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def create_augmentation_layers():
    """
    The same augmentation as create_augmenter(), written as Keras layers so it can
    run inside a tf.data pipeline (used for MobileNetV2). Still no flips.
    """
    return tf.keras.Sequential([
        layers.RandomRotation(config.ROTATION_RANGE / 360),   # factor = fraction of a full turn
        layers.RandomZoom(config.ZOOM_RANGE),
        layers.RandomTranslation(config.HEIGHT_SHIFT_RANGE, config.WIDTH_SHIFT_RANGE),
    ], name="augmentation")


def make_mobilenet_dataset(X, y=None, training=False):
    """
    Build a tf.data pipeline for MobileNetV2 from uint8 images.

    Images are converted to float one batch at a time, so the full 96x96 dataset
    never has to sit in memory as float32. Training batches are shuffled and augmented.
    """
    dataset = tf.data.Dataset.from_tensor_slices(X if y is None else (X, y))
    if training:
        dataset = dataset.shuffle(config.SHUFFLE_BUFFER, seed=config.RANDOM_SEED)
    dataset = dataset.batch(config.BATCH_SIZE)

    augmenter = create_augmentation_layers() if training else None

    def prepare_batch(images, labels=None):
        images = tf.cast(images, tf.float32)
        if augmenter is not None:
            images = augmenter(images, training=True)
        images = preprocess_input(images)   # MobileNetV2 expects pixels scaled to [-1, 1]
        return images if labels is None else (images, labels)

    dataset = dataset.map(prepare_batch, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.prefetch(tf.data.AUTOTUNE)   # prepare the next batch while the GPU trains
