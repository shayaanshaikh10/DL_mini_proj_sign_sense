"""
Predict the traffic sign in a single image.

    predict_image(image) -> (class_id, class_name, confidence, top3)

`image` can be a file path, a PIL image or a NumPy array (RGB).
The image gets exactly the same preprocessing as during training.

Run from the project root:
    python -m src.predict data/Test/00000.png
    python -m src.predict data/Test/00000.png --model mobilenetv2
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model

from src import config
from src.data_loader import load_sign_names

_loaded_models = {}   # keeps each model in memory after the first load


def get_model(model_name=config.MODEL_A_NAME):
    """Load a saved model once and reuse it on later calls."""
    if model_name not in _loaded_models:
        _loaded_models[model_name] = load_model(config.MODELS_DIR / f"{model_name}.keras")
    return _loaded_models[model_name]


def to_rgb_array(image):
    """Convert a file path, PIL image or NumPy array into an RGB uint8 array (H, W, 3)."""
    if isinstance(image, (str, Path)):
        image = Image.open(image)
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))   # also handles grayscale and transparent PNGs

    image = np.asarray(image)
    if image.ndim == 2:                          # grayscale array -> 3 channels
        image = np.stack([image] * 3, axis=-1)
    if image.shape[-1] == 4:                     # drop the transparency channel
        image = image[..., :3]
    return image.astype(np.uint8)


def preprocess_for_model(rgb_image, model_name):
    """Resize and scale one RGB image the same way as in training; returns a batch of 1."""
    is_mobilenet = model_name.startswith("mobilenet")
    size = config.MOBILENET_IMG_SIZE if is_mobilenet else config.IMG_SIZE
    resized = cv2.resize(rgb_image, (size, size), interpolation=cv2.INTER_AREA)
    batch = resized[np.newaxis].astype("float32")   # shape (1, size, size, 3)

    if is_mobilenet:
        return preprocess_input(batch)               # scale to [-1, 1]
    return batch / 255.0                             # scale to [0, 1]


def predict_image(image, model_name=config.MODEL_A_NAME):
    """
    Predict the sign in one image.

    Returns (class_id, class_name, confidence, top3), where confidence is 0-1
    and top3 is a list of (class_id, class_name, confidence) tuples.
    """
    sign_names = load_sign_names()
    batch = preprocess_for_model(to_rgb_array(image), model_name)
    probabilities = get_model(model_name).predict(batch, verbose=0)[0]

    # Indices of the 3 highest probabilities, largest first
    top_ids = np.argsort(probabilities)[::-1][:config.TOP_K]
    top3 = [(int(i), sign_names[int(i)], float(probabilities[i])) for i in top_ids]

    class_id, class_name, confidence = top3[0]
    return class_id, class_name, confidence, top3


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict the traffic sign in an image")
    parser.add_argument("image_path", help="path to an image file")
    parser.add_argument("--model", default=config.MODEL_A_NAME,
                        help="custom_cnn_bn (default), custom_cnn or mobilenetv2")
    args = parser.parse_args()

    class_id, class_name, confidence, top3 = predict_image(args.image_path, args.model)
    print(f"Prediction: {class_name} (class {class_id}), confidence {confidence:.1%}")
    for rank, (top_id, top_name, top_conf) in enumerate(top3, start=1):
        print(f"  {rank}. {top_name:45s} {top_conf:.1%}")
