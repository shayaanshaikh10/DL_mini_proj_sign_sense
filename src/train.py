"""
Train the custom CNN and save the best model plus its training history.

Run from the project root:
    python -m src.train                 # plain custom CNN
    python -m src.train --batchnorm     # optional variant with BatchNormalization

Outputs:
    models/custom_cnn.keras             best model (highest validation accuracy)
    results/history_custom_cnn.json     loss/accuracy per epoch + training time
"""

import argparse
import json
import time

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from src import config
from src.data_loader import set_seeds
from src.models import build_custom_cnn
from src.preprocess import create_augmenter, prepare_data


def get_callbacks(model_path):
    """Return the callbacks used during training."""
    return [
        # Stop when validation loss stops improving, and go back to the best epoch
        EarlyStopping(monitor="val_loss", patience=config.EARLY_STOP_PATIENCE,
                      restore_best_weights=True, verbose=1),
        # Save the model every time validation accuracy reaches a new best
        ModelCheckpoint(str(model_path), monitor="val_accuracy",
                        save_best_only=True, verbose=1),
        # Halve the learning rate when validation loss gets stuck
        ReduceLROnPlateau(monitor="val_loss", factor=config.REDUCE_LR_FACTOR,
                          patience=config.REDUCE_LR_PATIENCE, verbose=1),
    ]


def save_history(history, train_time_sec, save_path):
    """Save per-epoch metrics and total training time to a JSON file."""
    history_dict = {key: [float(v) for v in values] for key, values in history.history.items()}
    history_dict["train_time_sec"] = round(train_time_sec, 1)
    with open(save_path, "w") as file:
        json.dump(history_dict, file, indent=2)
    print(f"History saved to {save_path}")


def train_custom_cnn(use_batchnorm=False):
    """Train the custom CNN with data augmentation and save the results."""
    set_seeds()
    config.MODELS_DIR.mkdir(exist_ok=True)
    config.RESULTS_DIR.mkdir(exist_ok=True)

    X_train, X_val, _, y_train, y_val, _ = prepare_data()   # test set is NOT used here
    model = build_custom_cnn(use_batchnorm)
    model.summary()

    model_path = config.MODELS_DIR / f"{model.name}.keras"
    # The augmenter creates new randomly rotated/zoomed/shifted batches every epoch
    train_batches = create_augmenter().flow(X_train, y_train,
                                            batch_size=config.BATCH_SIZE, seed=config.RANDOM_SEED)

    start_time = time.time()
    history = model.fit(
        train_batches,
        validation_data=(X_val, y_val),   # validation images are NOT augmented
        epochs=config.EPOCHS,
        callbacks=get_callbacks(model_path),
    )
    train_time = time.time() - start_time
    print(f"Training took {train_time / 60:.1f} minutes. Best model saved to {model_path}")

    save_history(history, train_time, config.RESULTS_DIR / f"history_{model.name}.json")
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the SignSense custom CNN")
    parser.add_argument("--batchnorm", action="store_true",
                        help="add BatchNormalization after each convolution")
    args = parser.parse_args()
    train_custom_cnn(use_batchnorm=args.batchnorm)
