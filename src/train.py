"""
Train a model and save the best version plus its training history.

Run from the project root:
    python -m src.train                          # custom CNN (baseline)
    python -m src.train --batchnorm              # custom CNN + BatchNormalization
    python -m src.train --model mobilenetv2      # MobileNetV2 transfer learning (use a GPU)

Outputs:
    models/<name>.keras             best model (highest validation accuracy)
    results/history_<name>.json     loss/accuracy per epoch + training time
"""

import argparse
import json
import time

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from src import config
from src.data_loader import set_seeds
from src.models import build_custom_cnn, build_mobilenet, unfreeze_top_layers
from src.preprocess import create_augmenter, make_mobilenet_dataset, prepare_data


def get_callbacks(model_path, best_val_accuracy=None):
    """
    Return the callbacks used during training.

    best_val_accuracy: the best score so far (used in fine-tuning, so the saved
    model is only replaced if fine-tuning actually beats the first stage).
    """
    return [
        # Stop when validation loss stops improving, and go back to the best epoch
        EarlyStopping(monitor="val_loss", patience=config.EARLY_STOP_PATIENCE,
                      restore_best_weights=True, verbose=1),
        # Save the model every time validation accuracy reaches a new best
        ModelCheckpoint(str(model_path), monitor="val_accuracy", save_best_only=True,
                        initial_value_threshold=best_val_accuracy, verbose=1),
        # Halve the learning rate when validation loss gets stuck
        ReduceLROnPlateau(monitor="val_loss", factor=config.REDUCE_LR_FACTOR,
                          patience=config.REDUCE_LR_PATIENCE, verbose=1),
    ]


def save_history(history_dict, train_time_sec, save_path):
    """Save per-epoch metrics and total training time to a JSON file."""
    history_dict = {key: [float(v) for v in values] for key, values in history_dict.items()}
    history_dict["train_time_sec"] = round(train_time_sec, 1)
    with open(save_path, "w") as file:
        json.dump(history_dict, file, indent=2)
    print(f"History saved to {save_path}")


def _make_output_folders():
    """Create models/ and results/ if they do not exist yet."""
    config.MODELS_DIR.mkdir(exist_ok=True)
    config.RESULTS_DIR.mkdir(exist_ok=True)


def train_custom_cnn(use_batchnorm=False):
    """Train the custom CNN with data augmentation and save the results."""
    set_seeds()
    _make_output_folders()

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

    save_history(history.history, train_time, config.RESULTS_DIR / f"history_{model.name}.json")
    return model, history


def _merge_histories(first, second):
    """Join two Keras history dicts (stage 1 + stage 2) into one."""
    return {key: first[key] + second.get(key, []) for key in first}


def train_mobilenet():
    """
    Train MobileNetV2 in two stages and save the results.

    Stage 1: base frozen, train only the new head (MOBILENET_EPOCHS).
    Stage 2: unfreeze the top FINE_TUNE_LAYERS layers and fine-tune with a tiny learning rate.
    """
    set_seeds()
    _make_output_folders()

    # Keep images as uint8 at 96x96; tf.data converts them batch by batch
    X_train, X_val, _, y_train, y_val, _ = prepare_data(config.MOBILENET_IMG_SIZE, normalize=False)
    train_data = make_mobilenet_dataset(X_train, y_train, training=True)
    val_data = make_mobilenet_dataset(X_val, y_val)

    model, base_model = build_mobilenet()
    model.summary()
    model_path = config.MOBILENET_PATH
    start_time = time.time()

    print("\n=== Stage 1: training the new classifier head (MobileNetV2 base frozen) ===")
    stage1 = model.fit(train_data, validation_data=val_data, epochs=config.MOBILENET_EPOCHS,
                       callbacks=get_callbacks(model_path))

    print(f"\n=== Stage 2: fine-tuning the top {config.FINE_TUNE_LAYERS} layers ===")
    unfreeze_top_layers(model, base_model)
    stage2 = model.fit(train_data, validation_data=val_data,
                       initial_epoch=len(stage1.epoch),
                       epochs=len(stage1.epoch) + config.FINE_TUNE_EPOCHS,
                       callbacks=get_callbacks(model_path, max(stage1.history["val_accuracy"])))

    train_time = time.time() - start_time
    print(f"Training took {train_time / 60:.1f} minutes. Best model saved to {model_path}")
    history = _merge_histories(stage1.history, stage2.history)
    save_history(history, train_time, config.RESULTS_DIR / "history_mobilenetv2.json")
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a SignSense model")
    parser.add_argument("--model", choices=["custom_cnn", "mobilenetv2"], default="custom_cnn",
                        help="which model to train")
    parser.add_argument("--batchnorm", action="store_true",
                        help="custom CNN only: add BatchNormalization after each convolution")
    args = parser.parse_args()

    if args.model == "mobilenetv2":
        train_mobilenet()
    else:
        train_custom_cnn(use_batchnorm=args.batchnorm)
