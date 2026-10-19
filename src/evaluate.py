"""
Evaluate a trained model on the official test set.

Run from the project root:
    python -m src.evaluate                       # custom CNN
    python -m src.evaluate --model custom_cnn_bn # BatchNorm variant
    python -m src.evaluate --model mobilenetv2   # transfer learning model

Saves to results/ (with the model name in each file name):
    curves_<name>.png                  training vs validation accuracy and loss
    confusion_matrix_<name>.png        43x43 heatmap (row-normalised)
    confusion_errors_<name>.png        same matrix with the diagonal removed
    misclassified_<name>.png           10 wrong predictions
    classification_report_<name>.csv   precision / recall / F1 per class
    worst_classes_<name>.csv           5 classes with the lowest F1
    metrics_<name>.json                test accuracy and loss
"""

import argparse
import json

import matplotlib
matplotlib.use("Agg")   # save plots to files without opening windows
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model

from src import config
from src.data_loader import load_sign_names, load_test_data, set_seeds
from src.preprocess import make_mobilenet_dataset, normalize_images

TRAIN_COLOR = "#3b6fb6"   # blue
VAL_COLOR = "#d9822b"     # orange (blue/orange stays distinct for colour-blind readers)


def load_history(model_name):
    """Read the training history JSON saved by train.py."""
    with open(config.RESULTS_DIR / f"history_{model_name}.json") as file:
        return json.load(file)


def plot_training_curves(history, model_name, save_path):
    """Plot training vs validation accuracy (left) and loss (right)."""
    epochs = range(1, len(history["accuracy"]) + 1)
    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, metric, title in [(ax_acc, "accuracy", "Accuracy"), (ax_loss, "loss", "Loss")]:
        ax.plot(epochs, history[metric], color=TRAIN_COLOR, linewidth=2, label="Train")
        ax.plot(epochs, history[f"val_{metric}"], color=VAL_COLOR, linewidth=2, label="Validation")
        ax.set_title(f"{title} per epoch")
        ax.set_xlabel("Epoch")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))   # whole epochs only
        ax.set_ylabel(title)
        ax.grid(color="#e0e0e0", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)

    fig.suptitle(f"Training curves: {model_name}")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(cm, title, save_path):
    """Draw a 43x43 confusion matrix heatmap (rows = true class, columns = predicted)."""
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm, cmap="Blues", square=True, cbar=True, ax=ax,
                xticklabels=range(config.NUM_CLASSES), yticklabels=range(config.NUM_CLASSES))
    ax.set_title(title)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_confusion_plots(y_true, y_pred, model_name):
    """Save the normalised confusion matrix and an 'errors only' version."""
    cm = confusion_matrix(y_true, y_pred, labels=range(config.NUM_CLASSES))
    # Divide each row by its total, so each cell is "fraction of class i predicted as j"
    cm_normalised = cm / cm.sum(axis=1, keepdims=True)
    plot_confusion_matrix(cm_normalised, f"Confusion matrix (row-normalised): {model_name}",
                          config.RESULTS_DIR / f"confusion_matrix_{model_name}.png")

    # Most predictions are correct, so the diagonal hides the mistakes.
    # Removing it makes the confusions between classes visible.
    errors_only = cm.copy()
    np.fill_diagonal(errors_only, 0)
    plot_confusion_matrix(errors_only, f"Mistakes only (counts, diagonal removed): {model_name}",
                          config.RESULTS_DIR / f"confusion_errors_{model_name}.png")
    return cm


def save_classification_report(y_true, y_pred, sign_names, model_name):
    """Save per-class precision, recall and F1 to CSV and return it as a DataFrame."""
    report = classification_report(y_true, y_pred, labels=range(config.NUM_CLASSES),
                                   output_dict=True, zero_division=0)
    rows = []
    for class_id in range(config.NUM_CLASSES):
        scores = report[str(class_id)]
        rows.append({"ClassId": class_id, "SignName": sign_names[class_id],
                     "precision": scores["precision"], "recall": scores["recall"],
                     "f1": scores["f1-score"], "support": int(scores["support"])})
    report_df = pd.DataFrame(rows).round(4)
    report_df.to_csv(config.RESULTS_DIR / f"classification_report_{model_name}.csv", index=False)
    return report_df


def _top_confusion(counts, class_id, sign_names):
    """Describe the largest off-diagonal count, e.g. '26: Traffic signals (22x)'."""
    counts = counts.copy()
    counts[class_id] = 0                       # ignore correct predictions
    other = int(np.argmax(counts))
    if counts[other] == 0:
        return "-"
    return f"{other}: {sign_names[other]} ({counts[other]}x)"


def find_worst_classes(report_df, cm, sign_names, model_name, count=5):
    """
    Return the classes with the lowest F1 and their most common confusions.

    predicted_as: what this sign is wrongly predicted as (hurts recall)
    mistaken_for_it: which other sign is wrongly predicted as this one (hurts precision)
    """
    worst = report_df.nsmallest(count, "f1").copy()
    worst["predicted_as"] = [_top_confusion(cm[c], c, sign_names) for c in worst["ClassId"]]
    worst["mistaken_for_it"] = [_top_confusion(cm[:, c], c, sign_names) for c in worst["ClassId"]]
    worst.to_csv(config.RESULTS_DIR / f"worst_classes_{model_name}.csv", index=False)
    return worst


def plot_misclassified(images, y_true, y_pred, y_prob, sign_names, model_name, count=10):
    """Show `count` randomly chosen wrong predictions with true and predicted labels."""
    wrong_indices = np.where(y_true != y_pred)[0]
    chosen = np.random.choice(wrong_indices, size=min(count, len(wrong_indices)), replace=False)
    fig, axes = plt.subplots(2, count // 2, figsize=(16, 8))

    for ax, index in zip(axes.flat, chosen):
        ax.imshow(images[index])
        ax.set_title(f"True: {sign_names[y_true[index]]}\n"
                     f"Pred: {sign_names[y_pred[index]]} ({y_prob[index].max():.0%})",
                     fontsize=8)
        ax.axis("off")

    fig.suptitle(f"Misclassified test images: {model_name}")
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / f"misclassified_{model_name}.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def report_results(model_name, y_true, y_prob, display_images):
    """
    Create every evaluation output from the model's predicted probabilities.

    y_true: true class ids (N,), y_prob: predicted probabilities (N, 43),
    display_images: test images for plotting (N, H, W, 3).
    Works for any model, so Phase 5 (MobileNetV2) reuses it unchanged.
    """
    set_seeds()
    sign_names = load_sign_names()
    y_pred = np.argmax(y_prob, axis=1)
    accuracy = float(np.mean(y_pred == y_true))

    plot_training_curves(load_history(model_name), model_name,
                         config.RESULTS_DIR / f"curves_{model_name}.png")
    cm = save_confusion_plots(y_true, y_pred, model_name)
    report_df = save_classification_report(y_true, y_pred, sign_names, model_name)
    worst = find_worst_classes(report_df, cm, sign_names, model_name)
    plot_misclassified(display_images, y_true, y_pred, y_prob, sign_names, model_name)

    with open(config.RESULTS_DIR / f"metrics_{model_name}.json", "w") as file:
        json.dump({"model": model_name, "test_accuracy": round(accuracy, 4),
                   "num_test_images": int(len(y_true)),
                   "num_wrong": int(np.sum(y_pred != y_true))}, file, indent=2)
    return accuracy, worst


def predict_test_set(model, model_name):
    """Return (test images, true labels, predicted probabilities) using the model's own preprocessing."""
    if model_name.startswith("mobilenet"):
        X_test, y_test = load_test_data(config.MOBILENET_IMG_SIZE)
        y_prob = model.predict(make_mobilenet_dataset(X_test), verbose=0)
    else:
        X_test, y_test = load_test_data(config.IMG_SIZE)
        y_prob = model.predict(normalize_images(X_test), batch_size=config.BATCH_SIZE, verbose=0)
    return X_test, y_test, y_prob


def evaluate_model(model_name="custom_cnn"):
    """Load a saved model, predict the test set, and save all evaluation outputs."""
    model = load_model(config.MODELS_DIR / f"{model_name}.keras")
    X_test, y_test, y_prob = predict_test_set(model, model_name)
    accuracy, worst = report_results(model_name, y_test, y_prob, X_test)

    print(f"\n{model_name}: test accuracy = {accuracy:.2%} "
          f"({int(np.sum(np.argmax(y_prob, axis=1) != y_test))} wrong out of {len(y_test)})")
    print("\n5 worst classes (lowest F1):")
    print(worst[["ClassId", "SignName", "precision", "recall", "f1",
                 "predicted_as", "mistaken_for_it"]].to_string(index=False))
    print(f"\nPlots and tables saved to {config.RESULTS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained SignSense model")
    parser.add_argument("--model", default="custom_cnn",
                        help="model name: custom_cnn, custom_cnn_bn or mobilenetv2")
    args = parser.parse_args()
    evaluate_model(args.model)
