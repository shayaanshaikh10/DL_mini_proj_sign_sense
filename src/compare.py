"""
Phase 6: compare all trained models in one table and one chart.

Run from the project root (after training and evaluating every model):
    python -m src.compare

Saves:
    results/comparison.csv          the comparison table
    results/comparison_chart.png    test accuracy per model
"""

import json

import matplotlib
matplotlib.use("Agg")   # save plots to files without opening windows
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import load_model

from src import config

# The models to compare, with details that cannot be read from the saved files
MODELS = [
    {"name": "custom_cnn", "label": "Custom CNN (baseline)",
     "input": f"{config.IMG_SIZE}x{config.IMG_SIZE}", "hardware": "laptop CPU"},
    {"name": "custom_cnn_bn", "label": "Custom CNN + BatchNorm (Model A)",
     "input": f"{config.IMG_SIZE}x{config.IMG_SIZE}", "hardware": "laptop CPU"},
    {"name": "mobilenetv2", "label": "MobileNetV2 fine-tuned (Model B)",
     "input": f"{config.MOBILENET_IMG_SIZE}x{config.MOBILENET_IMG_SIZE}",
     "hardware": "Colab T4 GPU"},
]

# MobileNetV2 run 1 (20 layers fine-tuned, lr 1e-5) was overwritten by run 2.
# Its numbers were copied from the Colab training and evaluation output.
MOBILENET_RUN1 = {
    "Model": "MobileNetV2 light fine-tuning (run 1)", "Input size": "96x96",
    "Params": None, "Train time (min)": 5.6, "Hardware": "Colab T4 GPU",
    "Best val accuracy (%)": 77.93, "Test accuracy (%)": 66.75, "Model file (MB)": None,
}


def _read_json(path):
    """Load a JSON file."""
    with open(path) as file:
        return json.load(file)


def collect_model_row(model_info):
    """Gather all comparison numbers for one trained model."""
    name = model_info["name"]
    model_path = config.MODELS_DIR / f"{name}.keras"
    history = _read_json(config.RESULTS_DIR / f"history_{name}.json")
    metrics = _read_json(config.RESULTS_DIR / f"metrics_{name}.json")
    model = load_model(model_path)

    return {
        "Model": model_info["label"],
        "Input size": model_info["input"],
        "Params": model.count_params(),
        "Train time (min)": round(history["train_time_sec"] / 60, 1),
        "Hardware": model_info["hardware"],
        "Best val accuracy (%)": round(max(history["val_accuracy"]) * 100, 2),
        "Test accuracy (%)": round(metrics["test_accuracy"] * 100, 2),
        "Model file (MB)": round(model_path.stat().st_size / 1e6, 1),
    }


def build_comparison_table():
    """Build the comparison DataFrame for all models (including MobileNetV2 run 1)."""
    rows = [collect_model_row(info) for info in MODELS]
    # MobileNetV2 run 1 has the same architecture, so the same parameter count
    run1 = dict(MOBILENET_RUN1, Params=rows[-1]["Params"])
    rows.insert(2, run1)
    return pd.DataFrame(rows)


def plot_accuracy_chart(table, save_path):
    """Horizontal bar chart of test accuracy per model, with the value on each bar."""
    fig, ax = plt.subplots(figsize=(9, 3.6))
    labels = table["Model"][::-1]
    values = table["Test accuracy (%)"][::-1]

    bars = ax.barh(labels, values, color="#3b6fb6", height=0.6)
    ax.bar_label(bars, labels=[f"{v:.2f}%" for v in values], padding=4, fontsize=9)

    ax.set_xlim(0, 105)
    ax.set_xlabel("Test accuracy (%) on 12,630 GTSRB test images")
    ax.set_title("Test accuracy by model")
    ax.grid(axis="x", color="#e0e0e0", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    """Create and save the comparison table and chart."""
    table = build_comparison_table()
    table.to_csv(config.RESULTS_DIR / "comparison.csv", index=False)
    plot_accuracy_chart(table, config.RESULTS_DIR / "comparison_chart.png")

    pd.set_option("display.width", 200)
    print(table.to_string(index=False))
    print(f"\nSaved comparison.csv and comparison_chart.png to {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
