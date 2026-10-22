"""
Draw the SignSense block diagram (used in the report, section 3.1).

Run from the project root:
    python scripts/make_block_diagram.py
Saves results/block_diagram.png
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src import config

BOX_COLOR = "#e8eef8"     # light blue fill
EDGE_COLOR = "#3b6fb6"    # blue border
TEXT_COLOR = "#1a1a1a"


def draw_box(ax, x, y, width, height, title, lines):
    """Draw a rounded box with a bold title and smaller description lines."""
    box = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.02,rounding_size=0.08",
                         facecolor=BOX_COLOR, edgecolor=EDGE_COLOR, linewidth=1.5)
    ax.add_patch(box)
    ax.text(x + width / 2, y + height - 0.22, title, ha="center", va="top",
            fontsize=10.5, fontweight="bold", color=TEXT_COLOR)
    ax.text(x + width / 2, y + height - 0.55, "\n".join(lines), ha="center", va="top",
            fontsize=8.5, color=TEXT_COLOR, linespacing=1.4)


def draw_arrow(ax, start, end):
    """Draw an arrow between two (x, y) points."""
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="-|>", color="#444444", linewidth=1.4))


def main():
    """Draw all boxes and arrows and save the figure."""
    fig, ax = plt.subplots(figsize=(11, 4.0))
    ax.set_xlim(0, 11.2)
    ax.set_ylim(0, 4.0)
    ax.axis("off")

    # Row 1 (left to right): input -> preprocessing -> augmentation -> classifier
    draw_box(ax, 0.1, 2.5, 2.3, 1.4, "Input image", ["GTSRB dataset (43 classes)", "or test image in the app"])
    draw_box(ax, 2.9, 2.5, 2.5, 1.4, "Preprocessing", ["OpenCV: BGR to RGB", "Resize 32x32 / 96x96", "Scale pixels, one-hot labels"])
    draw_box(ax, 5.9, 2.5, 2.5, 1.4, "Data augmentation", ["Training images only", "Rotate, zoom, shift, shear", "No flips"])

    # Classifier group: one of the two models
    group = FancyBboxPatch((8.85, 0.95), 2.25, 2.95, boxstyle="round,pad=0.02,rounding_size=0.1",
                           facecolor="none", edgecolor="#888888", linestyle="--", linewidth=1.2)
    ax.add_patch(group)
    ax.text(9.97, 3.78, "Classifier (one of)", ha="center", va="top", fontsize=9, style="italic")
    draw_box(ax, 9.0, 2.35, 1.95, 1.1, "Model A", ["Custom CNN + BatchNorm"])
    draw_box(ax, 9.0, 1.1, 1.95, 1.1, "Model B", ["MobileNetV2 (fine-tuned)"])

    # Row 2 (right to left): softmax -> prediction -> output
    draw_box(ax, 5.9, 0.1, 2.5, 1.4, "Softmax output", ["Probabilities for", "43 sign classes"])
    draw_box(ax, 2.9, 0.1, 2.5, 1.4, "Prediction", ["Sign name + confidence", "Top-3, warning below 50%"])
    draw_box(ax, 0.1, 0.1, 2.3, 1.4, "Output", ["Streamlit web app", "Evaluation metrics"])

    draw_arrow(ax, (2.45, 3.2), (2.85, 3.2))
    draw_arrow(ax, (5.45, 3.2), (5.85, 3.2))
    draw_arrow(ax, (8.45, 3.2), (8.82, 3.2))
    draw_arrow(ax, (9.97, 0.92), (8.45, 0.8))
    draw_arrow(ax, (5.85, 0.8), (5.45, 0.8))
    draw_arrow(ax, (2.85, 0.8), (2.45, 0.8))

    fig.tight_layout()
    save_path = config.RESULTS_DIR / "block_diagram.png"
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Saved {save_path}")


if __name__ == "__main__":
    main()
