"""
Build notebooks/signsense_colab.ipynb: the whole SignSense pipeline in one notebook.

The notebook first writes the project modules (src/*.py) with %%writefile cells,
so the code is readable inside the notebook, then runs every phase step by step.

Run this again whenever code in src/ changes:
    python scripts/build_full_notebook.py
"""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src import config
from scripts.build_colab_notebook import code_cell, markdown_cell

NOTEBOOK_PATH = config.BASE_DIR / "notebooks" / "signsense_colab.ipynb"

# Files written into the Colab runtime, in a sensible reading order
PROJECT_FILES = [
    "src/__init__.py", "src/config.py", "src/data_loader.py", "src/preprocess.py",
    "src/models.py", "src/train.py", "src/evaluate.py", "src/compare.py",
    "src/predict.py", "scripts/download_data.py", "data/signnames.csv",
]


def writefile_cell(relative_path):
    """Create a %%writefile cell that recreates one project file."""
    content = (config.BASE_DIR / relative_path).read_text(encoding="utf-8")
    return code_cell(f"%%writefile {relative_path}\n{content}")


def intro_cells():
    """Title, overview and setup cells."""
    return [
        markdown_cell(
            "# SignSense: Deep CNN-Based Traffic Sign Recognition\n\n"
            "**Deep Learning mini project** - Shayaan Shaikh, Yash Rane, Dishant Sasane\n\n"
            "This notebook runs the whole project end to end on the GTSRB dataset (43 classes):\n"
            "data exploration, a custom CNN (Model A), MobileNetV2 transfer learning (Model B),\n"
            "evaluation, comparison and predictions.\n\n"
            "**How to run:** `Runtime -> Change runtime type -> T4 GPU`, then `Runtime -> Run all`.\n"
            "Cell 3.1 asks for your Kaggle username and key (from `kaggle.json`).\n"
            "The full run takes roughly 30-40 minutes on a free T4 GPU."),
        markdown_cell("## 1. Setup"),
        code_cell("# Check the GPU and create the project folders\n"
                  "!nvidia-smi -L\n"
                  "import tensorflow as tf\n"
                  "print('TensorFlow', tf.__version__, '| GPU:', tf.config.list_physical_devices('GPU'))\n"
                  "!mkdir -p /content/signsense/src /content/signsense/scripts /content/signsense/data\n"
                  "%cd /content/signsense"),
    ]


def code_section_cells():
    """Cells that write every project module."""
    cells = [markdown_cell(
        "## 2. Project code\n\n"
        "Each cell below writes one module of the project. The same files are in the\n"
        "`src/` folder of the repository. `config.py` holds every path and hyperparameter.")]
    cells += [writefile_cell(path) for path in PROJECT_FILES]
    cells.append(code_cell("# Make the modules importable in this notebook\n"
                           "import os, sys\n"
                           "sys.path.insert(0, os.getcwd())\n"
                           "from src import config"))
    return cells


def data_cells():
    """Dataset download and exploration cells."""
    return [
        markdown_cell("## 3. Dataset\n\n### 3.1 Download GTSRB from Kaggle"),
        code_cell("# Kaggle login (values are in kaggle.json; the key stays hidden)\n"
                  "import getpass\n"
                  "os.environ['KAGGLE_USERNAME'] = input('Kaggle username: ')\n"
                  "os.environ['KAGGLE_KEY'] = getpass.getpass('Kaggle key: ')\n"
                  "!python scripts/download_data.py"),
        markdown_cell("### 3.2 Explore the data\n\n"
                      "Images are loaded with OpenCV, converted BGR -> RGB and resized to 32x32."),
        code_cell("from src.data_loader import (set_seeds, load_sign_names, load_train_data,\n"
                  "                             plot_sample_images, plot_class_distribution)\n"
                  "from src.preprocess import normalize_images, plot_augmented_examples\n\n"
                  "set_seeds()\n"
                  "config.RESULTS_DIR.mkdir(exist_ok=True)\n"
                  "sign_names = load_sign_names()\n"
                  "X, y = load_train_data()\n"
                  "print('Training images:', X.shape, '| labels:', y.shape)\n"
                  "plot_sample_images(X, y, sign_names, save_path=config.RESULTS_DIR / 'sample_images.png');"),
        markdown_cell("The classes are **imbalanced**: some have about 200 images, others over 2,000."),
        code_cell("plot_class_distribution(y, save_path=config.RESULTS_DIR / 'class_distribution.png');"),
        markdown_cell("### 3.3 Data augmentation\n\n"
                      "Random rotation (10 degrees), zoom (0.15), shift (0.1) and shear (0.1).\n"
                      "**No flips**: flipping would turn \"turn left\" into \"turn right\"."),
        code_cell("turn_left = normalize_images(X[y == 34][:1])[0]   # class 34 = Turn left ahead\n"
                  "plot_augmented_examples(turn_left, save_path=config.RESULTS_DIR / 'augmentation_examples.png');\n"
                  "del X, y   # free memory before training"),
    ]


def _show_results_code(model_name):
    """Code that evaluates a model and displays its saved plots and worst classes."""
    return ("from src.evaluate import evaluate_model\n"
            "from IPython.display import Image, display\n"
            "import pandas as pd\n\n"
            f"evaluate_model('{model_name}')\n"
            "for plot in ['curves', 'confusion_matrix', 'confusion_errors', 'misclassified']:\n"
            f"    display(Image(f'results/{{plot}}_{model_name}.png'))")


def model_cells():
    """Training and evaluation cells for both models."""
    return [
        markdown_cell("## 4. Model A: custom CNN\n\n"
                      "Two blocks of `Conv(3x3) -> Conv(3x3) -> MaxPool -> Dropout(0.25)` (32 then 64 filters),\n"
                      "then `Flatten -> Dense(256) -> Dropout(0.5) -> Dense(43, softmax)`, with\n"
                      "BatchNormalization after each convolution. Adam (lr 0.001), batch 64, up to 20 epochs,\n"
                      "EarlyStopping, ModelCheckpoint and ReduceLROnPlateau.\n\n"
                      "Set `TRAIN_BASELINE = True` to also train the plain CNN without BatchNorm (~10 more minutes)."),
        code_cell("from src.train import train_custom_cnn\n\n"
                  "TRAIN_BASELINE = False\n"
                  "if TRAIN_BASELINE:\n"
                  "    train_custom_cnn(use_batchnorm=False)\n"
                  "train_custom_cnn(use_batchnorm=True);"),
        markdown_cell("### 4.1 Evaluate Model A on the official test set"),
        code_cell(_show_results_code(config.MODEL_A_NAME)),
        markdown_cell("## 5. Model B: MobileNetV2 transfer learning\n\n"
                      "ImageNet-pretrained MobileNetV2 at 96x96 with a new head\n"
                      "(`GlobalAveragePooling -> Dropout(0.3) -> Dense(43)`).\n"
                      "Stage 1 trains the head with the base frozen; stage 2 fine-tunes the top 60 layers\n"
                      "with a learning rate of 1e-4. Data flows through a `tf.data` pipeline to save memory."),
        code_cell("from src.train import train_mobilenet\n"
                  "train_mobilenet();"),
        markdown_cell("### 5.1 Evaluate Model B on the official test set"),
        code_cell(_show_results_code(config.MODEL_B_NAME)),
    ]


def comparison_and_demo_cells():
    """Comparison, prediction demo and conclusion cells."""
    return [
        markdown_cell("## 6. Comparison"),
        code_cell("from src.compare import main as compare_models\n"
                  "compare_models()\n"
                  "display(Image('results/comparison_chart.png'))"),
        markdown_cell(
            "**Our results** (official test set, 12,630 images):\n\n"
            "| Model | Params | Test accuracy |\n|---|---|---|\n"
            "| Custom CNN (baseline) | 486K | 98.05% |\n"
            "| **Custom CNN + BatchNorm (Model A)** | 487K | **98.60%** |\n"
            "| MobileNetV2, light fine-tuning (run 1) | 2.31M | 66.75% |\n"
            "| MobileNetV2, fine-tuned (Model B) | 2.31M | 88.30% |\n\n"
            "The small custom CNN wins: ImageNet features suit large, detailed photos rather than tiny,\n"
            "blurry signs, and GlobalAveragePooling loses position information (left vs right arrows).\n"
            "Fine-tuning was essential for MobileNetV2. Exact numbers vary slightly between runs."),
        markdown_cell("## 7. Predict single images"),
        code_cell("%matplotlib inline\n"
                  "import random\n"
                  "import matplotlib.pyplot as plt\n"
                  "from pathlib import Path\n"
                  "from src.data_loader import find_image_folder\n"
                  "from src.predict import predict_image\n\n"
                  "test_df = pd.read_csv(config.TEST_CSV)\n"
                  "test_dir = find_image_folder(config.TEST_DIR)\n"
                  "fig, axes = plt.subplots(2, 4, figsize=(14, 7))\n"
                  "for ax, row in zip(axes.flat, test_df.sample(8, random_state=1).itertuples()):\n"
                  "    image_path = test_dir / Path(row.Path).name\n"
                  "    class_id, name, confidence, top3 = predict_image(image_path)\n"
                  "    ax.imshow(plt.imread(image_path))\n"
                  "    verdict = 'correct' if class_id == row.ClassId else 'WRONG'\n"
                  "    ax.set_title(f'{name}\\n{confidence:.0%} ({verdict})', fontsize=9)\n"
                  "    ax.axis('off')\n"
                  "plt.tight_layout()"),
        markdown_cell("## 8. Demo app\n\n"
                      "The Streamlit app (`app/app.py`) runs on a laptop, not in Colab:\n\n"
                      "```\nstreamlit run app/app.py\n```\n\n"
                      "Download `models/custom_cnn_bn.keras` and `models/mobilenetv2.keras` from this\n"
                      "runtime into the project's `models/` folder first (or use the models trained locally)."),
    ]


def has_outputs(path):
    """True if the notebook already contains cell outputs (e.g. a finished Colab run)."""
    if not path.exists():
        return False
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return any(cell.get("outputs") for cell in notebook["cells"])


def main():
    """Assemble and write the notebook."""
    if has_outputs(NOTEBOOK_PATH) and "--force" not in sys.argv:
        print(f"{NOTEBOOK_PATH.name} contains outputs from a run; not overwriting it. "
              "Use --force to replace it (the outputs will be lost).")
        return
    cells =intro_cells() + code_section_cells() + data_cells() + model_cells() \
        + comparison_and_demo_cells()
    notebook = {
        "cells": cells,
        "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
                     "kernelspec": {"display_name": "Python 3", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"Notebook written to {NOTEBOOK_PATH} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
