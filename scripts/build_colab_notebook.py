"""
Build notebooks/train_mobilenet_colab.ipynb with the project code embedded inside it.

Why: when the notebook runs on a Colab runtime from VS Code, the runtime cannot
see files on your laptop and the upload button does not work. So the code is
zipped, turned into text (base64) and stored in a notebook cell.

Run this again whenever code in src/ or scripts/ changes:
    python scripts/build_colab_notebook.py
"""

import base64
import io
import json
import sys
import zipfile
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src import config

NOTEBOOK_PATH = config.BASE_DIR / "notebooks" / "train_mobilenet_colab.ipynb"


def zip_project_code():
    """Zip src/, scripts/ and signnames.csv in memory and return it as base64 text."""
    files = sorted(config.BASE_DIR.glob("src/*.py")) + sorted(config.BASE_DIR.glob("scripts/*.py"))
    files.append(config.SIGNNAMES_CSV)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for file in files:
            archive.write(file, file.relative_to(config.BASE_DIR).as_posix())
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def markdown_cell(text):
    """Create a notebook markdown cell."""
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code_cell(code):
    """Create a notebook code cell."""
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": code}


def build_cells(code_b64):
    """Return the list of notebook cells."""
    return [
        markdown_cell(
            "# SignSense - Phase 5: train MobileNetV2 on a Colab GPU\n\n"
            "Works in the browser **and** in VS Code (Colab kernel). No file uploads needed.\n\n"
            "1. Choose a Colab runtime with a **T4 GPU**.\n"
            "2. Run the cells from top to bottom. Cell 5 shows live training progress.\n"
            "3. Have your Kaggle username and key ready (open `kaggle.json` on your laptop)."),
        code_cell("# 1. Check that a GPU is available\n"
                  "!nvidia-smi -L\n"
                  "import tensorflow as tf\n"
                  "print('TensorFlow', tf.__version__)\n"
                  "print('GPU:', tf.config.list_physical_devices('GPU'))"),
        code_cell("# 2. Unpack the project code (stored below as zipped text, made by\n"
                  "#    scripts/build_colab_notebook.py on the laptop)\n"
                  "import base64, io, os, zipfile\n"
                  f"CODE_ZIP = '{code_b64}'\n"
                  "os.makedirs('/content/signsense', exist_ok=True)\n"
                  "zipfile.ZipFile(io.BytesIO(base64.b64decode(CODE_ZIP))).extractall('/content/signsense')\n"
                  "%cd /content/signsense\n"
                  "!ls -R src scripts"),
        code_cell("# 3. Kaggle login (values are in kaggle.json on your laptop; the key is hidden)\n"
                  "import getpass\n"
                  "os.environ['KAGGLE_USERNAME'] = input('Kaggle username: ')\n"
                  "os.environ['KAGGLE_KEY'] = getpass.getpass('Kaggle key: ')"),
        code_cell("# 4. Download the GTSRB dataset (~600 MB)\n"
                  "!python scripts/download_data.py"),
        code_cell("# 5. Train MobileNetV2 (live progress below)\n"
                  "#    Reads all images at 96x96 first (a few minutes), then\n"
                  "#    Stage 1: 10 epochs training the new head\n"
                  "#    Stage 2: 15 epochs fine-tuning the top 60 MobileNetV2 layers\n"
                  "!python -m src.train --model mobilenetv2"),
        code_cell("# 6. Evaluate on the official test set\n"
                  "!python -m src.evaluate --model mobilenetv2"),
        code_cell("# 7. Quick look at the curves and mistakes\n"
                  "from IPython.display import Image, display\n"
                  "display(Image('results/curves_mobilenetv2.png'))\n"
                  "display(Image('results/misclassified_mobilenetv2.png'))"),
        markdown_cell(
            "## 8. Bring the results back to your laptop\n\n"
            "The next cell prints the trained model and results as one long block of text.\n"
            "After it finishes: **save this notebook (Ctrl+S)**, then on your laptop run\n\n"
            "`python scripts/import_colab_outputs.py`\n\n"
            "It reads that text from the saved notebook and recreates `models/mobilenetv2.keras` "
            "and the `results/*mobilenetv2*` files."),
        code_cell("# 8. Print the outputs as text (do not clear this output before saving!)\n"
                  "import glob\n"
                  "buffer = io.BytesIO()\n"
                  "with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:\n"
                  "    for path in glob.glob('models/mobilenetv2*.keras') + glob.glob('results/*mobilenetv2*'):\n"
                  "        archive.write(path)\n"
                  "print('SIGNSENSE_OUTPUTS_START')\n"
                  "print(base64.b64encode(buffer.getvalue()).decode('ascii'))\n"
                  "print('SIGNSENSE_OUTPUTS_END')"),
    ]


def main():
    """Write the notebook file."""
    notebook = {
        "cells": build_cells(zip_project_code()),
        "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
                     "kernelspec": {"display_name": "Python 3", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1))
    print(f"Notebook written to {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
