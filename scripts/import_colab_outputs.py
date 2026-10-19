"""
Recreate the MobileNetV2 model and results from the saved Colab notebook.

Cell 8 of notebooks/train_mobilenet_colab.ipynb prints the outputs as text.
After saving the notebook (Ctrl+S), run from the project root:
    python scripts/import_colab_outputs.py
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
START_MARKER = "SIGNSENSE_OUTPUTS_START"
END_MARKER = "SIGNSENSE_OUTPUTS_END"


def collect_output_text(notebook):
    """Join the printed text of every cell output in the notebook."""
    pieces = []
    for cell in notebook["cells"]:
        for output in cell.get("outputs", []):
            text = output.get("text", "")
            pieces.append("".join(text) if isinstance(text, list) else text)
    return "".join(pieces)


def main():
    """Find the encoded outputs in the notebook and extract them into the project."""
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    all_text = collect_output_text(notebook)

    if START_MARKER not in all_text or END_MARKER not in all_text:
        print("No outputs found. Run cell 8 in the notebook, save it (Ctrl+S), then try again.")
        return

    encoded = all_text.split(START_MARKER)[1].split(END_MARKER)[0]
    encoded = "".join(encoded.split())   # remove line breaks and spaces
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(encoded))) as archive:
        archive.extractall(config.BASE_DIR)
        for name in archive.namelist():
            print(f"  extracted {name}")
    print("Done. You can now run: python -m src.evaluate --model mobilenetv2")


if __name__ == "__main__":
    main()
