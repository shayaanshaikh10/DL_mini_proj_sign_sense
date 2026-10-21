"""
Render the key functions of the project as code images (used in the report, chapter 5).

The code is read directly from src/, so the images always match the real code.

Run from the project root:
    python scripts/make_code_images.py
Saves results/code/<name>.png
"""

import ast
import sys
from pathlib import Path

from pygments import highlight
from pygments.formatters import ImageFormatter
from pygments.lexers import PythonLexer

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src import config

OUTPUT_DIR = config.RESULTS_DIR / "code"

# image name -> (source file, functions to include)
CODE_EXCERPTS = {
    "code_augmentation": ("src/preprocess.py", ["split_train_val", "create_augmenter"]),
    "code_custom_cnn": ("src/models.py", ["_conv_block", "build_custom_cnn"]),
    "code_mobilenet": ("src/models.py", ["build_mobilenet", "unfreeze_top_layers"]),
    "code_training": ("src/train.py", ["get_callbacks", "train_custom_cnn"]),
    "code_predict": ("src/predict.py", ["preprocess_for_model", "predict_image"]),
}


def function_without_docstring(source_lines, node):
    """Return a function's source lines, leaving out its docstring (comments are kept)."""
    lines = source_lines[node.lineno - 1:node.end_lineno]
    first = node.body[0]
    has_docstring = (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                     and isinstance(first.value.value, str))
    if has_docstring:
        start, end = first.lineno - node.lineno, first.end_lineno - node.lineno
        lines = lines[:start] + lines[end + 1:]
    return "\n".join(lines)


def extract_functions(file_path, function_names):
    """Return the source code of the named functions from a Python file (without docstrings)."""
    source = (config.BASE_DIR / file_path).read_text(encoding="utf-8")
    source_lines = source.splitlines()
    pieces = [function_without_docstring(source_lines, node) for node in ast.parse(source).body
              if isinstance(node, ast.FunctionDef) and node.name in function_names]
    return f"# {file_path}\n\n" + "\n\n\n".join(pieces)


def render_code_image(code, save_path):
    """Render code as a PNG with syntax highlighting and line numbers."""
    formatter = ImageFormatter(font_name="Consolas", font_size=26, line_numbers=True,
                               line_number_bg="#f0f0f0", line_number_fg="#888888",
                               image_pad=24, line_pad=6, style="default")
    save_path.write_bytes(highlight(code, PythonLexer(), formatter))


def main():
    """Create one image per excerpt."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, (file_path, functions) in CODE_EXCERPTS.items():
        code = extract_functions(file_path, functions)
        render_code_image(code, OUTPUT_DIR / f"{name}.png")
        print(f"Saved {name}.png ({code.count(chr(10)) + 1} lines)")


if __name__ == "__main__":
    main()
