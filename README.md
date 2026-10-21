# SignSense: Deep CNN-Based Traffic Sign Recognition

SignSense classifies an image of a road traffic sign into one of **43 classes**
(stop, yield, speed limits, turn signs, ...) using deep learning. It compares a
custom CNN trained from scratch with MobileNetV2 transfer learning, and includes
a Streamlit web demo.

Deep Learning mini project (B.E. AI & Data Science) by **Shayaan Shaikh, Yash Rane and Dishant Sasane**.

## Results

Official GTSRB test set (12,630 images, never used for training):

| Model | Input | Parameters | Test accuracy |
|---|---|---|---|
| Custom CNN (baseline) | 32x32 | 486K | 98.05% |
| **Custom CNN + BatchNorm (Model A)** | 32x32 | 487K | **98.60%** |
| MobileNetV2, light fine-tuning | 96x96 | 2.31M | 66.75% |
| MobileNetV2, fine-tuned (Model B) | 96x96 | 2.31M | 88.30% |

The small custom CNN beats MobileNetV2 while being about 5x smaller.
See [results/comparison_analysis.md](results/comparison_analysis.md) for the full analysis.

## Dataset

[GTSRB - German Traffic Sign Recognition Benchmark](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)
on Kaggle: 39,209 training images and 12,630 test images in 43 classes. The classes are
imbalanced (210 to 2,250 images per class). Training data is split 80/20 into train and
validation sets (stratified, `random_state=42`).

## Project structure

```
├── app/app.py                  Streamlit demo app
├── data/                       dataset (downloaded) + signnames.csv
├── models/                     trained .keras models (not in git, see below)
├── notebooks/
│   ├── signsense_colab.ipynb        full pipeline, end to end (Colab)
│   └── train_mobilenet_colab.ipynb  MobileNetV2 training only (Colab, works from VS Code)
├── results/                    plots, metrics, comparison table, report content
├── scripts/
│   ├── download_data.py        download GTSRB from Kaggle
│   ├── check_dataset.py        verify the dataset structure
│   ├── explore_data.py         sample images, class distribution, augmentation plots
│   ├── build_full_notebook.py  regenerate notebooks/signsense_colab.ipynb from src/
│   ├── build_colab_notebook.py regenerate notebooks/train_mobilenet_colab.ipynb
│   └── import_colab_outputs.py bring MobileNetV2 results back from the Colab notebook
└── src/
    ├── config.py               all paths and hyperparameters
    ├── data_loader.py          load images and labels (with .npz cache)
    ├── preprocess.py           normalisation, one-hot, split, augmentation, tf.data
    ├── models.py               build_custom_cnn(), build_mobilenet()
    ├── train.py                train a model, save best weights + history
    ├── evaluate.py             test accuracy, curves, confusion matrix, worst classes
    ├── compare.py              comparison table and chart
    └── predict.py              predict_image() for a single image
```

## Option 1: run everything on Google Colab (easiest)

1. Open [notebooks/signsense_colab.ipynb](notebooks/signsense_colab.ipynb) in
   [Google Colab](https://colab.research.google.com) (File -> Upload notebook).
2. `Runtime -> Change runtime type -> T4 GPU`.
3. `Runtime -> Run all`. When asked, enter your Kaggle username and key
   (from your `kaggle.json`, see "Kaggle API key" below).

The notebook writes the project code, downloads the data, trains and evaluates both
models, compares them and shows predictions. It takes roughly 30-40 minutes.

## Option 2: run locally

Requires Python 3.11 (TensorFlow 2.16 does not support Python 3.13+).

**1. Install** (with [uv](https://docs.astral.sh/uv/)):

```
uv sync
.venv\Scripts\activate          (Windows)
source .venv/bin/activate       (Linux / macOS)
```

or with pip:

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**2. Kaggle API key:** on kaggle.com go to Settings -> API -> Create New Token and save
`kaggle.json` to `C:\Users\<you>\.kaggle\` (Windows) or `~/.kaggle/` (Linux / macOS).

**3. Download and check the data:**

```
python scripts/download_data.py
python scripts/check_dataset.py
python scripts/explore_data.py
```

**4. Train and evaluate** (each command run from the project root):

```
python -m src.train --batchnorm                 Model A, about 15 min on a laptop CPU
python -m src.evaluate --model custom_cnn_bn

python -m src.train                             optional: baseline CNN without BatchNorm
python -m src.evaluate --model custom_cnn

python -m src.train --model mobilenetv2         Model B: needs a GPU (about 9 min on a T4;
python -m src.evaluate --model mobilenetv2        several hours on a CPU), see Colab notes

python -m src.compare                           comparison table and chart
```

The first data load reads all images and saves a cache in `data/`, so later runs start in seconds.
All plots and tables are saved to `results/`.

**MobileNetV2 without a local GPU:** open `notebooks/train_mobilenet_colab.ipynb` with a Colab
runtime (in the browser, or in VS Code with the Google Colab extension) and run all cells.
Then save the notebook and run `python scripts/import_colab_outputs.py` to copy the trained
model and results into `models/` and `results/`.

## Demo app

```
streamlit run app/app.py
```

Opens at http://localhost:8501. Upload a sign image (or click **Try a random test image**),
choose a model in the sidebar, and see the predicted sign, the confidence, a "not sure"
warning below 50% and the top-3 predictions. The models expect a cropped sign like the
GTSRB images, not a full street scene.

To predict from the command line:

```
python -m src.predict data/Test/00000.png
python -m src.predict data/Test/00000.png --model mobilenetv2
```

## Notes

- Trained models (`models/*.keras`) are not stored in git because of their size. Train them
  with the commands above, or download them from a Colab run.
- Validation accuracy (99.9%) is higher than test accuracy (98.6%) because GTSRB's training
  set contains ~30 near-identical photos of each physical sign, so a random split puts very
  similar images in both train and validation. The official test set is the honest measure.
- Random seeds are fixed (`RANDOM_SEED = 42` in `src/config.py`), but GPU training can still
  vary slightly between runs.
