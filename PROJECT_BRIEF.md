# SignSense: Deep CNN-Based Traffic Sign Recognition

## 1. Context

- **Type:** College mini project for the subject **Deep Learning** (B.E. AI & Data Science, final year).
- **Team:** 3 students (Shayaan Shaikh, Yash Rane, Dishant Sasane).
- **Skill level:** Beginner to intermediate in Python and deep learning. Code must be easy to read and understand.
- **Deliverables:** working code (Jupyter/Colab notebook plus a small Python package), trained model, web demo, report and PPT content.
- **Constraint:** the project should be "not too complex". Prefer simple, well-explained solutions over clever ones.

## 2. Problem Statement

Build a deep learning system that looks at an image of a road traffic sign and correctly classifies it into one of **43 classes** (stop, yield, speed limits, turn signs, etc.). The system should be accurate, easy to demo, and well evaluated.

## 3. Objectives

1. Build and train a custom **CNN** from scratch that reaches **95%+ test accuracy**.
2. Build a second model using **transfer learning (MobileNetV2)** and compare it to the custom CNN.
3. Evaluate properly: accuracy and loss curves, confusion matrix, per-class precision/recall/F1, and examples of wrong predictions.
4. Build a **web demo** (Streamlit preferred, Gradio acceptable) where a user uploads a sign image and gets the predicted sign name plus confidence.
5. Produce clean, commented code and a short README so the project is reproducible.

## 4. Dataset

- **Name:** GTSRB (German Traffic Sign Recognition Benchmark), from Kaggle (for example the dataset slug `meowmeowmeowmeowmeow/gtsrb-german-traffic-sign`; verify the exact structure after downloading).
- **Size:** about 39,000 training images and about 12,600 test images, 43 classes.
- **Expected structure (verify and adapt):**
  - `Train/<class_id>/*.png` where `class_id` is a folder named `0` to `42`
  - `Test/*.png` with labels in `Test.csv` (columns include `ClassId` and `Path`)
  - `Meta/` and `Meta.csv` (reference images of each sign)
- **Known issues to handle:**
  - Class imbalance (some classes have about 200 images, others about 2,000). Show a bar chart and mention it in the report.
  - Image sizes vary, and some images are dark or blurry.
  - Class folder names are strings: sort them numerically (`0, 1, 2, ... 42`), not alphabetically.
- **Class names:** see `data/signnames.csv` (columns: `ClassId,SignName`).

## 5. Tech Stack

- **Language:** Python 3.10+
- **Deep learning:** TensorFlow / Keras
- **Data and utilities:** NumPy, pandas, OpenCV (`cv2`), scikit-learn, Pillow
- **Visualization:** Matplotlib, Seaborn
- **Demo app:** Streamlit (preferred) or Gradio
- **Training environment:** Google Colab (free T4 GPU). Code should also run locally on CPU, only slower.
- **Do not use:** exotic libraries, custom training loops, or complex architectures (no Vision Transformers, no object detection in the main scope).

## 6. Project Structure

```
├── PROJECT_BRIEF.md
├── README.md
├── requirements.txt
├── data/            (Train/, Test/, Test.csv, signnames.csv)
├── notebooks/signsense_colab.ipynb
├── src/             config.py, data_loader.py, preprocess.py, models.py,
│                    train.py, evaluate.py, predict.py
├── scripts/         download_data.py, check_dataset.py
├── models/          saved .keras files
├── results/         plots, confusion matrices, comparison table
└── app/app.py       Streamlit demo
```

## 7. Detailed Requirements

### 7.1 Data loading
- Load every training image with OpenCV, convert **BGR to RGB**, resize to **32x32** for the custom CNN (a configurable `IMG_SIZE`).
- Store images in a NumPy array `X` (shape `N x 32 x 32 x 3`) and labels in `y`.
- Show: 10 random sample images with their sign names, and a bar chart of images per class.

### 7.2 Preprocessing
- Convert pixel values to `float32` and divide by 255.
- One-hot encode labels with `to_categorical(y, 43)`.
- Split: **80% train / 20% validation**, using `train_test_split(..., stratify=y, random_state=42)`.
- Test data: load from `Test.csv` using the same preprocessing. Never train on test data.

### 7.3 Data augmentation
- Use `ImageDataGenerator` or Keras preprocessing layers:
  - rotation about 10 degrees, zoom about 0.15, width/height shift about 0.1, shear about 0.1
- **Do NOT use horizontal or vertical flip** (it changes the meaning of signs, e.g. turn left vs turn right).
- Show a small grid of augmented examples.

### 7.4 Model A: Custom CNN
```
Input (32, 32, 3)
Conv2D(32, 3x3, relu) -> Conv2D(32, 3x3, relu) -> MaxPooling2D -> Dropout(0.25)
Conv2D(64, 3x3, relu) -> Conv2D(64, 3x3, relu) -> MaxPooling2D -> Dropout(0.25)
Flatten -> Dense(256, relu) -> Dropout(0.5) -> Dense(43, softmax)
```
- Optimizer: Adam (learning rate 0.001)
- Loss: categorical cross-entropy; metric: accuracy
- Epochs: 15 to 25, batch size 64
- Callbacks: `EarlyStopping(patience=5, restore_best_weights=True)`, `ModelCheckpoint` (save best), optionally `ReduceLROnPlateau`
- Print `model.summary()` and save the training history.
- Optional improvement: add `BatchNormalization` after convolution layers and report whether it helps.

### 7.5 Model B: Transfer Learning (MobileNetV2)
- Resize images to **96x96** (or 128x128), and use MobileNetV2's `preprocess_input`.
- Load `MobileNetV2(weights="imagenet", include_top=False)`, **freeze the base**, add `GlobalAveragePooling2D -> Dropout(0.3) -> Dense(43, softmax)`.
- Train the head for about 5 to 10 epochs. Optionally unfreeze the last 20 layers and fine-tune with a low learning rate (1e-5) for a few epochs.
- Make sure memory use stays reasonable (use a `tf.data` pipeline or batch generator instead of loading all large images into memory if needed).

### 7.6 Evaluation (for both models)
- Plot training vs validation **accuracy** and **loss** curves.
- Report accuracy on the official **test set**.
- **Confusion matrix** (43x43 heatmap), plus a `classification_report` (precision, recall, F1 per class).
- List the **5 worst classes** (lowest F1) and explain likely reasons (similar-looking signs, blur, darkness).
- Show 10 **misclassified images** with true vs predicted label.
- Final **comparison table** saved to `results/comparison.csv`:

| Model | Input size | Params | Train time | Test accuracy | Model file size |
|-------|-----------|--------|-----------|---------------|-----------------|
| Custom CNN | 32x32 | ? | ? | ? | ? |
| MobileNetV2 | 96x96 | ? | ? | ? | ? |

### 7.7 Prediction function
- `predict_image(image)` accepts a file path, PIL image or NumPy array, applies the same preprocessing as training, and returns `(class_id, class_name, confidence, top3)`.

### 7.8 Web demo (Streamlit)
- Title "SignSense", short description.
- File uploader (jpg/png), shows the uploaded image.
- Shows predicted sign name, confidence %, and a **top-3 predictions bar chart**.
- Sidebar dropdown to choose the model (Custom CNN or MobileNetV2).
- Load the model once (`@st.cache_resource`).
- Handle errors (invalid file, very low confidence: show a "not sure" warning below about 50%).

## 8. Coding Style Rules (important)

- **Beginner-friendly code:** simple functions, clear variable names, no clever one-liners.
- Add **short comments** explaining what each block does and why, and a docstring for each function.
- Keep functions small (about 30 lines or fewer). Use `config.py` for all constants (no magic numbers).
- Set random seeds (`np.random.seed(42)`, `tf.random.set_seed(42)`) for reproducibility.
- Use relative paths (via `config.py`), not hardcoded absolute paths.
- Provide `requirements.txt` with pinned or compatible versions.
- When something fails, explain the cause in simple words and fix it.

## 9. Workflow

Work in **phases**. After each phase: summarize what was done in 3 to 5 lines, show how to run/verify it, and **wait for confirmation** before starting the next phase.

1. **Setup:** folder structure, `requirements.txt`, `config.py`, `signnames.csv`; check the dataset structure.
2. **Data:** `data_loader.py` and `preprocess.py`, plus exploration plots (samples and class distribution).
3. **Custom CNN:** `models.py` (Model A), `train.py`, train it, save the best model and history.
4. **Evaluation:** `evaluate.py` with all plots, metrics and misclassified examples for Model A.
5. **Transfer learning:** Model B, train, and evaluate with the same `evaluate.py`.
6. **Comparison:** comparison table and a short written analysis.
7. **Demo app:** `predict.py` and `app/app.py` (Streamlit), with run instructions.
8. **Colab notebook and docs:** `signsense_colab.ipynb` end to end, `README.md`, and report/PPT content.

## 10. Report and PPT Content (Phase 8)

Create `results/report_content.md` with ready-to-paste text for: Abstract (150 words); Introduction and motivation; Problem statement and objectives; Literature survey (3 to 5 related works; citations must be verified by the team, none invented); Dataset description; Methodology; Implementation details; Results and discussion; Custom CNN vs MobileNetV2 comparison; Conclusion; Limitations and future scope (YOLO real-time detection, video input, Indian traffic sign datasets, mobile/edge deployment); and a viva Q&A list of 15 questions with short answers.

## 11. Success Criteria

- Custom CNN test accuracy is **95% or higher** (stretch goal: 97%+).
- MobileNetV2 results are reported honestly, even if lower than the custom CNN.
- The Streamlit app runs with one command (`streamlit run app/app.py`) and predicts correctly on sample images.
- Everything is reproducible from the README in under 30 minutes on Colab.
- Code is readable by a beginner.

## 12. Out of Scope (unless requested later)

Real-time webcam detection, object detection (YOLO/SSD), mobile app deployment, and training on additional datasets. These can be listed as "future scope" in the report.
