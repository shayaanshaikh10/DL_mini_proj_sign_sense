"""
SignSense demo app.

Run from the project root:
    streamlit run app/app.py
"""

import json
import random
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

# Make "src" importable (Streamlit runs this file from the app/ folder)
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src import config
from src.data_loader import find_image_folder, load_sign_names
from src.predict import get_model, predict_image

# Dropdown label -> saved model name
MODEL_OPTIONS = {
    "Custom CNN + BatchNorm (Model A)": config.MODEL_A_NAME,
    "MobileNetV2 transfer learning (Model B)": config.MODEL_B_NAME,
}


@st.cache_resource(max_entries=2)
def load_cached_model(model_name):
    """Load each model only once, even when the app reruns."""
    return get_model(model_name)


@st.cache_data
def load_test_accuracy(model_name):
    """Read the model's test accuracy from results/metrics_<name>.json (None if missing)."""
    metrics_path = config.RESULTS_DIR / f"metrics_{model_name}.json"
    if not metrics_path.exists():
        return None
    return json.loads(metrics_path.read_text())["test_accuracy"]


@st.cache_data
def load_test_samples():
    """Return a list of (image path, true class id) from Test.csv, or [] if the data is missing."""
    try:
        test_dir = find_image_folder(config.TEST_DIR)
        test_df = pd.read_csv(config.TEST_CSV)
    except (FileNotFoundError, OSError):
        return []
    return [(str(test_dir / Path(p).name), int(c)) for p, c in zip(test_df["Path"], test_df["ClassId"])]


def show_reference_sign(class_id):
    """Show the clean reference picture of a sign from data/Meta/, if available."""
    meta_path = config.META_DIR / f"{class_id}.png"
    if meta_path.exists():
        st.image(str(meta_path), caption="Reference sign", width=90)


# ---------------------------------------------------------------
# Page header
# ---------------------------------------------------------------
st.set_page_config(page_title="SignSense", page_icon=":material/traffic:")
st.title(":material/traffic: SignSense")
st.write("Pick a random image from the GTSRB test set (12,630 German traffic signs the model "
         "never saw during training) and see which of the 43 sign types the model predicts.")

# ---------------------------------------------------------------
# Sidebar: model choice
# ---------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    model_label = st.selectbox("Model", list(MODEL_OPTIONS))
    model_name = MODEL_OPTIONS[model_label]
    accuracy = load_test_accuracy(model_name)
    if accuracy is not None:
        st.caption(f"Test accuracy: {accuracy:.2%} on 12,630 images")

# ---------------------------------------------------------------
# Choose a random test image (only images from the 43 known classes,
# so the true label is always known and shown next to the prediction)
# ---------------------------------------------------------------
samples = load_test_samples()
if not samples:
    st.error("Test images not found. Run `python scripts/download_data.py` first.")
    st.stop()

if st.button("Try a random test image", icon=":material/shuffle:", type="primary"):
    st.session_state["sample"] = random.choice(samples)

if "sample" not in st.session_state:
    st.info("Click **Try a random test image** to start.", icon=":material/shuffle:")
    st.stop()

image_path, true_class_id = st.session_state["sample"]
image = Image.open(image_path).convert("RGB")

# ---------------------------------------------------------------
# Predict and show the results
# ---------------------------------------------------------------
with st.spinner("Loading model..."):
    load_cached_model(model_name)
class_id, class_name, confidence, top3 = predict_image(image, model_name)

image_col, result_col = st.columns(2)
with image_col:
    st.image(image, caption="Test image", width=220)

with result_col:
    st.subheader(class_name)
    st.metric("Confidence", f"{confidence:.1%}")
    if confidence < config.LOW_CONFIDENCE:
        st.warning("Not sure: the model's confidence is below 50%. "
                   "The image may be very dark or blurry.", icon=":material/help:")
    if true_class_id == class_id:
        st.success("Correct! This matches the true label.", icon=":material/check_circle:")
    else:
        true_name = load_sign_names()[true_class_id]
        st.error(f"Wrong. The true sign is: {true_name}.", icon=":material/cancel:")
    show_reference_sign(class_id)

st.subheader("Top 3 predictions")
top3_table = pd.DataFrame({
    "Sign": [name for _, name, _ in top3],
    "Confidence (%)": [round(conf * 100, 2) for _, _, conf in top3],
})
st.bar_chart(top3_table, x="Sign", y="Confidence (%)", horizontal=True,
             sort="-Confidence (%)", color="#3b6fb6")
