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
from PIL import Image, UnidentifiedImageError

# Make "src" importable (Streamlit runs this file from the app/ folder)
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src import config
from src.data_loader import find_image_folder
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
st.write("Upload a photo of a German traffic sign and a deep learning model "
         "will tell you which of the 43 sign types it is.")

# ---------------------------------------------------------------
# Sidebar: model choice and a random test image
# ---------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    model_label = st.selectbox("Model", list(MODEL_OPTIONS))
    model_name = MODEL_OPTIONS[model_label]
    accuracy = load_test_accuracy(model_name)
    if accuracy is not None:
        st.caption(f"Test accuracy: {accuracy:.2%} on 12,630 images")

    st.divider()
    samples = load_test_samples()
    if samples and st.button("Try a random test image", icon=":material/shuffle:"):
        st.session_state["sample"] = random.choice(samples)
    st.caption("Tip: the models expect a cropped sign, like the GTSRB images, "
               "not a full street scene.")

# ---------------------------------------------------------------
# Choose the input image: an upload wins over a random sample
# ---------------------------------------------------------------
uploaded_file = st.file_uploader("Upload a sign image", type=["png", "jpg", "jpeg", "ppm"])
true_class_id = None

if uploaded_file is not None:
    image_source = uploaded_file
elif "sample" in st.session_state:
    image_source, true_class_id = st.session_state["sample"]
else:
    st.info("Upload an image, or use **Try a random test image** in the sidebar.",
            icon=":material/upload:")
    st.stop()

try:
    image = Image.open(image_source).convert("RGB")
except (UnidentifiedImageError, OSError):
    st.error("This file could not be read as an image. Please upload a PNG or JPG file.")
    st.stop()

# ---------------------------------------------------------------
# Predict and show the results
# ---------------------------------------------------------------
with st.spinner("Loading model..."):
    load_cached_model(model_name)
class_id, class_name, confidence, top3 = predict_image(image, model_name)

image_col, result_col = st.columns(2)
with image_col:
    st.image(image, caption="Your image", width=220)

with result_col:
    st.subheader(class_name)
    st.metric("Confidence", f"{confidence:.1%}")
    if confidence < config.LOW_CONFIDENCE:
        st.warning("Not sure: the model's confidence is below 50%. "
                   "The image may be unclear or not a GTSRB sign.", icon=":material/help:")
    if true_class_id is not None:
        if true_class_id == class_id:
            st.success("Correct! This matches the true label.", icon=":material/check_circle:")
        else:
            st.error(f"Wrong. The true label is class {true_class_id}.", icon=":material/cancel:")
    show_reference_sign(class_id)

st.subheader("Top 3 predictions")
top3_table = pd.DataFrame({
    "Sign": [name for _, name, _ in top3],
    "Confidence (%)": [round(conf * 100, 2) for _, _, conf in top3],
})
st.bar_chart(top3_table, x="Sign", y="Confidence (%)", horizontal=True,
             sort="-Confidence (%)", color="#3b6fb6")
