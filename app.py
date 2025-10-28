import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import gdown, os

# ================================
# Load the trained model
# ================================

model_path = "rice_cnn_model.h5"
file_id = "YOUR_FILE_ID"

if not os.path.exists(model_path):
    st.write("📥 Downloading model from Google Drive...")
    gdown.download(f"https://drive.google.com/file/d/1b7fcYIAKC1Xo67LcMQEmcEMdJLxM0Esn", model_path, quiet=False)

# Verify file size
if os.path.getsize(model_path) < 1000000:  # less than 1 MB
    st.error("Model file seems incomplete. Please re-upload to Google Drive.")
else:
    st.success("Model downloaded successfully!")

from tensorflow.keras.models import load_model
model = load_model(model_path)

# ================================
# Define class names (in same order as during training)
# ================================
# Replace with your actual folder names if different
class_names = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']

# ================================
# Streamlit UI
# ================================
st.set_page_config(page_title="Rice Variety Classifier", page_icon="🌾", layout="centered")

st.title("🌾 Rice Variety Classification App")
st.write("Upload an image of rice grain and the model will predict its variety.")

uploaded_file = st.file_uploader("Upload a rice grain image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img = img.resize((128, 128))  # same as model training size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Display result
    st.success(f"**Predicted Variety:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")
