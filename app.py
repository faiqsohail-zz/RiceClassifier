import os
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import gdown

# —————————————————————————
# App configuration
# —————————————————————————
st.set_page_config(page_title="Rice Variety Classifier", layout="centered")
st.title("🌾 Rice Variety Classification App")
st.write("Upload a rice grain image and the trained CNN model will predict its variety.")

# —————————————————————————
# Model download/setup
# —————————————————————————
model_filename = "rice_cnn_model.h5"
# From your Drive link: https://drive.google.com/file/d/1b7fcYIAKC1Xo67LcMQEmcEMdJLxM0Esn/view?usp=drive_link
file_id = "1b7fcYIAKC1Xo67LcMQEmcEMdJLxM0Esn"
download_url = f"https://drive.google.com/uc?id={file_id}"

if not os.path.exists(model_filename):
    st.write("📥 Downloading model…")
    gdown.download(download_url, model_filename, quiet=False)
else:
    st.write("✅ Model file already present.")

# Basic check of file size
min_size_bytes = 5 * 1024 * 1024  # e.g. require at least 5 MB for a valid model
if os.path.exists(model_filename):
    size = os.path.getsize(model_filename)
    st.write(f"Downloaded model size: {size / (1024*1024):.2f} MB")
    if size < min_size_bytes:
        st.error("⚠️ Model file too small — might be incorrect download. Please check your Drive sharing link.")
        st.stop()
else:
    st.error("❌ Model file not found.")
    st.stop()

# Load the model
model = load_model(model_filename)
st.success("✅ Model loaded successfully!")

# —————————————————————————
# Define classes (ensure same order as during training)
# —————————————————————————
class_names = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']

# —————————————————————————
# Image upload & prediction
# —————————————————————————
uploaded_file = st.file_uploader("Upload a rice grain image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess: resize, convert, normalise
    img_height, img_width = 128, 128
    image = image.resize((img_width, img_height))
    img_array = np.array(image)
    if img_array.shape[-1] == 4:
        # if image has alpha channel, drop it
        img_array = img_array[..., :3]
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    preds = model.predict(img_array)
    pred_index = np.argmax(preds, axis=1)[0]
    predicted_class = class_names[pred_index]
    confidence = preds[0][pred_index] * 100
    
    # Show result
    st.success(f"**Predicted Variety:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")
