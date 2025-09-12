import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from data_preprocessing import train_generator  # to load class indices
from PIL import Image

# Load model
model = load_model("cnn_model.h5")

# Class labels
class_labels = list(train_generator.class_indices.keys())

st.title("🌱 Crop Disease Detection using CNN")
st.write("Upload a plant leaf image to predict its disease category.")

# File uploader
uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = img.resize((128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    disease = class_labels[predicted_class]

    st.success(f"✅ Predicted Disease: **{disease}**")
