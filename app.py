import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

st.title("🩻 Pneumonia Detection from Chest X-Ray)")

model = load_model("pneumonia_resnet50.h5")

uploaded_file = st.file_uploader("Upload a Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(150,150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    label = "Pneumonia" if prediction > 0.5 else "Normal"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    st.image(img, caption="Uploaded Chest X-ray", use_container_width=True)
    st.markdown(f"### 🔍 Prediction: **{label}**")
    st.markdown(f"**Confidence Score:** {confidence:.2f}")

