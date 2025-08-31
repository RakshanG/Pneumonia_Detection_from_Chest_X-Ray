import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="pneumonia_resnet50.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Preprocessing function
def preprocess_image(image: Image.Image):
    img = image.resize((224, 224))  # ResNet input size
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    return img_array

# Prediction function
def predict(image: Image.Image):
    img_array = preprocess_image(image)

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], img_array)

    # Run inference
    interpreter.invoke()

    # Get output tensor
    prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]

    label = "Pneumonia" if prediction > 0.5 else "Normal"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    return label, float(confidence)

# Streamlit UI
st.title("🩻 Pneumonia Detection from Chest X-Ray")
st.write("Upload a Chest X-ray Image")

uploaded_file = st.file_uploader("Upload X-ray", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Chest X-ray", use_container_width=True)

    label, confidence = predict(image)
    st.markdown(f"### 🔍 Prediction: **{label}**")
    st.markdown(f"### Confidence Score: **{confidence:.2f}**")

