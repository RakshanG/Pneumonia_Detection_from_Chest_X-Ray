🩻 Pneumonia Detection from Chest X-Ray

This project uses a ResNet50 CNN model (converted to TFLite) to classify chest X-ray images as Normal or Pneumonia.
It comes with a Streamlit web app for easy image upload and real-time prediction.

Features

1.Built with TensorFlow/Keras + ResNet50
2.Converted to TensorFlow Lite (.tflite) for smaller size & faster inference
3.Interactive Streamlit app for uploading chest X-ray images
4.Predicts with confidence score (probability)

Chest_X-Ray_CNN/
│── app.py                     
│── CNN_Chest_X-Ray.ipynb      
│── pneumonia_resnet50.tflite  
│── requirements.txt           
│── README.md                  

Installation
Clone the repo and install dependencies:

git clone https://github.com/RakshanG/Pneumonia_Detection_from_Chest_X-Ray.git
cd Pneumonia_Detection_from_Chest_X-Ray
pip install -r requirements.txt

Run the App

Start the Streamlit app:

streamlit run app.py


Then open your browser at http://localhost:8501

Example Prediction

If you upload an image like this:

File: person28_virus_62.jpeg

The app outputs:
Prediction: Pneumonia
Confidence Score: 0.93

Requirements
Main dependencies:
TensorFlow
Streamlit
Pillow
Numpy

Acknowledgements
Dataset: Chest X-Ray Images (Pneumonia)
Base Model: ResNet50
