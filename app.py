import streamlit as st
import gdown
import pickle
import os
import numpy as np

# Title
st.title("🚗 Gas Consumption Prediction App")

# Download model if not already present
model_file = "gas_consumption_model.pkl"
file_id = "1UClYlkrOBEoZvVHAIQSce4lgthih0OVy"
url = f"https://drive.google.com/uc?id={file_id}"

if not os.path.exists(model_file):
    with st.spinner("Downloading model..."):
        gdown.download(url, model_file, quiet=False)

# Load the model
try:
    with open(model_file, "rb") as f:
        model = pickle.load(f)
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# Sample input fields - change based on your model's expected input
st.subheader("Enter input values:")
feature1 = st.number_input("Feature 1", min_value=0.0)
feature2 = st.number_input("Feature 2", min_value=0.0)
feature3 = st.number_input("Feature 3", min_value=0.0)
feature4 = st.number_input("Feature 4", min_value=0.0)

# Predict
if st.button("Predict"):
    try:
        input_data = np.array([[feature1, feature2, feature3, feature4]])
        prediction = model.predict(input_data)[0]
        st.success(f"Predicted Gas Consumption: {prediction:.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

