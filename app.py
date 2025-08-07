import streamlit as st
import pickle
import gdown
import numpy as np
import os

# Title of the app
st.title("Gas Consumption Prediction App")

# Download the model from Google Drive if not already downloaded
model_file = "gas_consumption_model.pkl"
file_id = "1UClYlkrOBEoZvVHAIQSce4lgthih0OVy"  # Your model's file ID from Google Drive
url = f"https://drive.google.com/uc?id={file_id}"

if not os.path.exists(model_file):
    st.info("Downloading model...")
    gdown.download(url, model_file, quiet=False)

# Load the model
try:
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# Input fields for features (adjust number of inputs based on your model)
st.header("Enter Input Features")

feature1 = st.number_input("Feature 1 (e.g., Family Size)", min_value=1, max_value=20, value=4)
feature2 = st.number_input("Feature 2 (e.g., Income)", min_value=0.0, step=0.1, value=5000.0)
feature3 = st.number_input("Feature 3 (e.g., Number of Appliances)", min_value=0, max_value=50, value=5)

# Add more inputs here if your model requires more features

# Make prediction
if st.button("Predict Gas Consumption"):
    input_data = np.array([[feature1, feature2, feature3]])  # Update this as per your model
    try:
        prediction = model.predict(input_data)[0]
        st.subheader("Predicted Gas Consumption:")
        st.success(f"{prediction:.2f} units")
    except Exception as e:
        st.error(f"Error making prediction: {e}")
