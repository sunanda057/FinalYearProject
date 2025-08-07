import streamlit as st
import pandas as pd
import numpy as np
import joblib
import gdown

# Download and load the model from Google Drive
@st.cache_resource
def load_model():
    try:
        url = "https://drive.google.com/uc?id=1UClYlkrOBEoZvVHAIQSce4lgthih0OVy"
        output = "gas_model.joblib"
        gdown.download(url, output, quiet=False)
        model = joblib.load(output)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

# Load the model
model = load_model()

# Main App
st.title("Gas Consumption Prediction App")

if model is not None:
    st.write("Enter the details below to predict gas consumption:")

    # Example input fields (customize these based on your model features)
    income = st.number_input("Monthly Income (in INR)", min_value=0)
    family_size = st.number_input("Family Size", min_value=1)
    has_gas_connection = st.selectbox("Has Gas Connection?", ["Yes", "No"])
    uses_gas_cylinder = st.selectbox("Uses Gas Cylinder?", ["Yes", "No"])
    
    # Map categorical to numerical
    gas_connection = 1 if has_gas_connection == "Yes" else 0
    gas_cylinder = 1 if uses_gas_cylinder == "Yes" else 0

    # Predict Button
    if st.button("Predict Gas Consumption"):
        input_data = pd.DataFrame({
            'income': [income],
            'family_size': [family_size],
            'gas_connection': [gas_connection],
            'gas_cylinder': [gas_cylinder]
        })

        prediction = model.predict(input_data)[0]
        st.success(f"Predicted Monthly Gas Consumption: {prediction:.2f} units")
else:
    st.warning("Model failed to load. Please check the file link or compatibility.")
