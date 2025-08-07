import streamlit as st
import pandas as pd
import joblib
import gdown

# Cache the model loading to prevent re-downloading every time
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

# App title
st.title("Gas Consumption Prediction App")

if model is not None:
    # Input fields
    income = st.number_input("Enter monthly income (in ₹):", min_value=0)
    family_size = st.number_input("Enter family size:", min_value=1)
    has_gas_connection = st.selectbox("Do you have a gas connection?", ["Yes", "No"])
    uses_gas_cylinder = st.selectbox("Do you use gas cylinder?", ["Yes", "No"])

    if st.button("Predict Gas Consumption"):
        # Prepare input data (as strings for categorical values)
        input_data = pd.DataFrame({
            'income': [income],
            'family_size': [family_size],
            'gas_connection': [has_gas_connection],   # keep as "Yes"/"No"
            'gas_cylinder': [uses_gas_cylinder]       # keep as "Yes"/"No"
        })

        try:
            prediction = model.predict(input_data)[0]
            st.success(f"Predicted Monthly Gas Consumption: {prediction:.2f} units")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
else:
    st.warning("Model could not be loaded. Please check the file or link.")
