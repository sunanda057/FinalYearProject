import streamlit as st
import pandas as pd
import joblib
import gdown

# Cache the model download and load
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

model = load_model()

st.title("Gas Consumption Prediction App")

if model is not None:
    # Collect all required inputs
    income = st.number_input("Total Monthly Income (₹)", min_value=0)
    family_members = st.number_input("No. of Family Members", min_value=1)
    adults = st.number_input("No. of Adults", min_value=0)
    children = st.number_input("No. of Children", min_value=0)
    city = st.text_input("City")
    state = st.text_input("State")

    if st.button("Predict Gas Consumption"):
        # Create input DataFrame with correct column names
        input_data = pd.DataFrame({
            'Total Monthly Income (₹)': [income],
            'No. of Family Members': [family_members],
            'No. of Adults': [adults],
            'No. of Children': [children],
            'City': [city],
            'State': [state]
        })

        try:
            prediction = model.predict(input_data)[0]
            st.success(f"Predicted Monthly Gas Consumption: {prediction:.2f} units")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
else:
    st.warning("Model could not be loaded.")
