import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="HIVision: Smart HIV Risk Predictor", layout="wide")
st.title("HIVision: Smart HIV Risk Predictor Dashboard")

# Load model
def load_model():
    return joblib.load("hiv_probability_model.pkl")

try:
    model = load_model()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Predict HIV Probability", "View Data"])

if page == "Predict HIV Probability":
    st.header("Predict HIV Probability")
    art_coverage = st.number_input("Estimated ART coverage among people living with HIV (%)", min_value=0.0, max_value=100.0, value=70.0)
    art_coverage_children = st.number_input("Estimated ART coverage among children (%)", min_value=0.0, max_value=100.0, value=60.0)
    new_cases = st.number_input("New Cases (Adults)", min_value=0.0, value=1000.0)
    deaths = st.number_input("Deaths", min_value=0.0, value=100.0)
    mother_to_child = st.number_input("Mother-to-Child Prevention (%)", min_value=0.0, max_value=100.0, value=80.0)
    who_region = st.number_input("WHO Region (Encoded)", min_value=0, max_value=10, value=1)

    if st.button("Predict"):
        # Prepare input for model (customize as per your model's requirements)
        X = [[art_coverage, art_coverage_children, new_cases, deaths, mother_to_child, who_region]]  # Update this line as needed
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(X)[0][1]
            st.success(f"Predicted HIV Probability: {probability:.2%}")
        elif hasattr(model, "predict"):
            prediction = model.predict(X)[0]
            st.success(f"Predicted HIV Probability: {prediction}")
        elif isinstance(model, np.ndarray):
            st.error("Loaded file is a NumPy array, not a trained model. Please check your 'hiv_probability_model.pkl' file.")
        else:
            st.error("Loaded object is not a valid model. Please check your 'hiv_probability_model.pkl' file.")

elif page == "View Data":
    st.header("Explore Data")
    data_file = st.selectbox(
        "Select a data file to view:",
        [
            "art_coverage_by_country_clean.csv",
            "art_pediatric_coverage_by_country_clean.csv",
            "no_of_cases_adults_15_to_49_by_country_clean.csv",
            "no_of_deaths_by_country_clean.csv",
            "no_of_people_living_with_hiv_by_country_clean.csv",
            "prevention_of_mother_to_child_transmission_by_country_clean.csv",
            "world_population.csv"
        ]
    )
    df = pd.read_csv(data_file)
    st.dataframe(df)
    st.write("Shape:", df.shape)
    st.write("Columns:", df.columns.tolist())

st.sidebar.info("Developed with Streamlit. Customize this app as needed!")


