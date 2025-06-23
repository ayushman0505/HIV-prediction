import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

st.set_page_config(page_title="HIVision: Smart HIV Risk Predictor", layout="wide")
st.title("HIVision: Smart HIV Risk Predictor Dashboard")

# WHO region options for one-hot encoding
WHO_REGIONS = [
    "Africa (AFRO)",
    "Americas (AMRO)",
    "South-East Asia (SEARO)",
    "Europe (EURO)",
    "Eastern Mediterranean (EMRO)",
    "Western Pacific (WPRO)"
]

# Load model and feature list
def load_model():
    return joblib.load("hiv_probability_model.pkl")

def load_feature_list():
    with open("model_features.txt", "r") as f:
        return [line.strip() for line in f.readlines()]

try:
    model = load_model()
    feature_list = load_feature_list()
except Exception as e:
    st.error(f"Failed to load model or feature list: {e}")
    st.stop()

# Sidebar for navigation
st.sidebar.title("Navigation")
st.sidebar.markdown("""
**WHO Region Codes:**
- Africa (AFRO)
- Americas (AMRO)
- South-East Asia (SEARO)
- Europe (EURO)
- Eastern Mediterranean (EMRO)
- Western Pacific (WPRO)
""")
page = st.sidebar.radio("Go to", ["Predict HIV Probability", "View Data"])

if page == "Predict HIV Probability":
    st.header("Predict HIV Probability")
    art_coverage = st.number_input("Estimated ART coverage among people living with HIV (%)", min_value=0.0, max_value=100.0, value=70.0)
    art_coverage_children = st.number_input("Estimated ART coverage among children (%)", min_value=0.0, max_value=100.0, value=60.0)
    new_cases = st.number_input("New Cases (Adults)", min_value=0.0, value=1000.0)
    deaths = st.number_input("Deaths", min_value=0.0, value=100.0)
    mother_to_child = st.number_input("Mother-to-Child Prevention (%)", min_value=0.0, max_value=100.0, value=80.0)
    who_region = st.selectbox("WHO Region", WHO_REGIONS)

    # Prepare input for model (one-hot encoding for region)
    # --- FIX: Always use model_features.txt for input columns ---
    input_dict = {}
    # Map display names to codes for robust matching
    region_map = {
        'Africa (AFRO)': 'AFR',
        'Americas (AMRO)': 'AMR',
        'South-East Asia (SEARO)': 'SEAR',
        'Europe (EURO)': 'EUR',
        'Eastern Mediterranean (EMRO)': 'EMR',
        'Western Pacific (WPRO)': 'WPR',
    }
    selected_code = region_map.get(who_region, who_region)
    # Fill input_dict with all features from model_features.txt
    for feat in feature_list:
        if feat == 'Estimated ART coverage among people living with HIV (%)_median':
            input_dict[feat] = art_coverage
        elif feat == 'Estimated ART coverage among children (%)_median':
            input_dict[feat] = art_coverage_children
        elif feat == 'New_Cases_Adults':
            input_dict[feat] = new_cases
        elif feat == 'Deaths':
            input_dict[feat] = deaths
        elif feat == 'Mother_to_Child_Prevention':
            input_dict[feat] = mother_to_child
        elif feat.startswith('WHO Region_'):
            # Set 1 for selected region, 0 otherwise
            input_dict[feat] = 1 if feat.endswith(selected_code) else 0
        else:
            input_dict[feat] = 0  # Default for any extra features
    # Build DataFrame in exact order
    X = pd.DataFrame([[input_dict[feat] for feat in feature_list]], columns=feature_list)

    if st.button("Predict"):
        # Debug: Show feature alignment
        st.write("Model expects features:", feature_list)
        st.write("Input DataFrame columns:", X.columns.tolist())
        # Check for mismatch
        if feature_list != list(X.columns):
            st.error("Feature mismatch! Please check model_features.txt and input construction.")
            st.stop()
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


