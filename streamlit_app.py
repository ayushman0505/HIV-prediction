import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="HIV Probability Model App", layout="wide")
st.title("HIV Probability Model Dashboard")

# Load model
def load_model():
    with open("hiv_probability_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Predict HIV Probability", "View Data"])

if page == "Predict HIV Probability":
    st.header("Predict HIV Probability")
    # Example input fields (customize as per your model's features)
    age = st.number_input("Age", min_value=0, max_value=100, value=30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    country = st.text_input("Country", "Kenya")
    # Add more fields as required by your model

    if st.button("Predict"):
        # Prepare input for model (customize as per your model's requirements)
        # Example: X = [[age, 1 if gender == 'Male' else 0, ...]]
        X = [[age, 1 if gender == 'Male' else 0]]  # Update this line as needed
        probability = model.predict_proba(X)[0][1]
        st.success(f"Predicted HIV Probability: {probability:.2%}")

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
