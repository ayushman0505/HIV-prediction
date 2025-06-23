# HIV Probability Prediction Web App

This project provides a web-based tool for predicting the probability of HIV infection using machine learning. It features both a Flask backend for model inference and a Streamlit dashboard for data exploration and interactive prediction.

## Key Features
- **Machine Learning Model**: Utilizes a trained model (`hiv_probability_model.pkl`) to predict HIV probability based on user input and epidemiological data.
- **Interactive Web Interface**: Users can input relevant health and demographic data to receive instant predictions.
- **Data Exploration**: The Streamlit app allows users to explore key HIV-related datasets, such as ART coverage, new cases, deaths, and more.

## Project Highlights
- Over **38 million people** globally are living with HIV as of 2023, with significant regional disparities in prevalence and treatment coverage.
- In 2023, approximately **1.3 million new HIV infections** were reported worldwide, highlighting the ongoing need for effective prevention and intervention strategies.
- The model leverages data such as ART coverage, new cases, deaths, and mother-to-child transmission rates to provide actionable insights for public health planning.

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. To run the Flask app:
   ```bash
   python app.py
   ```
3. To run the Streamlit dashboard:
   ```bash
   streamlit run streamlit_app.py
   ```

## Files
- `app.py`: Flask backend for prediction
- `streamlit_app.py`: Streamlit dashboard for data exploration and prediction
- `hiv_probability_model.pkl`: Trained machine learning model
- `*.csv`: Cleaned datasets for analysis

---
Developed for HIV probability prediction and data-driven public health insights.
