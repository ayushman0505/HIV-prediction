from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Project name
PROJECT_NAME = "HIVision: Smart HIV Risk Predictor"

# Load the trained model (this will load the 'hiv_probability_model.pkl' file)
model = joblib.load('hiv_probability_model.pkl')

@app.route('/')
def index():
    who_region_info = {
        0: 'Africa (AFRO)',
        1: 'Americas (AMRO)',
        2: 'South-East Asia (SEARO)',
        3: 'Europe (EURO)',
        4: 'Eastern Mediterranean (EMRO)',
        5: 'Western Pacific (WPRO)'
    }
    return render_template('index.html', project_name=PROJECT_NAME, who_region_info=who_region_info)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the HTML form with validation and tooltips
        def get_float(field, default=0.0):
            try:
                return float(request.form.get(field, default))
            except (ValueError, TypeError):
                return default
        def get_int(field, default=0):
            try:
                return int(request.form.get(field, default))
            except (ValueError, TypeError):
                return default

        estimated_art = get_float('estimated_art')
        estimated_art_children = get_float('estimated_art_children')
        new_cases = get_float('new_cases')
        deaths = get_float('deaths')
        mother_to_child = get_float('mother_to_child')
        who_region_encoded = get_int('who_region')

        # Create DataFrame from input data
        new_data = pd.DataFrame({
            'Estimated ART coverage among people living with HIV (%)_median': [estimated_art],
            'Estimated ART coverage among children (%)_median': [estimated_art_children],
            'New_Cases_Adults': [new_cases],
            'Deaths': [deaths],
            'Mother_to_Child_Prevention': [mother_to_child],
            'WHO_Region_Encoded': [who_region_encoded]
        })

        # Make prediction (probability)
        probability = model.predict_proba(new_data)[0][1] if hasattr(model, 'predict_proba') else model.predict(new_data)[0]
        prediction = model.predict(new_data)[0]

        # Risk category
        if probability < 0.33:
            risk = "Low"
        elif probability < 0.66:
            risk = "Medium"
        else:
            risk = "High"

        # Prepare summary
        input_summary = {
            'Estimated ART coverage (%)': estimated_art,
            'ART coverage among children (%)': estimated_art_children,
            'New Cases (Adults)': new_cases,
            'Deaths': deaths,
            'Mother-to-Child Prevention': mother_to_child,
            'WHO Region (encoded)': who_region_encoded
        }

        return render_template(
            'index.html',
            project_name=PROJECT_NAME,
            prediction_text=f'HIV Probability: {probability:.4f} ({risk} Risk)',
            input_summary=input_summary,
            who_region_info={
                0: 'Africa (AFRO)',
                1: 'Americas (AMRO)',
                2: 'South-East Asia (SEARO)',
                3: 'Europe (EURO)',
                4: 'Eastern Mediterranean (EMRO)',
                5: 'Western Pacific (WPRO)'
            }
        )

    except Exception as e:
        return render_template('index.html', project_name=PROJECT_NAME, prediction_text=f'Error: {str(e)}', who_region_info={
                0: 'Africa (AFRO)',
                1: 'Americas (AMRO)',
                2: 'South-East Asia (SEARO)',
                3: 'Europe (EURO)',
                4: 'Eastern Mediterranean (EMRO)',
                5: 'Western Pacific (WPRO)'
            })

if __name__ == "__main__":
    app.run(debug=True)
