from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model (this will load the 'hiv_probability_model.pkl' file)
model = joblib.load('hiv_probability_model.pkl')

# Step 1: Define a route to the homepage (HTML form for input)
@app.route('/')
def index():
    return render_template('index.html')

# Step 2: Define a route to handle prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the HTML form
        estimated_art = float(request.form['estimated_art'])
        estimated_art_children = float(request.form['estimated_art_children'])
        new_cases = float(request.form['new_cases'])
        deaths = float(request.form['deaths'])
        mother_to_child = float(request.form['mother_to_child'])
        who_region_encoded = int(request.form['who_region'])

        # Create DataFrame from input data
        new_data = pd.DataFrame({
            'Estimated ART coverage among people living with HIV (%)_median': [estimated_art],
            'Estimated ART coverage among children (%)_median': [estimated_art_children],
            'New_Cases_Adults': [new_cases],
            'Deaths': [deaths],
            'Mother_to_Child_Prevention': [mother_to_child],
            'WHO_Region_Encoded': [who_region_encoded]
        })

        # Make prediction
        prediction = model.predict(new_data)[0]

        # Return the prediction result
        return render_template('index.html', prediction_text=f'HIV Probability: {prediction:.4f}')

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)
