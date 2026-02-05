from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

model = joblib.load('Prediction.pkl')

expected_columns = [
    'Total_area', 'Bedrooms', 'Bathrooms', 'Floors', 'Balcony',
    'Location_Chennai', 'Location_Delhi', 'Location_Hyderabad',
    'Location_Mumbai', 'Location_Pune',
    'Parking_Yes',
    'Property_type_Independent House', 'Property_type_Villa',
    'Furnishing_status_Semi-Furnished', 'Furnishing_status_Unfurnished'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        location = request.form['location']
        property_type = request.form['property_type']
        area = float(request.form['area'])
        bedrooms = int(request.form['bhk'])
        bathrooms = int(request.form['bathrooms'])
        balcony = int(request.form['balcony'])
        furnished = request.form['furnishing']
        parking = request.form['parking']

    
        if not (605 <= area <= 3994):
            return render_template('index.html', error="❌ Area must be between 605 and 3994 sq ft.")
        if not (1 <= bedrooms <= 5):
            return render_template('index.html', error="❌ Bedrooms must be between 1 and 5.")
        if not (1 <= bathrooms <= 4):
            return render_template('index.html', error="❌ Bathrooms must be between 1 and 4.")

        floors = 1

        # DataFrame
        input_df = pd.DataFrame(columns=expected_columns)
        input_data = {
            'Total_area': area,
            'Bedrooms': bedrooms,
            'Bathrooms': bathrooms,
            'Floors': floors,
            'Balcony': balcony
        }

        input_data[f'Location_{location}'] = 1
        if parking == 'Yes':
            input_data['Parking_Yes'] = 1
        input_data[f'Property_type_{property_type}'] = 1
        if furnished == 'Semi-Furnished':
            input_data['Furnishing_status_Semi-Furnished'] = 1
        elif furnished == 'Unfurnished':
            input_data['Furnishing_status_Unfurnished'] = 1

        for col in expected_columns:
            if col not in input_data:
                input_data[col] = 0

        input_df.loc[0] = input_data

        prediction = model.predict(input_df)[0]
        return render_template('index.html', prediction=f"{round(prediction, 2)} Lakhs")

    except Exception as e:
        return render_template('index.html', error=f"⚠ Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
