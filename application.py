from flask import Flask, render_template, request, jsonify
import pickle
import json
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR, 'GBModel_v2.pkl'), 'rb'))

with open(os.path.join(BASE_DIR, 'meta_v2.json')) as f:
    meta = json.load(f)

df = pd.read_csv(os.path.join(BASE_DIR, 'dataset.csv'))

# Clean dataset columns once at startup
df['Mileage'] = df['Mileage'].str.extract(r'([\d.]+)').astype(float)
df['Engine']  = df['Engine'].str.extract(r'([\d.]+)').astype(float)
df['Power']   = df['Power'].str.extract(r'([\d.]+)').astype(float)
df.drop(columns=['Unnamed: 0', 'New_Price'], inplace=True, errors='ignore')
df.dropna(inplace=True)
df['car_age']      = 2024 - df['Year']
df['kms_per_year'] = df['Kilometers_Driven'] / (df['car_age'] + 1)


@app.route('/')
def index():
    return render_template('index.html', meta=meta, company_cars=meta['company_cars'])


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        name          = data['name']
        location      = data['location']
        year          = int(data['year'])
        kms_driven    = int(data['kms_driven'])
        fuel_type     = data['fuel_type']
        transmission  = data['transmission']
        owner_type    = data['owner_type']
        mileage       = float(data['mileage'])
        engine        = float(data['engine'])
        power         = float(data['power'])
        seats         = float(data['seats'])

        car_age      = 2024 - year
        kms_per_year = kms_driven / (car_age + 1)

        input_df = pd.DataFrame([[
            name, location, year, car_age, kms_driven,
            kms_per_year, fuel_type, transmission, owner_type,
            mileage, engine, power, seats
        ]], columns=[
            'Name', 'Location', 'Year', 'car_age', 'Kilometers_Driven',
            'kms_per_year', 'Fuel_Type', 'Transmission', 'Owner_Type',
            'Mileage', 'Engine', 'Power', 'Seats'
        ])

        prediction_lakh = model.predict(input_df)[0]
        prediction_lakh = max(0, prediction_lakh)
        prediction_rs   = prediction_lakh * 100000

        # Market comparison — similar cars
        similar = df[
            (df['Fuel_Type'] == fuel_type) &
            (df['Transmission'] == transmission) &
            (df['Year'].between(year - 2, year + 2))
        ]['Price']

        market_low  = round(float(similar.quantile(0.25)), 2) if len(similar) > 3 else None
        market_high = round(float(similar.quantile(0.75)), 2) if len(similar) > 3 else None
        market_avg  = round(float(similar.mean()), 2)         if len(similar) > 0 else None

        return jsonify({
            'success': True,
            'price_lakh': round(prediction_lakh, 2),
            'price_rs': int(prediction_rs),
            'market_low':  market_low,
            'market_high': market_high,
            'market_avg':  market_avg,
            'similar_count': len(similar)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/cars/<company>')
def get_cars(company):
    cars = meta['company_cars'].get(company, [])
    return jsonify(cars)


@app.route('/autofill', methods=['POST'])
def autofill():
    """Return typical values for a selected car model"""
    try:
        data = request.get_json()
        name = data['name']
        matches = df[df['Name'] == name]
        if len(matches) == 0:
            return jsonify({'success': False})
        row = matches.iloc[0]
        return jsonify({
            'success': True,
            'mileage': round(float(row['Mileage']), 1),
            'engine':  int(row['Engine']),
            'power':   round(float(row['Power']), 1),
            'seats':   int(row['Seats']),
            'fuel_type': row['Fuel_Type'],
            'transmission': row['Transmission']
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
