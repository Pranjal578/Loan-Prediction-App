from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('loan_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the form
        data = request.get_json()
        
        # Extract features in the same order as training
        features = [
            int(data['no_of_dependents']),
            int(data['education']),  # 1 for Graduate, 0 for Not Graduate
            int(data['self_employed']),  # 1 for Yes, 0 for No
            float(data['income_annum']),
            float(data['loan_amount']),
            int(data['loan_term']),
            int(data['cibil_score']),
            float(data['residential_assets_value']),
            float(data['commercial_assets_value']),
            float(data['luxury_assets_value']),
            float(data['bank_asset_value'])
        ]
        
        # Make prediction
        prediction = model.predict([features])[0]
        probability = model.predict_proba([features])[0]
        
        # Prepare response
        result = {
            'prediction': 'Approved' if prediction == 1 else 'Rejected',
            'probability': {
                'approved': round(probability[1] * 100, 2),
                'rejected': round(probability[0] * 100, 2)
            }
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
