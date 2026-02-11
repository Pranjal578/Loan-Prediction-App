from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load all trained models
try:
    model_lr = joblib.load('loan_model_lr.pkl')
    model_rf = joblib.load('loan_model_rf.pkl')
    model_xgb = joblib.load('loan_model_xgb.pkl')
    best_model = joblib.load('loan_model_best.pkl')
    print("✅ All models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    # If multiple models don't exist, try loading single model
    try:
        best_model = joblib.load('loan_model.pkl')
        model_lr = best_model
        model_rf = best_model
        model_xgb = best_model
        # print("✅ Single model loaded successfully!")
    except:
        best_model = None

# Model names mapping
MODEL_NAMES = {
    'lr': 'Logistic Regression',
    'rf': 'Random Forest',
    'xgb': 'XGBoost',
    'best': 'Best Model (Auto-selected)'
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # DEBUG: Print received data
        # print("=" * 50)
        # print("RECEIVED DATA:")
        # print(data)
        # print("=" * 50)

        # Determine which model to use
        model_choice = data.get('model', 'best')  # Default to best model
        
        if model_choice == 'lr':
            model = model_lr
            model_name = 'Logistic Regression'
        elif model_choice == 'rf':
            model = model_rf
            model_name = 'Random Forest'
        elif model_choice == 'xgb':
            model = model_xgb
            model_name = 'XGBoost'
        else:
            model = best_model
            model_name = 'Best Model'
        
        # Extract features in EXACT order as training
        # Order: no_of_dependents, education, self_employed, income_annum, loan_amount, 
        #        loan_term, cibil_score, residential_assets_value, commercial_assets_value, 
        #        luxury_assets_value, bank_asset_value
        
        # print("Extracting features...")
        features = [
            int(data['no_of_dependents']),
            int(data['education']),
            int(data['self_employed']),
            float(data['income_annum']),
            float(data['loan_amount']),
            int(data['loan_term']),
            int(data['cibil_score']),
            float(data['residential_assets_value']),
            float(data['commercial_assets_value']),
            float(data['luxury_assets_value']),
            float(data['bank_asset_value'])
        ]
        
        # print("Features extracted:", features)
        # print("Number of features:", len(features))
        
        # Make prediction
        # print("Making prediction...")
        prediction = model.predict([features])[0]
        probability = model.predict_proba([features])[0]
        
        # print("Prediction:", prediction)
        # print("Probability:", probability)
        
        # Prepare response - Convert numpy types to Python types for JSON serialization
        result = {
            'model_used': model_name,
            'prediction': 'Approved' if prediction == 1 else 'Rejected',
            'probability': {
                'approved': float(round(probability[1] * 100, 2)),
                'rejected': float(round(probability[0] * 100, 2))
            },
            'confidence': float(round(max(probability[0], probability[1]) * 100, 2))
        }
        
        # print("Result:", result)
        # print("=" * 50)
        
        return jsonify(result)
    
    except KeyError as e:
        error_msg = f'Missing required field: {str(e)}'
        # print(f"❌ KeyError: {error_msg}")
        return jsonify({'error': error_msg}), 400
    except Exception as e:
        error_msg = f'Prediction error: {str(e)}'
        # print(f"❌ Exception: {error_msg}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': error_msg}), 400

@app.route('/compare', methods=['POST'])
def compare():
    """Compare predictions from all three models"""
    try:
        # Get data from the form
        data = request.get_json()
        
        # Extract features in EXACT order
        features = [
            int(data['no_of_dependents']),
            int(data['education']),
            int(data['self_employed']),
            float(data['income_annum']),
            float(data['loan_amount']),
            int(data['loan_term']),
            int(data['cibil_score']),
            float(data['residential_assets_value']),
            float(data['commercial_assets_value']),
            float(data['luxury_assets_value']),
            float(data['bank_asset_value'])
        ]
        
        # Get predictions from all models
        results = {}
        
        for model, name in [(model_lr, 'Logistic Regression'), 
                            (model_rf, 'Random Forest'), 
                            (model_xgb, 'XGBoost')]:
            prediction = model.predict([features])[0]
            probability = model.predict_proba([features])[0]
            
            results[name] = {
                'prediction': 'Approved' if prediction == 1 else 'Rejected',
                'probability': {
                    'approved': float(round(probability[1] * 100, 2)),
                    'rejected': float(round(probability[0] * 100, 2))
                },
                'confidence': float(round(max(probability[0], probability[1]) * 100, 2))
            }
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get information about available models"""
    return jsonify({
        'available_models': MODEL_NAMES,
        'default_model': 'best'
    })

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)