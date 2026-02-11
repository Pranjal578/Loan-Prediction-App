# Loan Prediction Web Application

A machine learning-powered web application that predicts loan approval based on various financial and personal factors.

## Features

- 🎯 Real-time loan approval prediction
- 📊 Probability scores for approval/rejection
- 💼 Clean and modern user interface
- 📱 Responsive design for all devices

## Live Link

```link
<https://loan-prediction-dep-y9qw.onrender.com>
```

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Copy Your Trained Model

Copy your `loan_model.pkl` file to this directory:

```bash
# From your current project directory
copy loan_model.pkl loan_app/
```

### 3. Run the Application

```bash
python app.py
```

### 4. Open in Browser

Navigate to: `http://127.0.0.1:5000/`

## Project Structure

```text
loan_app/
├── app.py                  # Flask backend
├── templates/
│   └── index.html         # Frontend interface
├── loan_model.pkl         # Your trained ML model (copy here)
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## How to Use

1. Fill in all the loan application details:
   - Number of dependents
   - Education level
   - Employment status
   - Annual income
   - Loan amount and term
   - CIBIL score
   - Asset values

2. Click "Predict Loan Approval"

3. View the prediction result with probability scores

## Input Fields Explanation

- **Number of Dependents**: How many people depend on you financially (0-10)
- **Education**: Graduate or Not Graduate
- **Self Employed**: Are you self-employed? (Yes/No)
- **Annual Income**: Your yearly income in ₹
- **Loan Amount**: The amount you want to borrow in ₹
- **Loan Term**: Duration of the loan in months
- **CIBIL Score**: Your credit score (300-900)
- **Residential Assets Value**: Value of your residential properties in ₹
- **Commercial Assets Value**: Value of your commercial properties in ₹
- **Luxury Assets Value**: Value of luxury items (cars, jewelry, etc.) in ₹
- **Bank Asset Value**: Your total bank balance/deposits in ₹

## Model Details

- **Algorithm**: Logistic Regression
- **Accuracy**: ~80%
- **Training Data**: 4,269 loan records
- **Features**: 11 input features

## Future Enhancements

- [ ] Add more ML models (Random Forest, XGBoost)
- [ ] Feature importance visualization
- [ ] Historical predictions tracking
- [ ] User authentication
- [ ] PDF report generation
- [ ] API endpoint for integration

## Troubleshooting

### Model not found error

Make sure `loan_model.pkl` is in the same directory as `app.py`

### Port already in use

Change the port in `app.py`:

```python
app.run(debug=True, port=5001)
```

### Import errors

Reinstall dependencies:

```bash
pip install -r requirements.txt --upgrade
```

## License

MIT License - Feel free to use and modify for your projects!
