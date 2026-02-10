from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
model = joblib.load("loan_model.pkl")

@app.route('/', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        values = [float(x) for x in request.form.values()]
        prediction = model.predict([values])[0]
        result = "Approved" if prediction == 1 else "Not Approved"
        return render_template('index.html', result=result)
    return render_template('index.html')

app.run(debug=True)