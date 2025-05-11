from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('home.html', prediction=None)

@app.route('/', methods=['POST'])
def predict():
    prediction = None  # Default to None, so no prediction is shown until form is submitted

    if request.method == 'POST':
        # Collect form data
        features = [
            float(request.form.get('Pregnancies', 0)),
            float(request.form.get('Glucose', 0)),
            float(request.form.get('BloodPressure', 0)),
            float(request.form.get('SkinThickness', 0)),
            float(request.form.get('Insulin', 0)),
            float(request.form.get('BMI', 0)),
            float(request.form.get('DiabetesPedigreeFunction', 0)),
            float(request.form.get('Age', 0)),
        ]

        # Make prediction
        input_data = pd.DataFrame([features], columns=[
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ])
        prediction = model.predict(input_data)[0]  # 0 or 1

    return render_template('home.html', prediction=prediction)
    
if __name__ == '__main__':
    app.run(debug=True)
