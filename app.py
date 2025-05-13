from flask import Flask, render_template, request
from flask_wtf import FlaskForm
from wtforms import DecimalField, IntegerField, SubmitField
from wtforms.validators import InputRequired, NumberRange
import pandas as pd
import joblib
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'  # Replace with a secure key for production

# Define form class with validation for diabetes prediction inputs
class DiabetesForm(FlaskForm):
    pregnancies = IntegerField('Pregnancies', validators=[InputRequired(), NumberRange(min=0)])
    glucose = DecimalField('Glucose', validators=[InputRequired(), NumberRange(min=0)])
    blood_pressure = DecimalField('Blood Pressure', validators=[InputRequired(), NumberRange(min=0)])
    skin_thickness = DecimalField('Skin Thickness', validators=[InputRequired(), NumberRange(min=0)])
    insulin = DecimalField('Insulin', validators=[InputRequired(), NumberRange(min=0)])
    bmi = DecimalField('BMI', validators=[InputRequired(), NumberRange(min=0)])
    diabetes_pedigree = DecimalField('Diabetes Pedigree Function', validators=[InputRequired(), NumberRange(min=0)])
    age = IntegerField('Age', validators=[InputRequired(), NumberRange(min=0)])
    submit = SubmitField('Predict')

# Load the trained model with error handling
model_path = 'model.pkl'
model = None
if os.path.exists(model_path):
    try:
        model = joblib.load(model_path)
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print(f"Model file {model_path} not found. Please run train_model.py to generate it.")

@app.route('/', methods=['GET', 'POST'])
def home():
    form = DiabetesForm()
    prediction = None
    error = None

    if form.validate_on_submit():
        if model:
            try:
                # Extract form data into a list for prediction
                features = [
                    form.pregnancies.data,
                    form.glucose.data,
                    form.blood_pressure.data,
                    form.skin_thickness.data,
                    form.insulin.data,
                    form.bmi.data,
                    form.diabetes_pedigree.data,
                    form.age.data
                ]

                # Create DataFrame matching the model's expected input
                input_data = pd.DataFrame([features], columns=[
                    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
                ])

                # Make prediction (0 = No Risk, 1 = Risk Detected)
                prediction = model.predict(input_data)[0]

            except Exception as e:
                error = f"Prediction failed: {str(e)}. Please ensure model.pkl is compatible."
        else:
            error = "Model not loaded. Please ensure model.pkl exists and is valid."

    return render_template('home.html', form=form, prediction=prediction, error=error)

if __name__ == "__main__":
    app.run(debug=False)
