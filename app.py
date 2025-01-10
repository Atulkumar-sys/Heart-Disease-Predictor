from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load your trained model
model = joblib.load('heart_disease_predictor.pkl')

@app.route('/')
def index():
    return render_template('index.html', prediction=None)  # Ensure no prediction by default

@app.route('/predict', methods=['POST'])
def predict():
    # Get the values from the form
    data = {
        'age': float(request.form.get('age')),
        'sex': float(request.form.get('sex')),
        'chest pain type': float(request.form.get('chest_pain_type')),
        'resting bp s': float(request.form.get('resting_bp_s')),
        'cholesterol': float(request.form.get('cholesterol')),
        'fasting blood sugar': float(request.form.get('fasting_blood_sugar')),
        'max heart rate': float(request.form.get('max_heart_rate')),
        'oldpeak': float(request.form.get('oldpeak')),
        'ST slope': float(request.form.get('st_slope')),
        'exercise angina': float(request.form.get('exercise_angina'))
    }

    # Creating a DataFrame
    input_data = pd.DataFrame([data])
    
    # Make prediction
    prediction = model.predict(input_data)

    return render_template('index.html', prediction=prediction[0])  # Show prediction on the same page

if __name__ == '__main__':
    app.run(debug=True)