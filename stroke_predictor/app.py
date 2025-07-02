from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load your model
model = joblib.load(r"D:\final\stroke_predictor\stroke_model.pkl")

@app.route('/')
def home():
    return render_template('home.html')

original_cols = joblib.load(r'D:\final\stroke_predictor\columns.pkl')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    result = None
    if request.method == 'POST':
        input_dict = {
            'gender': [request.form['gender']],                      
            'age': [float(request.form['age'])],                     
            'hypertension': [int(request.form['hypertension'])],    
            'heart_disease': [int(request.form['heart_disease'])],  
            'ever_married': [request.form['ever_married']],          
            'work_type': [request.form['work_type']],               
            'Residence_type': [request.form['Residence_type']],     
            'avg_glucose_level': [float(request.form['avg_glucose_level'])],
            'bmi': [float(request.form['bmi'])],
            'smoking_status': [request.form['smoking_status']]      
        }

        # Convert to DataFrame
        input_df = pd.DataFrame(input_dict)

        # One-hot encode same as during training
        input_encoded = pd.get_dummies(input_df, drop_first=True)

        # Align columns to what the model expects
        input_encoded = input_encoded.reindex(columns=original_cols, fill_value=0)

        # Predict
        prediction = model.predict(input_encoded)[0]
        result = "Stroke Risk Detected" if prediction == 1 else "No Stroke Risk"

    return render_template('predict.html', result=result)



if __name__ == '__main__':
    app.run(debug=True)
