from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load("stroke_model.pkl")

# These must match the columns used during training
feature_columns = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
                   'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']

@app.route('/')
def home():
    return "Stroke Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])

        # Encode the categorical columns like during training
        df_encoded = pd.get_dummies(df)
        df_encoded = df_encoded.reindex(columns=model.feature_names_in_, fill_value=0)

        prediction = model.predict(df_encoded)
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)})
if __name__ == '__main__':
    app.run(debug=True)
