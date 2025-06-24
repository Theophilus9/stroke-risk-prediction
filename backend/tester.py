import requests
import json

# URL of your local Flask server
url = "http://127.0.0.1:5000/predict"

# Sample input (must match model features)
data = {
    "gender": "Male",
    "age": 67,
    "hypertension": 0,
    "heart_disease": 1,
    "ever_married": "Yes",
    "work_type": "Private",
    "Residence_type": "Urban",
    "avg_glucose_level": 228.69,
    "bmi": 36.6,
    "smoking_status": "formerly smoked"
}

# Send POST request
response = requests.post(url, json=data)

# Print the result
print("Response:", response.json())
