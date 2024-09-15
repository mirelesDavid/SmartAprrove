from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import pickle
import re
from datetime import datetime

# Create the Flask app
app = Flask(__name__)

# Load the saved scaler and label encoder
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

label_encoder = joblib.load('label_encoder.pkl')

# Import the mappings from the mappings.py file (this assumes you have a mappings.py file)
from mappings import loan_status_map, grade_map, sub_grade_map, home_ownership_map, term_map

# Reverse loan_status_map to map predicted numerical values back to loan status strings
inverse_loan_status_map = {v: k for k, v in loan_status_map.items()}

# Define the LoanPredictionModel class
class LoanPredictionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LoanPredictionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return self.softmax(x)

# Initialize and load the trained model
model = LoanPredictionModel(input_size=21, output_size=5)  # Adjust input/output sizes accordingly
model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
model.eval()

# Function to convert employment length to numeric
def emp_length_to_numeric(emp_length):
    if pd.isnull(emp_length):
        return 0
    if emp_length == '10+ years':
        return 10
    if emp_length == '< 1 year':
        return 0.5
    else:
        return int(re.search(r'\d+', emp_length).group())

# Function to handle date conversions (string to year)
def handle_date(date_str):
    try:
        return pd.to_datetime(date_str, format='%b-%Y').year
    except Exception:
        return 0  # Default to 0 if the date cannot be parsed

# Route for the homepage (data input form)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    form_data = request.json  # Usar request.json ya que estás enviando JSON

    # Verificar los datos recibidos
    print("Datos recibidos en Flask:", form_data)

    try:
        # Convertir los datos a DataFrame
        user_input = pd.DataFrame([{
            'loan_amnt': float(form_data.get('loan_amnt', 0)),
            'funded_amnt': float(form_data.get('funded_amnt', 0)),
            'funded_amnt_inv': float(form_data.get('funded_amnt_inv', 0)),
            'term': float(form_data.get('term', 0)),
            'int_rate': float(form_data.get('int_rate', 0)),
            'installment': float(form_data.get('installment', 0)),
            'grade': form_data.get('grade', ""),
            'sub_grade': form_data.get('sub_grade', ""),
            'emp_length': form_data.get('emp_length', ""),
            'home_ownership': form_data.get('home_ownership', ""),
            'annual_inc': float(form_data.get('annual_inc', 0)),
            'verification_status': form_data.get('verification_status', ""),
            'issue_d': form_data.get('issue_d', ""),
            'purpose': form_data.get('purpose', ""),
            'dti': float(form_data.get('dti', 0)),
            'delinq_2yrs': int(form_data.get('delinq_2yrs', 0)),
            'earliest_cr_line': form_data.get('earliest_cr_line', ""),
            'fico_range_low': float(form_data.get('fico_range_low', 0)),
            'fico_range_high': float(form_data.get('fico_range_high', 0)),
            'last_pymnt_d': form_data.get('last_pymnt_d', ""),
            'next_pymnt_d': form_data.get('next_pymnt_d', "")
        }])

        # Verifica que los datos han sido correctamente convertidos
        print("Datos convertidos en DataFrame:", user_input)

        # Apply the same transformations as in the original script
        # Convert emp_length to numeric
        user_input['emp_length'] = user_input['emp_length'].apply(emp_length_to_numeric)

        # Apply mappings
        user_input['grade'] = user_input['grade'].map(grade_map)
        user_input['sub_grade'] = user_input['sub_grade'].map(sub_grade_map)
        user_input['home_ownership'] = user_input['home_ownership'].map(home_ownership_map)
        user_input['term'] = user_input['term'].map(term_map)

        # Handle date columns (convert to year)
        user_input['issue_d'] = user_input['issue_d'].apply(handle_date)
        user_input['earliest_cr_line'] = user_input['earliest_cr_line'].apply(handle_date)
        user_input['last_pymnt_d'] = user_input['last_pymnt_d'].apply(handle_date)
        user_input['next_pymnt_d'] = user_input['next_pymnt_d'].apply(handle_date)

        # Convert any remaining object columns to numeric (if needed)
        for col in user_input.columns:
            if user_input[col].dtype == 'object':
                user_input[col] = pd.to_numeric(user_input[col], errors='coerce')

        # Fill any NaNs created during conversion
        user_input.fillna(0, inplace=True)

        # Scale the input data
        user_input_scaled = scaler.transform(user_input)

        # Convert the scaled data into a tensor
        user_input_tensor = torch.tensor(user_input_scaled, dtype=torch.float32)

        # Run the model to make predictions
        with torch.no_grad():
            output = model(user_input_tensor)
            _, predicted = torch.max(output, 1)

        # Inverse transform the predicted label using the reverse loan_status_map
        predicted_label_numeric = predicted.item()
        predicted_label = inverse_loan_status_map.get(predicted_label_numeric, "Unknown")

        print("Resultado de la predicción:", predicted_label)

        return {"prediction": predicted_label}

    except Exception as e:
        # Capturar cualquier error y devolver un mensaje de error claro
        print(f"Error durante el procesamiento de la predicción: {str(e)}")
        return {"error": f"Error durante el procesamiento: {str(e)}"}, 500


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
