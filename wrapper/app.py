from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
import joblib

# Mappings (moved from mappings.py)
loan_status_map = {
    "Charged Off": 0,
    "Current": 1,
    "Fully Paid": 2,
    "In Grace Period": 3,
    "Late (31-120 days)": 4
}

grade_map = {
    "A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5
}

sub_grade_map = {
    "A1": 0, "A2": 1, "A3": 2, "A4": 3, "A5": 4,
    "B1": 5, "B2": 6, "B3": 7, "B4": 8, "B5": 9,
    "C1": 10, "C2": 11, "C3": 12, "C4": 13, "C5": 14,
    "D1": 15, "D2": 16, "D3": 17, "D4": 18, "D5": 19,
    "E1": 20, "E2": 21, "E3": 22, "E4": 23, "E5": 24,
    "F1": 25, "F2": 26, "F3": 27, "F4": 28, "F5": 29
}

home_ownership_map = {
    "RENT": 0, "OWN": 1, "MORTGAGE": 2
}

verification_status_map = {
    "Not Verified": 0, "Source Verified": 1, "Verified": 2
}

pymnt_plan_map = {
    "n": 0, "y": 1
}

debt_settlement_flag_map = {
    "N": 0, "Y": 1
}

settlement_status_map = {
    "ACTIVE": 0, "BROKEN": 1, "COMPLETE": 2
}

initial_list_status_map = {
    "w": 0, "f": 1
}

application_type_map = {
    "Individual": 0, "Joint App": 1
}

term_map = {
    "36 months": 36, "60 months": 60
}

verification_status_joint_map = {
    "Not Verified": 0, "Source Verified": 1, "Verified": 2,
    np.nan: -1  # 'Not Applicable' case
}

# Load the saved model and scaler from the 'model' directory
with open('model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load the LabelEncoder from the 'model' directory
label_encoder = joblib.load('model/label_encoder.pkl')

# Define the loan status mapping (invert for readability)
inverse_loan_status_map = {v: k for k, v in loan_status_map.items()}

# PyTorch model definition
class LoanPredictionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LoanPredictionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return self.softmax(x)

# Initialize the model and load its state from the 'model' directory
num_features = 124  # Adjust according to actual number of features
num_classes = len(label_encoder.classes_)
model = LoanPredictionModel(input_size=num_features, output_size=num_classes)
model.load_state_dict(torch.load('model/model.pth'))
model.eval()

# Create a Flask application
app = Flask(__name__)

# Home route to display the form
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Collect form data
        input_data = request.form.to_dict()

        # Convert the form data into a DataFrame for processing
        df = pd.DataFrame([input_data])

        # Preprocess input
        df.fillna(0, inplace=True)
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Scale the input
        X_test_scaled = scaler.transform(df)
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

        # Make prediction
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            _, predicted = torch.max(test_outputs, 1)

        # Inverse transform the predicted labels back to the original form
        predicted_label = label_encoder.inverse_transform(predicted.numpy())[0]
        predicted_label_str = inverse_loan_status_map[predicted_label]

        # Return the predicted result to the user
        return render_template('index.html', prediction=predicted_label_str)

    # On GET, render the form
    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
