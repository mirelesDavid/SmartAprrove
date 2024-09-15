import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
from sklearn.preprocessing import StandardScaler
import joblib
import re

# Import the mappings from the mappings.py file
from mappings import (
    loan_status_map, grade_map, sub_grade_map, home_ownership_map, term_map
)

keep = [
    'loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'term', 'int_rate', 'installment', 
    'grade', 'sub_grade', 'emp_length', 'home_ownership', 'annual_inc', 'verification_status', 
    'issue_d', 'loan_status', 'purpose', 'dti', 'delinq_2yrs', 'earliest_cr_line', 'fico_range_low', 
    'fico_range_high', 'last_pymnt_d', 'next_pymnt_d'
]

# Load the dataset
df = pd.read_csv('primo1k.csv', low_memory=False)
df = df[keep]

# Function to convert emp_length to numeric
def emp_length_to_numeric(emp_length):
    if pd.isnull(emp_length):
        return 0
    if emp_length == '10+ years':
        return 10
    if emp_length == '< 1 year':
        return 0.5
    else:
        return int(re.search(r'\d+', emp_length).group())

# Apply emp_length_to_numeric function
df['emp_length'] = df['emp_length'].apply(emp_length_to_numeric)

# Apply the mappings to the dataset
df['loan_status'] = df['loan_status'].map(loan_status_map)
df['grade'] = df['grade'].map(grade_map)
df['sub_grade'] = df['sub_grade'].map(sub_grade_map)
df['home_ownership'] = df['home_ownership'].map(home_ownership_map)
df['term'] = df['term'].map(term_map)

# Handle date columns (converting to datetime and extracting year)
date_columns = [
    'issue_d', 'earliest_cr_line', 'last_pymnt_d', 'next_pymnt_d'
]
for col in date_columns:
    df[col] = pd.to_datetime(df[col], format='%b-%Y', errors='coerce').dt.year

# Convert any remaining object columns to numeric (if needed)
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Fill any NaNs that were created during the conversion
df.fillna(0, inplace=True)

# Separate features from the target variable (loan_status)
y_test_actual = df['loan_status']  # For evaluation later
X_test = df.drop('loan_status', axis=1)

# Load the saved scaler and apply it to the test data
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
X_test_scaled = scaler.transform(X_test)

# Convert test data to PyTorch tensors
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

# Load the model
# Define the LoanPredictionModel with the forward method
class LoanPredictionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LoanPredictionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  # Changed from 64 to 128
        self.fc2 = nn.Linear(128, 64)          # Changed from 64 to 32
        self.fc3 = nn.Linear(64, 32)           # Changed from 32 to 16
        self.fc4 = nn.Linear(32, output_size)  # Output size remains 5
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return self.softmax(x)



# Load the LabelEncoder to determine the number of classes
label_encoder = joblib.load('label_encoder.pkl')
num_classes = len(label_encoder.classes_)

# Initialize the model with the correct output size
model = LoanPredictionModel(input_size=X_test.shape[1], output_size=num_classes)

# Load the saved model state safely
model.load_state_dict(torch.load('model.pth', weights_only=True))
model.eval()

# Run predictions on the test set
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    _, predicted = torch.max(test_outputs, 1)

# Inverse transform the predicted labels back to the original form
predicted_labels = label_encoder.inverse_transform(predicted.numpy())

# Invert the loan_status_map to map numeric values back to their original loan status descriptions
inverse_loan_status_map = {v: k for k, v in loan_status_map.items()}

# Convert actual and predicted values back to the original string format using the inverted map
y_test_actual_str = y_test_actual.map(inverse_loan_status_map)
predicted_labels_str = pd.Series(predicted_labels).map(inverse_loan_status_map)

# Output the actual vs predicted results in a readable format with string labels
results_df = pd.DataFrame({
    'Actual': y_test_actual_str,
    'Predicted': predicted_labels_str
})

# Display the first few rows of results
print(results_df.head(100))

# Calculate accuracy
accuracy = (predicted_labels == y_test_actual.values).sum() / len(y_test_actual)
print(f"Model accuracy on test set: {accuracy:.4f}")

# Check class distribution in predictions
print(f"Class distribution in predictions: {np.bincount(predicted.numpy())}")

# Save the results to CSV with string labels
results_df.to_csv('test_results.csv', index=False)
print("Results saved to 'test_results.csv'")
