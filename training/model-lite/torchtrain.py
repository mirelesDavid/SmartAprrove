import pandas as pd
import numpy as np
import re
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import joblib
import matplotlib.pyplot as plt

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
df = pd.read_csv('accepted_2007_to_2018Q4.csv', low_memory=False)
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

# Save the columns (features) to columns.pkl
columns = df.columns.to_list()
print(f"Columns used in training: {columns}")  # Track columns used during training
print(f"Number of columns: {len(columns)}")    # Ensure this matches 125 columns
with open('columns.pkl', 'wb') as f:
    pickle.dump(columns, f)

# Separate target variable (loan_status) and features
y = df['loan_status']
X = df.drop('loan_status', axis=1)

# Check for any discrepancies in column count before proceeding
print(f"X_train columns: {X.shape[1]}")  # Ensure this matches the saved column count

# Save the columns (features) to columns.pkl
columns = df.columns.to_list()
with open('columns-afterdrop.pkl', 'wb') as f:
    pickle.dump(columns, f)

# Check class distribution
print(f"Class distribution in y_train: {np.bincount(y)}")

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler for later use
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Label Encoding for loan_status
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Save the LabelEncoder
joblib.dump(label_encoder, 'label_encoder.pkl')

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Build the PyTorch model
class LoanPredictionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LoanPredictionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  # Increased from 64
        self.fc2 = nn.Linear(128, 64)          # Increased from 32
        self.fc3 = nn.Linear(64, 32)           # Increased from 16
        self.fc4 = nn.Linear(32, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return self.softmax(x)


# Initialize the model, loss function, and optimizer
input_size = X_train.shape[1]  # Make sure this matches the expected input size (125)
output_size = len(np.unique(y_encoded))

print(f"Input size: {input_size}, Output size: {output_size}")

model = LoanPredictionModel(input_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Lowered learning rate

# Track loss over epochs
losses = []

# Training the model
epochs = 50
batch_size = 32

for epoch in range(epochs):
    model.train()
    
    permutation = torch.randperm(X_train_tensor.size()[0])
    
    epoch_loss = 0
    for i in range(0, X_train_tensor.size()[0], batch_size):
        optimizer.zero_grad()

        indices = permutation[i:i+batch_size]
        batch_X, batch_y = X_train_tensor[indices], y_train_tensor[indices]

        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()

    avg_loss = epoch_loss / (X_train_tensor.size()[0] // batch_size)
    losses.append(avg_loss)
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss}")

# Plot the loss over epochs
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.show()

# Save the model
torch.save(model.state_dict(), 'model.pth')

# Evaluating the model on the test set
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    _, predicted = torch.max(test_outputs, 1)
    accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)

print(f"Model accuracy on test set: {accuracy}")

# Check distribution of predictions
print(f"Class distribution in predictions: {np.bincount(predicted.numpy())}")
