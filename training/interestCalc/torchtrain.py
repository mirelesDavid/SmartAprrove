import pandas as pd
import numpy as np
import re
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import joblib
import matplotlib.pyplot as plt

# Import the mappings from the mappings.py file
from mappings import (
    grade_map, sub_grade_map, home_ownership_map, term_map, purpose_map
)

keep = [
    'loan_amnt', 'term', 'int_rate', 'installment', 'grade', 'sub_grade', 'emp_length', 
    'home_ownership', 'annual_inc', 'purpose', 'dti', 'fico_range_high', 'fico_range_low'
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

# Convert to numeric
df['emp_length'] = df['emp_length'].apply(emp_length_to_numeric)
df['grade'] = df['grade'].map(grade_map)
df['sub_grade'] = df['sub_grade'].map(sub_grade_map)
df['home_ownership'] = df['home_ownership'].map(home_ownership_map)
df['term'] = df['term'].map(term_map)
df['purpose'] = df['purpose'].map(purpose_map)

# Fill any NaNs that were created during the conversion
df.fillna(0, inplace=True)

# Separate target variables (int_rate, installment) and features
y = df[['int_rate', 'installment']]
X = df.drop(['int_rate', 'installment'], axis=1)

# Check for any discrepancies in column count before proceeding
print(f"X_train columns: {X.shape[1]}")

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler for later use
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)  # Use float for regression
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

# Build the PyTorch model
class LoanPredictionModel(nn.Module):
    def __init__(self, input_size):
        super(LoanPredictionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 2)  # Output 2 values (int_rate, installment)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)  # No activation since it's regression
        return x

# Initialize the model, loss function, and optimizer
input_size = X_train.shape[1]
model = LoanPredictionModel(input_size)
criterion = nn.MSELoss()  # Use MSELoss for regression
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Track loss over epochs
losses = []

# Training the model
epochs = 35
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
    test_loss = criterion(test_outputs, y_test_tensor)
    print(f"Test MSE Loss: {test_loss.item()}")

# Predicting new inputs
def predict_new_input(model, input_data, scaler):
    model.eval()
    input_scaled = scaler.transform(np.array(input_data).reshape(1, -1))
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
    
    with torch.no_grad():
        prediction = model(input_tensor)
    
    predicted_int_rate = prediction[0][0].item()
    predicted_installment = prediction[0][1].item()
    
    return predicted_int_rate, predicted_installment

# Example user input
# Example user input, ensuring it has the correct number of features
# Assuming the following order of features: 
# ['loan_amnt', 'term', 'grade', 'sub_grade', 'emp_length', 'home_ownership', 'annual_inc', 'purpose', 'dti', 'fico_range_high', 'fico_range_low']
new_user_input = [10000, 36, 0, 3, 5, 1, 60000, 4, 15, 700, 720]

# The input needs to match the number of features used for training (11 in this case).
predicted_int_rate, predicted_installment = predict_new_input(model, new_user_input, scaler)
print(f"Predicted int_rate: {predicted_int_rate}, Predicted installment: {predicted_installment}")