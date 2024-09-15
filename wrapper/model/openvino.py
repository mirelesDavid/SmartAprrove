import torch
import torch.nn as nn

class LoanPredictionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LoanPredictionModel, self).__init__()
        # Fully connected layer 1 (input -> 128 units)
        self.fc1 = nn.Linear(input_size, 128)
        # Fully connected layer 2 (128 -> 64 units)
        self.fc2 = nn.Linear(128, 64)
        # Fully connected layer 3 (64 -> 32 units)
        self.fc3 = nn.Linear(64, 32)
        # Output layer (32 -> output size, determined by the number of classes in target)
        self.fc4 = nn.Linear(32, output_size)
        
        # Activation functions
        self.relu = nn.ReLU()
        # Softmax for output probabilities
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Pass the input through each layer with ReLU activations
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        # Apply softmax to output layer for classification
        return self.softmax(x)

# Initialize the model with correct dimensions
input_size = 21  # Adjusted based on error
output_size = 5   # Adjusted based on error

model = LoanPredictionModel(input_size, output_size)
model.load_state_dict(torch.load('model2.pth', weights_only=True))  # Load the trained weights more securely
model.eval()

# Create a dummy input tensor with the correct shape
dummy_input = torch.randn(1, input_size)  # Shape should match the input size (125 features)

# Export the model to ONNX format
torch.onnx.export(model, dummy_input, "loan_model.onnx", opset_version=11)
