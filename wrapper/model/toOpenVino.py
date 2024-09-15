import onnxruntime as ort
import numpy as np

# Load the ONNX model
onnx_model_path = 'model.onnx'
session = ort.InferenceSession(onnx_model_path)

# Get model input details
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
input_type = session.get_inputs()[0].type

print(f"Input name: {input_name}")
print(f"Input shape: {input_shape}")
print(f"Input type: {input_type}")

# Create dummy input matching the model's input shape (change shape if needed)
# Assuming the input shape is [1, 3, 224, 224] for a single batch of images (example for image models)
dummy_input = np.random.randn(*input_shape).astype(np.float32)

# Run inference
outputs = session.run(None, {input_name: dummy_input})

# Print the output (you can check output shape and values)
print("Output:", outputs)
