import numpy as np

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# XOR dataset
X = np.array([[0,0], [0,1], [1,0], [1,1]])  # Inputs
y = np.array([[0], [1], [1], [0]])          # Expected outputs

# Initialize network parameters
np.random.seed(1)
input_size = 2
hidden_size = 4
output_size = 1
learning_rate = 0.01
epochs = 10000

# Initialize weights and biases
W1 = np.random.randn(input_size, hidden_size)    # Input to hidden weights
b1 = np.zeros((1, hidden_size))                 # Hidden layer bias
W2 = np.random.randn(hidden_size, output_size)  # Hidden to output weights
b2 = np.zeros((1, output_size))                 # Output layer bias

# Training the network
for epoch in range(epochs):
    # Forward propagation
    hidden_input = np.dot(X, W1) + b1
    hidden_output = sigmoid(hidden_input)
    
    output_input = np.dot(hidden_output, W2) + b2
    predicted_output = sigmoid(output_input)
    
    # Calculate error
    error = y - predicted_output
    
    # Backpropagation
    # Output layer gradients
    d_predicted_output = error * sigmoid_derivative(predicted_output)
    
    # Hidden layer gradients
    error_hidden = np.dot(d_predicted_output, W2.T)
    d_hidden_output = error_hidden * sigmoid_derivative(hidden_output)
    
    # Update weights and biases
    W2 += learning_rate * np.dot(hidden_output.T, d_predicted_output)
    b2 += learning_rate * np.sum(d_predicted_output, axis=0, keepdims=True)
    W1 += learning_rate * np.dot(X.T, d_hidden_output)
    b1 += learning_rate * np.sum(d_hidden_output, axis=0, keepdims=True)
    
    # Print error every 1000 epochs
    if epoch % 1000 == 0:
        loss = np.mean(np.abs(error))
        print(f'Epoch {epoch}, Loss: {loss}')

# Test the trained network
print("\nTesting XOR function:")
for i in range(len(X)):
    # Forward pass for testing
    hidden_input = np.dot(X[i:i+1], W1) + b1
    hidden_output = sigmoid(hidden_input)
    output_input = np.dot(hidden_output, W2) + b2
    predicted = sigmoid(output_input)
    
    # Convert to binary output (threshold at 0.5)
    binary_output = 1 if predicted[0][0] >= 0.5 else 0
    print(f"Input: {X[i]}, Predicted: {binary_output}, Actual: {y[i][0]}")