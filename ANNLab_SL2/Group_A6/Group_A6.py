import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def train_neural_network(X, y, learning_rate=0.1, epochs=10000):
    input_neurons = X.shape[1]
    hidden_neurons = 4
    output_neurons = y.shape[1]

    hidden_weights = np.random.uniform(size=(input_neurons, hidden_neurons))
    hidden_bias = np.random.uniform(size=(1, hidden_neurons))
    output_weights = np.random.uniform(size=(hidden_neurons, output_neurons))
    output_bias = np.random.uniform(size=(1, output_neurons))

    for _ in range(epochs):
        #Forward pass (calculate predictions)
        # Input -> Hidden Layer
        hidden_input = np.dot(X, hidden_weights) + hidden_bias
        hidden_output = sigmoid(hidden_input)
         

        #BP - learn from mistakes
        # Hidden -> Output Layer
        output_input = np.dot(hidden_output, output_weights) + output_bias
        predicted_output = sigmoid(output_input)
        
        #Error Cal
        error = y - predicted_output 
        output_adjustment = error * sigmoid_derivative(predicted_output) # Scale error by sigmoid derivative

        hidden_error = output_adjustment.dot(output_weights.T)
        hidden_adjustment = hidden_error * sigmoid_derivative(hidden_output)

        # Output layer W&B
        output_weights += learning_rate * hidden_output.T.dot(output_adjustment)
        output_bias += learning_rate * np.sum(output_adjustment, axis=0, keepdims=True)

        # Hidde Layer 
        hidden_weights += learning_rate * X.T.dot(hidden_adjustment)
        hidden_bias += learning_rate * np.sum(hidden_adjustment, axis=0, keepdims=True)

    return predicted_output

X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
y = np.array([[0], [1], [1], [0]])

predicted_output = train_neural_network(X, y, learning_rate=0.1, epochs=10000)

print("Predicted Output:\n", predicted_output)