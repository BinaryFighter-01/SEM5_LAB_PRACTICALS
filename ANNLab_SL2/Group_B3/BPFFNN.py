import numpy as np

class BPFeedForwardNN:
    def __init__(self, layer_sizes, learning_rate=0.01):
        """
        Initialize the feed-forward neural network.
        layer_sizes: List of integers representing the number of nodes in each layer
                    (e.g., [2, 4, 1] for 2 input, 4 hidden, 1 output).
        learning_rate: Step size for weight updates.
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.num_layers = len(layer_sizes)
        self.weights = []
        self.biases = []
        self.initialize_parameters()

    def initialize_parameters(self):
        """Initialize weights and biases with random values."""
        np.random.seed(1)  # For reproducibility
        for i in range(self.num_layers - 1):
            # Weights: shape (current_layer_size, next_layer_size)
            w = np.random.randn(self.layer_sizes[i], self.layer_sizes[i+1]) * 0.01
            self.weights.append(w)
            # Biases: shape (1, next_layer_size)
            b = np.zeros((1, self.layer_sizes[i+1]))
            self.biases.append(b)

    def sigmoid(self, x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to avoid overflow

    def sigmoid_derivative(self, x):
        """Derivative of sigmoid function."""
        return x * (1 - x)

    def forward_propagation(self, X):
        """
        Perform forward propagation.
        X: Input data, shape (n_samples, input_size)
        Returns: Activations and weighted inputs for each layer
        """
        activations = [X]
        z_values = []

        for i in range(self.num_layers - 1):
            # Compute weighted input: z = X * W + b
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            z_values.append(z)
            # Apply activation function
            a = self.sigmoid(z)
            activations.append(a)

        return activations, z_values

    def backward_propagation(self, X, y, activations, z_values):
        """
        Perform backward propagation to compute gradients.
        X: Input data
        y: Target output
        activations: List of layer activations from forward pass
        z_values: List of weighted inputs from forward pass
        Returns: Gradients for weights and biases
        """
        m = X.shape[0]  # Number of samples
        weight_gradients = [np.zeros_like(w) for w in self.weights]
        bias_gradients = [np.zeros_like(b) for b in self.biases]

        # Compute output layer error
        delta = (activations[-1] - y) * self.sigmoid_derivative(activations[-1])
        
        # Backward pass through layers
        for i in range(self.num_layers - 2, -1, -1):
            # Compute gradients for weights and biases
            weight_gradients[i] = np.dot(activations[i].T, delta) / m
            bias_gradients[i] = np.sum(delta, axis=0, keepdims=True) / m

            if i > 0:  # No delta for input layer
                # Propagate error backward
                delta = np.dot(delta, self.weights[i].T) * self.sigmoid_derivative(activations[i])

        return weight_gradients, bias_gradients

    def update_parameters(self, weight_gradients, bias_gradients):
        """Update weights and biases using gradients."""
        for i in range(self.num_layers - 1):
            self.weights[i] -= self.learning_rate * weight_gradients[i]
            self.biases[i] -= self.learning_rate * bias_gradients[i]

    def train(self, X, y, epochs):
        """
        Train the neural network.
        X: Input data, shape (n_samples, input_size)
        y: Target output, shape (n_samples, output_size)
        epochs: Number of training iterations
        """
        for epoch in range(epochs):
            # Forward propagation
            activations, z_values = self.forward_propagation(X)
            
            # Compute loss (mean squared error)
            loss = np.mean((activations[-1] - y) ** 2)
            
            # Backward propagation
            weight_gradients, bias_gradients = self.backward_propagation(X, y, activations, z_values)
            
            # Update parameters
            self.update_parameters(weight_gradients, bias_gradients)
            
            # Print loss every 1000 epochs
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")

    def predict(self, X):
        """
        Predict output for input data.
        X: Input data, shape (n_samples, input_size)
        Returns: Predicted output
        """
        activations, _ = self.forward_propagation(X)
        return activations[-1]

# Example usage
if __name__ == "__main__":
    # Generate synthetic dataset (XOR-like binary classification)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Inputs
    y = np.array([[0], [1], [1], [0]])              # Outputs (XOR)

    # Initialize neural network: 2 input, 4 hidden, 1 output
    layer_sizes = [2, 4, 1]
    nn = BPFeedForwardNN(layer_sizes, learning_rate=0.1)

    # Train the network
    print("Training Neural Network...")
    nn.train(X, y, epochs=10000)

    # Test predictions
    print("\nTesting Predictions:")
    predictions = nn.predict(X)
    for i in range(len(X)):
        binary_pred = 1 if predictions[i][0] >= 0.5 else 0
        print(f"Input: {X[i]}, Predicted: {binary_pred}, Actual: {y[i][0]}")