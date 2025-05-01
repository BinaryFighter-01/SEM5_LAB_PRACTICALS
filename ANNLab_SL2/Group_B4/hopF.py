import numpy as np

class HopfieldNetwork:
    def __init__(self, size):
        """
        Initialize the Hopfield Network.
        size: Dimension of the vectors (number of neurons).
        """
        self.size = size
        self.weights = np.zeros((size, size))

    def train(self, patterns):
        """
        Train the network by storing patterns using Hebbian learning.
        patterns: List of binary vectors (-1, +1).
        """
        for pattern in patterns:
            # Convert pattern to column vector
            pattern = np.array(pattern).reshape(self.size, 1)
            # Update weights: W = W + pattern * pattern^T (Hebbian rule)
            self.weights += np.dot(pattern, pattern.T)
        # Set diagonal to zero (no self-connections)
        np.fill_diagonal(self.weights, 0)
        # Normalize weights
        self.weights /= self.size

    def energy(self, state):
        """
        Compute the energy of a given state.
        state: Binary vector (-1, +1).
        """
        state = np.array(state).reshape(self.size, 1)
        return -0.5 * np.dot(state.T, np.dot(self.weights, state)).item()

    def update(self, state, max_iterations=100):
        """
        Update the state asynchronously until convergence or max iterations.
        state: Initial state vector.
        max_iterations: Maximum number of iterations.
        Returns: Final state after convergence.
        """
        state = np.array(state).copy()
        for _ in range(max_iterations):
            prev_state = state.copy()
            # Update each neuron asynchronously
            for i in range(self.size):
                # Compute input to neuron i: sum(W[i,j] * state[j])
                activation = np.dot(self.weights[i], state)
                # Apply sign function: +1 if activation >= 0, else -1
                state[i] = 1 if activation >= 0 else -1
            # Check for convergence
            if np.array_equal(state, prev_state):
                break
        return state

    def test_retrieval(self, test_pattern, original_patterns):
        """
        Test pattern retrieval and compare with original patterns.
        test_pattern: Input pattern (possibly noisy).
        original_patterns: List of stored patterns for comparison.
        """
        print(f"\nTesting pattern: {test_pattern}")
        retrieved = self.update(test_pattern)
        print(f"Retrieved pattern: {retrieved}")
        # Find closest original pattern
        for i, pattern in enumerate(original_patterns):
            if np.array_equal(retrieved, pattern):
                print(f"Matches stored pattern {i}: {pattern}")
                return
        print("No match with stored patterns")

# Example usage
if __name__ == "__main__":
    # Define four 5-dimensional binary vectors (-1, +1)
    patterns = [
        [1, -1, 1, -1, 1],   # Pattern 0
        [-1, 1, -1, 1, -1],  # Pattern 1
        [1, 1, -1, -1, 1],   # Pattern 2
        [-1, -1, 1, 1, -1]   # Pattern 3
    ]

    # Initialize Hopfield Network
    size = len(patterns[0])  # Vector dimension = 5
    hopfield = HopfieldNetwork(size)

    # Train the network with the patterns
    print("Training Hopfield Network with patterns:")
    for i, p in enumerate(patterns):
        print(f"Pattern {i}: {p}")
    hopfield.train(patterns)

    # Test retrieval with original patterns
    print("\nTesting retrieval with original patterns:")
    for pattern in patterns:
        hopfield.test_retrieval(pattern, patterns)

    # Test retrieval with noisy patterns
    print("\nTesting retrieval with noisy patterns:")
    noisy_patterns = [
        [1, -1, 1, -1, -1],  # Pattern 0 with 1 bit flipped
        [-1, 1, -1, -1, -1], # Pattern 1 with 1 bit flipped
        [1, 1, -1, 1, 1],    # Pattern 2 with 1 bit flipped
        [-1, -1, 1, -1, -1]  # Pattern 3 with 1 bit flipped
    ]
    for noisy in noisy_patterns:
        hopfield.test_retrieval(noisy, patterns)