import numpy as np

class ART1:
    def __init__(self, input_size, max_clusters, vigilance):
        """
        Initialize ART1 network.
        input_size: Number of input nodes (size of binary input vector).
        max_clusters: Maximum number of clusters allowed.
        vigilance: Vigilance parameter (0 to 1) controlling cluster similarity.
        """
        self.input_size = input_size
        self.max_clusters = max_clusters
        self.vigilance = vigilance
        self.L = 2  # Parameter for bottom-up weights
        self.clusters = 0  # Current number of clusters
        # Initialize bottom-up and top-down weights
        self.bottom_up = np.zeros((max_clusters, input_size))
        self.top_down = np.zeros((max_clusters, input_size))
        self.active_clusters = []  # Track active cluster indices

    def initialize_weights(self, cluster_idx):
        """Initialize weights for a new cluster."""
        # Bottom-up weights: small initial values
        self.bottom_up[cluster_idx] = self.L / (self.L - 1 + self.input_size)
        # Top-down weights: all 1s initially
        self.top_down[cluster_idx] = np.ones(self.input_size)

    def norm(self, vector):
        """Compute L1 norm (sum of elements) for binary vector."""
        return np.sum(vector)

    def train(self, patterns):
        """
        Train ART1 on a list of binary input patterns.
        patterns: List of binary input vectors.
        """
        for pattern in patterns:
            self.process_pattern(pattern)

    def process_pattern(self, pattern):
        """Process a single input pattern."""
        # Ensure pattern is binary
        pattern = np.array(pattern, dtype=int)
        if not np.all(np.logical_or(pattern == 0, pattern == 1)):
            print("Error: Input pattern must be binary (0 or 1).")
            return

        # If no clusters exist, create first cluster
        if self.clusters == 0:
            self.initialize_weights(0)
            self.top_down[0] = pattern
            self.clusters = 1
            self.active_clusters.append(0)
            print(f"Created cluster 0 for pattern {pattern}")
            return

        # Compute activation for each cluster (bottom-up)
        activations = []
        for j in range(self.clusters):
            activation = np.sum(self.bottom_up[j] * pattern)
            activations.append((activation, j))

        # Sort clusters by activation (highest first)
        activations.sort(reverse=True)
        matched = False

        # Try matching with existing clusters
        for _, j in activations:
            # Compute intersection of input and top-down weights
            intersection = np.minimum(pattern, self.top_down[j])
            similarity = self.norm(intersection) / self.norm(pattern)

            # Check if similarity meets vigilance threshold
            if similarity >= self.vigilance:
                # Resonance: Update weights
                self.top_down[j] = np.minimum(self.top_down[j], pattern)
                self.bottom_up[j] = (self.L * self.top_down[j]) / (self.L - 1 + self.norm(self.top_down[j]))
                matched = True
                print(f"Pattern {pattern} matched cluster {j}, updated weights")
                break

        # If no match and clusters available, create new cluster
        if not matched and self.clusters < self.max_clusters:
            new_cluster = self.clusters
            self.initialize_weights(new_cluster)
            self.top_down[new_cluster] = pattern
            self.clusters += 1
            self.active_clusters.append(new_cluster)
            print(f"Created cluster {new_cluster} for pattern {pattern}")
        elif not matched:
            print(f"Pattern {pattern} not matched, max clusters reached")

    def predict(self, pattern):
        """Predict cluster for a given pattern."""
        pattern = np.array(pattern, dtype=int)
        if not np.all(np.logical_or(pattern == 0, pattern == 1)):
            return None, "Input pattern must be binary (0 or 1)."

        max_activation = -1
        best_cluster = None

        # Find cluster with highest activation
        for j in range(self.clusters):
            activation = np.sum(self.bottom_up[j] * pattern)
            if activation > max_activation:
                intersection = np.minimum(pattern, self.top_down[j])
                similarity = self.norm(intersection) / self.norm(pattern)
                if similarity >= self.vigilance:
                    max_activation = activation
                    best_cluster = j

        if best_cluster is not None:
            return best_cluster, f"Pattern {pattern} matches cluster {best_cluster}"
        return None, f"Pattern {pattern} does not match any cluster"

# Example usage
if __name__ == "__main__":
    # Define 3x3 binary patterns (flattened to 9 elements)
    patterns = [
        [1, 1, 1, 0, 1, 0, 0, 1, 0],  # T-like pattern
        [1, 1, 1, 0, 1, 0, 0, 1, 1],  # Similar to T
        [0, 1, 0, 1, 1, 1, 0, 1, 0],  # Cross-like pattern
        [1, 0, 1, 0, 1, 0, 1, 0, 1],  # Checkerboard pattern
    ]

    # Initialize ART1 network
    input_size = 9  # 3x3 grid
    max_clusters = 5
    vigilance = 0.7
    art = ART1(input_size, max_clusters, vigilance)

    # Train the network
    print("Training ART1...")
    art.train(patterns)

    # Test predictions
    print("\nTesting predictions:")
    test_patterns = patterns + [[1, 1, 1, 0, 0, 0, 0, 0, 0]]  # Add a new pattern
    for pattern in test_patterns:
        cluster, message = art.predict(pattern)
        print(message)