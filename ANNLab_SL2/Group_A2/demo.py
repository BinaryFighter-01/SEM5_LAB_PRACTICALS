import numpy as np

class McCullochPittsNeuron:
    def __init__(self, weights, threshold):
        self.weights = np.array(weights)
        self.threshold = threshold
    
    def activate(self, inputs):
        # Calculate weighted sum
        weighted_sum = np.dot(self.weights, inputs)
        # Apply threshold function
        return 1 if weighted_sum >= self.threshold else 0

def andnot_mcp(inputs):
    """
    Implements ANDNOT (A ∧ ¬B) function using McCulloch-Pitts neuron
    Input: [A, B]
    Output: 1 if A=1 and B=0, 0 otherwise
    """
    # For ANDNOT (A ∧ ¬B):
    # Weight for A = 1 (positive contribution)
    # Weight for B = -1 (inhibitory)
    # Threshold = 1 (fires only when A is 1 and B is 0)
    
    neuron = McCullochPittsNeuron(weights=[1, -1], threshold=1)
    return neuron.activate(inputs)

# Test the ANDNOT function
def test_andnot():
    test_cases = [
        [0, 0],  # A=0, B=0
        [0, 1],  # A=0, B=1
        [1, 0],  # A=1, B=0
        [1, 1]   # A=1, B=1
    ]
    
    print("Testing ANDNOT (A ∧ ¬B) function:")
    print("A B | Output")
    print("----------")
    for inputs in test_cases:
        output = andnot_mcp(inputs)
        print(f"{inputs[0]} {inputs[1]} |   {output}")

# Run the test
if __name__ == "__main__":
    test_andnot()