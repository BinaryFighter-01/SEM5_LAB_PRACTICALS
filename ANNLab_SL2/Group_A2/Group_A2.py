import numpy as np

# McCulloch-Pitts Neuron for ANDNOT function
def mcCulloch_pitts_andnot(A, B):
    weights = np.array([1, -1])  # Weights for A and B
    inputs = np.array([A, B])    # Inputs
    threshold = 1                # Threshold value

    # Compute weighted sum
    weighted_sum = np.dot(weights, inputs)

    # Apply activation function (Step function)
    output = 1 if weighted_sum >= threshold else 0
    return output

# Testing ANDNOT function with all possible inputs
print("A  B  NOT B  | ANDNOT (A AND NOT B)")
print("------------------------------------")
for A in [0, 1]:
    for B in [0, 1]:
        NOT_B = 1 - B  # NOT B is the inverse of B
        print(A, B, "  ", NOT_B, "   | ", mcCulloch_pitts_andnot(A, B))
