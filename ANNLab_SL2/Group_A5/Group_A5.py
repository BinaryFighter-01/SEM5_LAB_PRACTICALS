import numpy as np

# Input and output patterns (two pairs of vectors)
X = np.array([[1, 1, 1, 1], [-1, -1, -1, -1]])  # X vectors (4D)
Y = np.array([[-1, -1], [-1, -1]])               # Y vectors (2D)

# Weight matrix calculation
W = np.dot(Y.T, X)  # W = Y transpose * X for X-to-Y recall

# Activation function (strictly -1 or 1)
def activate(v):
    return np.where(v >= 0, 1, -1)  # If >= 0, return 1; else -1

# Bidirectional recall functions
def bam_x_to_y(x, max_iter=10):
    y = np.zeros(2)  # Start with a neutral Y
    for _ in range(max_iter):  # Iterate to stabilize
        y_new = activate(np.dot(W, x))  # Compute Y from X
        if np.array_equal(y, y_new):  # Stop if no change
            break
        y = y_new
    return y

def bam_y_to_x(y, max_iter=10):
    x = np.zeros(4)  # Start with a neutral X
    for _ in range(max_iter):  # Iterate to stabilize
        x_new = activate(np.dot(W.T, y))  # Compute X from Y using W transpose
        if np.array_equal(x, x_new):  # Stop if no change
            break
        x = x_new
    return x

# Take user input for test pattern (X or Y)
choice = input("Enter 'X' to test an X vector (4D) or 'Y' to test a Y vector (2D): ").strip().upper()

if choice == 'X':
    x_test = list(map(int, input("Enter a 4D binary vector (e.g., 1 -1 -1 1): ").split()))
    x_test = np.array(x_test)
    y_pred = bam_x_to_y(x_test)
    print("Input X:", x_test)
    print("Predicted Y:", y_pred)
elif choice == 'Y':
    y_test = list(map(int, input("Enter a 2D binary vector (e.g., -1 -1): ").split()))
    y_test = np.array(y_test)
    x_pred = bam_y_to_x(y_test)
    print("Input Y:", y_test)
    print("Predicted X:", x_pred)
else:
    print("Invalid choice! Please enter 'X' or 'Y'.")

# Test with stored patterns to verify
print("\nVerifying stored patterns:")
for i in range(2):
    y_from_x = bam_x_to_y(X[i])
    x_from_y = bam_y_to_x(Y[i])
    print(f"X{i}: {X[i]} -> Predicted Y: {y_from_x}, Actual Y: {Y[i]}")
    print(f"Y{i}: {Y[i]} -> Predicted X: {x_from_y}, Actual X: {X[i]}")