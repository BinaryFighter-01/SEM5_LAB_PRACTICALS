# Perceptron to classify points based on AND logic
# Learning rate = 0.3

import numpy as np
import matplotlib.pyplot as plt

# Training data for AND logic
X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
Y = np.array([-1, -1, -1, 1])  # -1 for False, 1 for True (AND)

w = np.zeros(X.shape[1])  # Initialize weights (0, 0)
b = 0                     # Initialize bias

# Perceptron Training Loop
for _ in range(6):  # 6 epochs
    for i in range(X.shape[0]):
        y_pred = np.sign(np.dot(X[i], w) + b)  # Predict output
        
        if y_pred != Y[i]:  # If prediction is wrong
            w += 0.3 * Y[i] * X[i]  # Update weights
            b += 0.3 * Y[i]         # Update bias

# Define Plot Boundaries
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

Z = np.sign(np.dot(np.c_[xx.ravel(), yy.ravel()], w) + b)
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.8)  # Plot Decision Boundary
plt.scatter(X[:, 0], X[:, 1], c=(Y + 1) // 2, edgecolor='k')  # Convert Y (-1,1) to (0,1)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Perceptron Decision Regions for AND Logic')
plt.grid(True)
plt.show()
