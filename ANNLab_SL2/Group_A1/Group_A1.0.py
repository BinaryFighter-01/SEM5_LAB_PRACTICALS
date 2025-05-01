import numpy as np
import matplotlib.pyplot as plt

# Define activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

# Function to plot the activation functions
def plot_af(function, function_name):
    x = np.linspace(-5, 5, 100)  
    y = function(x)
    plt.plot(x, y, label=function_name)

# Plot all activation functions
plot_af(sigmoid, 'Sigmoid')
plot_af(relu, 'ReLU')
plot_af(tanh, 'Tanh')
plot_af(leaky_relu, 'Leaky ReLU')

# Labels and title
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Activation Functions')
plt.legend()
plt.show()
