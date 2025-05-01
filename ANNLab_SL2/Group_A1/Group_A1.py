import numpy as np 
import matplotlib.pyplot as plt 
 
# Define the activation functions 
def sigmoid(x): 
    return 1 / (1 + np.exp(-x)) 
 
def tanh(x): 
    return np.tanh(x) 
 
def relu(x): 
    return np.maximum(0, x) 
 
def leaky_relu(x, alpha=0.01): 
    return np.where(x > 0, x, alpha * x) 
 
# Generate values for the x-axis 
x = np.linspace(-10, 10, 400) 
 
# Plot each activation function 
plt.figure(figsize=(10, 8)) 
 
# Sigmoid plot 
plt.subplot(2, 2, 1) 
plt.plot(x, sigmoid(x), label='Sigmoid', color='y') 
plt.title('Sigmoid Activation Function') 
plt.grid(True) 
plt.legend() 
 
# Tanh plot 
plt.subplot(2, 2, 2) 
plt.plot(x, tanh(x), label='Tanh', color='g') 
plt.title('Tanh Activation Function') 
plt.grid(True) 
plt.legend() 
 
# ReLU plot 
plt.subplot(2, 2, 3) 
plt.plot(x, relu(x), label='ReLU', color='b') 
plt.title('ReLU Activation Function') 
plt.grid(True) 
plt.legend() 
# Leaky ReLU plot 
plt.subplot(2, 2, 4) 
plt.plot(x, leaky_relu(x), label='Leaky ReLU', color='purple') 
plt.title('Leaky ReLU Activation Function') 
plt.grid(True) 
plt.legend() 
# Display the plot 
plt.tight_layout() 
plt.show()     