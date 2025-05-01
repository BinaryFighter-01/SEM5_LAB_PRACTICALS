import numpy as np  # Import numpy for array operations

j = int(input("Enter a Number (0-9): "))  # Get user input (0-9) and convert to integer
step_function = lambda x: 1 if x >= 0 else 0  # Define step function: 1 if x >= 0, else 0

training_data = [                           # Training data: inputs and labels (1 = even, 0 = odd)
    {'input': [1, 1, 0, 0, 0, 0], 'label': 1},
    {'input': [1, 1, 0, 0, 0, 1], 'label': 0},
    {'input': [1, 1, 0, 0, 1, 0], 'label': 1},
    {'input': [1, 1, 0, 1, 1, 1], 'label': 0},
    {'input': [1, 1, 0, 1, 0, 0], 'label': 1},
    {'input': [1, 1, 0, 1, 0, 1], 'label': 0},
    {'input': [1, 1, 0, 1, 1, 0], 'label': 1},
    {'input': [1, 1, 0, 1, 1, 1], 'label': 0},
    {'input': [1, 1, 1, 0, 0, 0], 'label': 1},
    {'input': [1, 1, 1, 0, 0, 1], 'label': 0},
]

weights = np.array([0, 0, 0, 0, 0, 1])  # Initial weights for the perceptron

for data in training_data:  # Loop through training data
    input = np.array(data['input'])  # Convert input to numpy array
    label = data['label']  # Get the expected label (1 or 0)
    output = step_function(np.dot(input, weights))  # Compute output using weights
    error = label - output  # Calculate error (difference from label)
    weights += input * error  # Update weights based on error

input = np.array([int(x) for x in list('{0:06b}'.format(j))])  # Convert input number to 6-bit binary
output = "odd" if step_function(np.dot(input, weights)) == 0 else "even"  # Predict odd/even
print(j, " is ", output)  # Print the result