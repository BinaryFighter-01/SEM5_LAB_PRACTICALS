import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Generate synthetic dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train).reshape(-1, 1)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test).reshape(-1, 1)

# Define the neural network
class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.layer3(x))
        return x

# Initialize model, loss, and optimizer
model = NeuralNetwork(input_size=X.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
print("Training PyTorch Neural Network...")
model.train()
epochs = 20
batch_size = 32
n_batches = len(X_train) // batch_size

for epoch in range(epochs):
    for i in range(0, len(X_train), batch_size):
        X_batch = X_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]
        
        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Print loss every 5 epochs
    if (epoch + 1) % 5 == 0:
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_train)
            val_loss = criterion(val_outputs, y_train)
        model.train()
        print(f"Epoch {epoch+1}, Loss: {val_loss.item():.4f}")

# Evaluate the model
print("\nEvaluating on test set:")
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    test_loss = criterion(test_outputs, y_test)
    test_pred = (test_outputs >= 0.5).float()
    test_accuracy = (test_pred == y_test).float().mean().item()
print(f"Test Loss: {test_loss.item():.4f}, Test Accuracy: {test_accuracy:.4f}")

# Scikit-learn accuracy
y_pred = test_pred.numpy().flatten()
sklearn_accuracy = accuracy_score(y_test.numpy().flatten(), y_pred)
print(f"Scikit-learn Test Accuracy: {sklearn_accuracy:.4f}")