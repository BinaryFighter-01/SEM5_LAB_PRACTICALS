import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.metrics import accuracy_score

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Load and preprocess MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize and reshape
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# One-hot encode labels
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Define CNN architecture
def create_model():
    inputs = tf.keras.Input(shape=(28, 28, 1))
    # Conv1
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    # Conv2
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    # Flatten and Dense
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(10, activation='softmax')(x)
    return tf.keras.Model(inputs, outputs)

# Initialize model
model = create_model()

# Define loss and optimizer
loss_fn = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Training function
@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_fn(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, predictions

# Evaluation function
@tf.function
def test_step(images, labels):
    predictions = model(images, training=False)
    loss = loss_fn(labels, predictions)
    return loss, predictions

# Training loop
print("Training Core TensorFlow CNN...")
epochs = 10
batch_size = 128
steps_per_epoch = len(X_train) // batch_size

for epoch in range(epochs):
    # Shuffle training data
    indices = np.random.permutation(len(X_train))
    X_train_shuffled = X_train[indices]
    y_train_shuffled = y_train[indices]
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Batch training
    for i in range(0, len(X_train), batch_size):
        X_batch = X_train_shuffled[i:i+batch_size]
        y_batch = y_train_shuffled[i:i+batch_size]
        
        loss, predictions = train_step(X_batch, y_batch)
        running_loss += loss.numpy()
        
        # Compute batch accuracy
        predicted_classes = tf.argmax(predictions, axis=1)
        true_classes = tf.argmax(y_batch, axis=1)
        correct += tf.reduce_sum(tf.cast(predicted_classes == true_classes, tf.float32)).numpy()
        total += y_batch.shape[0]
    
    epoch_loss = running_loss / steps_per_epoch
    epoch_accuracy = correct / total
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

# Evaluate on test set
print("\nEvaluating on test set:")
test_loss = 0.0
y_true, y_pred = [], []

for i in range(0, len(X_test), batch_size):
    X_batch = X_test[i:i+batch_size]
    y_batch = y_test[i:i+batch_size]
    
    loss, predictions = test_step(X_batch, y_batch)
    test_loss += loss.numpy()
    
    predicted_classes = tf.argmax(predictions, axis=1).numpy()
    true_classes = tf.argmax(y_batch, axis=1).numpy()
    y_pred.extend(predicted_classes)
    y_true.extend(true_classes)

test_loss /= (len(X_test) // batch_size)
test_accuracy = accuracy_score(y_true, y_pred)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
print(f"Scikit-learn Test Accuracy: {test_accuracy:.4f}")