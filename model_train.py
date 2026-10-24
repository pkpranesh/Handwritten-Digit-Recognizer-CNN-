import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

print(f"Training samples : {X_train.shape}")
print(f"Testing samples  : {X_test.shape}")
print(f"Image shape      : {X_train[0].shape}")
print(f"Pixel range      : {X_train.min()} to {X_train.max()}")

#normalize
X_train = X_train / 255
X_test  = X_test  / 255

#reshape
X_train = X_train.reshape(-1, 28, 28, 1)
X_test  = X_test.reshape(-1, 28, 28, 1)


print(f"X_train shape: {X_train.shape}")
print(f"X_test shape : {X_test.shape}")

model = Sequential([
    # First Conv Block
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D((2,2)),
    
    # Second Conv Block
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    
    # Flatten + Dense
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.compile(
    optimizer = 'adam',
    loss      = 'sparse_categorical_crossentropy',
    metrics   = ['accuracy']
)

model.summary()

history = model.fit(
    X_train, y_train,
    epochs          = 10,
    batch_size      = 32,
    validation_split= 0.1,
    verbose         = 1
)

# ── Evaluate on Test Set ──────────────────────────────────
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy : {test_accuracy*100:.2f}%")
print(f"Test Loss     : {test_loss:.4f}")

# ── Plot Training History ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Accuracy plot
axes[0].plot(history.history['accuracy'],     label='Train')
axes[0].plot(history.history['val_accuracy'], label='Validation')
axes[0].set_title('Model Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()

# Loss plot
axes[1].plot(history.history['loss'],     label='Train')
axes[1].plot(history.history['val_loss'], label='Validation')
axes[1].set_title('Model Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

# ── Save Model ────────────────────────────────────────────
os.makedirs('model', exist_ok=True)
model.save('model/mnist_cnn.h5')
print("✅ Model saved!")