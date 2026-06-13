import cv2
import numpy as np
import os
from tensorflow import keras
from tensorflow.keras import layers

print("="*50)
print("CREATING SKIN CLASSIFIER MODEL")
print("="*50)

# Create models folder
os.makedirs("models", exist_ok=True)

# Build model
print("Building neural network...")
model = keras.Sequential([
    layers.Input(shape=(64, 64, 3)),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Create sample training data
print("Creating training data...")
X = np.random.random((100, 64, 64, 3))
y = np.random.randint(0, 2, 100)

# Train model
print("Training model (this will take about 30 seconds)...")
model.fit(X, y, epochs=10, verbose=1)

# Save model
model.save("models/skin_binary_classifier.h5")
print("\n" + "="*50)
print("SUCCESS! Model saved to models/skin_binary_classifier.h5")
print("="*50)