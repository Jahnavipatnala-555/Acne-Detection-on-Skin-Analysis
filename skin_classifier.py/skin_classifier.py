"""
BINARY CLASSIFIER: Is this image showing skin or not?

WHY WE NEED THIS:
- Before analyzing skin conditions, we must confirm the image actually contains skin
- This prevents errors when someone uploads a picture of a cat or a car
- It's the first "gate" in our analysis pipeline

HOW IT WORKS:
1. Takes an image as input
2. Analyzes color patterns (skin has specific color ranges)
3. Uses a neural network trained on skin vs non-skin images
4. Returns: True (is skin) or False (not skin) with confidence score
"""

import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import os


class SkinBinaryClassifier:
    """
    This class determines if an image contains human skin or not.
    
    Think of it like a bouncer at a club - it only lets "skin images" through
    to the next stage of analysis.
    """
    
    def __init__(self):
        # The trained model (starts as None, we load/create it later)
        self.model = None
        
        # Path where we save the trained model
        self.model_path = "models/skin_binary_classifier.h5"
        
        # Size we resize all images to (neural networks need consistent sizes)
        self.image_size = (64, 64)
        
        # Try to load existing model, or create new one
        self._load_or_create_model()
    
    def _load_or_create_model(self):
        """
        Attempts to load a pre-trained model.
        If none exists, creates a new untrained model.
        """
        # Create models folder if it doesn't exist
        os.makedirs("models", exist_ok=True)
        
        if os.path.exists(self.model_path):
            # Load existing trained model
            print("Loading existing skin classifier model...")
            self.model = keras.models.load_model(self.model_path)
        else:
            # Create new model architecture
            print("Creating new skin classifier model...")
            self.model = self._build_model()
    
    def _build_model(self):
        """
        Builds the neural network architecture.
        
        ARCHITECTURE EXPLANATION:
        - Conv2D layers: Look for patterns in the image (edges, colors, textures)
        - MaxPooling2D: Shrinks the image to focus on important features
        - Flatten: Converts 2D image data to 1D for final decision
        - Dense layers: Makes the final yes/no decision
        
        Think of it like:
        1. First look at small details (Conv2D)
        2. Step back to see bigger picture (MaxPooling)
        3. Repeat a few times
        4. Make final decision (Dense)
        """
        model = keras.Sequential([
            # Input: 64x64 pixel image with 3 color channels (Red, Green, Blue)
            layers.Input(shape=(64, 64, 3)),
            
            # First convolution block - finds basic patterns
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            
            # Second convolution block - finds more complex patterns
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            
            # Third convolution block - finds even more complex patterns
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            
            # Flatten: convert 2D feature maps to 1D vector
            layers.Flatten(),
            
            # Dense layer: combine all features
            layers.Dense(128, activation='relu'),
            
            # Dropout: prevents overfitting (randomly ignores some neurons during training)
            layers.Dropout(0.5),
            
            # Output layer: single neuron, sigmoid gives probability 0-1
            # Close to 1 = skin, close to 0 = not skin
            layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile: set up how the model learns
        model.compile(
            optimizer='adam',           # Algorithm for updating weights
            loss='binary_crossentropy', # Loss function for yes/no problems
            metrics=['accuracy']        # What we measure during training
        )
        
        return model
    
    def is_skin_by_color(self, image):
        """
        Quick color-based check for skin presence.
        
        Human skin has specific color ranges in YCrCb color space:
        - Y (brightness): varies widely
        - Cr (red-difference): typically 133-173
        - Cb (blue-difference): typically 77-127
        
        This is a fast preliminary check before using the neural network.
        """
        # Convert BGR (OpenCV format) to YCrCb color space
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        
        # Define skin color range in YCrCb
        lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin = np.array([255, 173, 127], dtype=np.uint8)
        
        # Create mask where skin-colored pixels are white, others are black
        skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
        
        # Calculate percentage of image that's skin-colored
        skin_percentage = np.sum(skin_mask > 0) / skin_mask.size
        
        # If more than 15% of image is skin-colored, likely contains skin
        return skin_percentage > 0.15, skin_percentage
    
    def predict(self, image):
        """
        Main prediction function - determines if image contains skin.
        
        PROCESS:
        1. Quick color check (fast)
        2. If color check passes, use neural network (more accurate)
        3. Combine results for final decision
        
        Returns:
        - is_skin: True/False
        - confidence: 0-100 percentage
        - message: explanation of result
        """
        # First: quick color-based check
        color_is_skin, skin_percentage = self.is_skin_by_color(image)
        
        if not color_is_skin:
            return {
                "is_skin": False,
                "confidence": (1 - skin_percentage) * 100,
                "message": "Image does not appear to contain human skin. Please upload a clear face photo."
            }
        
        # Second: neural network prediction
        # Resize image to expected size
        resized = cv2.resize(image, self.image_size)
        
        # Normalize pixel values from 0-255 to 0-1 (neural networks work better with small numbers)
        normalized = resized / 255.0
        
        # Add batch dimension (model expects batch of images, we have just 1)
        batched = np.expand_dims(normalized, axis=0)
        
        # Get prediction from model
        prediction = self.model.predict(batched, verbose=0)[0][0]
        
        # Combine color analysis with neural network prediction
        combined_confidence = (prediction + skin_percentage) / 2
        
        is_skin = combined_confidence > 0.5
        
        return {
            "is_skin": is_skin,
            "confidence": combined_confidence * 100,
            "message": "Skin detected! Proceeding with analysis." if is_skin else "Could not confirm skin in image."
        }
    
    def train(self, skin_images, non_skin_images, epochs=10):
        """
        Train the model on labeled images.
        
        Parameters:
        - skin_images: list of images that ARE skin
        - non_skin_images: list of images that are NOT skin
        - epochs: how many times to go through all training data
        
        More epochs = better learning, but takes longer and might overfit
        """
        # Prepare training data
        X = []  # Images
        y = []  # Labels (1 = skin, 0 = not skin)
        
        # Process skin images
        for img in skin_images:
            resized = cv2.resize(img, self.image_size)
            normalized = resized / 255.0
            X.append(normalized)
            y.append(1)  # Label: this IS skin
        
        # Process non-skin images
        for img in non_skin_images:
            resized = cv2.resize(img, self.image_size)
            normalized = resized / 255.0
            X.append(normalized)
            y.append(0)  # Label: this is NOT skin
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Shuffle the data (important for training)
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        # Train the model
        print(f"Training on {len(X)} images for {epochs} epochs...")
        self.model.fit(X, y, epochs=epochs, validation_split=0.2, verbose=1)
        
        # Save the trained model
        self.model.save(self.model_path)
        print(f"Model saved to {self.model_path}")


# Quick test when running this file directly
if __name__ == "__main__":
    classifier = SkinBinaryClassifier()
    print("Skin Binary Classifier initialized successfully!")
    print("This classifier determines if an image contains human skin or not.")
