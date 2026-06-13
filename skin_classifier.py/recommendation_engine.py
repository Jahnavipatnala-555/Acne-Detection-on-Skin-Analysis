"""
TRAINING SCRIPT
Trains the binary classifier on sample skin/non-skin data
Run this first if no pre-trained model exists
"""

import cv2
import numpy as np
import os
from skin_classifier import SkinBinaryClassifier

def create_sample_training_data():
    """
    Creates synthetic training data for demonstration.
    In production, you would use real labeled images.
    """
    skin_images = []
    non_skin_images = []
    
    print("Creating synthetic training data...")
    
    # Create synthetic "skin" images (various shades of skin color)
    for i in range(50):
        # Base skin colors in BGR
        skin_colors = [
            (80, 140, 200),   # Light skin
            (70, 120, 180),   # Medium light
            (60, 100, 160),   # Medium
            (50, 80, 140),    # Tan
            (40, 60, 120),    # Dark
        ]
        
        color = skin_colors[i % len(skin_colors)]
        
        # Create base image
        img = np.full((64, 64, 3), color, dtype=np.uint8)
        
        # Add texture variation
        noise = np.random.randint(-20, 20, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Add some random spots (simulating pores/texture)
        for _ in range(np.random.randint(5, 20)):
            x, y = np.random.randint(0, 64, 2)
            cv2.circle(img, (x, y), 1, (color[0]-20, color[1]-20, color[2]-20), -1)
        
        skin_images.append(img)
    
    # Create synthetic "non-skin" images
    for i in range(50):
        # Random non-skin colors
        if i < 15:
            # Blue/Green (nature, objects)
            color = (np.random.randint(100, 255), np.random.randint(100, 255), np.random.randint(0, 100))
        elif i < 30:
            # Grays/Whites/Blacks
            val = np.random.randint(0, 255)
            color = (val, val, val)
        else:
            # Random colors
            color = tuple(np.random.randint(0, 255, 3))
        
        img = np.full((64, 64, 3), color, dtype=np.uint8)
        
        # Add patterns
        if i % 3 == 0:
            # Stripes
            for j in range(0, 64, 8):
                cv2.line(img, (0, j), (64, j), (255-color[0], 255-color[1], 255-color[2]), 2)
        elif i % 3 == 1:
            # Grid
            for j in range(0, 64, 16):
                cv2.line(img, (0, j), (64, j), (255-color[0], 255-color[1], 255-color[2]), 1)
                cv2.line(img, (j, 0), (j, 64), (255-color[0], 255-color[1], 255-color[2]), 1)
        
        non_skin_images.append(img)
    
    print(f"Created {len(skin_images)} skin samples and {len(non_skin_images)} non-skin samples")
    
    return skin_images, non_skin_images

def main():
    """Main training function."""
    print("="*50)
    print("SKIN CLASSIFIER TRAINING")
    print("="*50)
    
    # Check if model already exists
    if os.path.exists("models/skin_binary_classifier.h5"):
        print("\nModel already exists at models/skin_binary_classifier.h5")
        response = input("Do you want to retrain? (y/n): ")
        if response.lower() != 'y':
            print("Training cancelled.")
            return
    
    # Create classifier
    classifier = SkinBinaryClassifier()
    
    # Create training data
    skin_images, non_skin_images = create_sample_training_data()
    
    # Train the model
    print("\nStarting training...")
    classifier.train(skin_images, non_skin_images, epochs=20)
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE!")
    print("="*50)
    print("Model saved to: models/skin_binary_classifier.h5")
    print("\nYou can now run the main application!")

if __name__ == "__main__":
    main()