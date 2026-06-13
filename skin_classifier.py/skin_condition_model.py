"""
SKIN CONDITION DETECTION MODEL

This is the main analysis engine. It detects:

ACNE TYPES:
- Pustules: Red bumps with white/yellow pus centers
- Papules: Small red/pink bumps without visible pus
- Nodules: Large, painful lumps deep under skin
- Cystic: Deep, pus-filled, painful bumps
- Comedones: Blackheads and whiteheads

SKIN CONDITIONS:
- Hyperpigmentation: Dark patches/spots on skin
- Rosacea: Redness, visible blood vessels, sometimes bumps

SKIN TYPES:
- Oily: Shiny, enlarged pores
- Dry: Flaky, tight feeling
- Combination: Oily T-zone, dry cheeks
- Normal: Balanced, few imperfections

HOW IT WORKS:
1. Detects face in image using MediaPipe
2. Analyzes different face regions (forehead, cheeks, chin, nose)
3. Uses color analysis to detect redness, darkness, oiliness
4. Uses texture analysis to detect bumps, roughness
5. Combines all analysis for final diagnosis
"""

import cv2
import numpy as np
import mediapipe as mp
from tensorflow import keras
from tensorflow.keras import layers
import os


class SkinConditionAnalyzer:
    """
    Analyzes skin for various conditions, acne types, and skin type.
    """
    
    def __init__(self):
        # Initialize MediaPipe Face Mesh for facial landmark detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,      # For single images (not video)
            max_num_faces=1,             # We only need one face
            min_detection_confidence=0.5  # Minimum confidence to detect face
        )
        
        # Model for condition classification
        self.model = None
        self.model_path = "models/skin_condition_model.h5"
        self._load_or_create_model()
        
        # Define regions of face for analysis
        # These are indices from MediaPipe's 468 facial landmarks
        self.face_regions = {
            'forehead': [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288],
            'left_cheek': [234, 93, 132, 58, 172, 136, 150, 149, 176, 148],
            'right_cheek': [454, 323, 361, 288, 397, 365, 379, 378, 400, 377],
            'nose': [1, 2, 98, 327, 168, 6, 197, 195, 5],
            'chin': [152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234],
            't_zone': [10, 151, 9, 8, 168, 6, 197, 195, 5, 1, 2]  # Forehead + Nose
        }
    
    def _load_or_create_model(self):
        """Load existing model or create new one."""
        os.makedirs("models", exist_ok=True)
        
        if os.path.exists(self.model_path):
            print("Loading skin condition model...")
            self.model = keras.models.load_model(self.model_path)
        else:
            print("Creating new skin condition model...")
            self.model = self._build_model()
    
    def _build_model(self):
        """
        Build multi-output neural network for skin analysis.
        
        This model has multiple outputs because we're predicting multiple things:
        - Acne types (5 types)
        - Skin conditions (3 conditions)
        - Skin type (4 types)
        """
        # Input layer
        inputs = layers.Input(shape=(128, 128, 3))
        
        # Shared convolutional layers (extract features used by all outputs)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Flatten()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        
        # Output 1: Acne types (5 classes, each can be present or not)
        # Using sigmoid because multiple acne types can be present simultaneously
        acne_output = layers.Dense(256, activation='relu')(x)
        acne_output = layers.Dense(5, activation='sigmoid', name='acne_types')(acne_output)
        
        # Output 2: Skin conditions (hyperpigmentation, rosacea, normal)
        condition_output = layers.Dense(128, activation='relu')(x)
        condition_output = layers.Dense(3, activation='sigmoid', name='conditions')(condition_output)
        
        # Output 3: Skin type (oily, dry, combination, normal)
        # Using softmax because skin can only be ONE type
        skin_type_output = layers.Dense(128, activation='relu')(x)
        skin_type_output = layers.Dense(4, activation='softmax', name='skin_type')(skin_type_output)
        
        model = keras.Model(
            inputs=inputs,
            outputs=[acne_output, condition_output, skin_type_output]
        )
        
        model.compile(
            optimizer='adam',
            loss={
                'acne_types': 'binary_crossentropy',
                'conditions': 'binary_crossentropy',
                'skin_type': 'categorical_crossentropy'
            },
            metrics=['accuracy']
        )
        
        return model
    
    def detect_face_landmarks(self, image):
        """
        Detect facial landmarks using MediaPipe.
        Returns 468 points on the face.
        """
        # Convert BGR to RGB (MediaPipe expects RGB)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process image
        results = self.face_mesh.process(rgb_image)
        
        if not results.multi_face_landmarks:
            return None
        
        # Get landmarks for first face
        landmarks = results.multi_face_landmarks[0]
        
        # Convert normalized coordinates to pixel coordinates
        h, w = image.shape[:2]
        points = []
        for landmark in landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            points.append((x, y))
        
        return points
    
    def extract_region(self, image, landmarks, region_indices):
        """
        Extract a specific region of the face based on landmarks.
        """
        if landmarks is None:
            return None
        
        # Get points for this region
        points = [landmarks[i] for i in region_indices if i < len(landmarks)]
        
        if len(points) < 3:
            return None
        
        # Create bounding box
        points_array = np.array(points)
        x_min, y_min = points_array.min(axis=0)
        x_max, y_max = points_array.max(axis=0)
        
        # Add padding
        padding = 10
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(image.shape[1], x_max + padding)
        y_max = min(image.shape[0], y_max + padding)
        
        # Extract region
        region = image[y_min:y_max, x_min:x_max]
        
        return region if region.size > 0 else None
    
    def analyze_texture(self, region):
        """
        Analyze skin texture for bumps, roughness, etc.
        
        Uses Laplacian variance to detect texture:
        - High variance = rough/bumpy texture
        - Low variance = smooth texture
        """
        if region is None or region.size == 0:
            return {"roughness": 0, "variance": 0}
        
        # Convert to grayscale
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        
        # Apply Laplacian (edge detection)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        
        # Calculate variance (measure of texture)
        variance = laplacian.var()
        
        # Normalize to 0-100 scale
        roughness = min(100, variance / 10)
        
        return {
            "roughness": roughness,
            "variance": variance
        }
    
    def analyze_color(self, region):
        """
        Analyze skin color for redness, darkness, oiliness.
        
        Analyzes in multiple color spaces:
        - RGB: For general color analysis
        - HSV: For saturation (redness) and value (brightness)
        - LAB: For accurate darkness/lightness analysis
        """
        if region is None or region.size == 0:
            return {
                "redness": 0,
                "darkness": 0,
                "oiliness": 0
            }
        
        # Convert to different color spaces
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB)
        
        # Analyze redness (high red channel relative to others)
        b, g, r = cv2.split(region)
        redness = np.mean(r) - (np.mean(g) + np.mean(b)) / 2
        redness = max(0, min(100, redness))
        
        # Analyze darkness using LAB L channel (0 = black, 255 = white)
        l_channel = lab[:, :, 0]
        # Invert so higher = darker
        darkness = 100 - (np.mean(l_channel) / 255 * 100)
        
        # Analyze oiliness using HSV saturation and value
        # Oily skin tends to have higher brightness (reflection)
        h, s, v = cv2.split(hsv)
        # High brightness with low saturation can indicate oily/shiny skin
        oiliness = np.mean(v) / 255 * 100 - np.mean(s) / 255 * 50
        oiliness = max(0, min(100, oiliness))
        
        return {
            "redness": redness,
            "darkness": darkness,
            "oiliness": oiliness
        }
    
    def detect_spots(self, region):
        """
        Detect spots (acne, dark spots) using blob detection.
        
        Blobs are areas that differ from surrounding skin:
        - Red blobs might be acne
        - Dark blobs might be hyperpigmentation
        """
        if region is None or region.size == 0:
            return {"count": 0, "spots": []}
        
        # Convert to grayscale
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        
        # Set up blob detector parameters
        params = cv2.SimpleBlobDetector_Params()
        
        # Filter by area (size of spots)
        params.filterByArea = True
        params.minArea = 10
        params.maxArea = 500
        
        # Filter by circularity (spots tend to be round)
        params.filterByCircularity = True
        params.minCircularity = 0.5
        
        # Filter by convexity
        params.filterByConvexity = True
        params.minConvexity = 0.5
        
        # Create detector
        detector = cv2.SimpleBlobDetector_create(params)
        
        # Detect blobs
        keypoints = detector.detect(gray)
        
        spots = []
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            size = kp.size
            spots.append({"x": x, "y": y, "size": size})
        
        return {
            "count": len(spots),
            "spots": spots
        }
    
    def classify_acne_type(self, region_analysis):
        """
        Classify acne type based on region analysis.
        
        Classification rules:
        - Pustules: High redness + detected spots + medium texture
        - Papules: Medium redness + few spots + low texture
        - Nodules: Very high redness + high texture + large spots
        - Cystic: Very high redness + deep darkness + large spots
        - Comedones: Low redness + many small spots + low texture
        """
        results = {
            "pustules": 0,
            "papules": 0,
            "nodules": 0,
            "cystic": 0,
            "comedones": 0
        }
        
        redness = region_analysis["color"]["redness"]
        roughness = region_analysis["texture"]["roughness"]
        spot_count = region_analysis["spots"]["count"]
        
        # Pustules: Red, bumpy, with visible pus
        if redness > 40 and spot_count > 3 and roughness > 30:
            results["pustules"] = min(100, redness + spot_count * 5)
        
        # Papules: Red bumps without visible pus
        if redness > 30 and redness < 60 and roughness > 20 and roughness < 50:
            results["papules"] = min(100, redness + roughness)
        
        # Nodules: Deep, painful, large bumps
        if redness > 50 and roughness > 50:
            results["nodules"] = min(100, (redness + roughness) / 2)
        
        # Cystic: Deep, pus-filled, severe
        if redness > 60 and roughness > 40 and spot_count > 2:
            results["cystic"] = min(100, redness + roughness / 2)
        
        # Comedones: Blackheads/whiteheads, low inflammation
        if redness < 30 and spot_count > 5:
            results["comedones"] = min(100, spot_count * 10)
        
        return results
    
    def determine_skin_type(self, regional_analyses):
        """
        Determine overall skin type based on analysis of different face regions.
        
        Logic:
        - Oily: High oiliness in T-zone and cheeks
        - Dry: Low oiliness, high roughness
        - Combination: Oily T-zone, dry/normal cheeks
        - Normal: Balanced across all metrics
        """
        # Get T-zone analysis (forehead + nose)
        t_zone_oiliness = 0
        cheek_oiliness = 0
        overall_roughness = 0
        
        for region_name, analysis in regional_analyses.items():
            if region_name in ['forehead', 'nose', 't_zone']:
                t_zone_oiliness = max(t_zone_oiliness, analysis["color"]["oiliness"])
            if region_name in ['left_cheek', 'right_cheek']:
                cheek_oiliness = max(cheek_oiliness, analysis["color"]["oiliness"])
            overall_roughness += analysis["texture"]["roughness"]
        
        overall_roughness /= max(1, len(regional_analyses))
        
        # Determine skin type
        if t_zone_oiliness > 60 and cheek_oiliness > 50:
            return {"type": "oily", "confidence": t_zone_oiliness}
        elif t_zone_oiliness > 50 and cheek_oiliness < 40:
            return {"type": "combination", "confidence": (t_zone_oiliness + (100 - cheek_oiliness)) / 2}
        elif overall_roughness > 40 and t_zone_oiliness < 40:
            return {"type": "dry", "confidence": overall_roughness}
        else:
            return {"type": "normal", "confidence": 70}
    
    def detect_conditions(self, regional_analyses):
        """
        Detect skin conditions: hyperpigmentation, rosacea.
        """
        conditions = {
            "hyperpigmentation": {"detected": False, "severity": 0, "areas": []},
            "rosacea": {"detected": False, "severity": 0, "areas": []},
            "normal": {"detected": True, "confidence": 100}
        }
        
        for region_name, analysis in regional_analyses.items():
            # Check for hyperpigmentation (dark patches)
            if analysis["color"]["darkness"] > 40:
                conditions["hyperpigmentation"]["detected"] = True
                conditions["hyperpigmentation"]["severity"] = max(
                    conditions["hyperpigmentation"]["severity"],
                    analysis["color"]["darkness"]
                )
                conditions["hyperpigmentation"]["areas"].append(region_name)
                conditions["normal"]["detected"] = False
            
            # Check for rosacea (persistent redness, especially on cheeks)
            if region_name in ['left_cheek', 'right_cheek'] and analysis["color"]["redness"] > 50:
                conditions["rosacea"]["detected"] = True
                conditions["rosacea"]["severity"] = max(
                    conditions["rosacea"]["severity"],
                    analysis["color"]["redness"]
                )
                conditions["rosacea"]["areas"].append(region_name)
                conditions["normal"]["detected"] = False
        
        if conditions["normal"]["detected"]:
            conditions["normal"]["confidence"] = 100 - (
                conditions["hyperpigmentation"]["severity"] +
                conditions["rosacea"]["severity"]
            ) / 2
        
        return conditions
    
    def analyze(self, image):
        """
        Main analysis function - performs complete skin analysis.
        
        Returns comprehensive results including:
        - Acne types detected
        - Skin conditions
        - Skin type
        - Severity scores
        - Affected areas
        """
        # Step 1: Detect face landmarks
        landmarks = self.detect_face_landmarks(image)
        
        if landmarks is None:
            return {
                "success": False,
                "error": "No face detected. Please ensure your face is clearly visible."
            }
        
        # Step 2: Analyze each face region
        regional_analyses = {}
        
        for region_name, region_indices in self.face_regions.items():
            region = self.extract_region(image, landmarks, region_indices)
            
            if region is not None:
                regional_analyses[region_name] = {
                    "texture": self.analyze_texture(region),
                    "color": self.analyze_color(region),
                    "spots": self.detect_spots(region)
                }
        
        if not regional_analyses:
            return {
                "success": False,
                "error": "Could not analyze face regions. Please try a clearer photo."
            }
        
        # Step 3: Classify acne types for each region
        acne_results = {
            "pustules": 0,
            "papules": 0,
            "nodules": 0,
            "cystic": 0,
            "comedones": 0
        }
        
        for region_name, analysis in regional_analyses.items():
            region_acne = self.classify_acne_type(analysis)
            for acne_type, score in region_acne.items():
                acne_results[acne_type] = max(acne_results[acne_type], score)
        
        # Step 4: Determine skin type
        skin_type = self.determine_skin_type(regional_analyses)
        
        # Step 5: Detect conditions
        conditions = self.detect_conditions(regional_analyses)
        
        # Step 6: Calculate overall severity
        acne_severity = sum(acne_results.values()) / len(acne_results)
        
        return {
            "success": True,
            "acne_types": acne_results,
            "skin_type": skin_type,
            "conditions": conditions,
            "regional_analysis": regional_analyses,
            "overall_severity": acne_severity,
            "summary": self._generate_summary(acne_results, skin_type, conditions)
        }
    
    def _generate_summary(self, acne_results, skin_type, conditions):
        """Generate human-readable summary of analysis."""
        summary = []
        
        # Skin type
        summary.append(f"Skin Type: {skin_type['type'].capitalize()} ({skin_type['confidence']:.0f}% confidence)")
        
        # Acne
        detected_acne = [name for name, score in acne_results.items() if score > 20]
        if detected_acne:
            summary.append(f"Acne Detected: {', '.join(a.capitalize() for a in detected_acne)}")
        else:
            summary.append("Acne: Minimal to none detected")
        
        # Conditions
        if conditions["hyperpigmentation"]["detected"]:
            summary.append(f"Hyperpigmentation: Detected ({conditions['hyperpigmentation']['severity']:.0f}% severity)")
        if conditions["rosacea"]["detected"]:
            summary.append(f"Rosacea: Detected ({conditions['rosacea']['severity']:.0f}% severity)")
        
        return summary


# Test when running directly
if __name__ == "__main__":
    analyzer = SkinConditionAnalyzer()
    print("Skin Condition Analyzer initialized successfully!")
