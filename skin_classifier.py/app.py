import cv2
import numpy as np
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# OpenCV Face Detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Skin Analyzer</title>
</head>
<body style="font-family: Arial; text-align:center; padding:20px;">
    <h1>AI Skin Analyzer</h1>

    <input type="file" id="fileInput">
    <br><br>
    <button onclick="upload()">Analyze</button>

    <h3>Results:</h3>
    <pre id="result"></pre>

<script>
async function upload() {
    let file = document.getElementById('fileInput').files[0];

    if (!file) {
        alert("Select an image first!");
        return;
    }

    let fd = new FormData();
    fd.append('image', file);

    let res = await fetch('/analyze', {
        method: 'POST',
        body: fd
    });

    let data = await res.json();
    document.getElementById('result').innerText = JSON.stringify(data, null, 2);
}
</script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML)

@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files['image']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({'error': 'Invalid image'})

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect face
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    if len(faces) == 0:
        return jsonify({'error': 'No face detected'})

    # Take first face
    x, y, w, h = faces[0]
    face = img[y:y+h, x:x+w]

    # Convert to HSV
    hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)

    brightness = np.mean(hsv[:, :, 2])
    saturation = np.mean(hsv[:, :, 1])

    # Skin Type Detection
    if brightness > 160:
        skin_type = 'oily'
    elif brightness < 100:
        skin_type = 'dry'
    elif saturation < 100:
        skin_type = 'combination'
    else:
        skin_type = 'normal'

    # Redness Detection
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    red_mask = mask1 + mask2
    redness = float(min(100, (np.sum(red_mask > 0) / red_mask.size) * 300))

    # Pigmentation Detection
    gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_face, 60, 255, cv2.THRESH_BINARY_INV)
    pigmentation = float(min(100, (np.sum(thresh > 0) / thresh.size) * 200))

    # Acne Types
    acne_types = {
        'pustules': redness * 0.6 if redness > 20 else 0,
        'papules': redness * 0.4 if redness > 15 else 0,
        'nodules': redness * 0.3 if redness > 40 else 0,
        'cystic': redness * 0.2 if redness > 50 else 0,
        'comedones': 30 if skin_type == 'oily' else 10
    }

    # Severity
    if redness < 20:
        severity = "Mild"
    elif redness < 50:
        severity = "Moderate"
    else:
        severity = "Severe"

    return jsonify({
        'skin_type': skin_type,
        'redness': redness,
        'pigmentation': pigmentation,
        'severity': severity,
        'acne_types': acne_types
    })

if __name__ == '__main__':
    app.run(debug=True)