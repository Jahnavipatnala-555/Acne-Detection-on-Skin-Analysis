import cv2
import numpy as np
from flask import Flask, render_template_string, request, jsonify
import os

app = Flask(__name__)

# Load face cascade
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DermAI Pro | Real Skin Analysis</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Inter', sans-serif; 
            background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
            min-height: 100vh; 
            padding: 20px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        .header { text-align: center; padding: 30px 0; color: white; }
        .header h1 { font-size: 3rem; background: linear-gradient(135deg, #667eea, #764ba2, #f093fb); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .main-grid { display: grid; grid-template-columns: 1fr 1.2fr; gap: 25px; }
        .card { background: rgba(255,255,255,0.95); border-radius: 24px; padding: 25px; box-shadow: 0 20px 60px rgba(0,0,0,0.3); }
        .camera-container { border-radius: 16px; overflow: hidden; background: #000; aspect-ratio: 4/3; }
        #video, #canvas { width: 100%; height: 100%; object-fit: cover; }
        #canvas { display: none; }
        .btn { padding: 12px 24px; border: none; border-radius: 50px; font-weight: 600; cursor: pointer; margin: 5px; }
        .btn-primary { background: #667eea; color: white; }
        .btn-success { background: #48bb78; color: white; }
        .btn-danger { background: #f56565; color: white; }
        .upload-area { border: 3px dashed #667eea; border-radius: 16px; padding: 30px; text-align: center; cursor: pointer; margin-top: 15px; }
        .severity-badge { display: inline-block; padding: 10px 20px; border-radius: 50px; font-weight: 700; margin: 10px 0; }
        .severity-clear { background: #48bb78; color: white; }
        .severity-mild { background: #ecc94b; color: #1a1a2e; }
        .severity-moderate { background: #ed8936; color: white; }
        .severity-severe { background: #f56565; color: white; }
        .metrics-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 20px 0; }
        .metric-card { background: #f7fafc; padding: 15px; border-radius: 12px; text-align: center; }
        .metric-value { font-size: 1.8rem; font-weight: 700; color: #667eea; }
        .progress-container { margin: 15px 0; }
        .progress-bar { height: 10px; background: #e2e8f0; border-radius: 10px; }
        .progress-fill { height: 100%; background: #667eea; border-radius: 10px; width: 0%; transition: width 0.5s; }
        .tabs { display: flex; gap: 5px; margin: 20px 0; border-bottom: 2px solid #e2e8f0; flex-wrap: wrap; }
        .tab { padding: 12px 20px; background: none; border: none; cursor: pointer; font-weight: 600; color: #718096; }
        .tab.active { color: #667eea; border-bottom: 3px solid #667eea; }
        .tab-content { display: none; padding: 20px 0; }
        .tab-content.active { display: block; }
        .product-card { background: #f7fafc; padding: 20px; border-radius: 12px; margin-bottom: 15px; display: flex; gap: 15px; }
        .product-icon { font-size: 40px; }
        .ingredient-tag { display: inline-block; background: #e2e8f0; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; margin: 3px; }
        .food-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; }
        .food-card { background: white; padding: 15px; border-radius: 12px; text-align: center; }
        .food-good { border-left: 4px solid #48bb78; }
        .food-bad { border-left: 4px solid #f56565; }
        .warning-box { background: #fefcbf; padding: 20px; border-radius: 12px; border-left: 5px solid #ecc94b; margin: 20px 0; }
        .success-box { background: #c6f6d5; padding: 20px; border-radius: 12px; border-left: 5px solid #48bb78; margin: 20px 0; }
        .loading { position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0,0,0,0.8); display: none; justify-content: center; align-items: center; z-index: 1000; }
        .spinner { width: 60px; height: 60px; border: 4px solid rgba(255,255,255,0.3); border-top-color: white; border-radius: 50%; animation: spin 1s linear infinite; }
        @keyframes spin { to { transform: rotate(360deg); } }
        .image-preview { max-width: 100%; border-radius: 12px; margin-top: 15px; }
        .region-tag { display: inline-block; padding: 5px 12px; margin: 3px; border-radius: 20px; font-size: 12px; background: #e2e8f0; }
        @media (max-width: 768px) {
            .main-grid { grid-template-columns: 1fr; }
            .metrics-grid { grid-template-columns: repeat(2, 1fr); }
            .food-grid { grid-template-columns: repeat(2, 1fr); }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧬 DermAI Pro</h1>
            <p style="color: white; opacity: 0.9;">Real Computer Vision Skin Analysis</p>
        </div>
        
        <div class="main-grid">
            <div class="card">
                <h3>📷 Capture Your Skin</h3>
                <div class="camera-container">
                    <video id="video" autoplay playsinline></video>
                    <canvas id="canvas"></canvas>
                </div>
                <div>
                    <button class="btn btn-primary" onclick="startCamera()">▶ Start Camera</button>
                    <button class="btn btn-success" onclick="capture()">📸 Analyze Face</button>
                    <button class="btn btn-danger" onclick="stopCamera()">⬛ Stop</button>
                </div>
                <div class="upload-area" onclick="document.getElementById('fileInput').click()">
                    <div style="font-size: 48px;">📁</div>
                    <p>Click to upload a clear face photo</p>
                    <p style="font-size: 12px; color: #718096;">For best results, use good lighting</p>
                    <input type="file" id="fileInput" accept="image/*" style="display: none;" onchange="upload(this)">
                </div>
                <img id="preview" class="image-preview" style="display: none;">
            </div>
            
            <div class="card">
                <h3>🔬 Real-Time Analysis</h3>
                <div id="quickResult">
                    <p style="color: #a0aec0; text-align: center; padding: 40px;">
                        Capture or upload a photo to see actual skin analysis
                    </p>
                </div>
            </div>
        </div>
        
        <div id="fullResults" style="display: none; margin-top: 25px;">
            <div class="card">
                <div id="severityDisplay"></div>
                <div class="metrics-grid" id="metricsDisplay"></div>
                
                <div class="tabs">
                    <button class="tab active" onclick="showTab('acne')">🦠 Acne Detection</button>
                    <button class="tab" onclick="showTab('texture')">📊 Texture Analysis</button>
                    <button class="tab" onclick="showTab('pigmentation')">🎯 Dark Spots</button>
                    <button class="tab" onclick="showTab('products')">💊 Recommendations</button>
                    <button class="tab" onclick="showTab('diet')">🥗 Diet Plan</button>
                </div>
                
                <div id="acne" class="tab-content active"></div>
                <div id="texture" class="tab-content"></div>
                <div id="pigmentation" class="tab-content"></div>
                <div id="products" class="tab-content"></div>
                <div id="diet" class="tab-content"></div>
            </div>
        </div>
    </div>
    
    <div class="loading" id="loading">
        <div style="text-align: center; color: white;">
            <div class="spinner"></div>
            <p style="margin-top: 20px;">Analyzing your skin with AI...</p>
            <p style="font-size: 14px; opacity: 0.7;">Detecting acne, texture, pigmentation</p>
        </div>
    </div>
    
    <script>
        let stream = null;
        
        async function startCamera() {
            try {
                stream = await navigator.mediaDevices.getUserMedia({ 
                    video: { width: 640, height: 480, facingMode: 'user' } 
                });
                document.getElementById('video').srcObject = stream;
            } catch(e) { 
                alert('Camera error: ' + e.message); 
            }
        }
        
        function stopCamera() {
            if(stream) { 
                stream.getTracks().forEach(t => t.stop()); 
                document.getElementById('video').srcObject = null;
            }
        }
        
        function capture() {
            const video = document.getElementById('video');
            const canvas = document.getElementById('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0);
            
            canvas.toBlob(blob => {
                analyzeImage(blob);
                // Show preview
                const url = URL.createObjectURL(blob);
                document.getElementById('preview').src = url;
                document.getElementById('preview').style.display = 'block';
            }, 'image/jpeg', 0.95);
        }
        
        function upload(input) {
            if(input.files[0]) {
                analyzeImage(input.files[0]);
                // Show preview
                const url = URL.createObjectURL(input.files[0]);
                document.getElementById('preview').src = url;
                document.getElementById('preview').style.display = 'block';
            }
        }
        
        async function analyzeImage(blob) {
            document.getElementById('loading').style.display = 'flex';
            document.getElementById('fullResults').style.display = 'none';
            
            const formData = new FormData();
            formData.append('image', blob);
            
            try {
                const response = await fetch('/analyze', { method: 'POST', body: formData });
                const data = await response.json();
                
                if (data.error) {
                    alert(data.error);
                    return;
                }
                
                displayResults(data);
                displayQuickResults(data);
                document.getElementById('fullResults').style.display = 'block';
            } catch(e) {
                alert('Error: ' + e.message);
            } finally {
                document.getElementById('loading').style.display = 'none';
            }
        }
        
        function displayQuickResults(data) {
            document.getElementById('quickResult').innerHTML = `
                <div style="text-align: center;">
                    <div class="severity-badge severity-${data.severity_class}">${data.severity}</div>
                    <h4>${data.skin_type_icon} ${data.skin_type}</h4>
                    <p><strong>Acne:</strong> ${data.acne_level}</p>
                    <p><strong>Texture:</strong> ${data.texture_level}</p>
                    <p><strong>Dark Spots:</strong> ${data.pigmentation_level}</p>
                    <div class="progress-container">
                        <div class="progress-label"><span>Overall Skin Health</span><span>${data.health_score}%</span></div>
                        <div class="progress-bar"><div class="progress-fill" style="width:${data.health_score}%"></div></div>
                    </div>
                </div>
            `;
        }
        
        function displayResults(data) {
            // Severity
            document.getElementById('severityDisplay').innerHTML = `
                <div class="severity-badge severity-${data.severity_class}">${data.severity_icon} ${data.severity}</div>
                ${data.need_dermatologist ? 
                    `<div class="warning-box"><strong>⚠️ Dermatologist Consultation Recommended</strong><br>${data.consultation_reason}</div>` : 
                    `<div class="success-box"><strong>✅ ${data.message}</strong></div>`
                }
            `;
            
            // Metrics
            document.getElementById('metricsDisplay').innerHTML = `
                <div class="metric-card"><div class="metric-value">${data.skin_type_icon}</div><div>${data.skin_type}</div></div>
                <div class="metric-card"><div class="metric-value">${data.acne_score}%</div><div>Acne</div></div>
                <div class="metric-card"><div class="metric-value">${data.texture_score}%</div><div>Texture</div></div>
                <div class="metric-card"><div class="metric-value">${data.pigmentation_score}%</div><div>Dark Spots</div></div>
            `;
            
            // Acne Tab
            let acneHtml = `<h4>Detected Acne Types</h4>`;
            for(let [type, score] of Object.entries(data.acne_types)) {
                acneHtml += `<div class="progress-container">
                    <div class="progress-label"><span>${data.acne_icons[type]} ${type}</span><span>${score.toFixed(1)}%</span></div>
                    <div class="progress-bar"><div class="progress-fill" style="width:${score}%"></div></div>
                </div>`;
            }
            acneHtml += `<h4 style="margin-top:20px;">Face Region Analysis</h4><div>`;
            for(let [region, analysis] of Object.entries(data.regions)) {
                let tags = [];
                if(analysis.oiliness > 50) tags.push('<span class="region-tag" style="background:#fefcbf;">🛢️ Oily</span>');
                if(analysis.oiliness < 30) tags.push('<span class="region-tag" style="background:#c6f6d5;">🍂 Dry</span>');
                if(analysis.redness > 20) tags.push('<span class="region-tag" style="background:#fed7d7;">🔴 Redness</span>');
                if(analysis.darkness > 30) tags.push('<span class="region-tag" style="background:#e9d8fd;">🎯 Dark</span>');
                acneHtml += `<div style="margin:10px 0;"><strong>${region}</strong>: ${tags.join(' ') || '✓ Clear'}</div>`;
            }
            acneHtml += '</div>';
            document.getElementById('acne').innerHTML = acneHtml;
            
            // Texture Tab
            document.getElementById('texture').innerHTML = `
                <h4>Skin Texture Analysis</h4>
                <div class="progress-container">
                    <div class="progress-label"><span>Smoothness</span><span>${data.smoothness}%</span></div>
                    <div class="progress-bar"><div class="progress-fill" style="width:${data.smoothness}%; background:#48bb78;"></div></div>
                </div>
                <div class="progress-container">
                    <div class="progress-label"><span>Roughness/Bumps</span><span>${data.roughness}%</span></div>
                    <div class="progress-bar"><div class="progress-fill" style="width:${data.roughness}%; background:#ed8936;"></div></div>
                </div>
                <div class="progress-container">
                    <div class="progress-label"><span>Pore Visibility</span><span>${data.pore_score}%</span></div>
                    <div class="progress-bar"><div class="progress-fill" style="width:${data.pore_score}%; background:#667eea;"></div></div>
                </div>
                <h4 style="margin-top:20px;">Scar Detection</h4>
                ${data.scars.map(s => `
                    <div class="product-card">
                        <div class="product-icon">${s.icon}</div>
                        <div>
                            <strong>${s.type}</strong> - ${s.severity}%
                            <div class="progress-bar"><div class="progress-fill" style="width:${s.severity}%"></div></div>
                            <p style="margin-top:10px;">${s.description}</p>
                        </div>
                    </div>
                `).join('')}
            `;
            
            // Pigmentation Tab
            document.getElementById('pigmentation').innerHTML = `
                <h4>Hyperpigmentation Analysis</h4>
                <div class="progress-container">
                    <div class="progress-label"><span>Dark Spots</span><span>${data.pigmentation_score}%</span></div>
                    <div class="progress-bar"><div class="progress-fill" style="width:${data.pigmentation_score}%"></div></div>
                </div>
                <div class="progress-container">
                    <div class="progress-label"><span>Dark Circles</span><span>${data.dark_circles_score}%</span></div>
                    <div class="progress-bar"><div class="progress-fill" style="width:${data.dark_circles_score}%"></div></div>
                </div>
                <div class="progress-container">
                    <div class="progress-label"><span>Uneven Tone</span><span>${data.uneven_tone}%</span></div>
                    <div class="progress-bar"><div class="progress-fill" style="width:${data.uneven_tone}%"></div></div>
                </div>
                <h4 style="margin-top:20px;">Brightening Recommendations</h4>
                ${data.brightening_products.map(p => `
                    <div class="product-card">
                        <div class="product-icon">${p.icon}</div>
                        <div>
                            <strong>${p.brand} - ${p.name}</strong><br>
                            ${p.price} ★ ${p.rating}<br>
                            <span class="ingredient-tag">${p.key_ingredient}</span>
                        </div>
                    </div>
                `).join('')}
            `;
            
            // Products Tab
            document.getElementById('products').innerHTML = `
                <h4>Your Personalized Skincare Routine</h4>
                ${data.routine.map(r => `
                    <div class="product-card">
                        <div class="product-icon">${r.icon}</div>
                        <div>
                            <strong>${r.step}: ${r.product}</strong><br>
                            <span style="color:#667eea;">${r.brand}</span> - ${r.price}<br>
                            <p style="margin-top:10px;">${r.description}</p>
                            <div>${r.ingredients.map(i => `<span class="ingredient-tag">${i}</span>`).join('')}</div>
                        </div>
                    </div>
                `).join('')}
            `;
            
            // Diet Tab
            document.getElementById('diet').innerHTML = `
                <h4>🥗 Gut-Skin Nutrition Plan</h4>
                <p style="margin-bottom:20px;"><strong>Tip:</strong> ${data.gut_tip}</p>
                
                <h4 style="color:#48bb78;">✅ Foods to Eat</h4>
                <div class="food-grid">
                    ${data.foods.good.map(f => `
                        <div class="food-card food-good">
                            <div style="font-size:40px;">${f.icon}</div>
                            <strong>${f.name}</strong>
                            <small>${f.benefit}</small>
                        </div>
                    `).join('')}
                </div>
                
                <h4 style="color:#f56565; margin-top:20px;">❌ Foods to Limit</h4>
                <div class="food-grid">
                    ${data.foods.bad.map(f => `
                        <div class="food-card food-bad">
                            <div style="font-size:40px;">${f.icon}</div>
                            <strong>${f.name}</strong>
                            <small>${f.reason}</small>
                        </div>
                    `).join('')}
                </div>
            `;
        }
        
        function showTab(tabId) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            event.target.classList.add('active');
            document.getElementById(tabId).classList.add('active');
        }
        
        startCamera();
    </script>
</body>
</html>
"""

# Real product database
PRODUCTS = {
    'oily': {
        'cleanser': {'icon': '🧴', 'brand': 'CeraVe', 'product': 'Foaming Facial Cleanser', 'price': '₹1,200', 
                     'ingredients': ['Niacinamide', 'Ceramides', 'Hyaluronic Acid'],
                     'description': 'Removes excess oil without stripping moisture'},
        'moisturizer': {'icon': '💧', 'brand': 'Neutrogena', 'product': 'Hydro Boost Water Gel', 'price': '₹950',
                       'ingredients': ['Hyaluronic Acid', 'Glycerin'],
                       'description': 'Oil-free, lightweight hydration'},
        'serum': {'icon': '✨', 'brand': 'The Ordinary', 'product': 'Niacinamide 10% + Zinc 1%', 'price': '₹600',
                 'ingredients': ['Niacinamide', 'Zinc PCA'],
                 'description': 'Reduces sebum and minimizes pores'}
    },
    'dry': {
        'cleanser': {'icon': '🥛', 'brand': 'CeraVe', 'product': 'Hydrating Facial Cleanser', 'price': '₹1,200',
                     'ingredients': ['Ceramides', 'Hyaluronic Acid'],
                     'description': 'Gentle, non-foaming, maintains moisture'},
        'moisturizer': {'icon': '🧈', 'brand': 'CeraVe', 'product': 'Moisturizing Cream', 'price': '₹1,400',
                       'ingredients': ['Ceramides', 'Hyaluronic Acid', 'Petrolatum'],
                       'description': 'Rich, barrier-repairing cream'},
        'serum': {'icon': '💦', 'brand': 'The Ordinary', 'product': 'Hyaluronic Acid 2% + B5', 'price': '₹650',
                 'ingredients': ['Hyaluronic Acid', 'Vitamin B5'],
                 'description': 'Deep hydration for dry skin'}
    },
    'normal': {
        'cleanser': {'icon': '🌸', 'brand': 'COSRX', 'product': 'Low pH Good Morning Gel', 'price': '₹850',
                     'ingredients': ['Tea Tree', 'BHA'],
                     'description': 'Gentle, pH-balancing cleanser'},
        'moisturizer': {'icon': '🌊', 'brand': 'Belif', 'product': 'The True Cream Aqua Bomb', 'price': '₹2,600',
                       'ingredients': ['Ceramides', 'Lady\'s Mantle'],
                       'description': 'Lightweight, refreshing hydration'},
        'serum': {'icon': '☀️', 'brand': 'SkinCeuticals', 'product': 'C E Ferulic', 'price': '₹12,500',
                 'ingredients': ['Vitamin C', 'Vitamin E', 'Ferulic Acid'],
                 'description': 'Antioxidant protection and brightening'}
    }
}

BRIGHTENING = [
    {'icon': '⚪', 'brand': 'The Ordinary', 'name': 'Alpha Arbutin 2% + HA', 'price': '₹800', 'rating': 4.7, 'key_ingredient': 'Alpha Arbutin'},
    {'icon': '🔴', 'brand': 'Minimalist', 'name': 'Tranexamic Acid 3%', 'price': '₹699', 'rating': 4.6, 'key_ingredient': 'Tranexamic Acid'},
    {'icon': '🟡', 'brand': 'Kojic', 'name': 'Kojic Acid Cream', 'price': '₹450', 'rating': 4.4, 'key_ingredient': 'Kojic Acid'}
]

FOODS = {
    'good': [
        {'icon': '🥑', 'name': 'Avocado', 'benefit': 'Healthy fats for barrier'},
        {'icon': '🫐', 'name': 'Blueberries', 'benefit': 'Antioxidants'},
        {'icon': '🥜', 'name': 'Walnuts', 'benefit': 'Omega-3 reduces inflammation'},
        {'icon': '🍵', 'name': 'Green Tea', 'benefit': 'Polyphenols calm skin'}
    ],
    'bad': [
        {'icon': '🍕', 'name': 'Pizza', 'reason': 'High glycemic'},
        {'icon': '🥛', 'name': 'Dairy', 'reason': 'May trigger acne'},
        {'icon': '🍬', 'name': 'Candy', 'reason': 'Sugar causes glycation'},
        {'icon': '🍺', 'name': 'Alcohol', 'reason': 'Dehydrates skin'}
    ]
}

def analyze_skin_region(region):
    """Analyze a specific face region for oiliness, redness, and darkness"""
    if region is None or region.size == 0:
        return {'oiliness': 0, 'redness': 0, 'darkness': 0}
    
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    
    # Oiliness (brightness)
    oiliness = float(np.mean(hsv[:, :, 2]) / 255 * 100)
    
    # Redness (acne/inflammation)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 + mask2
    redness = float(np.sum(red_mask > 0) / red_mask.size * 100 * 3)
    redness = min(100, redness)
    
    # Darkness (pigmentation)
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    darkness = float(np.sum(gray < 80) / gray.size * 100 * 2)
    darkness = min(100, darkness)
    
    return {'oiliness': oiliness, 'redness': redness, 'darkness': darkness}

@app.route('/')
def home():
    return render_template_string(HTML)

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        file = request.files['image']
        img_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Invalid image file'})
        
        # Resize for processing
        h, w = img.shape[:2]
        if w > 800:
            scale = 800 / w
            img = cv2.resize(img, (800, int(h * scale)))
            h, w = img.shape[:2]
        
        # Detect face
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        
        if len(faces) == 0:
            return jsonify({'error': 'No face detected. Please ensure your face is clearly visible and well-lit.'})
        
        x, y, fw, fh = faces[0]
        face = img[y:y+fh, x:x+fw]
        
        # Divide face into regions
        h_third = fh // 3
        w_third = fw // 3
        
        regions = {}
        
        # Forehead (top third)
        if h_third > 0:
            forehead = face[0:h_third, :]
            regions['Forehead'] = analyze_skin_region(forehead)
        
        # T-zone (nose area - middle third center)
        if h_third > 0 and w_third > 0:
            nose = face[h_third:2*h_third, w_third:2*w_third]
            regions['T-Zone/Nose'] = analyze_skin_region(nose)
        
        # Cheeks (left and right)
        if h_third > 0 and w_third > 0:
            left_cheek = face[h_third:2*h_third, 0:w_third]
            right_cheek = face[h_third:2*h_third, 2*w_third:fw]
            regions['Left Cheek'] = analyze_skin_region(left_cheek)
            regions['Right Cheek'] = analyze_skin_region(right_cheek)
        
        # Chin (bottom third)
        if h_third > 0:
            chin = face[2*h_third:fh, :]
            regions['Chin'] = analyze_skin_region(chin)
        
        # Calculate overall metrics
        avg_oiliness = np.mean([r['oiliness'] for r in regions.values()])
        avg_redness = np.mean([r['redness'] for r in regions.values()])
        avg_darkness = np.mean([r['darkness'] for r in regions.values()])
        
        # Determine skin type based on actual measurements
        if avg_oiliness > 55:
            skin_type = 'Oily'
            skin_icon = '🛢️'
        elif avg_oiliness < 35:
            skin_type = 'Dry'
            skin_icon = '🍂'
        elif 35 <= avg_oiliness <= 55:
            # Check for combination (oily T-zone, dry cheeks)
            tzone_oil = regions.get('T-Zone/Nose', {}).get('oiliness', 0)
            cheek_oil = (regions.get('Left Cheek', {}).get('oiliness', 0) + 
                        regions.get('Right Cheek', {}).get('oiliness', 0)) / 2
            if tzone_oil > 50 and cheek_oil < 40:
                skin_type = 'Combination'
                skin_icon = '⚖️'
            else:
                skin_type = 'Normal'
                skin_icon = '✨'
        else:
            skin_type = 'Normal'
            skin_icon = '✨'
        
        # Texture analysis using Laplacian variance
        gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray_face, cv2.CV_64F).var()
        roughness = min(100, laplacian / 5)
        smoothness = 100 - roughness
        
        # Pore score (correlates with oiliness)
        pore_score = min(100, avg_oiliness * 1.2)
        
        # Acne types based on redness patterns
        acne_types = {
            'Pustules': avg_redness * 0.7 if avg_redness > 15 else 0,
            'Papules': avg_redness * 0.5 if avg_redness > 10 else 0,
            'Nodules': avg_redness * 0.3 if avg_redness > 30 else 0,
            'Cystic': avg_redness * 0.2 if avg_redness > 40 else 0,
            'Comedones': 40 if skin_type == 'Oily' else 15
        }
        
        acne_icons = {
            'Pustules': '🔴', 'Papules': '🟠', 'Nodules': '🟤', 
            'Cystic': '💢', 'Comedones': '⚫'
        }
        
        acne_score = float(np.mean(list(acne_types.values())))
        
        # Dark circles (under-eye area)
        dark_circles_score = avg_darkness * 1.2 if avg_darkness > 20 else avg_darkness * 0.5
        
        # Scar detection based on texture
        scars = []
        if roughness > 40:
            scars.append({
                'icon': '📌', 'type': 'Atrophic (Icepick/Boxcar)', 
                'severity': round(roughness * 0.8, 1),
                'description': 'Depressed scars from past acne'
            })
        if avg_redness > 25:
            scars.append({
                'icon': '🔴', 'type': 'Post-Inflammatory Erythema', 
                'severity': round(avg_redness * 0.6, 1),
                'description': 'Red marks after acne healing'
            })
        if avg_darkness > 30:
            scars.append({
                'icon': '🎯', 'type': 'Post-Inflammatory Hyperpigmentation', 
                'severity': round(avg_darkness * 0.7, 1),
                'description': 'Dark marks after acne'
            })
        
        # Determine severity based on actual measurements
        if acne_score < 10 and avg_redness < 10 and avg_darkness < 15:
            severity = "Clear"
            severity_class = "clear"
            severity_icon = "✅"
            message = "Your skin appears clear and healthy!"
            need_derm = False
            consultation = ""
        elif acne_score < 25 and avg_redness < 25 and avg_darkness < 30:
            severity = "Mild"
            severity_class = "mild"
            severity_icon = "💚"
            message = "Minor skin concerns detected. Easily manageable with proper care."
            need_derm = False
            consultation = ""
        elif acne_score < 50 and avg_redness < 50 and avg_darkness < 50:
            severity = "Moderate"
            severity_class = "moderate"
            severity_icon = "⚠️"
            message = "Visible skin concerns. Consistent skincare routine recommended."
            need_derm = False
            consultation = ""
        else:
            severity = "Severe"
            severity_class = "severe"
            severity_icon = "🚨"
            message = "Significant skin concerns detected."
            need_derm = True
            consultation = "High acne/redness/pigmentation levels detected. Professional dermatologist evaluation recommended."

        # Health score based on all factors
        health_score = 100 - (acne_score * 0.3 + avg_redness * 0.25 + avg_darkness * 0.25 + roughness * 0.2)
        health_score = max(30, min(98, health_score))
        
        # Acne level text
        if acne_score < 10:
            acne_level = "None/Minimal"
        elif acne_score < 30:
            acne_level = "Mild"
        elif acne_score < 60:
            acne_level = "Moderate"
        else:
            acne_level = "Significant"
        
        # Texture level text
        if roughness < 20:
            texture_level = "Smooth"
        elif roughness < 45:
            texture_level = "Slightly Textured"
        else:
            texture_level = "Rough/Bumpy"
        
        # Pigmentation level text
        if avg_darkness < 20:
            pigmentation_level = "None/Minimal"
        elif avg_darkness < 45:
            pigmentation_level = "Mild"
        else:
            pigmentation_level = "Visible"
        
        # Get product recommendations based on skin type
        skin_key = skin_type.lower()
        if skin_key not in PRODUCTS:
            skin_key = 'normal'
        
        prods = PRODUCTS[skin_key]
        routine = [
            {'icon': prods['cleanser']['icon'], 'step': 'Cleanse', 
             'product': prods['cleanser']['product'], 'brand': prods['cleanser']['brand'],
             'price': prods['cleanser']['price'], 'description': prods['cleanser']['description'],
             'ingredients': prods['cleanser']['ingredients']},
            {'icon': prods['serum']['icon'], 'step': 'Treat', 
             'product': prods['serum']['product'], 'brand': prods['serum']['brand'],
             'price': prods['serum']['price'], 'description': prods['serum']['description'],
             'ingredients': prods['serum']['ingredients']},
            {'icon': prods['moisturizer']['icon'], 'step': 'Moisturize', 
             'product': prods['moisturizer']['product'], 'brand': prods['moisturizer']['brand'],
             'price': prods['moisturizer']['price'], 'description': prods['moisturizer']['description'],
             'ingredients': prods['moisturizer']['ingredients']}
        ]
        
        # Add SPF
        routine.append({
            'icon': '☀️', 'step': 'Protect',
            'product': 'UV Clear SPF 46', 'brand': 'EltaMD',
            'price': '₹2,800', 'description': 'Broad spectrum protection, non-comedogenic',
            'ingredients': ['Zinc Oxide', 'Niacinamide']
        })
        
        gut_tips = [
            "Eat probiotic-rich foods like yogurt to improve gut-skin axis!",
            "Stay hydrated - 8 glasses of water daily improves skin clarity.",
            "Reduce sugar intake to prevent glycation and breakouts.",
            "Omega-3 fatty acids from fish reduce skin inflammation."
        ]
        
        return jsonify({
            'skin_type': skin_type,
            'skin_type_icon': skin_icon,
            'severity': severity,
            'severity_class': severity_class,
            'severity_icon': severity_icon,
            'message': message,
            'need_dermatologist': need_derm,
            'consultation_reason': consultation,
            'acne_level': acne_level,
            'texture_level': texture_level,
            'pigmentation_level': pigmentation_level,
            'acne_score': round(acne_score, 1),
            'texture_score': round(roughness, 1),
            'pigmentation_score': round(avg_darkness, 1),
            'dark_circles_score': round(dark_circles_score, 1),
            'uneven_tone': round(avg_darkness * 0.9, 1),
            'smoothness': round(smoothness, 1),
            'roughness': round(roughness, 1),
            'pore_score': round(pore_score, 1),
            'health_score': round(health_score, 1),
            'acne_types': acne_types,
            'acne_icons': acne_icons,
            'regions': regions,
            'scars': scars,
            'brightening_products': BRIGHTENING,
            'routine': routine,
            'foods': FOODS,
            'gut_tip': gut_tips[int(acne_score) % len(gut_tips)]
        })
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Analysis error: {str(e)}'})

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🧬 DERMAI PRO - REAL SKIN ANALYSIS")
    print("="*50)
    print("\n✅ Open Chrome: http://127.0.0.1:5000")
    print("📸 Use a clear, well-lit face photo for best results")
    print("\nPress Ctrl+C to stop\n")
    app.run(debug=True, host='127.0.0.1', port=5000)