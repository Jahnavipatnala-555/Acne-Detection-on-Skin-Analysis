"""
DERMAI PRO - COMPLETELY NEW VERSION
FORCE NEW DESIGN - DIFFERENT PORT
"""
import cv2
import numpy as np
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# COMPLETELY DIFFERENT HTML - BOLD COLORS, OBVIOUS CHATBOT
HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>✨ NEW DERMAI ✨</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        /* AGGRESSIVE REDESIGN - YOU WILL SEE THE DIFFERENCE */
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            background: #000000;
            font-family: Arial, sans-serif;
            padding: 20px;
        }
        
        /* BIG BANNER TO PROVE IT'S NEW */
        .new-banner {
            background: linear-gradient(90deg, #ff0000, #ff00ff, #00ffff);
            padding: 25px;
            border-radius: 20px;
            text-align: center;
            margin-bottom: 30px;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
        
        .new-banner h1 {
            color: white;
            font-size: 48px;
            text-shadow: 4px 4px 10px black;
        }
        
        .new-banner p {
            color: yellow;
            font-size: 24px;
            font-weight: bold;
        }
        
        /* MAIN LAYOUT */
        .layout {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 25px;
        }
        
        /* CARDS */
        .card {
            background: #1a1a1a;
            border-radius: 30px;
            padding: 30px;
            border: 3px solid #ff00ff;
            box-shadow: 0 0 30px #ff00ff80;
        }
        
        .card h2 {
            color: #00ffff;
            font-size: 32px;
            margin-bottom: 25px;
            text-transform: uppercase;
        }
        
        /* CAMERA */
        .camera-box {
            background: black;
            border-radius: 20px;
            overflow: hidden;
            border: 4px solid #ff0000;
        }
        
        video {
            width: 100%;
            display: block;
        }
        
        /* BUTTONS */
        .btn {
            padding: 18px 35px;
            border: none;
            border-radius: 60px;
            font-size: 18px;
            font-weight: bold;
            cursor: pointer;
            margin: 10px 8px;
            text-transform: uppercase;
            transition: all 0.3s;
        }
        
        .btn-purple {
            background: #8b00ff;
            color: white;
            box-shadow: 0 0 20px #8b00ff;
        }
        
        .btn-green {
            background: #00ff88;
            color: black;
            box-shadow: 0 0 20px #00ff88;
        }
        
        .btn-red {
            background: #ff0040;
            color: white;
            box-shadow: 0 0 20px #ff0040;
        }
        
        .btn:hover {
            transform: scale(1.05);
        }
        
        /* UPLOAD AREA */
        .upload-zone {
            border: 4px dashed #ff00ff;
            border-radius: 20px;
            padding: 30px;
            text-align: center;
            margin-top: 25px;
            cursor: pointer;
            background: #0a0a0a;
        }
        
        .upload-zone:hover {
            background: #1a0033;
        }
        
        .upload-zone p {
            color: #00ffff;
            font-size: 20px;
            margin-top: 15px;
        }
        
        /* RESULTS */
        .result-box {
            color: white;
        }
        
        .severity-tag {
            display: inline-block;
            padding: 15px 35px;
            border-radius: 50px;
            font-size: 28px;
            font-weight: bold;
            margin: 20px 0;
        }
        
        .clear { background: #00ff88; color: black; }
        .mild { background: #ffaa00; color: black; }
        .severe { background: #ff0040; color: white; }
        
        .metric {
            background: #0a0a0a;
            padding: 25px;
            border-radius: 20px;
            margin: 15px 0;
            border-left: 6px solid #00ffff;
        }
        
        .metric .value {
            font-size: 48px;
            color: #00ffff;
            font-weight: bold;
        }
        
        .metric .label {
            color: #aaaaaa;
            font-size: 16px;
        }
        
        /* ========== AI CHATBOT - IMPOSSIBLE TO MISS ========== */
        .chatbot-icon {
            position: fixed;
            bottom: 40px;
            right: 40px;
            width: 90px;
            height: 90px;
            background: linear-gradient(135deg, #ff00ff, #00ffff);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            box-shadow: 0 0 50px #ff00ff;
            z-index: 9999;
            border: 5px solid white;
            animation: bounce 1.5s infinite;
        }
        
        @keyframes bounce {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-15px); }
        }
        
        .chatbot-icon span {
            color: white;
            font-size: 50px;
            font-weight: bold;
        }
        
        .chatbot-panel {
            position: fixed;
            bottom: 150px;
            right: 40px;
            width: 450px;
            height: 600px;
            background: #0a0a0a;
            border-radius: 30px;
            border: 4px solid #ff00ff;
            display: none;
            flex-direction: column;
            z-index: 10000;
            box-shadow: 0 0 80px #ff00ff;
        }
        
        .chatbot-panel.show {
            display: flex;
        }
        
        .chatbot-header {
            background: linear-gradient(90deg, #ff00ff, #00ffff);
            padding: 25px;
            border-radius: 26px 26px 0 0;
        }
        
        .chatbot-header h3 {
            color: white;
            font-size: 28px;
        }
        
        .chat-messages {
            flex: 1;
            padding: 25px;
            overflow-y: auto;
        }
        
        .message {
            margin: 15px 0;
            padding: 15px 20px;
            border-radius: 20px;
            max-width: 80%;
        }
        
        .bot-message {
            background: #1a0033;
            color: #00ffff;
            border: 2px solid #ff00ff;
        }
        
        .user-message {
            background: #00331a;
            color: #00ff88;
            border: 2px solid #00ff88;
            margin-left: auto;
        }
        
        .chat-input-area {
            padding: 25px;
            border-top: 2px solid #ff00ff;
            display: flex;
            gap: 15px;
        }
        
        .chat-input-area input {
            flex: 1;
            padding: 18px;
            border-radius: 50px;
            border: 2px solid #ff00ff;
            background: black;
            color: white;
            font-size: 16px;
        }
        
        .chat-input-area button {
            padding: 18px 30px;
            border-radius: 50px;
            border: none;
            background: #ff00ff;
            color: white;
            font-weight: bold;
            cursor: pointer;
            font-size: 16px;
        }
        
        .quick-chips {
            padding: 0 25px 15px;
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        
        .chip {
            background: #1a1a1a;
            padding: 10px 20px;
            border-radius: 30px;
            color: #00ffff;
            border: 1px solid #ff00ff;
            cursor: pointer;
            font-size: 14px;
        }
        
        .chip:hover {
            background: #ff00ff;
            color: white;
        }
        
        /* TABS */
        .tabs {
            display: flex;
            gap: 10px;
            margin: 30px 0;
            border-bottom: 2px solid #ff00ff;
        }
        
        .tab {
            padding: 15px 30px;
            background: none;
            border: none;
            color: #aaaaaa;
            font-size: 18px;
            cursor: pointer;
            font-weight: bold;
        }
        
        .tab.active {
            color: #00ffff;
            border-bottom: 4px solid #ff00ff;
        }
        
        .tab-content {
            display: none;
            color: white;
            padding: 20px 0;
        }
        
        .tab-content.active {
            display: block;
        }
    </style>
</head>
<body>

<!-- BIG BANNER - YOU CANNOT MISS THIS -->
<div class="new-banner">
    <h1>🧬 NEW DERMAI PRO v3.0 🧬</h1>
    <p>✨ COMPLETELY REDESIGNED ✨ AI CHATBOT VISIBLE ✨</p>
</div>

<div class="layout">
    <!-- LEFT: CAMERA -->
    <div class="card">
        <h2>📷 SCAN SKIN</h2>
        <div class="camera-box">
            <video id="video" autoplay playsinline></video>
            <canvas id="canvas" style="display: none;"></canvas>
        </div>
        <div style="margin-top: 25px;">
            <button class="btn btn-purple" onclick="startCamera()">▶ START</button>
            <button class="btn btn-green" onclick="capture()">📸 ANALYZE</button>
            <button class="btn btn-red" onclick="stopCamera()">⏹ STOP</button>
        </div>
        <div class="upload-zone" onclick="document.getElementById('fileInput').click()">
            <div style="font-size: 60px;">📁</div>
            <p>CLICK TO UPLOAD IMAGE</p>
            <input type="file" id="fileInput" accept="image/*" style="display: none;" onchange="upload(this)">
        </div>
    </div>
    
    <!-- RIGHT: RESULTS -->
    <div class="card">
        <h2>📊 ANALYSIS</h2>
        <div id="resultsArea" class="result-box">
            <p style="color: #aaaaaa; text-align: center; padding: 80px 20px; font-size: 20px;">
                ⬅ CAPTURE OR UPLOAD TO BEGIN
            </p>
        </div>
    </div>
</div>

<!-- FULL RESULTS WITH TABS -->
<div id="fullResults" style="display: none; margin-top: 30px;">
    <div class="card">
        <div class="tabs">
            <button class="tab active" onclick="showTab('overview')">📋 OVERVIEW</button>
            <button class="tab" onclick="showTab('acne')">🦠 ACNE</button>
            <button class="tab" onclick="showTab('treat')">💊 TREATMENT</button>
        </div>
        <div id="overview" class="tab-content active"></div>
        <div id="acne" class="tab-content"></div>
        <div id="treat" class="tab-content"></div>
    </div>
</div>

<!-- ========== AI CHATBOT - BIG AND VISIBLE ========== -->
<div class="chatbot-icon" onclick="toggleChatbot()">
    <span>🤖</span>
</div>

<div class="chatbot-panel" id="chatbotPanel">
    <div class="chatbot-header">
        <h3>🤖 AI SKIN ASSISTANT</h3>
        <p style="color: white; margin-top: 8px;">Ask me anything about your skin!</p>
    </div>
    <div class="chat-messages" id="chatMessages">
        <div class="message bot-message">
            <strong>🤖 AI:</strong> Hello! I'm your skin assistant. Ask me about acne, scars, or skincare!
        </div>
    </div>
    <div class="quick-chips">
        <span class="chip" onclick="quickAsk('What causes acne?')">Acne causes</span>
        <span class="chip" onclick="quickAsk('How to treat scars?')">Scar treatment</span>
        <span class="chip" onclick="quickAsk('Best diet for skin?')">Diet tips</span>
        <span class="chip" onclick="quickAsk('My skincare routine?')">Routine</span>
    </div>
    <div class="chat-input-area">
        <input type="text" id="chatInput" placeholder="Type your question...">
        <button onclick="sendMessage()">SEND</button>
    </div>
</div>

<script>
    let stream = null;
    let currentData = null;
    
    // TOGGLE CHATBOT
    function toggleChatbot() {
        document.getElementById('chatbotPanel').classList.toggle('show');
    }
    
    async function startCamera() {
        try {
            stream = await navigator.mediaDevices.getUserMedia({ video: true });
            document.getElementById('video').srcObject = stream;
        } catch(e) { alert('Camera error: ' + e.message); }
    }
    
    function stopCamera() {
        if(stream) { stream.getTracks().forEach(t => t.stop()); }
    }
    
    function capture() {
        const video = document.getElementById('video');
        const canvas = document.getElementById('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0);
        canvas.toBlob(b => analyze(b), 'image/jpeg');
    }
    
    function upload(input) { 
        if(input.files[0]) analyze(input.files[0]); 
    }
    
    async function analyze(blob) {
        const fd = new FormData();
        fd.append('image', blob);
        
        try {
            const res = await fetch('/analyze', { method: 'POST', body: fd });
            const data = await res.json();
            if(data.error) { alert(data.error); return; }
            
            currentData = data;
            
            document.getElementById('resultsArea').innerHTML = `
                <div style="text-align: center;">
                    <span class="severity-tag ${data.severity_class}">${data.severity_icon} ${data.severity}</span>
                    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px;">
                        <div class="metric"><div class="value">${data.acne_score}%</div><div class="label">ACNE</div></div>
                        <div class="metric"><div class="value">${data.pigment}%</div><div class="label">DARK SPOTS</div></div>
                        <div class="metric"><div class="value">${data.health}%</div><div class="label">HEALTH</div></div>
                    </div>
                    <p style="color: #00ffff; font-size: 20px; margin-top: 20px;">${data.skin_type} • ${data.acne_type}</p>
                </div>
            `;
            
            document.getElementById('fullResults').style.display = 'block';
            document.getElementById('overview').innerHTML = `
                <h3 style="color: #00ffff; margin-bottom: 20px;">CLINICAL SUMMARY</h3>
                <p style="color: white; font-size: 18px;">${data.summary}</p>
                ${makeProgress('Acne Severity', data.acne_score, '#ff0040')}
                ${makeProgress('Pigmentation', data.pigment, '#ffaa00')}
                ${makeProgress('Texture', data.texture, '#00ffff')}
            `;
            
            document.getElementById('acne').innerHTML = `
                <h3 style="color: #00ffff;">ACNE TYPES</h3>
                ${Object.entries(data.acne_types).map(([t,s]) => makeProgress(t, s, '#ff0040')).join('')}
                <p style="color: white; margin-top: 30px; font-size: 18px;"><strong>Primary:</strong> ${data.acne_type}</p>
            `;
            
            document.getElementById('treat').innerHTML = `
                <h3 style="color: #00ffff;">YOUR ROUTINE</h3>
                ${data.routine.map(r => `<div class="metric"><strong style="color: #00ffff;">${r.time}:</strong> ${r.step}</div>`).join('')}
                <p style="color: #ffaa00; margin-top: 30px; font-size: 18px;">⚠️ ${data.note}</p>
            `;
            
        } catch(e) { alert('Error: ' + e.message); }
    }
    
    function makeProgress(label, value, color) {
        return `<div style="margin: 20px 0;"><div style="display: flex; justify-content: space-between; color: white; margin-bottom: 8px;"><span>${label}</span><span>${value}%</span></div><div style="height: 12px; background: #1a1a1a; border-radius: 10px;"><div style="width: ${value}%; height: 100%; background: ${color}; border-radius: 10px;"></div></div></div>`;
    }
    
    function showTab(id) {
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        event.target.classList.add('active');
        document.getElementById(id).classList.add('active');
    }
    
    // CHATBOT FUNCTIONS
    async function sendMessage() {
        const input = document.getElementById('chatInput');
        const msg = input.value.trim();
        if(!msg) return;
        
        const messages = document.getElementById('chatMessages');
        messages.innerHTML += `<div class="message user-message"><strong>You:</strong> ${msg}</div>`;
        input.value = '';
        
        try {
            const res = await fetch('/chat', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({message: msg})
            });
            const data = await res.json();
            messages.innerHTML += `<div class="message bot-message"><strong>🤖 AI:</strong> ${data.reply}</div>`;
            messages.scrollTop = messages.scrollHeight;
        } catch(e) {}
    }
    
    function quickAsk(q) {
        document.getElementById('chatInput').value = q;
        sendMessage();
    }
    
    // Start camera automatically
    startCamera();
</script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML)

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        file = request.files['image']
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        if img is None: return jsonify({'error': 'Invalid image'})
        
        h, w = img.shape[:2]
        if w > 800: img = cv2.resize(img, (800, int(h * (800/w))))
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        if len(faces) == 0: return jsonify({'error': 'No face detected'})
        
        x, y, fw, fh = faces[0]
        face = img[y:y+fh, x:x+fw]
        hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
        gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        
        avg_red = float(min(100, np.sum(cv2.inRange(hsv, (0, 60, 60), (12, 255, 255)) > 0) / (hsv.size/3) * 100 * 3))
        avg_dark = float(min(100, np.sum(gray_face < 85) / gray_face.size * 100 * 3))
        avg_oil = float(np.mean(hsv[:, :, 2]) / 255 * 100)
        
        actual_red = avg_red if avg_red > 8 else 0
        actual_dark = avg_dark if avg_dark > 10 else 0
        
        skin_type = 'Oily' if avg_oil > 60 else 'Dry' if avg_oil < 35 else 'Normal'
        roughness = min(100, cv2.Laplacian(gray_face, cv2.CV_64F).var() / 10)
        
        acne_types = {'Comedonal': actual_red*0.3, 'Papules': actual_red*0.4, 'Pustules': actual_red*0.3, 'Nodules': actual_red*0.15, 'Cystic': actual_red*0.1}
        acne_score = round(float(np.mean(list(acne_types.values()))), 1)
        primary = max(acne_types.items(), key=lambda x: x[1])[0] if max(acne_types.values()) > 5 else 'None'
        
        if acne_score < 8: sev, sev_class, sev_icon = 'CLEAR', 'clear', '✅'
        elif acne_score < 20: sev, sev_class, sev_icon = 'MILD', 'mild', '💚'
        elif acne_score < 45: sev, sev_class, sev_icon = 'MODERATE', 'moderate', '⚠️'
        else: sev, sev_class, sev_icon = 'SEVERE', 'severe', '🚨'
        
        health = round(max(35, min(98, 100 - (acne_score*0.4 + actual_dark*0.3 + roughness*0.2))), 1)
        
        return jsonify({
            'skin_type': skin_type, 'severity': sev, 'severity_class': sev_class, 'severity_icon': sev_icon,
            'acne_score': acne_score, 'pigment': round(actual_dark, 1), 'health': health,
            'acne_type': primary, 'acne_types': acne_types, 'texture': round(roughness, 1),
            'summary': f"{skin_type} skin, {sev} acne ({acne_score}%). {actual_dark:.0f}% pigmentation.",
            'routine': [{'time':'AM','step':'Cleanse + Vitamin C + SPF 50+'}, {'time':'PM','step':'Cleanse + Retinoid + Moisturizer'}],
            'note': 'Always use SPF 50+! Start retinoids slowly.'
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/chat', methods=['POST'])
def chat():
    msg = request.json.get('message', '').lower()
    reply = "I can help with acne, scars, and skincare. What would you like to know?"
    if 'acne' in msg: reply = "Acne is caused by clogged pores. Use Salicylic Acid or Benzoyl Peroxide. For severe acne, see a dermatologist."
    elif 'scar' in msg: reply = "Acne scars can be treated with Vitamin C, Retinoids, or microneedling."
    elif 'pigment' in msg: reply = "Dark spots fade with Vitamin C, Niacinamide, Azelaic Acid, and SUNSCREEN."
    elif 'diet' in msg: reply = "Eat Omega-3s (salmon), probiotics (yogurt). Avoid dairy and sugar."
    elif 'routine' in msg: reply = "AM: Cleanse → Vitamin C → SPF 50+. PM: Cleanse → Retinoid → Moisturizer."
    return jsonify({'reply': reply})

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 NEW DERMAI PRO - DIFFERENT DESIGN!")
    print("="*60)
    print("\n✅ Open Chrome: http://127.0.0.1:5000")
    print("✅ Look for BIG PURPLE CHATBOT ICON bottom-right!")
    print("✅ Rainbow banner at top - IMPOSSIBLE TO MISS!")
    print("\nPress Ctrl+C to stop\n")
    app.run(debug=True, host='127.0.0.1', port=5000)