"""
Minimal Flask Backend - Bypasses model loading crash
Provides mock predictions until encoding issue is resolved
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
from PIL import Image
import io

app = Flask(__name__)

# Configure CORS
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:5173", "http://localhost:5174"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'status': 'operational',
        'service': 'Iris Stress Detection API (Minimal Mode)',
        'version': '1.0.0',
        'note': 'Using mock predictions - model loading disabled due to encoding issue'
    }), 200

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'operational',
        'model_status': 'mock_mode',
        'backend': 'Flask',
        'port': 5000
    }), 200

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        # Get uploaded image
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image provided'
            }), 400
        
        image_file = request.files['image']
        age = int(request.form.get('age', 30))
        
        # Read image
        image_bytes = image_file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({
                'success': False,
                'error': 'Invalid image format'
            }), 400
        
        # Mock detection results
        # Analyze image to detect tension rings (iris circles)
        height, width = img.shape[:2]
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Detect circles with STRICT parameters to avoid false positives
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,              # Higher dp = fewer circles detected
            minDist=50,          # Minimum distance between circles
            param1=100,          # Higher = stricter edge detection
            param2=50,           # Higher = stricter circle detection (was 30)
            minRadius=15,        # Minimum radius for tension rings
            maxRadius=min(height, width) // 3  # Maximum radius
        )
        
        # Count ONLY tension rings (exclude the main iris/pupil boundaries)
        if circles is not None and len(circles[0]) > 2:
            # More than 2 circles = likely has tension rings
            # Subtract 2 for pupil and iris outer boundary
            ring_count = len(circles[0]) - 2
            ring_count = max(0, min(ring_count, 3))  # Cap at 3 rings
        else:
            # 0-2 circles = normal eye (just pupil/iris boundaries)
            ring_count = 0
        
        # Additional check: analyze pixel intensity variance
        # Tension rings create alternating light/dark patterns
        # Calculate standard deviation - high variance suggests rings
        std_dev = np.std(gray)
        
        # Adjust ring count based on image characteristics
        if std_dev < 30 and ring_count > 0:
            # Low variance but circles detected = likely false positive
            ring_count = 0
        
        # KEY LOGIC: Tension rings → STRESSED (be conservative)
        if ring_count >= 2:
            stress_detected = True
            stress_probability = 0.85  # High confidence
        elif ring_count == 1:
            stress_detected = True
            stress_probability = 0.70  # Medium-high confidence
        else:
            # No rings detected → NORMAL
            stress_detected = False
            stress_probability = 0.25  # Low stress probability
        
        # Mock response matching real API format
        response = {
            'success': True,
            'detection': {
                'success': True,
                'method': 'mock',
                'pupil': {'center': [width//2, height//2], 'radius': 40},
                'iris': {'center': [width//2, height//2], 'radius': 80}
            },
            'measurements': {
                'pupil_diameter_mm': 4.5 if not stress_detected else 6.2,
                'is_dilated': stress_detected,
                'tension_rings': ring_count
            },
            'prediction': {
                'stress_detected': stress_detected,
                'stress_probability': stress_probability,
                'stress_level': 'STRESS' if stress_detected else 'NORMAL',
                'confidence': stress_probability if stress_detected else (1 - stress_probability),
                'note': 'Mock prediction - ML model temporarily disabled'
            },
            'metadata': {
                'age': age,
                'image_size': f"{width}x{height}",
                'mode': 'mock'
            }
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': 'Processing failed',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    print("\n" + "="*80)
    print("FLASK BACKEND - MINIMAL MODE")
    print("="*80)
    print("Running without ML model (mock predictions)")
    print("Server: http://localhost:5000")
    print("="*80 + "\n")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=True
    )
