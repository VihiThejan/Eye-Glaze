"""
Vercel Serverless Function Entry Point
Minimal Flask Backend for Iris Stress Detection
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2

app = Flask(__name__)

# Configure CORS - Allow all origins for Vercel deployment
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Disable Flask's default logger in production
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'status': 'operational',
        'service': 'Iris Stress Detection API',
        'version': '2.0.0',
        'deployment': 'Vercel Serverless',
        'message': 'Backend is running successfully'
    }), 200

@app.route('/api', methods=['GET'])
def api_root():
    return jsonify({
        'status': 'operational',
        'service': 'Iris Stress Detection API',
        'version': '2.0.0',
        'deployment': 'Vercel Serverless'
    }), 200

@app.route('/health', methods=['GET'])
@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'operational',
        'backend': 'Flask on Vercel',
        'endpoints': {
            'predict': '/predict (POST) - Image-based detection',
            'health': '/health (GET)'
        }
    }), 200

@app.route('/predict', methods=['POST', 'OPTIONS'])
@app.route('/api/predict', methods=['POST', 'OPTIONS'])
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
        
        # Analyze image to detect tension rings
        height, width = img.shape[:2]
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Detect circles with STRICT parameters
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=50,
            param1=100,
            param2=50,
            minRadius=15,
            maxRadius=min(height, width) // 3
        )
        
        # Count tension rings
        if circles is not None and len(circles[0]) > 2:
            ring_count = len(circles[0]) - 2
            ring_count = max(0, min(ring_count, 3))
        else:
            ring_count = 0
        
        # Analyze pixel intensity variance
        std_dev = np.std(gray)
        
        # Adjust ring count based on image characteristics
        if std_dev < 30 and ring_count > 0:
            ring_count = 0
        
        # Determine stress level
        if ring_count >= 2:
            stress_level = "STRESS"
            stress_detected = True
            stress_probability = 0.85
            confidence = 0.85
        elif ring_count == 1:
            stress_level = "PARTIAL_STRESS"
            stress_detected = True
            stress_probability = 0.70
            confidence = 0.70
        else:
            stress_level = "NORMAL"
            stress_detected = False
            stress_probability = 0.25
            confidence = 0.75
        
        # Calculate pupil diameter (mock)
        pupil_diameter_mm = round(3.5 + (ring_count * 0.8), 2)
        
        # Build response
        response = {
            'success': True,
            'prediction': {
                'stress_level': stress_level,
                'stress_detected': stress_detected,
                'stress_reason': f'{ring_count}_tension_rings' if ring_count > 0 else 'no_rings_detected',
                'stress_confidence_level': 'High' if confidence > 0.7 else 'Medium' if confidence > 0.5 else 'Low',
                'stress_probability': float(stress_probability),
                'stress_percentage': float(stress_probability * 100),
                'confidence': 'High' if confidence > 0.7 else 'Medium' if confidence > 0.5 else 'Low',
                'confidence_value': float(confidence * 100),
                'model_prediction': 'Image Analysis Detection',
                'needs_better_image': False,
                'is_potential_stress': stress_detected
            },
            'pupil_analysis': {
                'diameter_mm': pupil_diameter_mm,
                'stress_threshold': 4.0,
                'is_dilated': ring_count >= 2,
                'status': 'Dilated' if ring_count >= 2 else 'Normal',
                'recommended_range': {
                    'min': 3.0,
                    'max': 4.0,
                    'age_group': 'Below 60 years' if age < 60 else '60+ years'
                }
            },
            'iris_analysis': {
                'tension_rings_count': ring_count,
                'original_ring_count': ring_count,
                'ring_count_inferred': False,
                'has_stress_rings': ring_count > 0,
                'interpretation': f'{ring_count} tension ring(s) detected' if ring_count > 0 else 'No tension rings',
                'inference_note': None
            },
            'subject_info': {
                'age': age,
                'age_group': 'Below 60 years' if age < 60 else '60+ years'
            },
            'detection_info': {
                'pupil_detected': True,
                'iris_detected': True,
                'image_type': 'color' if len(img.shape) == 3 else 'grayscale',
                'detection_method': 'circle_detection',
                'total_circles_detected': len(circles[0]) if circles is not None else 0
            },
            'measurements': {
                'pupil_diameter_mm': pupil_diameter_mm,
                'ring_count': ring_count,
                'validation': 'Circle Detection Analysis',
                'conversion_factor': 0.05
            }
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': 'Processing failed',
            'message': str(e)
        }), 500

# Export the Flask app for Vercel
# Vercel's Python runtime expects a variable named 'app'
# No custom handler needed - Vercel handles WSGI automatically

# For local testing
if __name__ == '__main__':
    print("Starting Flask app for local testing...")
    print("Visit: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
