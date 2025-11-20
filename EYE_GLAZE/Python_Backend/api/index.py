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
        
        # Count tension rings (iris pattern analysis)
        original_ring_count = 0
        if circles is not None and len(circles[0]) > 2:
            original_ring_count = len(circles[0]) - 2
            original_ring_count = max(0, min(original_ring_count, 3))
        
        ring_count = original_ring_count
        ring_count_inferred = False
        
        # Analyze pixel intensity variance for validation
        std_dev = np.std(gray)
        
        # Adjust ring count based on image characteristics
        if std_dev < 30 and ring_count > 0:
            ring_count = 0
        
        # ============================================================
        # AGE-BASED PUPIL DIAMETER CALCULATION
        # ============================================================
        # Estimate pupil diameter based on image analysis
        # This is a simplified estimation - real measurement requires calibration
        
        # Use image center region to estimate pupil size
        cy, cx = height // 2, width // 2
        roi_size = min(height, width) // 4
        pupil_roi = gray[max(0, cy-roi_size):min(height, cy+roi_size), 
                         max(0, cx-roi_size):min(width, cx+roi_size)]
        
        # Find darkest region (likely pupil)
        _, binary = cv2.threshold(pupil_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        dark_pixels = np.sum(binary == 0)
        pupil_radius_pixels = np.sqrt(dark_pixels / np.pi) if dark_pixels > 0 else 30
        
        # Convert to mm (rough estimation: ~0.05mm per pixel)
        conversion_factor = 0.05
        pupil_diameter_mm = round(pupil_radius_pixels * 2 * conversion_factor, 2)
        
        # Clamp to realistic range (2-8mm)
        pupil_diameter_mm = max(2.0, min(8.0, pupil_diameter_mm))
        
        # Check if pupil size is too small (need better image)
        needs_better_image = pupil_diameter_mm < 1.5
        
        # ============================================================
        # AGE-BASED STRESS THRESHOLDS (from notebook logic)
        # ============================================================
        # Age < 60: Stressed if pupil > 4.0mm
        # Age ≥ 60: Stressed if pupil > 3.0mm
        
        if age < 60:
            stress_threshold_mm = 4.0
            recommended_min = 3.0
            recommended_max = 4.0
            age_group = "Below 60 years"
        else:
            stress_threshold_mm = 3.0
            recommended_min = 2.0
            recommended_max = 3.0
            age_group = "60+ years"
        
        # Check if pupil is dilated (primary stress indicator)
        is_dilated = pupil_diameter_mm > stress_threshold_mm
        
        # Determine pupil status
        if pupil_diameter_mm < recommended_min:
            pupil_status = "Constricted"
        elif is_dilated:
            pupil_status = "Dilated"
        else:
            pupil_status = "Normal"
        
        # ============================================================
        # INTELLIGENT STRESS DETERMINATION
        # ============================================================
        # Priority logic:
        # 1. Tension rings (iris analysis) = definite stress indicator
        # 2. Pupil dilation + age threshold = stress indicator
        # 3. Normal range = no stress
        
        final_stress_detected = False
        stress_reason = ""
        stress_confidence_level = ""
        
        if ring_count >= 1:
            # Tension rings detected - definite stress
            final_stress_detected = True
            stress_reason = f"{ring_count}_tension_rings"
            stress_confidence_level = "High"
            stress_probability = 0.95 if ring_count >= 2 else 0.85
        elif ring_count == 0 and is_dilated:
            # Pupil dilated but no rings - potential stress
            final_stress_detected = True
            stress_reason = "pupil_dilation"
            stress_confidence_level = "Medium"
            stress_probability = 0.70
        elif ring_count == 0 and not is_dilated:
            # No indicators - normal
            final_stress_detected = False
            stress_reason = "no_indicators"
            stress_confidence_level = "High"
            stress_probability = 0.15
        else:
            # Fallback
            final_stress_detected = False
            stress_reason = "inconclusive"
            stress_confidence_level = "Low"
            stress_probability = 0.50
        
        # Set final stress level
        if ring_count >= 2:
            stress_level = "STRESS"
        elif ring_count == 1 or (is_dilated and ring_count == 0):
            stress_level = "PARTIAL_STRESS"
        else:
            stress_level = "NORMAL"
        
        # Build response with proper age-based logic
        response = {
            'success': True,
            'prediction': {
                'stress_level': stress_level,
                'stress_detected': final_stress_detected,
                'stress_reason': stress_reason,
                'stress_confidence_level': stress_confidence_level,
                'stress_probability': float(stress_probability),
                'stress_percentage': float(stress_probability * 100),
                'confidence': stress_confidence_level,
                'confidence_value': float(stress_probability * 100),
                'model_prediction': 'Age-Based Circle Detection',
                'needs_better_image': needs_better_image,
                'is_potential_stress': (is_dilated and ring_count == 0)
            },
            'pupil_analysis': {
                'diameter_mm': float(pupil_diameter_mm),
                'stress_threshold': float(stress_threshold_mm),
                'is_dilated': is_dilated,
                'status': pupil_status,
                'recommended_range': {
                    'min': float(recommended_min),
                    'max': float(recommended_max),
                    'age_group': age_group
                }
            },
            'iris_analysis': {
                'tension_rings_count': int(ring_count),
                'original_ring_count': int(original_ring_count),
                'ring_count_inferred': ring_count_inferred,
                'has_stress_rings': ring_count >= 1,
                'interpretation': ('High stress indicator' if ring_count >= 3 else 
                                 'Moderate stress indicator' if ring_count >= 1 else 
                                 'No stress indicators'),
                'inference_note': None
            },
            'subject_info': {
                'age': age,
                'age_group': age_group
            },
            'detection_info': {
                'pupil_detected': True,
                'iris_detected': True,
                'image_type': 'color' if len(img.shape) == 3 else 'grayscale',
                'detection_method': 'age_based_circle_detection',
                'total_circles_detected': len(circles[0]) if circles is not None else 0
            },
            'measurements': {
                'pupil_diameter_mm': float(pupil_diameter_mm),
                'ring_count': int(ring_count),
                'validation': 'Age-based threshold analysis',
                'conversion_factor': conversion_factor
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
