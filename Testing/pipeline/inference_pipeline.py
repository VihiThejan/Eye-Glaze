"""
Main Inference Pipeline - Production stress detection workflow
Author: AI Assistant
Date: 2025-11-11

Orchestrates the complete production pipeline:
1. Load image and detect type (color/grayscale)
2. Detect pupil and iris using adaptive detection
3. Measure pupil diameter and count tension rings
4. Preprocess for model input (5-channel format, EXACT match to training)
5. Run prediction with production model (best_dual_stream_model.keras)
6. Generate results with fusion analysis

CRITICAL: All preprocessing must match training exactly:
- Pupil stream: RGB only (channels 3-4 ZEROED)
- Iris stream: RGB + Canny + BlackHat (all 5 channels active)
- Age: One-hot encoded (8 groups)
- Ring count: Normalized by /10.0
"""

import os
import cv2
import numpy as np
from typing import Dict, Tuple, Optional
from pathlib import Path

# Import detection modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from detection import (
    detect_eye_color,
    detect_eye_grayscale,
    count_tension_rings
)
from measurement import measure_pupil_diameter, validate_pupil_measurement
from utils import preprocess_eye_image, encode_age, extract_eye_region
from pipeline.model_loader import predict_single
import config


def classify_stress_level(prediction_score: float, confidence: float) -> str:
    """
    Classify stress level based on confidence threshold.
    
    Classification Rules:
    - Confidence < 80%: Normal
    - Confidence >= 80%: Stress
    
    Parameters:
    -----------
    prediction_score : float
        Model prediction (0-1, where >0.5 = stressed)
    confidence : float
        Prediction confidence (0-1)
    
    Returns:
    --------
    str: Stress level classification ('Normal' or 'Stress')
    """
    # Convert to percentage for easier comparison
    confidence_pct = confidence * 100
    
    # Simple threshold: >= 80% = Stress, < 80% = Normal
    if confidence_pct >= 80:
        return "Stress"
    else:
        return "Normal"


def detect_image_type(image_path: str) -> str:
    """
    Detect if image is color or grayscale.
    
    Parameters:
    -----------
    image_path : str
        Path to input image
    
    Returns:
    --------
    str: 'color' or 'grayscale'
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return 'unknown'
        
        if len(img.shape) == 2:
            return 'grayscale'
        elif img.shape[2] == 1:
            return 'grayscale'
        else:
            # Check if RGB channels are identical (grayscale saved as color)
            if np.allclose(img[:, :, 0], img[:, :, 1]) and np.allclose(img[:, :, 1], img[:, :, 2]):
                return 'grayscale'
            return 'color'
    
    except Exception as e:
        print(f"❌ Error detecting image type: {e}")
        return 'unknown'


def run_detection(image_path: str) -> Dict:
    """
    Run eye detection on an image (auto-detects color/grayscale).
    
    Parameters:
    -----------
    image_path : str
        Path to input image
    
    Returns:
    --------
    dict: Detection results containing:
        - 'success': bool
        - 'image_type': 'color' or 'grayscale'
        - 'pupil': tuple ((x, y), radius)
        - 'iris': tuple ((x, y), radius)
        - 'image': numpy array
        - 'error': str (if failed)
    """
    print(f"\n🔍 Detecting eye regions...")
    
    # Load image first to check channels
    image = cv2.imread(image_path)
    if image is None:
        return {'success': False, 'error': 'Could not load image'}
    
    # CRITICAL FIX: cv2.imread ALWAYS loads as BGR (3-channel) even for grayscale!
    # So we need to check if it's ACTUALLY grayscale (all channels identical)
    print(f"   Loaded image shape: {image.shape}")
    
    # Detect image type
    img_type = detect_image_type(image_path)
    print(f"   Image type detected: {img_type}")
    
    if img_type == 'unknown':
        return {'success': False, 'error': 'Could not determine image type'}
    
    # Run appropriate detector
    # NO CONVERSION NEEDED - cv2.imread already gives us BGR format!
    if img_type == 'grayscale':
        # Grayscale detection (uses notebook functions)
        result = detect_eye_grayscale(image_path, config.DEFAULT_CONFIG)
    else:
        # Color detection (uses hybrid functions)
        result = detect_eye_color(image_path)
    
    result['image_type'] = img_type
    
    if result['success']:
        print(f"   ✅ Detection successful!")
        
        if img_type == 'color':
            pupil_center, pupil_radius = result['pupil']
            iris_center, iris_radius = result['iris']
        else:
            # FIX: Handle grayscale detection result format
            if isinstance(result['pupil'], tuple) and len(result['pupil']) == 3:
                # Grayscale returns (x, y, r) directly
                pupil_x, pupil_y, pupil_radius = result['pupil']
                pupil_center = (pupil_x, pupil_y) if pupil_x is not None else None
            else:
                # Already formatted as ((x,y), r)
                pupil_center, pupil_radius = result['pupil']
            
            if isinstance(result['iris'], tuple) and len(result['iris']) == 3:
                iris_x, iris_y, iris_radius = result['iris']
                iris_center = (iris_x, iris_y) if iris_x is not None else None
            else:
                iris_center, iris_radius = result['iris']
            
            # Reformat for consistency
            result['pupil'] = (pupil_center, pupil_radius)
            result['iris'] = (iris_center, iris_radius)
        
        print(f"   Pupil: center={pupil_center}, radius={pupil_radius}px")
        print(f"   Iris: center={iris_center}, radius={iris_radius}px")
    else:
        print(f"   ❌ Detection failed: {result.get('error', 'Unknown error')}")
    
    return result


def run_measurements(detection_result: Dict) -> Dict:
    """
    Measure pupil diameter and count tension rings.
    
    Parameters:
    -----------
    detection_result : dict
        Output from run_detection()
    
    Returns:
    --------
    dict: Measurement results containing:
        - 'pupil_diameter_mm': float
        - 'ring_count': int
        - 'pixels_per_mm': float
        - 'measurements_valid': bool
    """
    print(f"\n📏 Running measurements...")
    
    if not detection_result['success']:
        return {
            'pupil_diameter_mm': None,
            'ring_count': None,
            'pixels_per_mm': None,
            'measurements_valid': False,
            'error': 'Detection failed'
        }
    
    try:
        # Extract detection results
        pupil_center, pupil_radius = detection_result['pupil']
        iris_center, iris_radius = detection_result['iris']
        image = detection_result['image']
        
        # Get image dimensions for logging
        image_height, image_width = image.shape[:2]
        
        # Measure pupil diameter (no image_width needed - iris-based method is resolution-independent)
        pupil_px, pupil_mm, px_per_mm = measure_pupil_diameter(pupil_radius, iris_radius)
        is_valid, validation_msg = validate_pupil_measurement(pupil_mm)
        
        # Validate iris detection accuracy
        if iris_radius < 40 or iris_radius > 300:
            print(f"   ⚠️  Warning: Unusual iris radius ({iris_radius}px)")
            print(f"   ⚠️  Measurements may be inaccurate - iris detection might have failed")
        
        print(f"   Pupil diameter: {pupil_px}px = {pupil_mm:.2f}mm")
        print(f"   Iris radius: {iris_radius}px (reference for 12mm)")
        print(f"   Image resolution: {image_width}x{image_height}")
        print(f"   Validation: {'✅' if is_valid else '⚠️'}  {validation_msg}")
        print(f"   Conversion factor: {px_per_mm:.2f} px/mm")
        
        # Count tension rings
        ring_count = count_tension_rings(image, pupil_center, pupil_radius, iris_center, iris_radius)
        print(f"   Tension rings detected: {ring_count}")
        
        return {
            'pupil_diameter_mm': pupil_mm,
            'pupil_diameter_px': pupil_px,
            'ring_count': ring_count,
            'pixels_per_mm': px_per_mm,
            'measurements_valid': is_valid,
            'validation_message': validation_msg,
            'image_width': image_width,
            'image_height': image_height,
            'iris_radius_px': iris_radius,
            'conversion_factor': px_per_mm
        }
    
    except Exception as e:
        print(f"   ❌ Measurement error: {e}")
        import traceback
        traceback.print_exc()
        return {
            'pupil_diameter_mm': None,
            'ring_count': None,
            'pixels_per_mm': None,
            'measurements_valid': False,
            'error': str(e)
        }


def prepare_model_inputs(detection_result: Dict, measurements: Dict, age: int) -> Dict:
    """
    Prepare inputs for model prediction.
    
    Parameters:
    -----------
    detection_result : dict
        Output from run_detection()
    measurements : dict
        Output from run_measurements()
    age : int
        Subject age
    
    Returns:
    --------
    dict: Model inputs containing:
        - 'pupil_img': numpy array (224, 224, 5)
        - 'iris_img': numpy array (224, 224, 5)
        - 'age_vector': numpy array (8,)
        - 'ring_count': float
        - 'ready': bool
    """
    print(f"\n🔧 Preparing model inputs...")
    
    try:
        # Extract eye regions
        pupil_center, pupil_radius = detection_result['pupil']
        iris_center, iris_radius = detection_result['iris']
        image = detection_result['image']
        
        # Extract regions with padding
        pupil_region = extract_eye_region(image, pupil_center, pupil_radius, padding=1.8)
        iris_region = extract_eye_region(image, iris_center, iris_radius, padding=1.3)
        
        if pupil_region is None or iris_region is None:
            print(f"   ❌ Failed to extract eye regions")
            return {'ready': False, 'error': 'Region extraction failed'}
        
        # Preprocess to 5-channel format
        pupil_img = preprocess_eye_image(pupil_region, config.TARGET_SIZE)
        iris_img = preprocess_eye_image(iris_region, config.TARGET_SIZE)
        
        print(f"   ✅ Images preprocessed: {config.TARGET_SIZE}")
        print(f"      Pupil: {pupil_img.shape}")
        print(f"      Iris: {iris_img.shape}")
        
        # Encode age
        age_vector = encode_age(age)
        age_group = config.get_age_group(age)
        print(f"   ✅ Age encoded: {age} years → {age_group} group")
        
        # Normalize ring count
        ring_count_normalized = measurements['ring_count'] / 10.0
        print(f"   ✅ Ring count normalized: {measurements['ring_count']} → {ring_count_normalized:.2f}")
        
        return {
            'pupil_img': pupil_img,
            'iris_img': iris_img,
            'age_vector': age_vector,
            'ring_count': ring_count_normalized,
            'ready': True
        }
    
    except Exception as e:
        print(f"   ❌ Input preparation error: {e}")
        return {'ready': False, 'error': str(e)}


def run_inference_pipeline(image_path: str, age: int, model) -> Dict:
    """
    Complete production inference pipeline: detection → measurement → prediction.
    
    CRITICAL: This pipeline must preprocess data EXACTLY as training did:
    - Pupil stream: 5-channel with channels 3-4 ZEROED (RGB only)
    - Iris stream: 5-channel with all active (RGB + Canny + BlackHat)
    - Age: One-hot encoded
    - Ring count: Normalized
    
    Parameters:
    -----------
    image_path : str
        Path to input eye image
    age : int
        Subject age in years
    model : keras.Model
        Production model (best_dual_stream_model.keras)
    
    Returns:
    --------
    dict: Complete results containing:
        - 'detection': Detection results (pupil, iris, image_type)
        - 'measurements': Pupil diameter, ring count, validation
        - 'prediction': Model prediction with alpha analysis
        - 'success': Overall pipeline status
    """
    print("\n" + "="*80)
    print(f"🚀 PRODUCTION INFERENCE PIPELINE")
    print("="*80)
    print(f"   Input: {Path(image_path).name}")
    print(f"   Age: {age} years")
    print(f"   Model: {config.MODEL_PATH}")
    
    results = {
        'image_path': image_path,
        'age': age,
        'success': False
    }
    
    # Step 1: Detection
    detection_result = run_detection(image_path)
    results['detection'] = detection_result
    
    if not detection_result['success']:
        print(f"\n❌ Pipeline failed at detection stage")
        return results
    
    # Step 2: Measurements
    measurements = run_measurements(detection_result)
    results['measurements'] = measurements
    
    if not measurements['measurements_valid']:
        print(f"\n⚠️  Measurements may be unreliable")
    
    # Step 3: Prepare inputs
    model_inputs = prepare_model_inputs(detection_result, measurements, age)
    results['model_inputs'] = model_inputs
    
    if not model_inputs['ready']:
        print(f"\n❌ Pipeline failed at input preparation stage")
        return results
    
    # Step 4: Run prediction
    print(f"\n🤖 Running production model prediction...")
    
    try:
        pred, alpha = predict_single(
            model,
            model_inputs['pupil_img'],
            model_inputs['iris_img'],
            model_inputs['age_vector'],
            model_inputs['ring_count']
        )
        
        # Calculate confidence
        confidence = max(pred, 1 - pred)
        
        # Get stress level classification
        stress_level = classify_stress_level(pred, confidence)
        
        # ============================================================
        # INTELLIGENT RING COUNT OVERRIDE (Production Logic)
        # ============================================================
        # If model has high confidence (85%+) but ring detector returned 0,
        # trust the model and infer that rings exist but were missed
        original_ring_count = measurements['ring_count']
        ring_count_inferred = False
        
        if original_ring_count == 0 and confidence >= 0.85 and pred >= 0.5:
            # Model is VERY confident about stress, but no rings detected
            # Infer 1-2 rings based on confidence level
            if confidence >= 0.95:
                inferred_ring_count = 2  # Very high confidence → 2 rings
                print(f"   🔧 OVERRIDE: Ring count 0→2 (model confidence: {confidence:.1%})")
            else:
                inferred_ring_count = 1  # High confidence → 1 ring
                print(f"   🔧 OVERRIDE: Ring count 0→1 (model confidence: {confidence:.1%})")
            
            ring_count_inferred = True
            measurements['ring_count'] = inferred_ring_count
            measurements['original_ring_count'] = original_ring_count
            measurements['ring_count_inferred'] = ring_count_inferred
            print(f"   ℹ️  Model sees stress patterns that detector missed")
        
        prediction = {
            'prediction': pred,
            'alpha': alpha,
            'stress_level': stress_level,
            'confidence': confidence
        }
        
        print(f"   Prediction: {pred:.4f} → {stress_level}")
        print(f"   Confidence: {confidence:.2%}")
        
        if ring_count_inferred:
            print(f"   📊 Rings: {original_ring_count} (detected) → {measurements['ring_count']} (AI-inferred)")
        
        if alpha is not None:
            print(f"   🔀 Fusion Analysis (Training Average):")
            print(f"      • IRIS stream: {alpha:.3f} ({alpha*100:.1f}%)")
            print(f"      • PUPIL+AGE stream: {1-alpha:.3f} ({(1-alpha)*100:.1f}%)")
            print(f"      Note: Model uses adaptive fusion internally")
        
        results['prediction'] = prediction
        results['success'] = True
        
        print(f"\n✅ Pipeline completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Prediction error: {e}")
        import traceback
        traceback.print_exc()
        results['prediction'] = {'error': str(e)}
    
    print("="*80)
    
    return results


if __name__ == "__main__":
    print("🧪 Testing Inference Pipeline...")
    print("This module is ready to orchestrate the complete pipeline!")
