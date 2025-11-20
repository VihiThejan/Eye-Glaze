"""
Detection modules for eye region analysis.

Supports:
- Grayscale detection: For Pupil dataset images (grayscale_eye.py)
- Color detection: For Iris Normal/Stressed dataset images (color_eye.py)
- Ring counting: For iris tension pattern analysis (ring_counter.py)
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional

# Grayscale detection (for grayscale pupil images)
from .grayscale_eye import detect_pupil_robust, detect_iris_robust

# Color detection (for color iris images) - HYBRID approach
from .color_eye import detect_pupil_hybrid, detect_iris_hybrid

# Ring counting and iris unwrapping
from .ring_counter import (
    unwrap_iris_region,
    detect_tension_rings_radial_profile,
    count_tension_rings as _count_tension_rings_notebook
)


def count_tension_rings(image: np.ndarray,
                       pupil_center: Tuple[int, int],
                       pupil_radius: int,
                       iris_center: Tuple[int, int],  # For backwards compatibility (unused)
                       iris_radius: int) -> int:
    """
    Wrapper for count_tension_rings to match pipeline signature.
    Note: iris_center parameter is kept for backwards compatibility but not used.
    The notebook version uses only pupil_center as origin.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image (BGR or grayscale)
    pupil_center : tuple
        (x, y) coordinates of pupil center
    pupil_radius : int
        Pupil radius in pixels
    iris_center : tuple
        (x, y) coordinates of iris center (unused, for backwards compatibility)
    iris_radius : int
        Iris radius in pixels
    
    Returns:
    --------
    int: Number of detected tension rings
    """
    # Call the notebook version (which returns tuple)
    ring_count, _, _ = _count_tension_rings_notebook(image, pupil_center, pupil_radius, iris_radius)
    return ring_count


def detect_eye_color(image_path: str, brown_iris_mode: bool = False) -> Dict:
    """
    High-level wrapper for color eye detection.
    Uses hybrid detection approach combining Normal + Stressed notebook methods.
    
    Parameters:
    -----------
    image_path : str
        Path to color eye image
    brown_iris_mode : bool
        If True, uses brown iris detection mode (area filtering)
    
    Returns:
    --------
    dict: {
        'success': bool,
        'pupil': ((x, y), radius),
        'iris': ((x, y), radius),
        'image': numpy array,
        'error': str (if failed)
    }
    """
    try:
        # Load image
        import cv2
        image = cv2.imread(image_path)
        if image is None:
            return {
                'success': False,
                'error': f"Failed to load image: {image_path}"
            }
        
        # Detect pupil (returns tuple: (center, radius))
        pupil_center, pupil_radius = detect_pupil_hybrid(image_path, brown_iris_mode=brown_iris_mode)
        
        if pupil_center is None or pupil_radius is None:
            return {
                'success': False,
                'error': "Pupil detection failed"
            }
        
        # Detect iris (returns tuple: (center, radius))
        iris_center, iris_radius = detect_iris_hybrid(image, pupil_center, pupil_radius)
        
        if iris_center is None or iris_radius is None:
            return {
                'success': False,
                'error': "Iris detection failed"
            }
        
        return {
            'success': True,
            'pupil': (pupil_center, pupil_radius),
            'iris': (iris_center, iris_radius),
            'image': image
        }
    
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def detect_eye_grayscale(image_path: str, config: dict) -> Dict:
    """
    High-level wrapper for grayscale eye detection.
    Uses robust detection methods from Pupil dataset notebook.
    
    Parameters:
    -----------
    image_path : str
        Path to grayscale eye image
    config : dict
        Configuration parameters for detection
    
    Returns:
    --------
    dict: {
        'success': bool,
        'pupil': (x, y, radius),
        'iris': (x, y, radius),
        'image': numpy array,
        'error': str (if failed)
    }
    """
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return {
                'success': False,
                'error': f"Failed to load image: {image_path}"
            }
        
        # IMPORTANT: Grayscale detection functions expect BGR image
        # They handle the conversion to grayscale internally
        
        # Detect pupil (pass BGR image, function converts internally)
        pupil_x, pupil_y, pupil_radius = detect_pupil_robust(image, config)
        
        if pupil_x is None:
            return {
                'success': False,
                'error': "Pupil detection failed"
            }
        
        # Detect iris (pass BGR image, function converts internally)
        iris_x, iris_y, iris_radius = detect_iris_robust(image, config)
        
        if iris_x is None:
            return {
                'success': False,
                'error': "Iris detection failed"
            }
        
        return {
            'success': True,
            'pupil': (pupil_x, pupil_y, pupil_radius),
            'iris': (iris_x, iris_y, iris_radius),
            'image': image
        }
    
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


__all__ = [
    # Grayscale detection (Pupil dataset)
    'detect_pupil_robust',
    'detect_iris_robust',
    
    # Color detection (Iris dataset) - HYBRID
    'detect_pupil_hybrid',
    'detect_iris_hybrid',
    
    # Ring counting and unwrapping
    'unwrap_iris_region',
    'detect_tension_rings_radial_profile',
    'count_tension_rings',
    
    # High-level wrappers for pipeline
    'detect_eye_color',
    'detect_eye_grayscale'
]

