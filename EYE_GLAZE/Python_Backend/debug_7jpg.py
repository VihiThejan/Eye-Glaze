import cv2
import numpy as np
import pickle
from pathlib import Path

# Load model
with open('ring_detection_model.pkl', 'rb') as f:
    clf = pickle.load(f)

def extract_features(img_path):
    img = cv2.imread(str(img_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    blurred = cv2.GaussianBlur(gray, (7, 7), 1.5)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(blurred)
    
    features = []
    
    # Circle detection
    circles = cv2.HoughCircles(enhanced, cv2.HOUGH_GRADIENT, dp=1.3, minDist=60,
                               param1=110, param2=40, minRadius=22, maxRadius=min(h,w)//2)
    circle_count = 0 if circles is None else len(circles[0])
    features.append(circle_count)
    
    # Radial analysis
    cy, cx = h // 2, w // 2
    roi_size = min(h, w) // 3
    y1, y2 = max(0, cy - roi_size), min(h, cy + roi_size)
    x1, x2 = max(0, cx - roi_size), min(w, cx + roi_size)
    iris_roi = enhanced[y1:y2, x1:x2]
    
    if iris_roi.shape[0] > 50 and iris_roi.shape[1] > 50:
        rh, rw = iris_roi.shape
        center = (rw//2, rh//2)
        max_rad = min(rh, rw) // 2
        
        intensities = []
        for radius in range(15, max_rad, 4):
            mask = np.zeros_like(iris_roi, dtype=np.uint8)
            cv2.circle(mask, center, radius, 255, 2)
            mean_int = cv2.mean(iris_roi, mask=mask)[0]
            intensities.append(mean_int)
        
        if len(intensities) > 3:
            radial_var = np.var(intensities)
            radial_mean = np.mean(intensities)
            radial_std = np.std(intensities)
        else:
            radial_var = radial_mean = radial_std = 0
    else:
        radial_var = radial_mean = radial_std = 0
    
    features.extend([radial_var, radial_mean, radial_std])
    
    # Edge features
    edges = cv2.Canny(iris_roi, 45, 135)
    edge_density = np.sum(edges > 0) / (iris_roi.shape[0] * iris_roi.shape[1])
    features.append(edge_density)
    
    # Texture
    texture_var = np.var(iris_roi)
    texture_mean = np.mean(iris_roi)
    features.extend([texture_var, texture_mean])
    
    # Gradients
    sobelx = cv2.Sobel(iris_roi, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(iris_roi, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(sobelx**2 + sobely**2)
    grad_mean = np.mean(gradient_mag)
    grad_std = np.std(gradient_mag)
    features.extend([grad_mean, grad_std])
    
    # FFT
    f_transform = np.fft.fft2(iris_roi)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.abs(f_shift)
    freq_mean = np.mean(magnitude_spectrum)
    freq_std = np.std(magnitude_spectrum)
    features.extend([freq_mean, freq_std])
    
    return np.array(features).reshape(1, -1)

# Test problematic image
img_path = Path(r'c:\Users\pasan\Downloads\Final Project\Final Project\EYE_GLAZE\TestImg_Categorized\NORMAL\7.JPG')
features = extract_features(img_path)

prediction = clf.predict(features)[0]
probabilities = clf.predict_proba(features)[0]

print("="*60)
print(f"Image: {img_path.name}")
print(f"Raw Prediction: {prediction} (0=NORMAL, 1=PARTIAL, 2=STRESS)")
print(f"Probabilities:")
print(f"  NORMAL (0):         {probabilities[0]:.1%}")
print(f"  PARTIAL_STRESS (1): {probabilities[1]:.1%}")
print(f"  STRESS (2):         {probabilities[2]:.1%}")
print("="*60)
