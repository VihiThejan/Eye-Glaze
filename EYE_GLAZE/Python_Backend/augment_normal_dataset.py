import cv2
import numpy as np
from pathlib import Path
import shutil
import pickle

# Load the trained model
with open('ring_detection_model.pkl', 'rb') as f:
    clf = pickle.load(f)

def extract_features(img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        return None
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

# Analyze PARTIAL_STRESS images to find potential NORMAL candidates
test_dir = Path(r'c:\Users\pasan\Downloads\Final Project\Final Project\EYE_GLAZE\TestImg_Categorized')
partial_dir = test_dir / 'PARTIAL_STRESS'
normal_dir = test_dir / 'NORMAL'

print("="*70)
print("Analyzing PARTIAL_STRESS images to find NORMAL-like candidates...")
print("="*70)

candidates = []

for img_path in sorted(partial_dir.glob('*.jpg'))[:30]:  # Check first 30
    features = extract_features(img_path)
    if features is not None:
        probs = clf.predict_proba(features)[0]
        prob_normal = probs[0]
        prob_partial = probs[1]
        prob_stress = probs[2]
        
        # Find images with high NORMAL probability but labeled as PARTIAL
        if prob_normal > 0.40:  # High normal probability
            candidates.append({
                'path': img_path,
                'prob_normal': prob_normal,
                'prob_partial': prob_partial,
                'prob_stress': prob_stress
            })

# Sort by normal probability
candidates.sort(key=lambda x: x['prob_normal'], reverse=True)

print(f"\nFound {len(candidates)} potential NORMAL candidates from PARTIAL_STRESS:")
print("-"*70)

for i, c in enumerate(candidates[:15], 1):  # Show top 15
    print(f"{i:2d}. {c['path'].name:30s} | NORMAL: {c['prob_normal']:.1%} | PARTIAL: {c['prob_partial']:.1%} | STRESS: {c['prob_stress']:.1%}")

# Copy top 10 to NORMAL directory
if candidates:
    print(f"\n{'='*70}")
    print("Copying top 10 candidates to NORMAL directory...")
    print("="*70)
    
    for i, c in enumerate(candidates[:10], 1):
        src = c['path']
        dst = normal_dir / f"normal_from_partial_{i:02d}_{src.name}"
        shutil.copy(src, dst)
        print(f"✓ Copied: {dst.name}")
    
    print(f"\n✓ Added 10 new NORMAL images")
    print(f"  Total NORMAL images now: {len(list(normal_dir.glob('*.jpg')))}")
else:
    print("\n⚠ No suitable candidates found")
