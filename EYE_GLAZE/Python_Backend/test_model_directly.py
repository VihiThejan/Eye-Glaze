import pickle
import cv2
import numpy as np
from pathlib import Path

# Load the model
print("Loading model...")
with open('ring_detection_model.pkl', 'rb') as f:
    model = pickle.load(f)

print(f"Model loaded: {type(model).__name__}")
print(f"Expected features: {model.n_features_in_}")
print(f"Classes: {model.classes_}")

# Feature extraction function (same as app.py)
def extract_features(gray_image):
    """Extract features from grayscale eye image"""
    features = []
    
    # 1. Circle detection features
    blurred = cv2.GaussianBlur(gray_image, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=50,
        param1=100,
        param2=30,
        minRadius=20,
        maxRadius=150
    )
    num_circles = len(circles[0]) if circles is not None else 0
    features.append(num_circles)
    
    # 2. Radial pattern analysis
    center_y, center_x = gray_image.shape[0] // 2, gray_image.shape[1] // 2
    radius = min(center_x, center_y) - 10
    
    angles = np.linspace(0, 2 * np.pi, 360)
    radial_intensities = []
    for angle in angles:
        x = int(center_x + radius * np.cos(angle))
        y = int(center_y + radius * np.sin(angle))
        if 0 <= x < gray_image.shape[1] and 0 <= y < gray_image.shape[0]:
            radial_intensities.append(gray_image[y, x])
    
    radial_var = np.var(radial_intensities) if radial_intensities else 0
    radial_mean = np.mean(radial_intensities) if radial_intensities else 0
    radial_std = np.std(radial_intensities) if radial_intensities else 0
    
    features.extend([radial_var, radial_mean, radial_std])
    
    # 3. Edge detection
    edges = cv2.Canny(gray_image, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    features.append(edge_density)
    
    # 4. Texture analysis (using standard deviation of intensity)
    texture_var = np.var(gray_image)
    texture_mean = np.mean(gray_image)
    features.extend([texture_var, texture_mean])
    
    # 5. Gradient magnitude
    grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    grad_mean = np.mean(grad_mag)
    grad_std = np.std(grad_mag)
    features.extend([grad_mean, grad_std])
    
    # 6. Frequency domain features
    f_transform = np.fft.fft2(gray_image)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.abs(f_shift)
    freq_mean = np.mean(magnitude_spectrum)
    freq_std = np.std(magnitude_spectrum)
    features.extend([freq_mean, freq_std])
    
    return np.array(features).reshape(1, -1)

# Test paths
test_dir = Path(r'c:\Users\pasan\Downloads\Final Project\Final Project\EYE_GLAZE\TestImg_Categorized')

print("\n" + "="*80)
print("TESTING NORMAL IMAGES (Should predict class 0)")
print("="*80)

for img_path in sorted(test_dir.glob('NORMAL/*.JPG'))[:5]:
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    features = extract_features(gray)
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    
    print(f"\n{img_path.name:15s}")
    print(f"  Prediction: {prediction} (0=NORMAL, 1=PARTIAL, 2=STRESS)")
    print(f"  Probabilities: Normal={probabilities[0]:.2%}, Partial={probabilities[1]:.2%}, Stress={probabilities[2]:.2%}")
    print(f"  Features: {features[0][:5]}...")  # Show first 5 features

print("\n" + "="*80)
print("TESTING STRESS IMAGES (Should predict class 2)")
print("="*80)

for img_path in sorted(test_dir.glob('STRESS/*.jpg'))[:5]:
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    features = extract_features(gray)
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    
    print(f"\n{img_path.name:35s}")
    print(f"  Prediction: {prediction} (0=NORMAL, 1=PARTIAL, 2=STRESS)")
    print(f"  Probabilities: Normal={probabilities[0]:.2%}, Partial={probabilities[1]:.2%}, Stress={probabilities[2]:.2%}")
    print(f"  Features: {features[0][:5]}...")  # Show first 5 features

print("\n" + "="*80)
