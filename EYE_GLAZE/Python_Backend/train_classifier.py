import cv2
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle

def extract_features(img_path):
    """Extract comprehensive features from eye image"""
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Preprocessing
    blurred = cv2.GaussianBlur(gray, (7, 7), 1.5)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(blurred)
    
    features = []
    
    # 1. Circle detection
    circles = cv2.HoughCircles(enhanced, cv2.HOUGH_GRADIENT, dp=1.3, minDist=60,
                               param1=110, param2=40, minRadius=22, maxRadius=min(h,w)//2)
    circle_count = 0 if circles is None else len(circles[0])
    features.append(circle_count)
    
    # 2. Radial variance
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
    
    # 3. Edge features
    edges = cv2.Canny(iris_roi, 45, 135)
    edge_density = np.sum(edges > 0) / (iris_roi.shape[0] * iris_roi.shape[1])
    features.append(edge_density)
    
    # 4. Texture features
    texture_var = np.var(iris_roi)
    texture_mean = np.mean(iris_roi)
    features.extend([texture_var, texture_mean])
    
    # 5. Gradient magnitude
    sobelx = cv2.Sobel(iris_roi, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(iris_roi, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(sobelx**2 + sobely**2)
    grad_mean = np.mean(gradient_mag)
    grad_std = np.std(gradient_mag)
    features.extend([grad_mean, grad_std])
    
    # 6. Frequency analysis (FFT)
    f_transform = np.fft.fft2(iris_roi)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.abs(f_shift)
    freq_mean = np.mean(magnitude_spectrum)
    freq_std = np.std(magnitude_spectrum)
    features.extend([freq_mean, freq_std])
    
    return np.array(features)

# Load dataset
print("Loading dataset...")
test_dir = Path(r'c:\Users\pasan\Downloads\Final Project\Final Project\EYE_GLAZE\TestImg_Categorized')

X = []
y = []
filenames = []

# Map categories to ring counts
category_mapping = {
    'NORMAL': 0,
    'PARTIAL_STRESS': 1,
    'STRESS': 2
}

for category, label in category_mapping.items():
    cat_path = test_dir / category
    if not cat_path.exists():
        continue
    
    images = list(cat_path.glob('*.jpg'))
    print(f"Processing {category}: {len(images)} images")
    
    for img_path in images:
        features = extract_features(img_path)
        if features is not None:
            X.append(features)
            y.append(label)
            filenames.append(img_path.name)

X = np.array(X)
y = np.array(y)

print(f"\nDataset: {len(X)} images")
print(f"Feature shape: {X.shape}")
print(f"Class distribution: NORMAL={np.sum(y==0)}, PARTIAL={np.sum(y==1)}, STRESS={np.sum(y==2)}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train Random Forest
print("\nTraining Random Forest classifier...")
clf = RandomForestClassifier(
    n_estimators=200,  # More trees
    max_depth=15,      # Deeper trees
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42, 
    class_weight='balanced'
)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy:.2%}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['NORMAL', 'PARTIAL', 'STRESS']))

# Save model
model_path = 'ring_detection_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(clf, f)

print(f"\nModel saved to {model_path}")
print("\nFeature importance:")
feature_names = ['circles', 'radial_var', 'radial_mean', 'radial_std', 'edge_density', 
                 'texture_var', 'texture_mean', 'grad_mean', 'grad_std', 'freq_mean', 'freq_std']
importances = clf.feature_importances_
for name, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
    print(f"{name:15s}: {imp:.4f}")
