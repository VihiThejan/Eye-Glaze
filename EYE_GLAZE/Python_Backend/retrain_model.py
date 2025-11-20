import cv2
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pickle

def extract_features(gray_image):
    """
    Extract features from grayscale eye image
    EXACT SAME FUNCTION AS IN app.py
    """
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
    
    return np.array(features)

# Load dataset
print("="*80)
print("RETRAINING MODEL WITH CORRECT FEATURE EXTRACTION")
print("="*80)

dataset_path = Path(r'c:\Users\pasan\Downloads\Final Project\Final Project\EYE_GLAZE\TestImg_Categorized')

X = []
y = []
labels = {'NORMAL': 0, 'PARTIAL_STRESS': 1, 'STRESS': 2}

for category_name, label in labels.items():
    category_path = dataset_path / category_name
    if not category_path.exists():
        print(f"Warning: {category_path} not found")
        continue
    
    image_files = list(category_path.glob('*.jpg')) + list(category_path.glob('*.JPG'))
    print(f"\nProcessing {category_name}: {len(image_files)} images")
    
    count = 0
    for img_path in image_files:
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = extract_features(gray)
        
        X.append(features)
        y.append(label)
        count += 1
        
        if count % 50 == 0:
            print(f"  Processed {count} images...")
    
    print(f"  Total processed: {count}")

X = np.array(X)
y = np.array(y)

print(f"\nTotal samples: {len(X)}")
print(f"Feature shape: {X.shape}")
print(f"Class distribution:")
for category_name, label in labels.items():
    count = np.sum(y == label)
    print(f"  {category_name}: {count} ({count/len(y)*100:.1f}%)")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.22, random_state=42, stratify=y)

print(f"\nTraining set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# Train Random Forest
print("\nTraining Random Forest Classifier...")
clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "="*80)
print("MODEL PERFORMANCE")
print("="*80)
print(f"\nOverall Accuracy: {accuracy:.2%}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['NORMAL', 'PARTIAL_STRESS', 'STRESS']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print("            Predicted:")
print("              NORMAL  PARTIAL  STRESS")
print(f"Actual NORMAL     {cm[0][0]:3d}     {cm[0][1]:3d}     {cm[0][2]:3d}")
print(f"       PARTIAL    {cm[1][0]:3d}     {cm[1][1]:3d}     {cm[1][2]:3d}")
print(f"       STRESS     {cm[2][0]:3d}     {cm[2][1]:3d}     {cm[2][2]:3d}")

# Feature importance
feature_names = ['circles', 'radial_var', 'radial_mean', 'radial_std', 'edge_density',
                'texture_var', 'texture_mean', 'grad_mean', 'grad_std', 'freq_mean', 'freq_std']
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]

print("\nFeature Importance:")
for i in range(len(feature_names)):
    print(f"  {i+1}. {feature_names[indices[i]]:15s}: {importances[indices[i]]:.4f}")

# Save model
model_path = 'ring_detection_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(clf, f)

print(f"\n✓ Model saved to {model_path}")
print("="*80)
