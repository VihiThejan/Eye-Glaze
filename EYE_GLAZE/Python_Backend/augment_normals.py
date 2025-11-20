import cv2
import numpy as np
from pathlib import Path

# Augment existing NORMAL images with MORE variations
test_dir = Path(r'c:\Users\pasan\Downloads\Final Project\Final Project\EYE_GLAZE\TestImg_Categorized')
normal_dir = test_dir / 'NORMAL'

print("="*70)
print("Creating MORE augmented NORMAL images from existing 5 samples")
print("="*70)

# Get only original images (not augmented ones)
all_images = list(normal_dir.glob('*.jpg'))
original_images = [f for f in all_images if not f.name.startswith('aug_')]

print(f"\nOriginal NORMAL images: {len(original_images)}")
print(f"Current total images: {len(all_images)}")

count = 0
for img_path in original_images:
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    
    base_name = img_path.stem
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    
    # Create MORE augmented versions (24 per original image)
    augmentations = []
    
    # Rotation variations (8 variations)
    for angle in [3, -3, 7, -7, 10, -10, 12, -12]:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        augmentations.append((rotated, f'aug_rot{angle}_{base_name}.jpg'))
    
    # Brightness variations (4 variations)
    for alpha, beta in [(1.15, 10), (0.85, -10), (1.2, 15), (0.8, -15)]:
        bright = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        augmentations.append((bright, f'aug_bright{int(alpha*100)}_{base_name}.jpg'))
    
    # Contrast variations (3 variations)
    for alpha in [1.2, 1.3, 0.9]:
        contrast = cv2.convertScaleAbs(img, alpha=alpha, beta=0)
        augmentations.append((contrast, f'aug_cont{int(alpha*100)}_{base_name}.jpg'))
    
    # Blur variations (2 variations)
    for ksize in [(3, 3), (5, 5)]:
        blurred = cv2.GaussianBlur(img, ksize, 0.5)
        augmentations.append((blurred, f'aug_blur{ksize[0]}_{base_name}.jpg'))
    
    # Sharpening variations (2 variations)
    kernel1 = np.array([[-0.5, -0.5, -0.5], [-0.5, 5.0, -0.5], [-0.5, -0.5, -0.5]])
    kernel2 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    for i, kernel in enumerate([kernel1, kernel2], 1):
        sharpened = cv2.filter2D(img, -1, kernel)
        augmentations.append((sharpened, f'aug_sharp{i}_{base_name}.jpg'))
    
    # Flip
    flipped = cv2.flip(img, 1)
    augmentations.append((flipped, f'aug_flip_{base_name}.jpg'))
    
    # Noise addition (2 variations)
    for noise_level in [5, 10]:
        noise = np.random.normal(0, noise_level, img.shape).astype(np.uint8)
        noisy = cv2.add(img, noise)
        augmentations.append((noisy, f'aug_noise{noise_level}_{base_name}.jpg'))
    
    # Translation (2 variations)
    for tx, ty in [(5, 5), (-5, -5)]:
        M_trans = np.float32([[1, 0, tx], [0, 1, ty]])
        shifted = cv2.warpAffine(img, M_trans, (w, h), borderMode=cv2.BORDER_REPLICATE)
        augmentations.append((shifted, f'aug_shift{tx}{ty}_{base_name}.jpg'))
    
    # Save augmented images
    for aug_img, aug_name in augmentations:
        save_path = normal_dir / aug_name
        cv2.imwrite(str(save_path), aug_img)
        count += 1
        print(f"  ✓ Created: {aug_name}")

print(f"\n{'='*70}")
print(f"✓ Created {count} augmented NORMAL images")
print(f"  Total NORMAL images now: {len(list(normal_dir.glob('*.jpg')))}")
print("="*70)
