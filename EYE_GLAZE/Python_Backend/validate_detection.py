import cv2
import numpy as np
from pathlib import Path

test_dir = Path(r'c:\Users\pasan\Downloads\Final Project\Final Project\EYE_GLAZE\TestImg_Categorized')

for category in ['STRESS', 'NORMAL']:
    cat_path = test_dir / category
    images = list(cat_path.glob('*.jpg'))[:10]
    
    rings = []
    for img_path in images:
        img = cv2.imread(str(img_path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        blurred = cv2.GaussianBlur(gray, (7, 7), 1.5)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(blurred)
        
        circles = cv2.HoughCircles(enhanced, cv2.HOUGH_GRADIENT, dp=1.3, minDist=60,
                                   param1=110, param2=40, minRadius=22, maxRadius=min(h,w)//2)
        
        circle_count = 0
        if circles is not None:
            circles_arr = np.uint16(np.around(circles[0]))
            if len(circles_arr) > 1:
                cx = np.mean(circles_arr[:, 0])
                cy = np.mean(circles_arr[:, 1])
                concentric = [c for c in circles_arr if np.sqrt((c[0]-cx)**2 + (c[1]-cy)**2) < 35]
                circle_count = len(concentric)
            else:
                circle_count = len(circles_arr)
        
        cy, cx = h // 2, w // 2
        roi_size = min(h, w) // 3
        y1, y2 = max(0, cy - roi_size), min(h, cy + roi_size)
        x1, x2 = max(0, cx - roi_size), min(w, cx + roi_size)
        iris_roi = enhanced[y1:y2, x1:x2]
        
        radial_variance = 0
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
                radial_variance = np.var(intensities)
        
        if circle_count >= 6:
            ring_count = 3
        elif circle_count >= 5:
            ring_count = 2 if radial_variance > 25 else 3
        elif circle_count >= 4:
            ring_count = 2
        elif circle_count == 3:
            if radial_variance > 40:
                ring_count = 2
            elif radial_variance > 25:
                ring_count = 1
            else:
                ring_count = 0
        elif circle_count == 2:
            if radial_variance > 50:
                ring_count = 1
            else:
                ring_count = 0
        else:
            if radial_variance > 60:
                ring_count = 1
            else:
                ring_count = 0
        
        ring_count = min(ring_count, 3)
        rings.append(ring_count)
    
    avg = np.mean(rings)
    dist = {i: rings.count(i) for i in range(4)}
    print(f'{category}: Avg={avg:.1f} | Distribution: {" ".join([f"{k}={v}" for k,v in sorted(dist.items())])}')

print("\n" + "="*50)
print("SIMPLIFIED DETECTION - READY FOR TESTING")
print("="*50)
