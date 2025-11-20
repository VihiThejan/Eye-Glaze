import requests
from pathlib import Path
import json

# Test ML-based detection with detailed output
api_url = "http://localhost:5000/predict"
test_dir = Path(r'c:\Users\pasan\Downloads\Final Project\Final Project\EYE_GLAZE\TestImg_Categorized')

print("Detailed ML Detection Test\n" + "="*60)

for category in ['NORMAL', 'STRESS']:
    cat_path = test_dir / category
    images = list(cat_path.glob('*.jpg'))[:3]
    
    print(f"\n{category} Images:")
    print("-" * 60)
    
    for img_path in images:
        with open(img_path, 'rb') as f:
            files = {'image': f}
            data = {'age': 30}
            response = requests.post(api_url, files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                ring_count = result['iris_analysis']['tension_rings_count']
                stress_level = result['prediction']['stress_level']
                confidence = result['prediction']['stress_probability']
                
                print(f"  {img_path.name[:40]:40s} | Rings: {ring_count} | {stress_level:15s} | Conf: {confidence:.2%}")
            else:
                print(f"  {img_path.name[:40]:40s} | ERROR: {response.status_code}")

print("\n" + "="*60)
