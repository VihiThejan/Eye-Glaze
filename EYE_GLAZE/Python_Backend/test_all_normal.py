import requests
from pathlib import Path

api_url = "http://localhost:5000/predict"
test_dir = Path(r'c:\Users\pasan\Downloads\Final Project\Final Project\EYE_GLAZE\TestImg_Categorized')

print("Testing ALL NORMAL Images\n" + "="*70)

cat_path = test_dir / 'NORMAL'
images = sorted(cat_path.glob('*.jpg'))

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
            
            status = "✓ OK" if ring_count == 0 else "✗ MISCLASSIFIED"
            print(f"{status:16s} | {img_path.name:10s} | Rings: {ring_count} | {stress_level:15s} | {confidence:.1%}")
        else:
            print(f"ERROR          | {img_path.name:10s} | Response: {response.status_code}")

print("="*70)
