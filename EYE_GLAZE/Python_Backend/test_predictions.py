import requests
import json

# Test NORMAL image
print("\n" + "="*50)
print("Testing NORMAL Image")
print("="*50)
normal_path = r'c:\Users\pasan\Downloads\Final Project\Final Project\EYE_GLAZE\TestImg_Categorized\NORMAL\1.JPG'
with open(normal_path, 'rb') as f:
    response = requests.post(
        'http://localhost:5000/predict',
        files={'image': f},
        data={'age': '30'}
    )
    result = response.json()
    print(json.dumps(result, indent=2))

# Test STRESS image
print("\n" + "="*50)
print("Testing STRESS Image")
print("="*50)
stress_path = r'c:\Users\pasan\Downloads\Final Project\Final Project\EYE_GLAZE\TestImg_Categorized\STRESS\1.jpg'
with open(stress_path, 'rb') as f:
    response = requests.post(
        'http://localhost:5000/predict',
        files={'image': f},
        data={'age': '30'}
    )
    result = response.json()
    print(json.dumps(result, indent=2))

print("\n" + "="*50)
