import requests
import json
import os
from pathlib import Path

dataset_path = Path(r'c:\Users\pasan\Downloads\Final Project\Final Project\EYE_GLAZE\TestImg_Categorized')

def test_category(category, num_samples=3):
    folder = dataset_path / category
    images = list(folder.glob('*.jpg')) + list(folder.glob('*.JPG'))
    
    print(f"\n{'='*60}")
    print(f"Testing {category} Category ({len(images)} total images)")
    print(f"{'='*60}")
    
    correct = 0
    for i, img_path in enumerate(images[:num_samples]):
        with open(img_path, 'rb') as f:
            response = requests.post(
                'http://localhost:5000/predict',
                files={'image': f},
                data={'age': '30'}
            )
            result = response.json()
            
            predicted = result['prediction']['stress_level']
            rings = result['measurements']['ring_count']
            prob = result['prediction']['stress_probability']
            
            # Check if prediction matches expected category
            if category == 'NORMAL':
                expected = predicted == 'NORMAL'
            elif category == 'STRESS':
                expected = predicted in ['STRESS', 'PARTIAL_STRESS']
            else:  # PARTIAL_STRESS
                expected = predicted in ['STRESS', 'PARTIAL_STRESS']
            
            status = "✓" if expected else "✗"
            correct += 1 if expected else 0
            
            print(f"{status} {img_path.name:15s} → {predicted:15s} (Rings: {rings}, Prob: {prob:.0%})")
    
    accuracy = (correct / num_samples) * 100 if num_samples > 0 else 0
    print(f"\nAccuracy: {correct}/{num_samples} ({accuracy:.0f}%)")
    return correct, num_samples

# Test each category
total_correct = 0
total_tested = 0

for category in ['NORMAL', 'STRESS', 'PARTIAL_STRESS']:
    correct, tested = test_category(category, num_samples=5)
    total_correct += correct
    total_tested += tested

print(f"\n{'='*60}")
print(f"OVERALL ACCURACY: {total_correct}/{total_tested} ({(total_correct/total_tested)*100:.0f}%)")
print(f"{'='*60}\n")
