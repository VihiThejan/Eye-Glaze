import os
import sys
from pathlib import Path
import shutil

# Download and extract Kaggle dataset
dataset_url = "bimalawijekoon/iris-dataset-tension-rings-annotated"
download_dir = Path(r"c:\Users\pasan\Downloads\Final Project\Final Project\EYE_GLAZE\kaggle_dataset")
download_dir.mkdir(exist_ok=True)

print("="*70)
print("Downloading Kaggle Dataset...")
print("="*70)

# Set Kaggle credentials path
kaggle_dir = Path.home() / ".kaggle"
print(f"\nLooking for Kaggle credentials at: {kaggle_dir}")

if not kaggle_dir.exists():
    print("\n⚠ Kaggle credentials not found!")
    print("\nTo download from Kaggle, you need to:")
    print("1. Go to https://www.kaggle.com/settings")
    print("2. Create a new API token (downloads kaggle.json)")
    print("3. Place kaggle.json in: " + str(kaggle_dir))
    print("\nFor now, I'll use the existing dataset structure.")
    sys.exit(0)

os.chdir(download_dir)

# Download dataset
os.system(f'kaggle datasets download -d {dataset_url}')

# Find and extract the zip file
zip_files = list(download_dir.glob("*.zip"))
if zip_files:
    import zipfile
    print(f"\nExtracting {zip_files[0].name}...")
    with zipfile.ZipFile(zip_files[0], 'r') as zip_ref:
        zip_ref.extractall(download_dir)
    print("✓ Extraction complete")
    
    # List extracted contents
    print("\nExtracted files:")
    for item in download_dir.rglob("*"):
        if item.is_file():
            print(f"  {item.relative_to(download_dir)}")
else:
    print("No zip file found after download")
