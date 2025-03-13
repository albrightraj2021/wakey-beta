"""
Setup script for the driver module
"""

import os
import sys
import subprocess
import importlib
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        "opencv-python",
        "numpy",
        "pillow",
        "requests",
        "pygame"
    ]
    
    optional_packages = [
        "dlib"
    ]
    
    missing = []
    for package in required_packages:
        try:
            importlib.import_module(package.replace("-", "_"))
            print(f"✓ {package} installed")
        except ImportError:
            print(f"✗ {package} not installed")
            missing.append(package)
    
    for package in optional_packages:
        try:
            importlib.import_module(package)
            print(f"✓ {package} installed (optional)")
        except ImportError:
            print(f"! {package} not installed (optional but recommended)")
    
    return missing

def create_directories():
    """Create necessary directories"""
    directories = [
        "models",
        "assets",
        "logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created {directory} directory")

def download_landmarks_file():
    """Download facial landmarks file"""
    landmarks_path = Path("shape_predictor_68_face_landmarks.dat")
    
    if landmarks_path.exists():
        print(f"✓ Facial landmarks file already exists")
        return
    
    print("Downloading facial landmarks file...")
    try:
        import requests
        import bz2
        
        # URL for the shape predictor
        url = "https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2"
        
        # Download the compressed file
        response = requests.get(url, stream=True)
        compressed_path = "shape_predictor_68_face_landmarks.dat.bz2"
        
        with open(compressed_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        
        # Decompress the file
        with open(landmarks_path, 'wb') as new_file, bz2.BZ2File(compressed_path, 'rb') as file:
            new_file.write(file.read())
        
        # Remove the compressed file
        os.remove(compressed_path)
        
        print(f"✓ Downloaded and decompressed facial landmarks file")
    except Exception as e:
        print(f"✗ Error downloading facial landmarks file: {e}")

def main():
    """Main setup function"""
    print("=== Driver Module Setup ===")
    
    # Check dependencies
    print("\nChecking dependencies...")
    missing_packages = check_dependencies()
    
    if missing_packages:
        print("\nMissing required packages. Install them with:")
        print(f"pip install {' '.join(missing_packages)}")
        print("\nSetup cannot continue without these packages.")
        sys.exit(1)
    
    # Create directories
    print("\nCreating directories...")
    create_directories()
    
    # Download landmarks file
    print("\nChecking facial landmarks file...")
    download_landmarks_file()
    
    print("\nSetup complete! You can now run the driver module with:")
    print("python driver_monitor.py --server http://your-server-url --driver-id YOUR_ID")

if __name__ == "__main__":
    main()
