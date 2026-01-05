"""
Verify all required files exist before deployment
"""

import os
import sys

print("="*70)
print("VERIFYING FILES FOR DEPLOYMENT")
print("="*70)

required_files = {
    'models/mobilenet_feature_extractor.tflite': 'CNN Model',
    'models/violence_detection_lstm.tflite': 'LSTM Model',
    'models/gender_classification.tflite': 'Gender Model',
    'yolov8n.pt': 'YOLO Model',
    'requirements.txt': 'Dependencies',
    'packages.txt': 'System Packages',
    '.streamlit/config.toml': 'Streamlit Config',
    'streamlit_dashboard_enhanced.py': 'Main App'
}

all_good = True
missing_files = []

for filepath, description in required_files.items():
    if os.path.exists(filepath):
        size = os.path.getsize(filepath) / (1024 * 1024)  # MB
        print(f"✅ {description:20} - {filepath:50} ({size:.2f} MB)")
    else:
        print(f"❌ {description:20} - {filepath:50} MISSING!")
        all_good = False
        missing_files.append(filepath)

print("="*70)

if all_good:
    print("✅ ALL FILES PRESENT!")
    print("\nYou can now deploy:")
    print("  git add -f models/*.tflite yolov8n.pt")
    print("  git add requirements.txt .gitignore packages.txt")
    print("  git commit -m 'Add models and dependencies'")
    print("  git push origin main")
else:
    print("❌ MISSING FILES!")
    print("\nMissing files:")
    for f in missing_files:
        print(f"  - {f}")
    print("\nFix:")
    if any('models/' in f for f in missing_files):
        print("  - Copy models from backup folder:")
        print("    cp backup/*.tflite models/")
    if 'yolov8n.pt' in missing_files:
        print("  - Download YOLOv8:")
        print("    python -c \"from ultralytics import YOLO; YOLO('yolov8n.pt')\"")
    sys.exit(1)

print("="*70)
