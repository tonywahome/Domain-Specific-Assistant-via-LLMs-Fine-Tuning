# Kaggle-Specific Configuration
# Add this cell at the beginning of your notebook on Kaggle

import os

# ========================================
# 1. HUGGINGFACE AUTHENTICATION (REQUIRED)
# ========================================
from huggingface_hub import login

# OPTION A: Direct token (replace with yours)
HF_TOKEN = "hf_YOUR_TOKEN_HERE"  # Get from: https://huggingface.co/settings/tokens
login(token=HF_TOKEN)

# OPTION B: Use Kaggle Secrets (more secure)
# from kaggle_secrets import UserSecretsClient
# HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
# login(token=HF_TOKEN)

print("✓ HuggingFace authentication successful!")

# ========================================
# 2. KAGGLE PATHS (AUTO-DETECTION)
# ========================================

# Auto-detect Kaggle dataset path
if os.path.exists('/kaggle/input'):
    # List available datasets
    available_datasets = os.listdir('/kaggle/input')
    print(f"\n📂 Available datasets: {available_datasets}")
    
    # Try to find Financial QA dataset
    dataset_candidates = [d for d in available_datasets if 'financial' in d.lower() or 'qa' in d.lower() or '10k' in d.lower()]
    
    if dataset_candidates:
        dataset_folder = dataset_candidates[0]
        RAW_DATA_PATH = f"/kaggle/input/{dataset_folder}/Financial-QA-10k.csv"
        print(f"✓ Auto-detected dataset: {RAW_DATA_PATH}")
    else:
        # Default path - UPDATE THIS with your dataset name
        RAW_DATA_PATH = "/kaggle/input/financial-qa-10k-dataset/Financial-QA-10k.csv"
        print(f"⚠️ Using default path: {RAW_DATA_PATH}")
        print("   If dataset not found, update RAW_DATA_PATH with correct name from above list")
else:
    # Fallback for local/other
    RAW_DATA_PATH = "../dataset/Financial-QA-10k.csv"
    print(f"📊 Using local path: {RAW_DATA_PATH}")

# Verify file exists
if os.path.exists(RAW_DATA_PATH):
    file_size = os.path.getsize(RAW_DATA_PATH) / (1024 * 1024)  # MB
    print(f"✓ Dataset verified: {file_size:.1f} MB")
else:
    print(f"\n❌ ERROR: Dataset not found at {RAW_DATA_PATH}")
    print("\nTroubleshooting:")
    print("1. Click 'Add Data' in right sidebar")
    print("2. Search for your uploaded dataset")
    print("3. Click 'Add' to attach it to this notebook")
    print("4. Update RAW_DATA_PATH variable with correct path")
    if os.path.exists('/kaggle/input'):
        print(f"\n📂 Current datasets: {os.listdir('/kaggle/input')}")

# ========================================
# 3. KAGGLE GPU VERIFICATION
# ========================================

import torch

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"\n✓ GPU Detected: {gpu_name}")
    print(f"✓ VRAM: {gpu_memory:.1f} GB")
    
    # Check if it's a supported GPU
    if 'T4' in gpu_name or 'P100' in gpu_name:
        print(f"✓ {gpu_name} is perfect for this project!")
    elif 'V100' in gpu_name or 'A100' in gpu_name:
        print(f"✓ {gpu_name} - Excellent! Training will be faster.")
    else:
        print(f"⚠️ {gpu_name} - Should work but not tested extensively")
else:
    print("\n❌ ERROR: No GPU detected!")
    print("Solution: Click 'Accelerator' dropdown → Select 'GPU T4 x2'")
    raise RuntimeError("GPU required for training. Please enable GPU in notebook settings.")

# ========================================
# 4. INTERNET VERIFICATION
# ========================================

try:
    import urllib.request
    urllib.request.urlopen('https://huggingface.co', timeout=5)
    print("\n✓ Internet access: Enabled")
except:
    print("\n❌ ERROR: No internet access")
    print("Solution: Settings → Internet → Toggle ON")
    print("Note: Required to download Gemma model (~2GB)")

# ========================================
# 5. KAGGLE OUTPUT PATHS
# ========================================

# Create output directories
OUTPUT_DIRS = {
    'checkpoints': '/kaggle/working/models/checkpoints',
    'final': '/kaggle/working/models/final',
    'logs': '/kaggle/working/outputs/logs',
    'results': '/kaggle/working/outputs/results'
}

for name, path in OUTPUT_DIRS.items():
    os.makedirs(path, exist_ok=True)
    print(f"✓ Created: {path}")

# ========================================
# 6. MODEL CONFIGURATION
# ========================================

MODEL_NAME = "google/gemma-2b"
MAX_SAMPLES = 5000
MAX_SEQ_LENGTH = 2048
TRAIN_RATIO = 0.90
VAL_RATIO = 0.05
TEST_RATIO = 0.05

print("\n" + "="*60)
print("✅ KAGGLE CONFIGURATION COMPLETE")
print("="*60)
print(f"\nDataset: {RAW_DATA_PATH}")
print(f"Model: {MODEL_NAME}")
print(f"Samples: {MAX_SAMPLES}")
print(f"Max sequence length: {MAX_SEQ_LENGTH}")
print(f"\nReady to proceed with preprocessing!")
print("="*60)
