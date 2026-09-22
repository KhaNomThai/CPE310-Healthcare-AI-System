#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
config.py -- System Configuration & Clinical Mappings

Central configuration module containing all constants, paths, hyperparameters,
disease taxonomy, clinical names/descriptions, and utility functions used
across the entire pipeline.
"""

# =============================================================================
# System Setup
# =============================================================================

import sys
import random
import warnings
import tempfile
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

# Suppress non-critical warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# macOS SSL certificate bypass (required for downloading pretrained weights)
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Matplotlib: configure cache and use non-interactive backend (macOS compatibility)
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mpl_cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402, F401


# =============================================================================
# Paths
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "Dataset"

# Auto-detect whether ISIC 2019 images are in a nested or flat directory.
# Some archive extractions produce a double-nested folder; this handles both.
ISIC_2019_DIR = DATASET_DIR / "ISIC 2019"
_isic_input_dir = ISIC_2019_DIR / "ISIC_2019_Training_Input"
if (_isic_input_dir / "ISIC_2019_Training_Input").exists():
    ISIC_2019_IMAGE_DIR = _isic_input_dir / "ISIC_2019_Training_Input"
else:
    ISIC_2019_IMAGE_DIR = _isic_input_dir
ISIC_2019_GT_CSV = ISIC_2019_DIR / "ISIC_2019_Training_GroundTruth.csv"


# =============================================================================
# Training Hyperparameters
# =============================================================================

IMAGE_SIZE = 224                   # Input image resolution (pixels)
BATCH_SIZE = 32                    # Mini-batch size
NUM_EPOCHS = 15                    # Maximum training epochs
LEARNING_RATE = 1e-4               # Initial learning rate for AdamW
WEIGHT_DECAY = 1e-2                # L2 regularization coefficient
EARLY_STOPPING_PATIENCE = 3        # Epochs to wait before early stopping
NUM_CLASSES = 8                    # Total disease classes
NUM_WORKERS = 2                    # DataLoader workers (limited for macOS fork safety)
PIN_MEMORY = True                  # Pin memory for faster CPU-to-GPU transfer
RANDOM_SEED = 42                   # Reproducibility seed

# ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# =============================================================================
# 8-Class Disease Taxonomy
# =============================================================================

# Indices 0-3 = Cancer/Pre-Cancer, Indices 4-7 = Benign
CLASS_CODES: List[str] = ["MEL", "BCC", "SCC", "AK", "NV", "BKL", "VASC", "DF"]
CLASS_TO_IDX: Dict[str, int] = {code: idx for idx, code in enumerate(CLASS_CODES)}

# Column order in ISIC 2019 one-hot CSV (differs from CLASS_CODES order above)
ISIC_2019_LABEL_COLS = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]

# Binary group indices for cancer screening
CANCER_INDICES = {0, 1, 2, 3}   # MEL, BCC, SCC, AK
BENIGN_INDICES = {4, 5, 6, 7}   # NV, BKL, VASC, DF

# Clinical names (English)
CLASS_NAMES_EN: Dict[int, str] = {
    0: "Melanoma (MEL)",
    1: "Basal Cell Carcinoma (BCC)",
    2: "Squamous Cell Carcinoma (SCC)",
    3: "Actinic Keratoses / Bowen's Disease (AK)",
    4: "Melanocytic Nevi (NV)",
    5: "Benign Keratosis-like Lesions (BKL)",
    6: "Vascular Lesions (VASC)",
    7: "Dermatofibroma (DF)",
}

# Clinical names (Thai)
CLASS_NAMES_TH: Dict[int, str] = {
    0: "มะเร็งเมลาโนมา",
    1: "มะเร็งเซลล์ฐาน (Basal Cell Carcinoma)",
    2: "มะเร็งเซลล์สความัส (Squamous Cell Carcinoma)",
    3: "โรคผิวหนังก่อนมะเร็ง (Actinic Keratoses)",
    4: "ไฝและขี้แมลงวัน (Melanocytic Nevi)",
    5: "รอยโรคกลุ่มเคราโตซิส (Benign Keratosis)",
    6: "รอยโรคหลอดเลือด (Vascular Lesions)",
    7: "เนื้องอกใยผิวหนัง (Dermatofibroma)",
}

# Clinical descriptions (English)
CLASS_DESCRIPTIONS_EN: Dict[int, str] = {
    0: "Melanoma is the most dangerous form of skin cancer. It develops from melanocytes and can metastasize to other organs if not treated early.",
    1: "Basal Cell Carcinoma is the most common type of skin cancer. It rarely metastasizes but can cause significant local tissue damage if untreated.",
    2: "Squamous Cell Carcinoma is the second most common skin cancer. It arises from squamous cells and can metastasize if left untreated.",
    3: "Actinic Keratoses are pre-cancerous rough, scaly patches caused by prolonged UV exposure. They can progress to Squamous Cell Carcinoma.",
    4: "Melanocytic Nevi (moles) are common benign growths of melanocytes. They are usually harmless but should be monitored for changes.",
    5: "Benign Keratosis-like Lesions include seborrheic keratoses, solar lentigines, and lichen planus-like keratoses. They are non-cancerous.",
    6: "Vascular Lesions are benign conditions involving blood vessels, such as cherry angiomas, angiokeratomas, and pyogenic granulomas.",
    7: "Dermatofibroma is a common benign dermal nodule, usually firm and painless, typically found on the extremities.",
}

# Clinical descriptions (Thai)
CLASS_DESCRIPTIONS_TH: Dict[int, str] = {
    0: "เมลาโนมาเป็นมะเร็งผิวหนังที่อันตรายที่สุด เกิดจากเซลล์เมลาโนไซต์ สามารถแพร่กระจายไปยังอวัยวะอื่นได้หากไม่ได้รับการรักษา",
    1: "มะเร็งเซลล์ฐานเป็นมะเร็งผิวหนังที่พบบ่อยที่สุด มักไม่แพร่กระจาย แต่สามารถทำลายเนื้อเยื่อบริเวณใกล้เคียงได้",
    2: "มะเร็งเซลล์สความัสเป็นมะเร็งผิวหนังที่พบบ่อยเป็นอันดับสอง สามารถแพร่กระจายได้หากไม่ได้รับการรักษา",
    3: "โรคผิวหนังก่อนมะเร็ง เกิดจากการสัมผัสรังสี UV เป็นเวลานาน มีโอกาสพัฒนาเป็นมะเร็งเซลล์สความัส",
    4: "ไฝและขี้แมลงวันเป็นรอยโรคที่พบบ่อยและไม่เป็นอันตราย ควรสังเกตการเปลี่ยนแปลงอยู่เสมอ",
    5: "รอยโรคกลุ่มเคราโตซิสที่ไม่ร้ายแรง เช่น seborrheic keratosis และ solar lentigo",
    6: "รอยโรคหลอดเลือดเป็นภาวะที่ไม่ร้ายแรง เช่น cherry angioma และ pyogenic granuloma",
    7: "เนื้องอกใยผิวหนังเป็นก้อนเนื้อที่ไม่ร้ายแรง มักพบบริเวณแขนขา มีลักษณะแข็งและไม่เจ็บ",
}


# =============================================================================
# Model Registry (Multi-Model Comparison)
# =============================================================================

MODEL_REGISTRY: Dict[str, Dict] = {
    "ResNet50": {
        "checkpoint": BASE_DIR / "model_resnet50.pth",
        "metrics_file": BASE_DIR / "metrics_resnet50.json",
        "cm_file": BASE_DIR / "cm_resnet50.png",
    },
    "EfficientNet-B3": {
        "checkpoint": BASE_DIR / "model_efficientnet_b3.pth",
        "metrics_file": BASE_DIR / "metrics_efficientnet_b3.json",
        "cm_file": BASE_DIR / "cm_efficientnet_b3.png",
    },
    "DenseNet121": {
        "checkpoint": BASE_DIR / "model_densenet121.pth",
        "metrics_file": BASE_DIR / "metrics_densenet121.json",
        "cm_file": BASE_DIR / "cm_densenet121.png",
    },
}

# Ordered list of model names (for consistent iteration)
MODEL_NAMES: List[str] = list(MODEL_REGISTRY.keys())


# =============================================================================
# Utility Functions
# =============================================================================

def set_seed(seed: int = RANDOM_SEED) -> None:
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.manual_seed(seed)


def get_device() -> torch.device:
    """Detect and return the best available compute device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[INFO] Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[INFO] Using Apple Silicon MPS GPU Acceleration")
    else:
        device = torch.device("cpu")
        print("[INFO] Using CPU (no GPU acceleration detected)")
    return device


def print_banner() -> None:
    """Print system startup banner."""
    print("=" * 80)
    print("  Skin Cancer Multi-class Classification & Screening System")
    print("  ISIC 2019 Dataset | 8-Class Disease Classification")
    print("  Cancer Group: MEL, BCC, SCC, AK")
    print("  Benign Group: NV, BKL, VASC, DF")
    print("  Models: ResNet50 | EfficientNet-B3 | DenseNet121")
    print("=" * 80)
    print(f"[INFO] PyTorch version: {torch.__version__}")
    print(f"[INFO] Number of classes: {NUM_CLASSES}")
    print(f"[INFO] Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"[INFO] Batch size: {BATCH_SIZE}")
    print(f"[INFO] Max epochs: {NUM_EPOCHS}")
    print(f"[INFO] Learning rate: {LEARNING_RATE}")
    print(f"[INFO] Random seed: {RANDOM_SEED}")
    print()
