#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Skin Cancer Multi-class Classification & Screening System
(ระบบจำแนกประเภทโรคผิวหนังและคัดกรองมะเร็ง)

Clinical Support Tool — ISIC 2019 Dataset (Superset of HAM10000)
8-Class Disease Classification with Binary Cancer Screening

Cancer / Pre-Cancer Group:
  Class 0 — MEL  : Melanoma
  Class 1 — BCC  : Basal Cell Carcinoma
  Class 2 — SCC  : Squamous Cell Carcinoma
  Class 3 — AK   : Actinic Keratoses / Bowen's Disease (Pre-cancerous)

Benign Group:
  Class 4 — NV   : Melanocytic Nevi
  Class 5 — BKL  : Benign Keratosis-like Lesions
  Class 6 — VASC : Vascular Lesions
  Class 7 — DF   : Dermatofibroma

Platform : macOS (Apple Silicon / MPS GPU Acceleration Supported)
Tech     : PyTorch + Transfer Learning (ResNet50) + Gradio Clinical Demo
Course   : CPE310 — Healthcare AI System
================================================================================

Usage:
    python skin_lesion_classifier.py              # Train model + Evaluate
    python skin_lesion_classifier.py --demo       # Launch Gradio Clinical Demo
    python skin_lesion_classifier.py --skip-train # Skip training, evaluate & demo
"""

# ==============================================================================
# SECTION 0: Imports
# ==============================================================================

# Standard library
import sys
import argparse
import warnings
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Data processing
import numpy as np
import pandas as pd
from PIL import Image

# Deep learning (PyTorch)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Image augmentation and pretrained models
import torchvision.transforms as transforms
import torchvision.models as models

# Model evaluation metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
    precision_score,
    accuracy_score,
)
from sklearn.utils.class_weight import compute_class_weight

# Progress bar
from tqdm import tqdm

# Suppress non-critical warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# macOS SSL certificate bypass (required for downloading pretrained weights)
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Matplotlib: use non-interactive backend (required for macOS compatibility)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ==============================================================================
# SECTION 1: Configuration & Clinical Mappings
# ==============================================================================

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "Dataset"

# --- Dataset Paths ---
# Auto-detect whether ISIC 2019 images are in a nested or flat directory structure.
# Some archive extractions produce a double-nested folder; this handles both cases.
ISIC_2019_DIR = DATASET_DIR / "ISIC 2019"
_isic_input_dir = ISIC_2019_DIR / "ISIC_2019_Training_Input"
if (_isic_input_dir / "ISIC_2019_Training_Input").exists():
    ISIC_2019_IMAGE_DIR = _isic_input_dir / "ISIC_2019_Training_Input"
else:
    ISIC_2019_IMAGE_DIR = _isic_input_dir
ISIC_2019_GT_CSV = ISIC_2019_DIR / "ISIC_2019_Training_GroundTruth.csv"

# --- Output Paths ---

MODEL_SAVE_PATH = BASE_DIR / "best_multiclass_cancer_model.pth"
CONFUSION_MATRIX_PATH = BASE_DIR / "multiclass_confusion_matrix.png"

# --- Training Hyperparameters ---
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

# --- ImageNet Normalization ---
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# --- 8-Class Disease Taxonomy ---
# Indices 0-3 = Cancer/Pre-Cancer, Indices 4-7 = Benign
CLASS_CODES: List[str] = ["MEL", "BCC", "SCC", "AK", "NV", "BKL", "VASC", "DF"]
CLASS_TO_IDX: Dict[str, int] = {code: idx for idx, code in enumerate(CLASS_CODES)}

# Column order in ISIC 2019 one-hot CSV (differs from CLASS_CODES order above)
ISIC_2019_LABEL_COLS = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]

# Binary group indices for cancer screening
CANCER_INDICES = {0, 1, 2, 3}   # MEL, BCC, SCC, AK
BENIGN_INDICES = {4, 5, 6, 7}   # NV, BKL, VASC, DF

# --- Clinical Names ---
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


# ==============================================================================
# SECTION 2: System Initialization
# ==============================================================================

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
    print("=" * 80)
    print(f"[INFO] PyTorch version: {torch.__version__}")
    print(f"[INFO] Number of classes: {NUM_CLASSES}")
    print(f"[INFO] Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"[INFO] Batch size: {BATCH_SIZE}")
    print(f"[INFO] Max epochs: {NUM_EPOCHS}")
    print(f"[INFO] Learning rate: {LEARNING_RATE}")
    print(f"[INFO] Random seed: {RANDOM_SEED}")
    print()


# ==============================================================================
# SECTION 3: Data Preparation (ISIC 2019)
# ==============================================================================

def resolve_image_path(image_id: str, image_dir: Path) -> Optional[Path]:
    """
    Resolve image file path, handling _downsampled suffix.
    ISIC 2019 has some images with _downsampled.jpg suffix.
    Returns the path if found, None otherwise.
    """
    # Try standard filename first
    standard_path = image_dir / f"{image_id}.jpg"
    if standard_path.exists():
        return standard_path

    # Try _downsampled variant
    downsampled_path = image_dir / f"{image_id}_downsampled.jpg"
    if downsampled_path.exists():
        return downsampled_path

    return None


def load_and_prepare_data() -> pd.DataFrame:
    """
    Load ISIC 2019 ground truth CSV (one-hot encoded) and prepare
    a DataFrame with columns: [image_id, dx, label, image_path].
    """
    print("[INFO] Loading ISIC 2019 Training Ground Truth...")
    print(f"[INFO] CSV path: {ISIC_2019_GT_CSV}")
    print(f"[INFO] Image directory: {ISIC_2019_IMAGE_DIR}")

    if not ISIC_2019_GT_CSV.exists():
        print(f"[ERROR] Ground truth CSV not found: {ISIC_2019_GT_CSV}")
        sys.exit(1)

    if not ISIC_2019_IMAGE_DIR.exists():
        print(f"[ERROR] Image directory not found: {ISIC_2019_IMAGE_DIR}")
        sys.exit(1)

    # Read one-hot encoded ground truth
    gt_df = pd.read_csv(ISIC_2019_GT_CSV)
    print(f"[INFO] Loaded {len(gt_df)} entries from ground truth CSV")

    # Convert one-hot to single diagnosis label
    gt_df["dx"] = gt_df[ISIC_2019_LABEL_COLS].idxmax(axis=1)

    # Drop UNK rows (should be 0 in training set, but defensive)
    unk_mask = gt_df["dx"] == "UNK"
    if unk_mask.sum() > 0:
        print(f"[WARNING] Dropping {unk_mask.sum()} UNK rows")
        gt_df = gt_df[~unk_mask].reset_index(drop=True)

    # Map class codes to integer indices
    gt_df["label"] = gt_df["dx"].map(CLASS_TO_IDX)

    # Validate: all dx codes should be in our taxonomy
    unmapped = gt_df["label"].isna()
    if unmapped.sum() > 0:
        unknown_codes = gt_df.loc[unmapped, "dx"].unique()
        print(f"[WARNING] Unknown diagnosis codes: {unknown_codes}")
        gt_df = gt_df[~unmapped].reset_index(drop=True)

    gt_df["label"] = gt_df["label"].astype(int)

    # Resolve image paths (handle _downsampled suffix)
    print("[INFO] Resolving image file paths...")
    valid_rows = []
    missing_count = 0

    for _, row in tqdm(gt_df.iterrows(), total=len(gt_df), desc="Resolving images"):
        img_path = resolve_image_path(row["image"], ISIC_2019_IMAGE_DIR)
        if img_path is not None:
            valid_rows.append({
                "image_id": row["image"],
                "dx": row["dx"],
                "label": row["label"],
                "image_path": str(img_path),
            })
        else:
            missing_count += 1

    if missing_count > 0:
        print(f"[WARNING] {missing_count} images not found and excluded")

    df = pd.DataFrame(valid_rows)
    print(f"[INFO] Total valid samples: {len(df)}")

    # Print class distribution
    print()
    print("[INFO] Class Distribution:")
    print("-" * 60)
    for idx, code in enumerate(CLASS_CODES):
        count = (df["label"] == idx).sum()
        group = "Cancer/Pre-Cancer" if idx in CANCER_INDICES else "Benign"
        pct = 100.0 * count / len(df)
        print(f"  [{idx}] {code:5s} | {CLASS_NAMES_EN[idx]:45s} | {count:6d} ({pct:5.1f}%) | {group}")
    print("-" * 60)

    # Print group summary
    cancer_count = df["label"].isin(CANCER_INDICES).sum()
    benign_count = df["label"].isin(BENIGN_INDICES).sum()
    print(f"  Cancer / Pre-Cancer total: {cancer_count:6d} ({100.0 * cancer_count / len(df):.1f}%)")
    print(f"  Benign total:              {benign_count:6d} ({100.0 * benign_count / len(df):.1f}%)")
    print()

    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified split into train/val/test sets.
    Maintains class distribution across all splits.
    """
    print("[INFO] Splitting data (stratified by class label)...")

    # First split: train+val vs test
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, random_state=RANDOM_SEED, stratify=df["label"]
    )

    # Second split: train vs val
    relative_val_size = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        random_state=RANDOM_SEED,
        stratify=train_val_df["label"],
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print(f"[INFO] Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    return train_df, val_df, test_df


def compute_class_weights_tensor(labels: np.ndarray, device: torch.device) -> torch.Tensor:
    """Compute balanced class weights for handling class imbalance."""
    unique_classes = np.unique(labels)
    weights = compute_class_weight("balanced", classes=unique_classes, y=labels)
    weight_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    print(f"[INFO] Class weights computed for {len(unique_classes)} classes")
    for idx, code in enumerate(CLASS_CODES):
        if idx < len(weights):
            print(f"  [{idx}] {code:5s}: {weights[idx]:.4f}")
    return weight_tensor


# ==============================================================================
# SECTION 4: Dataset & DataLoaders
# ==============================================================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

val_test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


class SkinLesionDataset(Dataset):
    """PyTorch Dataset for skin lesion images from ISIC 2019."""

    def __init__(self, dataframe: pd.DataFrame, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.dataframe.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        label = int(row["label"])
        if self.transform:
            image = self.transform(image)
        return image, label


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoader instances for train, validation, and test sets."""
    train_dataset = SkinLesionDataset(train_df, transform=train_transform)
    val_dataset = SkinLesionDataset(val_df, transform=val_test_transform)
    test_dataset = SkinLesionDataset(test_df, transform=val_test_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    print(f"[INFO] DataLoaders created (num_workers={NUM_WORKERS})")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    print(f"  Test batches:  {len(test_loader)}")
    return train_loader, val_loader, test_loader


# ==============================================================================
# SECTION 5: Model Architecture
# ==============================================================================

def build_model(device: torch.device, num_classes: int = NUM_CLASSES) -> nn.Module:
    """
    Build ResNet50 model with custom classifier head for multi-class
    skin lesion classification.
    """
    print("[INFO] Building ResNet50 model with custom classifier head...")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    # Freeze all layers except layer4 and fc
    for name, param in model.named_parameters():
        if "layer4" not in name and "fc" not in name:
            param.requires_grad = False

    # Replace classifier head
    in_features = model.fc.in_features  # 2048
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(512),
        nn.Dropout(p=0.3),
        nn.Linear(512, num_classes),
    )

    model = model.to(device)

    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    print(f"[INFO] Total parameters:     {total_params:,}")
    print(f"[INFO] Trainable parameters: {trainable_params:,}")
    print(f"[INFO] Frozen parameters:    {frozen_params:,}")

    return model


# ==============================================================================
# SECTION 6: Training Loop
# ==============================================================================

class EarlyStopping:
    """Early stopping to terminate training when validation metric stops improving."""

    def __init__(self, patience: int = EARLY_STOPPING_PATIENCE, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score: Optional[float] = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True
        return False


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> Tuple[float, float]:
    """Train model for one epoch. Returns (avg_loss, macro_f1)."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(loader, desc=f"  Epoch {epoch:02d} [TRAIN]", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = running_loss / len(loader.dataset)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, macro_f1


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> Tuple[float, float]:
    """Validate model. Returns (avg_loss, macro_f1)."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"  Epoch {epoch:02d} [VAL]  ", leave=False)
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(loader.dataset)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, macro_f1


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    class_weights: torch.Tensor,
) -> nn.Module:
    """Complete training pipeline with early stopping and LR scheduling."""
    print()
    print("=" * 80)
    print("  MODEL TRAINING")
    print("=" * 80)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-7
    )
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)

    best_val_f1 = 0.0
    start_time = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()

        train_loss, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_loss, val_f1 = validate(model, val_loader, criterion, device, epoch)

        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start

        print(
            f"  Epoch {epoch:02d}/{NUM_EPOCHS:02d} | "
            f"Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f} | "
            f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s"
        )

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  [SAVED] New best model (Val F1: {val_f1:.4f})")

        scheduler.step(val_f1)

        if early_stopping(val_f1):
            print(f"  [EARLY STOP] No improvement for {EARLY_STOPPING_PATIENCE} epochs")
            break

    total_time = time.time() - start_time
    print(f"  Training completed in {total_time:.1f}s")
    print(f"  Best validation macro F1: {best_val_f1:.4f}")
    print(f"  Model saved to: {MODEL_SAVE_PATH}")

    # Load best model
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=True))
    return model


# ==============================================================================
# SECTION 7: Evaluation & Medical Safety Analysis
# ==============================================================================

def evaluate_on_test_set(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference on test set. Returns (all_labels, all_preds, all_probs)."""
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="  Evaluating on test set"):
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            all_labels.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def print_evaluation_report(
    labels: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
) -> None:
    """Print comprehensive evaluation report with per-class and binary grouped metrics."""
    print()
    print("=" * 80)
    print("  EVALUATION REPORT")
    print("=" * 80)

    # Per-class classification report
    target_names = [f"{CLASS_CODES[i]} ({CLASS_NAMES_EN[i]})" for i in range(NUM_CLASSES)]
    print()
    print("[INFO] Per-Class Classification Report:")
    print("-" * 80)
    report = classification_report(labels, preds, target_names=target_names, zero_division=0)
    print(report)

    # Overall metrics
    overall_acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)

    print(f"[INFO] Overall Accuracy:  {overall_acc:.4f} ({overall_acc * 100:.1f}%)")
    print(f"[INFO] Macro F1 Score:    {macro_f1:.4f}")
    print(f"[INFO] Weighted F1 Score: {weighted_f1:.4f}")

    # --- Binary Grouped Metrics (Cancer Screening) ---
    print()
    print("-" * 80)
    print("[INFO] Binary Cancer Screening Metrics (Grouped)")
    print("-" * 80)

    # Convert to binary: 1 = Cancer/Pre-Cancer (indices 0-3), 0 = Benign (indices 4-7)
    binary_labels = np.array([1 if lbl in CANCER_INDICES else 0 for lbl in labels])
    binary_preds = np.array([1 if pred in CANCER_INDICES else 0 for pred in preds])

    cancer_sensitivity = recall_score(binary_labels, binary_preds, pos_label=1, zero_division=0)
    cancer_specificity = recall_score(binary_labels, binary_preds, pos_label=0, zero_division=0)
    cancer_ppv = precision_score(binary_labels, binary_preds, pos_label=1, zero_division=0)
    cancer_npv = precision_score(binary_labels, binary_preds, pos_label=0, zero_division=0)
    binary_accuracy = accuracy_score(binary_labels, binary_preds)
    false_negative_rate = 1.0 - cancer_sensitivity

    total_cancer = binary_labels.sum()
    total_benign = len(binary_labels) - total_cancer

    print(f"  Cancer / Pre-Cancer samples: {total_cancer}")
    print(f"  Benign samples:              {total_benign}")
    print()
    print(f"  Binary Accuracy:       {binary_accuracy:.4f} ({binary_accuracy * 100:.1f}%)")
    print(f"  Cancer Sensitivity:    {cancer_sensitivity:.4f} ({cancer_sensitivity * 100:.1f}%)")
    print(f"  Cancer Specificity:    {cancer_specificity:.4f} ({cancer_specificity * 100:.1f}%)")
    print(f"  Positive Predictive Value (PPV): {cancer_ppv:.4f}")
    print(f"  Negative Predictive Value (NPV): {cancer_npv:.4f}")
    print(f"  False Negative Rate:   {false_negative_rate:.4f} ({false_negative_rate * 100:.1f}%)")

    # Safety assessment
    print()
    if cancer_sensitivity >= 0.90:
        print("[PASSED] Cancer sensitivity >= 90% target")
    else:
        print(f"[BELOW TARGET] Cancer sensitivity {cancer_sensitivity:.1%} is below 90% target")

    if false_negative_rate <= 0.10:
        print("[PASSED] False negative rate <= 10% target")
    else:
        print(f"[BELOW TARGET] False negative rate {false_negative_rate:.1%} exceeds 10% threshold")

    print()


def plot_confusion_matrix(labels: np.ndarray, preds: np.ndarray) -> None:
    """Generate and save 8-class confusion matrix plot."""
    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
    fig, ax = plt.subplots(figsize=(12, 10))

    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, shrink=0.8)

    tick_labels = CLASS_CODES
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=tick_labels,
        yticklabels=tick_labels,
        ylabel="True Label",
        xlabel="Predicted Label",
        title="Multi-class Confusion Matrix (8 Classes)",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=9,
            )

    # Draw group separator lines
    ax.axhline(y=3.5, color="red", linewidth=2, linestyle="--")
    ax.axvline(x=3.5, color="red", linewidth=2, linestyle="--")

    # Add group labels
    ax.text(-1.5, 1.5, "Cancer", ha="center", va="center", fontsize=10,
            fontweight="bold", color="red", rotation=90)
    ax.text(-1.5, 5.5, "Benign", ha="center", va="center", fontsize=10,
            fontweight="bold", color="green", rotation=90)

    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Confusion matrix saved to: {CONFUSION_MATRIX_PATH}")


# ==============================================================================
# SECTION 8: Gradio Web Demo (Two-Level Clinical Screening Interface)
# ==============================================================================

GRADIO_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Sarabun:wght@300;400;500;600;700&display=swap');

:root {
    --slate-900: #0f172a;
    --slate-800: #1e293b;
    --slate-700: #334155;
    --slate-600: #475569;
    --slate-500: #64748b;
    --slate-400: #94a3b8;
    --slate-300: #cbd5e1;
    --slate-200: #e2e8f0;
    --slate-100: #f1f5f9;
    --slate-50:  #f8fafc;
    --red-700:   #b91c1c;
    --red-600:   #dc2626;
    --red-100:   #fee2e2;
    --red-50:    #fef2f2;
    --green-700: #15803d;
    --green-600: #16a34a;
    --green-100: #dcfce7;
    --green-50:  #f0fdf4;
}

* {
    font-family: 'Inter', 'Sarabun', -apple-system, sans-serif !important;
}

body, .gradio-container {
    background-color: var(--slate-50) !important;
}

.gradio-container {
    max-width: 1020px !important;
    margin: 0 auto !important;
}

.system-header {
    background: var(--slate-900);
    color: #ffffff;
    padding: 20px 28px;
    border-radius: 8px;
    margin-bottom: 20px;
    border-bottom: 3px solid var(--slate-700);
}

.system-header h1 {
    font-size: 17px;
    font-weight: 600;
    margin: 0 0 4px 0;
    letter-spacing: 0.3px;
    color: #ffffff;
}

.system-header .subtitle {
    font-size: 12px;
    color: var(--slate-400);
    margin: 0;
    font-weight: 400;
}

.notice-bar {
    background: var(--slate-100);
    border: 1px solid var(--slate-200);
    border-left: 3px solid var(--slate-500);
    border-radius: 4px;
    padding: 10px 14px;
    margin-bottom: 16px;
    font-size: 11px;
    color: var(--slate-600);
    line-height: 1.6;
}

.notice-bar strong {
    color: var(--slate-700);
}

.info-panel {
    background: #ffffff;
    border: 1px solid var(--slate-200);
    border-radius: 6px;
    margin-top: 14px;
    overflow: hidden;
}

.info-panel .panel-header {
    background: var(--slate-100);
    padding: 8px 14px;
    font-size: 11px;
    font-weight: 600;
    color: var(--slate-700);
    text-transform: uppercase;
    letter-spacing: 0.4px;
    border-bottom: 1px solid var(--slate-200);
}

.info-panel .panel-body {
    padding: 10px 14px;
}

.info-panel .class-row {
    display: flex;
    justify-content: space-between;
    padding: 3px 0;
    font-size: 11px;
    color: var(--slate-600);
    border-bottom: 1px solid var(--slate-100);
}

.info-panel .class-row:last-child {
    border-bottom: none;
}

.info-panel .class-row .code {
    font-weight: 600;
    color: var(--slate-800);
    min-width: 42px;
}

.class-group-label {
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    padding: 6px 0 3px 0;
}

.class-group-label.cancer {
    color: var(--red-700);
    border-top: 1px solid var(--red-100);
    margin-top: 4px;
}

.class-group-label.benign {
    color: var(--green-700);
    border-top: 1px solid var(--green-100);
    margin-top: 4px;
}
"""


def launch_gradio_demo(model: nn.Module, device: torch.device) -> None:
    """Launch Gradio web interface for clinical skin lesion analysis."""
    try:
        import gradio as gr
    except ImportError:
        print("[ERROR] Gradio is not installed. Run: pip install gradio")
        return

    model.eval()

    def predict(input_image):
        """Process uploaded image and return two-level classification result."""
        if input_image is None:
            return _placeholder_html()

        try:
            if not isinstance(input_image, Image.Image):
                input_image = Image.fromarray(input_image)
            img = input_image.convert("RGB")
            img_tensor = val_test_transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(img_tensor)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

            cancer_prob = float(sum(probs[i] for i in CANCER_INDICES))
            benign_prob = float(sum(probs[i] for i in BENIGN_INDICES))
            is_high_risk = cancer_prob >= 0.5

            cancer_subtypes = [
                (CLASS_CODES[i], CLASS_NAMES_EN[i], CLASS_NAMES_TH[i], float(probs[i]))
                for i in sorted(CANCER_INDICES)
            ]
            benign_subtypes = [
                (CLASS_CODES[i], CLASS_NAMES_EN[i], CLASS_NAMES_TH[i], float(probs[i]))
                for i in sorted(BENIGN_INDICES)
            ]

            cancer_subtypes.sort(key=lambda x: x[3], reverse=True)
            benign_subtypes.sort(key=lambda x: x[3], reverse=True)

            top_idx = int(np.argmax(probs))
            top_prob = float(probs[top_idx])

            return _build_result_html(
                is_high_risk, cancer_prob, benign_prob,
                cancer_subtypes, benign_subtypes,
                top_idx, top_prob,
            )

        except Exception as e:
            return f"<div style='padding:16px;color:#b91c1c;font-size:13px;font-family:Inter,sans-serif;'>Analysis Error: {str(e)}</div>"

    def _placeholder_html() -> str:
        return (
            "<div style='display:flex;align-items:center;justify-content:center;"
            "height:300px;color:#94a3b8;font-size:13px;font-family:Inter,sans-serif;"
            "text-align:center;line-height:1.8;'>"
            "Upload a dermoscopic image<br>and press Analyze to begin.</div>"
        )

    def _build_result_html(
        is_high_risk: bool,
        cancer_prob: float,
        benign_prob: float,
        cancer_subtypes: list,
        benign_subtypes: list,
        top_idx: int,
        top_prob: float,
    ) -> str:

        F = "font-family:'Inter','Sarabun',sans-serif;"

        if is_high_risk:
            risk_label = "HIGH RISK"
            risk_bg = "#b91c1c"
            risk_border = "#991b1b"
            banner_bg = "#fef2f2"
            banner_border = "#fecaca"
        else:
            risk_label = "LOW RISK"
            risk_bg = "#15803d"
            risk_border = "#166534"
            banner_bg = "#f0fdf4"
            banner_border = "#bbf7d0"

        # ---- 1. SCREENING RESULT ----
        html = f"""<div style="{F}font-size:13px;color:#0f172a;">

        <div style="background:{banner_bg};border:1px solid {banner_border};border-radius:6px;
            padding:18px 22px;margin-bottom:16px;">
            <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:14px;">
                <span style="font-size:11px;font-weight:600;color:#475569;text-transform:uppercase;
                    letter-spacing:0.6px;">Screening Result</span>
                <span style="display:inline-block;background:{risk_bg};border:1px solid {risk_border};
                    color:#ffffff;padding:4px 14px;border-radius:3px;font-size:11px;
                    font-weight:600;letter-spacing:0.8px;">{risk_label}</span>
            </div>
            <div style="display:flex;gap:20px;">
                <div style="flex:1;">
                    <div style="display:flex;justify-content:space-between;margin-bottom:5px;">
                        <span style="font-size:11px;font-weight:500;color:#64748b;">Cancer / Pre-Cancer</span>
                        <span style="font-size:11px;font-weight:600;color:#b91c1c;">{cancer_prob*100:.1f}%</span>
                    </div>
                    <div style="height:5px;background:#e2e8f0;border-radius:2px;overflow:hidden;">
                        <div style="height:100%;width:{cancer_prob*100:.1f}%;background:#dc2626;
                            border-radius:2px;"></div>
                    </div>
                </div>
                <div style="flex:1;">
                    <div style="display:flex;justify-content:space-between;margin-bottom:5px;">
                        <span style="font-size:11px;font-weight:500;color:#64748b;">Benign</span>
                        <span style="font-size:11px;font-weight:600;color:#15803d;">{benign_prob*100:.1f}%</span>
                    </div>
                    <div style="height:5px;background:#e2e8f0;border-radius:2px;overflow:hidden;">
                        <div style="height:100%;width:{benign_prob*100:.1f}%;background:#16a34a;
                            border-radius:2px;"></div>
                    </div>
                </div>
            </div>
        </div>
        """

        # ---- 2. PRIMARY DIAGNOSIS ----
        is_cancer_class = top_idx in CANCER_INDICES
        diag_accent = "#b91c1c" if is_cancer_class else "#15803d"

        html += f"""
        <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:6px;
            padding:16px 20px;margin-bottom:16px;border-left:3px solid {diag_accent};">
            <div style="font-size:10px;font-weight:600;color:#94a3b8;text-transform:uppercase;
                letter-spacing:0.6px;margin-bottom:8px;">Primary Diagnosis</div>
            <div style="font-size:15px;font-weight:600;color:#0f172a;margin-bottom:2px;">
                {CLASS_NAMES_EN[top_idx]}</div>
            <div style="font-size:12px;color:#475569;margin-bottom:8px;">
                {CLASS_NAMES_TH[top_idx]}</div>
            <div style="display:inline-block;background:#f1f5f9;border:1px solid #e2e8f0;
                border-radius:3px;padding:2px 10px;font-size:11px;font-weight:600;
                color:#334155;margin-bottom:10px;">Confidence: {top_prob*100:.1f}%</div>
            <div style="font-size:11px;color:#64748b;line-height:1.6;margin-top:6px;">
                {CLASS_DESCRIPTIONS_EN[top_idx]}</div>
            <div style="font-size:11px;color:#64748b;line-height:1.6;margin-top:4px;">
                {CLASS_DESCRIPTIONS_TH[top_idx]}</div>
        </div>
        """

        # ---- 3. DIFFERENTIAL ANALYSIS TABLE ----
        html += """
        <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:6px;
            overflow:hidden;margin-bottom:16px;">
            <div style="background:#f1f5f9;padding:9px 18px;border-bottom:1px solid #e2e8f0;">
                <span style="font-size:11px;font-weight:600;color:#334155;text-transform:uppercase;
                    letter-spacing:0.5px;">Differential Probability Analysis</span>
            </div>
            <div style="padding:14px 18px;">
        """

        # Cancer subtypes table
        html += """<div style="font-size:10px;font-weight:600;color:#b91c1c;text-transform:uppercase;
            letter-spacing:0.5px;padding-bottom:6px;border-bottom:1px solid #fee2e2;
            margin-bottom:8px;">Cancer / Pre-Cancer</div>"""

        for code, name_en, name_th, prob in cancer_subtypes:
            bar_w = max(prob * 100, 0.3)
            html += f"""
            <div style="display:flex;align-items:center;gap:10px;padding:5px 0;
                border-bottom:1px solid #f8fafc;">
                <span style="font-size:11px;font-weight:600;color:#0f172a;
                    min-width:36px;">{code}</span>
                <span style="font-size:11px;color:#475569;flex:1;
                    white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{name_en}</span>
                <div style="width:100px;height:4px;background:#fee2e2;border-radius:2px;
                    overflow:hidden;flex-shrink:0;">
                    <div style="height:100%;width:{bar_w:.1f}%;background:#dc2626;
                        border-radius:2px;"></div>
                </div>
                <span style="font-size:11px;font-weight:600;color:#b91c1c;
                    min-width:44px;text-align:right;">{prob*100:.1f}%</span>
            </div>"""

        # Benign subtypes table
        html += """<div style="font-size:10px;font-weight:600;color:#15803d;text-transform:uppercase;
            letter-spacing:0.5px;padding-bottom:6px;border-bottom:1px solid #dcfce7;
            margin-top:14px;margin-bottom:8px;">Benign</div>"""

        for code, name_en, name_th, prob in benign_subtypes:
            bar_w = max(prob * 100, 0.3)
            html += f"""
            <div style="display:flex;align-items:center;gap:10px;padding:5px 0;
                border-bottom:1px solid #f8fafc;">
                <span style="font-size:11px;font-weight:600;color:#0f172a;
                    min-width:36px;">{code}</span>
                <span style="font-size:11px;color:#475569;flex:1;
                    white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{name_en}</span>
                <div style="width:100px;height:4px;background:#dcfce7;border-radius:2px;
                    overflow:hidden;flex-shrink:0;">
                    <div style="height:100%;width:{bar_w:.1f}%;background:#16a34a;
                        border-radius:2px;"></div>
                </div>
                <span style="font-size:11px;font-weight:600;color:#15803d;
                    min-width:44px;text-align:right;">{prob*100:.1f}%</span>
            </div>"""

        html += """
            </div>
        </div>
        """

        # ---- 4. CLINICAL RECOMMENDATION ----
        if is_high_risk:
            rec_border = "#fecaca"
            rec_bg = "#fef2f2"
            rec_accent = "#b91c1c"
            rec_en = (
                "Elevated probability for malignant or pre-cancerous lesion detected. "
                "Referral to a dermatologist for clinical evaluation and possible biopsy "
                "is strongly recommended."
            )
            rec_th = (
                "ตรวจพบความเสี่ยงสูงต่อรอยโรคมะเร็งหรือก่อนมะเร็ง "
                "แนะนำให้ส่งต่อแพทย์ผิวหนังเพื่อประเมินทางคลินิก "
                "และพิจารณาตรวจชิ้นเนื้อยืนยันผลทางพยาธิวิทยา"
            )
        else:
            rec_border = "#bbf7d0"
            rec_bg = "#f0fdf4"
            rec_accent = "#15803d"
            rec_en = (
                "Analysis suggests a benign lesion with low malignancy probability. "
                "Periodic self-monitoring is recommended. Consult a physician if changes "
                "in size, color, border, or symptoms are observed."
            )
            rec_th = (
                "ผลการวิเคราะห์บ่งชี้ว่าเป็นรอยโรคที่ไม่ร้ายแรง มีความเสี่ยงต่ำต่อมะเร็ง "
                "แนะนำให้ติดตามสังเกตอาการเป็นระยะ หากพบการเปลี่ยนแปลงของขนาด สี ขอบ "
                "หรือมีอาการผิดปกติ ควรปรึกษาแพทย์"
            )

        html += f"""
        <div style="background:{rec_bg};border:1px solid {rec_border};border-left:3px solid {rec_accent};
            border-radius:4px;padding:14px 18px;margin-bottom:14px;">
            <div style="font-size:10px;font-weight:600;color:{rec_accent};text-transform:uppercase;
                letter-spacing:0.5px;margin-bottom:6px;">Recommendation</div>
            <div style="font-size:11px;color:#334155;line-height:1.7;margin-bottom:4px;">{rec_en}</div>
            <div style="font-size:11px;color:#475569;line-height:1.7;">{rec_th}</div>
        </div>
        """

        # ---- 5. DISCLAIMER ----
        html += """
        <div style="padding:8px 0;">
            <div style="font-size:9px;color:#94a3b8;line-height:1.5;text-align:center;">
                This system is intended for educational and research purposes only (CPE310).
                Results are probabilistic estimates and do not constitute medical diagnosis.
                Consult qualified healthcare professionals for all clinical decisions.
            </div>
        </div>

        </div>"""

        return html

    # ---- BUILD GRADIO INTERFACE ----
    with gr.Blocks(css=GRADIO_CSS, title="Skin Cancer Screening System") as demo:

        gr.HTML("""
        <div class="system-header">
            <h1>Skin Cancer Classification & Screening System</h1>
            <p class="subtitle">ISIC 2019 Dataset  |  8-Class Multi-type Classification  |  Binary Cancer Screening</p>
        </div>
        """)

        gr.HTML("""
        <div class="notice-bar">
            <strong>Notice:</strong>
            This system is an AI-assisted clinical screening prototype developed for academic purposes
            (CPE310 Healthcare AI System). All results are probabilistic and must not replace
            professional dermatological evaluation and histopathological diagnosis.
        </div>
        """)

        with gr.Row(equal_height=False):
            with gr.Column(scale=2):
                input_image = gr.Image(
                    type="pil",
                    label="Dermoscopic Image",
                    height=340,
                )
                submit_btn = gr.Button(
                    "Analyze",
                    variant="primary",
                    size="lg",
                )

                gr.HTML("""
                <div class="info-panel">
                    <div class="panel-header">Detectable Conditions</div>
                    <div class="panel-body">
                        <div class="class-group-label cancer">Cancer / Pre-Cancer</div>
                        <div class="class-row"><span class="code">MEL</span><span>Melanoma</span></div>
                        <div class="class-row"><span class="code">BCC</span><span>Basal Cell Carcinoma</span></div>
                        <div class="class-row"><span class="code">SCC</span><span>Squamous Cell Carcinoma</span></div>
                        <div class="class-row"><span class="code">AK</span><span>Actinic Keratoses</span></div>
                        <div class="class-group-label benign">Benign</div>
                        <div class="class-row"><span class="code">NV</span><span>Melanocytic Nevi</span></div>
                        <div class="class-row"><span class="code">BKL</span><span>Benign Keratosis</span></div>
                        <div class="class-row"><span class="code">VASC</span><span>Vascular Lesions</span></div>
                        <div class="class-row"><span class="code">DF</span><span>Dermatofibroma</span></div>
                    </div>
                </div>
                """)

            with gr.Column(scale=3):
                result_output = gr.HTML(
                    value=(
                        "<div style='display:flex;align-items:center;justify-content:center;"
                        "height:300px;color:#94a3b8;font-size:13px;font-family:Inter,sans-serif;"
                        "text-align:center;line-height:1.8;'>"
                        "Upload a dermoscopic image<br>and press Analyze to begin.</div>"
                    ),
                    label="Analysis Result",
                )

        submit_btn.click(
            fn=predict,
            inputs=[input_image],
            outputs=[result_output],
        )

    print("[INFO] Launching Gradio Clinical Demo...")
    print("[INFO] Access the interface at: http://127.0.0.1:7860")
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)


# ==============================================================================
# SECTION 9: Main Pipeline
# ==============================================================================

def main():
    """Main entry point for the skin cancer classification pipeline."""
    parser = argparse.ArgumentParser(
        description="Skin Cancer Multi-class Classification & Screening System"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Launch Gradio demo only (skip training and evaluation)",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training, run evaluation on test set and launch demo",
    )
    args = parser.parse_args()

    set_seed()
    print_banner()
    device = get_device()

    # --- Demo-only mode ---
    if args.demo:
        print("[INFO] Demo mode: loading pre-trained model...")
        if not MODEL_SAVE_PATH.exists():
            print(f"[ERROR] Model checkpoint not found: {MODEL_SAVE_PATH}")
            print("[ERROR] Please train the model first: python skin_lesion_classifier.py")
            sys.exit(1)
        model = build_model(device)
        model.load_state_dict(
            torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=True)
        )
        model.eval()
        launch_gradio_demo(model, device)
        return

    # --- Load and prepare data ---
    df = load_and_prepare_data()
    train_df, val_df, test_df = split_data(df)

    # --- Create DataLoaders ---
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df
    )

    # --- Build model ---
    model = build_model(device)

    # --- Train or skip ---
    if not args.skip_train:
        class_weights = compute_class_weights_tensor(
            train_df["label"].values, device
        )
        model = train_model(model, train_loader, val_loader, device, class_weights)
    else:
        print("[INFO] Skipping training (--skip-train flag set)")
        if MODEL_SAVE_PATH.exists():
            print(f"[INFO] Loading pre-trained model from: {MODEL_SAVE_PATH}")
            model.load_state_dict(
                torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=True)
            )
        else:
            print(f"[WARNING] No pre-trained model found at: {MODEL_SAVE_PATH}")
            print("[WARNING] Evaluation will use randomly initialized weights")

    # --- Evaluate on test set ---
    print()
    print("=" * 80)
    print("  TEST SET EVALUATION")
    print("=" * 80)
    labels, preds, probs = evaluate_on_test_set(model, test_loader, device)
    print_evaluation_report(labels, preds, probs)
    plot_confusion_matrix(labels, preds)

    # --- Launch demo ---
    print()
    print("[INFO] Starting Gradio Clinical Demo...")
    launch_gradio_demo(model, device)


if __name__ == "__main__":
    main()
