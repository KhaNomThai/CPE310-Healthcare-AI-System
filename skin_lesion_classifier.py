#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Skin Cancer Screening & Classification System (ระบบคัดกรองมะเร็งผิวหนัง)
Clinical Support Tool — HAM10000 Dataset
Target Category: Skin Malignancy & Pre-cancerous Lesions (Binary Classification)

Class 0: Non-Cancer / Benign (nv, bkl, vasc, df)
Class 1: Cancer / Pre-Cancer (mel: Melanoma, bcc: Basal Cell Carcinoma, akiec: Actinic Keratoses)

Platform : macOS (Apple Silicon / MPS GPU Acceleration Supported)
Tech     : PyTorch + Transfer Learning (ResNet50) + Gradio Clinical Demo
Course   : CPE310 — Healthcare AI System
================================================================================

Usage:
    python skin_lesion_classifier.py              # Train model + Evaluate on Test set
    python skin_lesion_classifier.py --demo       # Launch Gradio Clinical Demo
    python skin_lesion_classifier.py --skip-train # Skip training, evaluate & launch demo
"""

# ==============================================================================
# SECTION 0: นำเข้าไลบรารีทั้งหมด (Imports)
# ==============================================================================
import os
import sys
import argparse
import warnings
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
import torchvision.models as models

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

from tqdm import tqdm

# ปิด Warning ที่ไม่จำเป็น
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- แก้ปัญหา SSL Certificate บน macOS ---
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# ==============================================================================
# SECTION 1: ค่าคงที่และการตั้งค่า (Configuration & Clinical Mappings)
# ==============================================================================

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "Dataset"
IMAGE_DIRS = [
    DATASET_DIR / "HAM10000_images_part_1",
    DATASET_DIR / "HAM10000_images_part_2",
]
METADATA_CSV = DATASET_DIR / "HAM10000_metadata.csv"
MODEL_SAVE_PATH = BASE_DIR / "best_cancer_model.pth"
CONFUSION_MATRIX_PATH = BASE_DIR / "cancer_confusion_matrix.png"

# --- Hyperparameters ---
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
EARLY_STOPPING_PATIENCE = 3
NUM_CLASSES = 2  # Binary Classification: Non-Cancer (0) vs Cancer / Pre-Cancer (1)
NUM_WORKERS = 2  # เหมาะสมสำหรับ macOS (หลีกเลี่ยง fork crash)
PIN_MEMORY = True
RANDOM_SEED = 42

# --- ImageNet Normalization ---
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# --- Clinical Cancer Classification Mappings ---
# คลาสที่จัดเป็นกลุ่มมะเร็งและภาวะก่อนมะเร็ง (Malignant & Pre-cancerous)
CANCER_DX_CODES = {"mel", "bcc", "akiec"}
# คลาสที่จัดเป็นกลุ่มรอยโรคไม่ร้ายแรง (Benign / Non-Cancer)
BENIGN_DX_CODES = {"nv", "bkl", "vasc", "df"}

# Binary Class Labels
CLASS_NAMES_EN: Dict[int, str] = {
    0: "Non-Cancer (Benign)",
    1: "Cancer / Pre-Cancer (Malignant)",
}

CLASS_NAMES_TH: Dict[int, str] = {
    0: "ไม่พบความเสี่ยงมะเร็ง (รอยโรคชนิดไม่ร้ายแรง)",
    1: "พบความเสี่ยงมะเร็งผิวหนัง (Malignant / Pre-Cancer)",
}

# Sub-type descriptions for clinical reference
SUBTYPE_NAMES_EN: Dict[str, str] = {
    "mel": "Melanoma (Malignant)",
    "bcc": "Basal Cell Carcinoma (Malignant)",
    "akiec": "Actinic Keratoses / Bowen's Disease (Precancerous)",
    "nv": "Melanocytic Nevi (Benign)",
    "bkl": "Benign Keratosis-like Lesions (Benign)",
    "vasc": "Vascular Lesions (Benign)",
    "df": "Dermatofibroma (Benign)",
}

SUBTYPE_NAMES_TH: Dict[str, str] = {
    "mel": "มะเร็งผิวหนังเมลาโนมา",
    "bcc": "มะเร็งเซลล์ฐาน (Basal Cell Carcinoma)",
    "akiec": "ผิวหนังก่อนมะเร็ง / โรคโบเวน",
    "nv": "ไฝ / ขี้แมลงวัน",
    "bkl": "รอยโรคสะเก็ดไม่ร้ายแรง",
    "vasc": "รอยโรคหลอดเลือด",
    "df": "เนื้องอกเส้นใยผิวหนัง",
}


# ==============================================================================
# SECTION 2: ฟังก์ชันเริ่มต้นระบบ (System Initialization)
# ==============================================================================

def set_seed(seed: int = RANDOM_SEED) -> None:
    """ตั้งค่า Random Seed ทุกตัวให้คงที่เพื่อความ Reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """
    ตรวจสอบและเลือกอุปกรณ์ประมวลผลที่เหมาะสม
    ลำดับความสำคัญ: MPS (Apple Silicon GPU) → CUDA → CPU
    """
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        print("[INFO] Device selected: Apple Silicon GPU (Metal Performance Shaders)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[INFO] Device selected: NVIDIA GPU ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print("[INFO] Device selected: CPU")
    return device


def print_banner() -> None:
    """แสดง Banner ทางการตอนเริ่มระบบ"""
    banner = """
    ================================================================
    |          Skin Cancer Screening & Diagnostic System           |
    |          Clinical Decision Support Tool — HAM10000           |
    |          Target: Malignancy & Pre-cancer Detection           |
    |          ResNet50 Transfer Learning Architecture             |
    ================================================================
    """
    print(banner)


# ==============================================================================
# SECTION 3: Data Preparation & Preprocessing (Step 2.1)
# ==============================================================================

def load_and_prepare_data() -> pd.DataFrame:
    """
    อ่าน Metadata CSV, ค้นหาและจับคู่ Path รูปภาพ
    แปลงรหัสโรค 7 ชนิดเป็น Binary Class:
      - Class 1 (Cancer / Pre-Cancer): mel, bcc, akiec
      - Class 0 (Non-Cancer / Benign): nv, bkl, vasc, df
    """
    print("\n" + "=" * 60)
    print("Step 2.1: Data Preparation & Cancer-Focused Preprocessing")
    print("=" * 60)

    # --- 1. อ่าน Metadata ---
    if not METADATA_CSV.exists():
        raise FileNotFoundError(f"[ERROR] Metadata file not found: {METADATA_CSV}")

    df = pd.read_csv(METADATA_CSV)
    print(f"[INFO] Metadata loaded successfully: {len(df)} records")
    print(f"   Columns: {list(df.columns)}")

    # --- 2. สแกนหาไฟล์รูปภาพ ---
    image_id_to_path: Dict[str, str] = {}
    for img_dir in IMAGE_DIRS:
        if not img_dir.exists():
            print(f"[WARNING] Image directory not found: {img_dir}")
            continue
        for img_file in img_dir.iterdir():
            if img_file.suffix.lower() in (".jpg", ".jpeg", ".png"):
                image_id_to_path[img_file.stem] = str(img_file)

    print(f"[INFO] Image directory scan completed: Found {len(image_id_to_path)} files")

    # --- 3. แมป image_path เข้า DataFrame ---
    df["image_path"] = df["image_id"].map(image_id_to_path)

    missing_count = df["image_path"].isna().sum()
    if missing_count > 0:
        print(f"[WARNING] Missing image files for {missing_count} records — Dropped")
        df = df.dropna(subset=["image_path"]).reset_index(drop=True)

    # --- 4. แปลง Label เป็น Binary (Cancer vs Non-Cancer) ---
    df["subtype"] = df["dx"]
    df["label"] = df["dx"].apply(lambda x: 1 if x in CANCER_DX_CODES else 0)

    # --- 5. แสดงสถิติการกระจายตัวของข้อมูล ---
    total_samples = len(df)
    cancer_df = df[df["label"] == 1]
    benign_df = df[df["label"] == 0]

    print("\n[INFO] Binary Distribution (Screening Target):")
    print("-" * 60)
    cancer_pct = len(cancer_df) / total_samples * 100
    benign_pct = len(benign_df) / total_samples * 100
    print(f"  [0] Non-Cancer (Benign)        : {len(benign_df):>5d} ({benign_pct:5.1f}%)")
    print(f"  [1] Cancer / Pre-Cancer (Target): {len(cancer_df):>5d} ({cancer_pct:5.1f}%)")
    print(f"      Total Samples              : {total_samples:>5d}")

    print("\n[INFO] Sub-type Breakdown within Each Category:")
    print("-" * 60)
    print("  --- Category 1: Cancer & Pre-cancerous ---")
    for dx_code in sorted(CANCER_DX_CODES):
        count = (cancer_df["subtype"] == dx_code).sum()
        pct = count / total_samples * 100
        print(f"      • {dx_code.upper():<6s} | {count:>5d} ({pct:4.1f}%) | {SUBTYPE_NAMES_EN[dx_code]}")

    print("  --- Category 0: Non-Cancer (Benign) ---")
    for dx_code in sorted(BENIGN_DX_CODES):
        count = (benign_df["subtype"] == dx_code).sum()
        pct = count / total_samples * 100
        print(f"      • {dx_code.upper():<6s} | {count:>5d} ({pct:4.1f}%) | {SUBTYPE_NAMES_EN[dx_code]}")

    return df


def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    แบ่งข้อมูลแบบ Stratified Split บนคอลัมน์ subtype:
    Train 80% / Val 10% / Test 10% เพื่อกระจายชนิดย่อยให้เท่าเทียมกันทุกชุด
    """
    train_df, temp_df = train_test_split(
        df,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=df["subtype"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=temp_df["subtype"],
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print(f"\n[INFO] Data split completed (Stratified Split by Disease Subtype):")
    print(f"   Train : {len(train_df):>5d} samples ({len(train_df)/len(df)*100:.1f}%) [Cancer: {(train_df['label']==1).sum()}]")
    print(f"   Val   : {len(val_df):>5d} samples ({len(val_df)/len(df)*100:.1f}%) [Cancer: {(val_df['label']==1).sum()}]")
    print(f"   Test  : {len(test_df):>5d} samples ({len(test_df)/len(df)*100:.1f}%) [Cancer: {(test_df['label']==1).sum()}]")
    return train_df, val_df, test_df


def compute_class_weights_tensor(train_df: pd.DataFrame, device: torch.device) -> torch.Tensor:
    """
    คำนวณ Class Weights สำหรับ 2 คลาส เพื่อแก้ปัญหาความไม่สมดุลของข้อมูล
    โดยให้น้ำหนักกับคลาสมะเร็ง (Class 1) มากขึ้นเพื่อเพิ่ม Sensitivity
    """
    labels = train_df["label"].values
    class_labels = np.array([0, 1])
    weights = compute_class_weight(
        class_weight="balanced",
        classes=class_labels,
        y=labels,
    )
    weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

    print(f"\n[INFO] Computed Class Weights (Binary Imbalance Compensation):")
    print(f"   [0] Non-Cancer (Benign)        : {weights[0]:.4f}")
    print(f"   [1] Cancer / Pre-Cancer (Target): {weights[1]:.4f}")
    return weights_tensor


# ==============================================================================
# SECTION 4: PyTorch Dataset & Data Augmentation (Step 2.2)
# ==============================================================================

def get_train_transforms() -> transforms.Compose:
    """Data Augmentation สำหรับ Training Set"""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=180),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transforms() -> transforms.Compose:
    """Transform สำหรับ Validation และ Test Set"""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class SkinLesionDataset(Dataset):
    """Custom Dataset สำหรับโหลดภาพรอยโรคผิวหนัง"""

    def __init__(self, dataframe: pd.DataFrame, transform: Optional[transforms.Compose] = None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.dataframe.iloc[idx]
        image_path = row["image_path"]
        label = int(row["label"])
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"[WARNING] Could not load image: {image_path} — {e}")
            image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0))

        if self.transform:
            image = self.transform(image)
        return image, label


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """สร้าง DataLoaders สำหรับ Train, Val, Test"""
    print("\n" + "=" * 60)
    print("Step 2.2: DataLoader Creation & Data Augmentation")
    print("=" * 60)

    train_dataset = SkinLesionDataset(train_df, transform=get_train_transforms())
    val_dataset = SkinLesionDataset(val_df, transform=get_eval_transforms())
    test_dataset = SkinLesionDataset(test_df, transform=get_eval_transforms())

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=True,
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

    print(f"[INFO] DataLoaders initialized:")
    print(f"   Train: {len(train_dataset)} samples → {len(train_loader)} batches")
    print(f"   Val  : {len(val_dataset)} samples → {len(val_loader)} batches")
    print(f"   Test : {len(test_dataset)} samples → {len(test_loader)} batches")
    print(f"   Batch Size: {BATCH_SIZE} | Workers: {NUM_WORKERS} | Pin Memory: {PIN_MEMORY}")
    return train_loader, val_loader, test_loader


# ==============================================================================
# SECTION 5: Model Architecture (Step 2.3)
# ==============================================================================

def build_model(device: torch.device) -> nn.Module:
    """
    สร้างโมเดล ResNet50 Transfer Learning สำหรับ Binary Classification (2 classes)
    """
    print("\n" + "=" * 60)
    print("Step 2.3: Model Architecture — ResNet50 Transfer Learning (Binary)")
    print("=" * 60)

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    # Freeze earlier layers, fine-tune layer4 & classifier head
    for name, param in model.named_parameters():
        if "layer4" not in name and "fc" not in name:
            param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(512),
        nn.Dropout(p=0.3),
        nn.Linear(512, NUM_CLASSES),  # Output 2 classes (Non-cancer vs Cancer)
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"[INFO] ResNet50 model built successfully")
    print(f"   Total Parameters     : {total_params:>12,d}")
    print(f"   Trainable Parameters : {trainable_params:>12,d}")
    print(f"   Frozen Parameters    : {total_params - trainable_params:>12,d}")
    print(f"   Output Classes       : {NUM_CLASSES} (0: Benign, 1: Cancer/Pre-cancer)")
    print(f"   Device               : {device}")
    return model


# ==============================================================================
# SECTION 6: Training & Validation Loop (Step 2.3 Continued)
# ==============================================================================

class EarlyStopping:
    """ระบบหยุดเทรนอัตโนมัติเพื่อป้องกัน Overfitting"""

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
            return False
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
    """เทรนโมเดล 1 Epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(loader, desc=f"  [Train] Epoch {epoch+1:>2d}/{NUM_EPOCHS}", leave=True, ncols=100)
    for images, labels in progress_bar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        progress_bar.set_postfix({
            "Loss": f"{loss.item():.4f}",
            "Acc": f"{correct / total * 100:.1f}%",
        })

    return running_loss / total, correct / total


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> Tuple[float, float, float, float, np.ndarray, np.ndarray]:
    """Validate โมเดลบน Validation Set"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(loader, desc=f"  [Val]   Epoch {epoch+1:>2d}/{NUM_EPOCHS}", leave=True, ncols=100)
    for images, labels in progress_bar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        progress_bar.set_postfix({
            "Loss": f"{loss.item():.4f}",
            "Acc": f"{correct / total * 100:.1f}%",
        })

    avg_loss = running_loss / total
    accuracy = correct / total

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    cancer_recall = recall_score(all_labels, all_preds, pos_label=1, zero_division=0)

    return avg_loss, accuracy, f1, cancer_recall, all_preds, all_labels


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    device: torch.device,
) -> Dict[str, List[float]]:
    """Training Loop หลัก"""
    print("\n" + "=" * 60)
    print("Initiating Cancer-Focused Screening Model Training")
    print("=" * 60)
    print(f"   Epochs       : {NUM_EPOCHS}")
    print(f"   Batch Size   : {BATCH_SIZE}")
    print(f"   Learning Rate: {LEARNING_RATE}")
    print(f"   Optimizer    : AdamW (weight_decay={WEIGHT_DECAY})")
    print(f"   Early Stop   : Patience = {EARLY_STOPPING_PATIENCE}")
    print()

    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)
    best_f1 = 0.0
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [], "val_f1": [], "val_cancer_recall": [],
        "lr": [],
    }
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc, val_f1, val_cancer_recall, _, _ = validate(model, val_loader, criterion, device, epoch)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_f1)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["val_cancer_recall"].append(val_cancer_recall)
        history["lr"].append(current_lr)

        epoch_time = time.time() - epoch_start
        print(f"\n  [Epoch {epoch+1:>2d}/{NUM_EPOCHS} Summary] ({epoch_time:.0f}s):")
        print(f"     Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"     Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc*100:.2f}%")
        print(f"     Val F1 (Weighted): {val_f1:.4f} | Cancer Sensitivity (Recall): {val_cancer_recall*100:.2f}% | LR: {current_lr:.2e}")

        # Save best model checkpoint
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": val_f1,
                "val_acc": val_acc,
                "val_loss": val_loss,
                "val_cancer_recall": val_cancer_recall,
                "num_classes": NUM_CLASSES,
                "class_mapping": CLASS_NAMES_EN,
            }, str(MODEL_SAVE_PATH))
            print(f"     [INFO] Saved new best model checkpoint (Val F1: {val_f1:.4f})")

        if early_stopping(val_f1):
            print(f"\n  [INFO] Early Stopping triggered (no improvement for {EARLY_STOPPING_PATIENCE} consecutive epochs)")
            break

        print()

    total_time = time.time() - start_time
    print(f"\n[INFO] Training completed in {total_time/60:.1f} minutes")
    print(f"   Best Validation F1-Score: {best_f1:.4f}")
    print(f"   Model saved to: {MODEL_SAVE_PATH}")
    return history


# ==============================================================================
# SECTION 7: Evaluation & Clinical Safety Metrics (Step 2.4)
# ==============================================================================

@torch.no_grad()
def evaluate_on_test_set(model: nn.Module, test_loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ทดสอบบน Test Set และคืนค่า (y_pred, y_true, y_probs)"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    progress_bar = tqdm(test_loader, desc="  [Test Evaluation]", leave=True, ncols=100)
    for images, labels in progress_bar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def print_evaluation_report(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """พิมพ์รายงานการประเมินผลทางการแพทย์ (Clinical Screening Evaluation Report)"""
    print("\n" + "=" * 60)
    print("Step 2.4: Evaluation & Medical Safety Metrics")
    print("=" * 60)

    target_names = [CLASS_NAMES_EN[0], CLASS_NAMES_EN[1]]
    print("\nClassification Report:")
    print("-" * 80)
    print(classification_report(y_true, y_pred, target_names=target_names, digits=4, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Medical Metrics Calculation
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0   # Recall for Cancer
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0   # Recall for Benign
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0           # Positive Predictive Value (Precision)
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0           # Negative Predictive Value
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0           # False Negative Rate (Critical Safety Metric)

    print(f"\nOverall Clinical Screening Metrics:")
    print(f"   Accuracy                                : {accuracy*100:.2f}%")
    print(f"   Sensitivity / Recall (Cancer Detection) : {sensitivity*100:.2f}%  ({tp}/{tp+fn})")
    print(f"   Specificity (Benign Confirmation)       : {specificity*100:.2f}%  ({tn}/{tn+fp})")
    print(f"   Positive Predictive Value (PPV)         : {ppv*100:.2f}%")
    print(f"   Negative Predictive Value (NPV)         : {npv*100:.2f}%")
    print(f"   F1-Score (Weighted)                     : {f1*100:.2f}%")

    print("\n" + "=" * 60)
    print("Medical Safety & Critical Diagnostic Analysis")
    print("=" * 60)
    print("   In cancer screening, False Negatives (FN) represent the highest clinical risk")
    print("   (i.e. malignant lesions misclassified as benign).")
    print()

    safety_status = "[PASSED]" if sensitivity >= 0.80 else "[BELOW TARGET]"
    print(f"   True Positives  (Cancer correctly identified) : {tp:>4d}")
    print(f"   False Negatives (Cancer missed as Benign)     : {fn:>4d} (False Negative Rate: {fnr*100:.2f}%)")
    print(f"   True Negatives  (Benign correctly identified) : {tn:>4d}")
    print(f"   False Positives (Benign flagged for review)   : {fp:>4d}")
    print(f"   Screening Safety Status                       : {safety_status}")
    print()
    print("   Note: This AI tool is intended for initial screening support only.")
    print("   Final diagnosis must be rendered by a certified dermatologist with histopathological correlation.")


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """พล็อต Confusion Matrix 2x2 สำหรับ Binary Screening"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("[WARNING] Matplotlib or Seaborn not found — Skipping Confusion Matrix plot")
        return

    labels = ["Benign (0)", "Cancer (1)"]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # 1. Raw Counts
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        ax=axes[0], linewidths=1.0, linecolor="#cbd5e1",
        cbar=False,
    )
    axes[0].set_title("Confusion Matrix (Case Counts)", fontsize=13, fontweight="bold", pad=12)
    axes[0].set_xlabel("Predicted Class", fontsize=11, labelpad=8)
    axes[0].set_ylabel("True Ground Truth", fontsize=11, labelpad=8)

    # 2. Normalized Percentage
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)
    sns.heatmap(
        cm_normalized, annot=True, fmt=".2%", cmap="Oranges",
        xticklabels=labels, yticklabels=labels,
        ax=axes[1], linewidths=1.0, linecolor="#cbd5e1",
        vmin=0, vmax=1, cbar=False,
    )
    axes[1].set_title("Confusion Matrix (Normalized Rate)", fontsize=13, fontweight="bold", pad=12)
    axes[1].set_xlabel("Predicted Class", fontsize=11, labelpad=8)
    axes[1].set_ylabel("True Ground Truth", fontsize=11, labelpad=8)

    plt.suptitle(
        "Skin Cancer Screening System — Confusion Matrix\n"
        "HAM10000 Dataset · ResNet50 Binary Classifier",
        fontsize=14, fontweight="bold", y=1.03,
    )
    plt.tight_layout()
    plt.savefig(str(CONFUSION_MATRIX_PATH), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[INFO] Confusion Matrix saved to: {CONFUSION_MATRIX_PATH}")


# ==============================================================================
# SECTION 8: Gradio Web Demo (Step 2.5) — Professional Cancer Screening UI
# ==============================================================================

GRADIO_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Sarabun:wght@300;400;500;600;700&display=swap');

.gradio-container {
    font-family: 'Inter', 'Sarabun', system-ui, -apple-system, sans-serif !important;
    max-width: 980px !important;
    margin: 0 auto !important;
    background: #f8fafc !important;
}

.header-section {
    text-align: center;
    padding: 2rem 1.5rem 1.5rem;
    margin-bottom: 0.75rem;
    background: #ffffff;
    border-bottom: 1px solid #e2e8f0;
    border-radius: 12px;
}
.header-section h1 {
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    color: #0f172a !important;
    margin: 0 0 0.35rem !important;
    letter-spacing: -0.01em;
}
.header-section .subtitle {
    font-size: 0.92rem;
    color: #475569;
    font-weight: 400;
    margin: 0;
}
.header-section .tech-badge {
    display: inline-block;
    margin-top: 0.75rem;
    padding: 0.25rem 0.75rem;
    background: #f1f5f9;
    border: 1px solid #cbd5e1;
    border-radius: 4px;
    font-size: 0.75rem;
    color: #334155;
    font-weight: 500;
    letter-spacing: 0.02em;
}

.section-label {
    font-size: 0.75rem;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.5rem;
}

.analyze-btn {
    width: 100% !important;
    border-radius: 6px !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    padding: 0.65rem 1.25rem !important;
    background: #0f172a !important;
    border: 1px solid #0f172a !important;
    color: #ffffff !important;
    letter-spacing: 0.02em;
    transition: background-color 0.2s ease !important;
}
.analyze-btn:hover {
    background: #1e293b !important;
    border-color: #1e293b !important;
}

.risk-card {
    border-radius: 8px;
    padding: 1.25rem;
    margin-top: 0.5rem;
}
.risk-high {
    background: #fef2f2;
    border: 1px solid #fca5a5;
}
.risk-low {
    background: #f0fdf4;
    border: 1px solid #86efac;
}
.status-pill-high {
    display: inline-block;
    background: #991b1b;
    color: #ffffff;
    padding: 0.25rem 0.65rem;
    border-radius: 4px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.status-pill-low {
    display: inline-block;
    background: #166534;
    color: #ffffff;
    padding: 0.25rem 0.65rem;
    border-radius: 4px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.risk-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #0f172a;
    margin: 0.75rem 0 0.15rem;
}
.risk-subtitle {
    font-size: 0.88rem;
    color: #475569;
    margin: 0 0 1rem;
    font-weight: 500;
}
.recommendation-header {
    font-size: 0.75rem;
    font-weight: 600;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 0.4rem;
}
.recommendation {
    font-size: 0.85rem;
    color: #1e293b;
    line-height: 1.6;
    padding-left: 1.2rem;
    margin: 0;
}
.recommendation li {
    margin-bottom: 0.35rem;
}

.prob-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 1.25rem;
    margin-top: 0.75rem;
}
.prob-table {
    width: 100%;
    border-collapse: collapse;
}
.prob-table tr {
    border-bottom: 1px solid #f1f5f9;
}
.prob-table tr:last-child {
    border-bottom: none;
}
.prob-table td {
    padding: 0.5rem 0.4rem;
    font-size: 0.82rem;
}
.prob-table .dx-code {
    font-weight: 700;
    width: 160px;
    font-size: 0.82rem;
    color: #334155;
}
.prob-bar-cell {
    width: 48%;
}
.prob-bar-bg {
    width: 100%;
    height: 8px;
    background: #f1f5f9;
    border-radius: 4px;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 4px;
}
.prob-value {
    text-align: right;
    font-weight: 600;
    color: #0f172a;
    font-size: 0.82rem;
    width: 70px;
    font-family: 'SF Mono', 'Fira Code', monospace;
}

.info-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 0.75rem;
    margin-top: 0.5rem;
}
.info-item {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 0.85rem 1rem;
    text-align: left;
}
.info-item .info-code {
    font-weight: 700;
    font-size: 0.85rem;
    color: #0f172a;
    display: block;
}
.info-item .info-name {
    display: block;
    font-size: 0.78rem;
    color: #475569;
    margin-top: 0.25rem;
    line-height: 1.4;
}
.info-item .info-risk {
    display: inline-block;
    margin-top: 0.4rem;
    font-size: 0.68rem;
    font-weight: 600;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
}
.risk-tag-high {
    background: #fee2e2;
    color: #991b1b;
}
.risk-tag-moderate {
    background: #fef3c7;
    color: #92400e;
}
.risk-tag-low {
    background: #f1f5f9;
    color: #475569;
}

.disclaimer {
    text-align: justify;
    font-size: 0.75rem;
    color: #64748b;
    padding: 1.25rem 1rem;
    margin-top: 1rem;
    border-top: 1px solid #e2e8f0;
    line-height: 1.6;
}

footer { display: none !important; }
"""


def launch_gradio_demo(device: torch.device) -> None:
    """สร้างและรันหน้าเว็บเดโม Gradio สำหรับคัดกรองมะเร็งผิวหนัง (No Emojis)"""
    try:
        import gradio as gr
    except ImportError:
        print("[ERROR] Gradio not found — Please install: pip install gradio")
        return

    print("\n" + "=" * 60)
    print("Step 2.5: Launching Gradio Skin Cancer Screening Interface")
    print("=" * 60)

    if not MODEL_SAVE_PATH.exists():
        print(f"[ERROR] Model file not found: {MODEL_SAVE_PATH}")
        print("   Please train the model first with: python skin_lesion_classifier.py")
        return

    # โหลดโมเดล ResNet50 Binary
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(512),
        nn.Dropout(p=0.3),
        nn.Linear(512, NUM_CLASSES),
    )
    checkpoint = torch.load(str(MODEL_SAVE_PATH), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    best_epoch = checkpoint.get("epoch", 0)
    best_f1 = checkpoint.get("val_f1", 0.0)
    print(f"[INFO] Model checkpoint loaded successfully (Epoch {best_epoch}, Val F1: {best_f1:.4f})")

    inference_transform = get_eval_transforms()

    # ─── Prediction Function ───
    def predict(image):
        if image is None:
            placeholder = (
                '<div style="text-align:center; padding:3rem 1rem; color:#94a3b8; font-size:0.85rem;">'
                'กรุณานำเข้าภาพถ่ายรอยโรคผิวหนัง แล้วกดปุ่ม "วิเคราะห์ผลคัดกรองมะเร็ง" เพื่อเริ่มต้นการประเมินผล'
                '</div>'
            )
            return {}, placeholder

        try:
            pil_image = Image.fromarray(image).convert("RGB") if isinstance(image, np.ndarray) else image.convert("RGB")
            input_tensor = inference_transform(pil_image).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()[0]

            benign_prob = float(probs[0])
            cancer_prob = float(probs[1])

            results = {
                "Cancer / Pre-Cancer (Malignant)": cancer_prob,
                "Non-Cancer (Benign)": benign_prob,
            }

            predicted_class = int(np.argmax(probs))
            confidence = probs[predicted_class] * 100
            risk_html = _build_risk_html(predicted_class, confidence, benign_prob, cancer_prob)

            return results, risk_html

        except Exception as e:
            err = (
                '<div class="risk-card risk-high" style="padding:1.25rem;">'
                '<p style="color:#991b1b; font-weight:600; margin:0 0 0.25rem;">เกิดข้อผิดพลาดในการประเมินผล</p>'
                f'<p style="color:#475569; font-size:0.85rem; margin:0;">{str(e)}</p>'
                '</div>'
            )
            return {}, err

    def _build_risk_html(predicted_class: int, confidence: float, benign_prob: float, cancer_prob: float) -> str:
        is_cancer = (predicted_class == 1)

        if is_cancer:
            card_cls = "risk-high"
            status_pill = '<span class="status-pill-high">HIGH RISK ASSESSMENT · CANCER DETECTED</span>'
            title = "พบความเสี่ยงมะเร็งผิวหนังหรือภาวะก่อนมะเร็ง (Malignancy Suspected)"
            subtitle = "รอยโรคมีลักษณะทางคลินิกที่เข้าได้กับกลุ่มมะเร็งผิวหนัง (Melanoma, Basal Cell Carcinoma หรือ Actinic Keratoses)"
            recs = [
                "แนะนำส่งพบแพทย์เฉพาะทางด้านผิวหนัง (Dermatologist Consultation) เพื่อการตรวจประเมินโดยละเอียดโดยเร็ว",
                "พิจารณาทำการตรวจด้วยกล้องส่องรอยโรคผิวหนัง (Dermoscopy) และการตรวจชิ้นเนื้อ (Skin Biopsy) เพื่อยืนยันการวินิจฉัย",
                "หลีกเลี่ยงการแกะ เกา หรือสัมผัสรอยโรคแรงๆ และบันทึกภาพถ่ายเพื่อติดตามการเปลี่ยนแปลงของขนาดและสี",
                "ผลการวิเคราะห์จากระบบปัญญาประดิษฐ์นี้เป็นข้อมูลสนับสนุนการคัดกรองเบื้องต้น ไม่สามารถใช้เป็นข้อสรุปการวินิจฉัยโรคได้",
            ]
        else:
            card_cls = "risk-low"
            status_pill = '<span class="status-pill-low">LOW RISK ASSESSMENT · BENIGN PATTERN</span>'
            title = "ไม่พบความเสี่ยงมะเร็งผิวหนัง (Benign Lesion Pattern)"
            subtitle = "รอยโรคมีลักษณะทางคลินิกที่เข้าได้กับกลุ่มไม่ร้ายแรง (เช่น ไฝ, รอยโรคสะเก็ด, รอยโรคหลอดเลือด หรือเนื้องอกเส้นใย)"
            recs = [
                "ผลการวิเคราะห์เบื้องต้นพบรอยโรคที่มีแนวโน้มเป็นชนิดไม่ร้ายแรง (Benign Condition)",
                "แนะนำให้ติดตามและสังเกตการเปลี่ยนแปลงทางคลินิกของรอยโรคอย่างสม่ำเสมอตามเกณฑ์ ABCDE (Asymmetry, Border, Color, Diameter, Evolving)",
                "หากพบการเปลี่ยนแปลงของรอยโรคในอนาคต เช่น ขยายขนาดเร็ว มีเลือดออก หรือขอบเขตไม่เรียบ ควรเข้าพบแพทย์ทันที",
            ]

        cancer_pct = cancer_prob * 100
        benign_pct = benign_prob * 100

        rec_items = "".join(f"<li>{r}</li>" for r in recs)

        return f"""
        <div class="risk-card {card_cls}">
            <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:0.5rem;">
                {status_pill}
                <span style="font-size:0.78rem; color:#475569; font-weight:500;">
                    Confidence Score: <b style="color:#0f172a;">{confidence:.1f}%</b>
                </span>
            </div>
            <p class="risk-title">{title}</p>
            <p class="risk-subtitle">{subtitle}</p>
            <div style="margin-bottom:0.5rem;">
                <div class="recommendation-header">ข้อแนะนำการจัดการทางคลินิก (Clinical Management Recommendation)</div>
                <ol class="recommendation">
                    {rec_items}
                </ol>
            </div>
        </div>
        <div class="prob-card">
            <div class="section-label" style="margin-bottom:0.75rem;">ผลการประเมินความน่าจะเป็นทางสถิติ (Screening Probability Distribution)</div>
            <table class="prob-table">
                <tr style="{'background:#fef2f2;' if is_cancer else ''}">
                    <td class="dx-code" style="color:#991b1b; font-weight:700;">Cancer / Pre-Cancer</td>
                    <td class="prob-bar-cell">
                        <div class="prob-bar-bg">
                            <div class="prob-bar-fill" style="width:{cancer_pct}%; background:#dc2626;"></div>
                        </div>
                    </td>
                    <td class="prob-value" style="color:#991b1b; font-weight:700;">{cancer_pct:.1f}%</td>
                </tr>
                <tr style="{'background:#f0fdf4;' if not is_cancer else ''}">
                    <td class="dx-code" style="color:#166534; font-weight:700;">Non-Cancer (Benign)</td>
                    <td class="prob-bar-cell">
                        <div class="prob-bar-bg">
                            <div class="prob-bar-fill" style="width:{benign_pct}%; background:#16a34a;"></div>
                        </div>
                    </td>
                    <td class="prob-value" style="color:#166534; font-weight:700;">{benign_pct:.1f}%</td>
                </tr>
            </table>
        </div>
        """

    # ─── Gradio Blocks Layout ───
    with gr.Blocks(
        title="Skin Cancer Screening System — Clinical Decision Support Tool",
        css=GRADIO_CSS,
        theme=gr.themes.Base(
            font=gr.themes.GoogleFont("Inter"),
            font_mono=gr.themes.GoogleFont("Fira Code"),
            primary_hue=gr.themes.colors.slate,
            secondary_hue=gr.themes.colors.slate,
            neutral_hue=gr.themes.colors.slate,
            radius_size=gr.themes.sizes.radius_sm,
        ),
    ) as demo:
        gr.HTML(
            '<div class="header-section">'
            '<h1>Skin Cancer Screening & Diagnostic System</h1>'
            '<p class="subtitle">ระบบวิเคราะห์และคัดกรองมะเร็งผิวหนังอัตโนมัติสำหรับการสนับสนุนการตัดสินใจทางคลินิก</p>'
            '<span class="tech-badge">ResNet50 Architecture | HAM10000 Dataset | Binary Screening Mode</span>'
            '</div>'
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=2, min_width=320):
                gr.HTML('<div class="section-label">นำเข้าภาพถ่ายรอยโรค (Lesion Image Input)</div>')
                image_input = gr.Image(
                    label=None,
                    type="numpy",
                    height=320,
                    show_label=False,
                    sources=["upload", "clipboard"],
                )
                analyze_btn = gr.Button(
                    "วิเคราะห์ผลคัดกรองมะเร็ง (Analyze Screening)",
                    variant="primary",
                    size="lg",
                    elem_classes=["analyze-btn"],
                )

            with gr.Column(scale=3, min_width=400):
                gr.HTML('<div class="section-label">ผลการวิเคราะห์ทางคลินิก (Screening Results)</div>')
                label_output = gr.Label(
                    label=None,
                    num_top_classes=NUM_CLASSES,
                    show_label=False,
                )
                risk_output = gr.HTML(
                    value=(
                        '<div style="text-align:center; padding:3rem 1rem; color:#94a3b8; font-size:0.85rem;">'
                        'กรุณานำเข้าภาพถ่ายรอยโรคผิวหนัง แล้วกดปุ่ม "วิเคราะห์ผลคัดกรองมะเร็ง" เพื่อเริ่มต้นการประเมินผล'
                        '</div>'
                    )
                )

        analyze_btn.click(fn=predict, inputs=image_input, outputs=[label_output, risk_output])
        image_input.change(fn=predict, inputs=image_input, outputs=[label_output, risk_output])

        gr.HTML("""
        <div style="margin-top:1.25rem;">
            <div class="section-label" style="text-align:center; margin-bottom:0.75rem;">หมวดหมู่โรคมะเร็งและภาวะก่อนมะเร็งที่ระบบตรวจจับ (Target Lesion Categories)</div>
            <div class="info-grid">
                <div class="info-item">
                    <span class="info-code">MEL · Melanoma</span>
                    <span class="info-name">มะเร็งผิวหนังเมลาโนมา — มะเร็งผิวหนังชนิดร้ายแรงที่สุดที่มีโอกาสแพร่กระจายสูง</span>
                    <span class="info-risk risk-tag-high">ความเสี่ยงสูงมาก</span>
                </div>
                <div class="info-item">
                    <span class="info-code">BCC · Basal Cell Carcinoma</span>
                    <span class="info-name">มะเร็งเซลล์ฐาน — มะเร็งผิวหนังที่พบบ่อยที่สุด มีการลุกลามเฉพาะที่</span>
                    <span class="info-risk risk-tag-high">ความเสี่ยงสูง</span>
                </div>
                <div class="info-item">
                    <span class="info-code">AKIEC · Actinic Keratoses</span>
                    <span class="info-name">ภาวะก่อนมะเร็งผิวหนัง / โรคโบเวน — รอยโรคสะเก็ดที่มีความเสี่ยงพัฒนาเป็นมะเร็ง SCC</span>
                    <span class="info-risk risk-tag-moderate">ความเสี่ยงปานกลาง (Precancerous)</span>
                </div>
            </div>
        </div>
        """)

        gr.HTML(
            '<div class="disclaimer">'
            '<b>ข้อจำกัดความรับผิดชอบทางการแพทย์ (Medical Disclaimer):</b> '
            'ระบบปัญญาประดิษฐ์นี้จัดทำขึ้นเพื่อเป็นเครื่องมือสนับสนุนการคัดกรองเบื้องต้นทางวิชาการและการวิจัยทางการแพทย์เท่านั้น '
            'ผลลัพธ์ที่ได้จากการประเมินไม่สามารถนำไปใช้ทดแทนการวินิจฉัย คำแนะนำ หรือการตัดสินใจทางการแพทย์โดยแพทย์ผู้เชี่ยวชาญเฉพาะทางด้านผิวหนังได้ '
            'การตัดสินใจทางการรักษาทั้งหมดต้องผ่านการประเมินทางคลินิกและการตรวจชิ้นเนื้อทางพยาธิวิทยาโดยแพทย์ผู้ดูแลเป็นสำคัญ<br/>'
            'CPE310 Healthcare AI System | ResNet50 Transfer Learning Architecture | HAM10000 Dataset'
            '</div>'
        )

    print("\n[INFO] Launching Gradio Web Demo: http://127.0.0.1:7860")
    print("   Press Ctrl+C to terminate\n")
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, show_error=True)


# ==============================================================================
# SECTION 9: Main Pipeline (รวมทุก Step)
# ==============================================================================

def main():
    global NUM_EPOCHS, BATCH_SIZE

    parser = argparse.ArgumentParser(description="Skin Cancer Screening & Diagnostic System — HAM10000")
    parser.add_argument("--demo", action="store_true", help="รันเฉพาะเดโม Gradio (ต้องมีไฟล์ checkpoint โมเดลก่อน)")
    parser.add_argument("--skip-train", action="store_true", help="ข้ามการเทรน ไปประเมินผล+เดโมเลย")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help=f"จำนวน Epochs (default: {NUM_EPOCHS})")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help=f"Batch Size (default: {BATCH_SIZE})")
    args = parser.parse_args()

    NUM_EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size

    print_banner()
    set_seed()
    device = get_device()

    if args.demo:
        launch_gradio_demo(device)
        return

    # Step 2.1: Data Preparation
    df = load_and_prepare_data()
    train_df, val_df, test_df = split_data(df)
    class_weights = compute_class_weights_tensor(train_df, device)

    # Step 2.2: DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(train_df, val_df, test_df)

    if not args.skip_train:
        # Step 2.3: Model + Training
        model = build_model(device)
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
        )
        history = train_model(
            model, train_loader, val_loader,
            criterion, optimizer, scheduler, device,
        )
    else:
        print("\n[INFO] Skipping training — Loading model from checkpoint")

    # Step 2.4: Evaluation on Test Set
    print("\n" + "=" * 60)
    print("[INFO] Loading Best Model Checkpoint for Test Evaluation")
    print("=" * 60)

    if not MODEL_SAVE_PATH.exists():
        print(f"[ERROR] Model checkpoint not found: {MODEL_SAVE_PATH}")
        print("   Please train the model first.")
        return

    eval_model = build_model(device)
    checkpoint = torch.load(str(MODEL_SAVE_PATH), map_location=device, weights_only=False)
    eval_model.load_state_dict(checkpoint["model_state_dict"])
    print(f"[INFO] Loaded Best Model (Epoch {checkpoint.get('epoch', '?')}, Val F1: {checkpoint.get('val_f1', 0):.4f})")

    y_pred, y_true, y_probs = evaluate_on_test_set(eval_model, test_loader, device)
    print_evaluation_report(y_true, y_pred)
    plot_confusion_matrix(y_true, y_pred)

    # Step 2.5: Gradio Demo
    print("\n" + "=" * 60)
    print("[INFO] Launching Gradio Demo")
    print("=" * 60)
    try:
        launch_gradio_demo(device)
    except KeyboardInterrupt:
        print("\n\nSystem terminated by user.")
    except Exception as e:
        print(f"\n[ERROR] Unable to launch demo: {e}")
        print("   You can run the demo later with: python skin_lesion_classifier.py --demo")


if __name__ == "__main__":
    main()
