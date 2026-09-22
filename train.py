#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py -- Training Loop

Implements the training pipeline including single-epoch training,
validation, early stopping, and learning rate scheduling.
"""

import time
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm

from config import (
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_EPOCHS,
    EARLY_STOPPING_PATIENCE,
    MODEL_REGISTRY,
)


# =============================================================================
# Early Stopping
# =============================================================================

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


# =============================================================================
# Single Epoch Training
# =============================================================================

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


# =============================================================================
# Validation
# =============================================================================

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


# =============================================================================
# Full Training Pipeline
# =============================================================================

def train_model(
    model: nn.Module,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    class_weights: torch.Tensor,
) -> Tuple[nn.Module, float]:
    """
    Complete training pipeline with early stopping and LR scheduling.

    Returns:
        Tuple of (trained model, training_time_seconds).
    """
    save_path = MODEL_REGISTRY[model_name]["checkpoint"]

    print()
    print("=" * 80)
    print(f"  TRAINING: {model_name}")
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
            torch.save(model.state_dict(), save_path)
            print(f"  [SAVED] New best model (Val F1: {val_f1:.4f})")

        scheduler.step(val_f1)

        if early_stopping(val_f1):
            print(f"  [EARLY STOP] No improvement for {EARLY_STOPPING_PATIENCE} epochs")
            break

    total_time = time.time() - start_time
    print(f"  Training completed in {total_time:.1f}s")
    print(f"  Best validation macro F1: {best_val_f1:.4f}")
    print(f"  Model saved to: {save_path}")

    # Load best checkpoint
    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    return model, total_time
