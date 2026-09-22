#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate.py -- Evaluation & Model Comparison

Provides test-set inference, per-class and binary grouped metrics reporting,
confusion matrix generation, metrics persistence (JSON), and multi-model
comparison tables.
"""

import json
from pathlib import Path
from typing import Tuple, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
    precision_score,
    accuracy_score,
)

from config import (
    BASE_DIR,
    CLASS_CODES,
    CLASS_NAMES_EN,
    NUM_CLASSES,
    CANCER_INDICES,
    BENIGN_INDICES,
    MODEL_REGISTRY,
    MODEL_NAMES,
)


# =============================================================================
# Test Set Inference
# =============================================================================

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


# =============================================================================
# Evaluation Report
# =============================================================================

def print_evaluation_report(
    model_name: str,
    labels: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
) -> None:
    """Print comprehensive evaluation report with per-class and binary grouped metrics."""
    print()
    print("=" * 80)
    print(f"  EVALUATION REPORT: {model_name}")
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

    # Binary grouped metrics (cancer screening)
    print()
    print("-" * 80)
    print("[INFO] Binary Cancer Screening Metrics (Grouped)")
    print("-" * 80)

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


# =============================================================================
# Confusion Matrix
# =============================================================================

def plot_confusion_matrix(
    model_name: str,
    labels: np.ndarray,
    preds: np.ndarray,
    save_path: Optional[Path] = None,
) -> None:
    """Generate and save 8-class confusion matrix plot."""
    if save_path is None:
        save_path = MODEL_REGISTRY[model_name]["cm_file"]

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
        title=f"Confusion Matrix -- {model_name}",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=9,
            )

    # Group separator lines (cancer vs benign)
    ax.axhline(y=3.5, color="red", linewidth=2, linestyle="--")
    ax.axvline(x=3.5, color="red", linewidth=2, linestyle="--")

    ax.text(-1.5, 1.5, "Cancer", ha="center", va="center", fontsize=10,
            fontweight="bold", color="red", rotation=90)
    ax.text(-1.5, 5.5, "Benign", ha="center", va="center", fontsize=10,
            fontweight="bold", color="green", rotation=90)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Confusion matrix saved to: {save_path}")


# =============================================================================
# Metrics Persistence (JSON)
# =============================================================================

def save_model_metrics(
    model_name: str,
    labels: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    training_time: float = 0.0,
    total_params: int = 0,
    trainable_params: int = 0,
) -> None:
    """Save evaluation metrics for a model to JSON."""
    overall_acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)

    binary_labels = np.array([1 if lbl in CANCER_INDICES else 0 for lbl in labels])
    binary_preds = np.array([1 if pred in CANCER_INDICES else 0 for pred in preds])

    cancer_sensitivity = recall_score(binary_labels, binary_preds, pos_label=1, zero_division=0)
    cancer_specificity = recall_score(binary_labels, binary_preds, pos_label=0, zero_division=0)
    false_negative_rate = 1.0 - cancer_sensitivity

    # Per-class F1
    per_class_f1 = {}
    per_class_f1_arr = f1_score(labels, preds, average=None, zero_division=0)
    for idx, code in enumerate(CLASS_CODES):
        if idx < len(per_class_f1_arr):
            per_class_f1[code] = round(float(per_class_f1_arr[idx]), 4)

    metrics = {
        "model_name": model_name,
        "overall_accuracy": round(float(overall_acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "cancer_sensitivity": round(float(cancer_sensitivity), 4),
        "cancer_specificity": round(float(cancer_specificity), 4),
        "false_negative_rate": round(float(false_negative_rate), 4),
        "per_class_f1": per_class_f1,
        "training_time_seconds": round(training_time, 1),
        "total_params": total_params,
        "trainable_params": trainable_params,
    }

    save_path = MODEL_REGISTRY[model_name]["metrics_file"]
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Metrics saved to: {save_path}")


def load_all_metrics() -> Dict[str, Dict]:
    """Load metrics JSON for all models that have been evaluated."""
    all_metrics = {}
    for name in MODEL_NAMES:
        metrics_path = MODEL_REGISTRY[name]["metrics_file"]
        if metrics_path.exists():
            with open(metrics_path, "r", encoding="utf-8") as f:
                all_metrics[name] = json.load(f)
    return all_metrics


# =============================================================================
# Model Comparison Table
# =============================================================================

def print_comparison_table() -> None:
    """Print a side-by-side comparison table of all evaluated models."""
    all_metrics = load_all_metrics()
    if not all_metrics:
        print("[WARNING] No model metrics found for comparison.")
        return

    print()
    print("=" * 80)
    print("  MODEL COMPARISON")
    print("=" * 80)

    names = list(all_metrics.keys())
    col_width = 18

    # Header
    header = f"  {'Metric':<28s}"
    for name in names:
        header += f"{name:>{col_width}s}"
    print(header)
    print("  " + "-" * (28 + col_width * len(names)))

    # Metrics rows
    rows = [
        ("Overall Accuracy", "overall_accuracy", True),
        ("Macro F1", "macro_f1", True),
        ("Weighted F1", "weighted_f1", True),
        ("Cancer Sensitivity", "cancer_sensitivity", True),
        ("Cancer Specificity", "cancer_specificity", True),
        ("False Negative Rate", "false_negative_rate", False),
        ("Training Time (s)", "training_time_seconds", None),
        ("Total Parameters", "total_params", None),
    ]

    for label, key, higher_is_better in rows:
        values = []
        for name in names:
            val = all_metrics[name].get(key, 0)
            values.append(val)

        # Find best value
        if higher_is_better is True:
            best_val = max(values)
        elif higher_is_better is False:
            best_val = min(values)
        else:
            best_val = None

        row_str = f"  {label:<28s}"
        for val in values:
            if key == "total_params":
                formatted = f"{val / 1e6:.1f}M"
            elif key == "training_time_seconds":
                minutes = val / 60
                formatted = f"{minutes:.1f} min"
            else:
                formatted = f"{val * 100:.1f}%"

            if best_val is not None and val == best_val:
                formatted = f"*{formatted}"
            row_str += f"{formatted:>{col_width}s}"
        print(row_str)

    print("  " + "-" * (28 + col_width * len(names)))
    print("  * = Best value for this metric")
    print()
