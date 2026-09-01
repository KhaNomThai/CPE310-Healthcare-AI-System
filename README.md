# Skin Cancer Screening & Classification System
## Clinical Decision Support Tool — Binary Malignancy Screening

An AI-driven **Binary Classification** system (Cancer vs. Non-Cancer) for dermatoscopic skin lesion analysis using the HAM10000 dataset, powered by Deep Learning (**ResNet50 Transfer Learning**).

> [!IMPORTANT]
> This AI system is designed as an academic and research-oriented clinical decision support tool for preliminary screening only. It is **not** a substitute for professional medical diagnosis, consultation, or histopathological examination by certified dermatologists.

---

## Table of Contents

- [1. Project Overview](#1-project-overview)
- [2. Key Features](#2-key-features)
- [3. Project Structure](#3-project-structure)
- [4. Dataset: HAM10000](#4-dataset-ham10000)
- [5. Technology Stack & Dependencies](#5-technology-stack--dependencies)
- [6. Model Architecture](#6-model-architecture)
- [7. End-to-End Pipeline](#7-end-to-end-pipeline)
- [8. Detailed Pipeline Steps](#8-detailed-pipeline-steps)
- [9. Summary of AI/ML Techniques](#9-summary-of-aiml-techniques)
- [10. Installation & Usage](#10-installation--usage)
- [11. Platform & Environment Configuration](#11-platform--environment-configuration)
- [12. Experimental Results & Metrics](#12-experimental-results--metrics)
- [13. Strengths & Limitations](#13-strengths--limitations)
- [14. Summary](#14-summary)

---

## 1. Project Overview

This project implements an end-to-end **Skin Cancer Screening System** designed for binary classification of skin lesions into two clinical categories:

| Class | Category | Included Lesion Subtypes |
|-------|----------|--------------------------|
| **Class 1** | **Cancer / Pre-Cancer (Malignant)** | `mel` (Melanoma), `bcc` (Basal Cell Carcinoma), `akiec` (Actinic Keratoses / Bowen's Disease) |
| **Class 0** | **Non-Cancer (Benign)** | `nv` (Melanocytic Nevi), `bkl` (Benign Keratosis), `vasc` (Vascular Lesion), `df` (Dermatofibroma) |

The model utilizes **Transfer Learning** on a **ResNet50** architecture pre-trained on ImageNet. Developed as part of the **CPE310 — Healthcare AI System** course.

> [!NOTE]
> The current system focuses on **Binary Classification** (Cancer vs. Non-Cancer) instead of a 7-way multi-class categorization to align with real-world clinical screening workflows.

---

## 2. Key Features

| Feature | Description |
|---------|-------------|
| **Binary Cancer Screening** | Differentiates Cancer/Pre-Cancer from Benign lesions |
| **Transfer Learning** | ResNet50 backbone pre-trained on ImageNet V2 |
| **Apple Silicon GPU Acceleration** | Native support for MPS (Metal Performance Shaders) on macOS |
| **Data Augmentation** | 6 techniques including Flip, Rotation, ColorJitter, and Affine transformations |
| **Class Balancing** | Balanced class-weighted Cross-Entropy Loss to counter data imbalance |
| **Early Stopping** | Patience of 3 epochs based on Validation F1-Score |
| **Learning Rate Scheduling** | `ReduceLROnPlateau` for automated learning rate reduction |
| **Clinical Safety Metrics** | Evaluates Sensitivity (Recall), Specificity, PPV, NPV, and False Negative Rate (FNR) |
| **Professional Web UI** | Clean, clinical Gradio interface with custom CSS (no casual emojis) |
| **Clinical Risk Assessment** | Generates HIGH RISK / LOW RISK cards with ABCDE management guidelines |

---

## 3. Project Structure

```
Healthcare AI System/
├── Dataset/
│   ├── HAM10000_images_part_1/        # Image folder part 1 (~5,000 images)
│   ├── HAM10000_images_part_2/        # Image folder part 2 (~5,015 images)
│   ├── HAM10000_metadata.csv          # Metadata: image_id, dx, etc. (563 KB)
│   ├── hmnist_28_28_L.csv             # Grayscale 28×28 pixel data
│   ├── hmnist_28_28_RGB.csv           # RGB 28×28 pixel data
│   ├── hmnist_8_8_L.csv              # Grayscale 8×8 pixel data
│   └── hmnist_8_8_RGB.csv            # RGB 8×8 pixel data
├── skin_lesion_classifier.py          # Main script (~1,358 lines, ~56 KB)
├── best_cancer_model.pth              # Trained binary screening model checkpoint (~227 MB)
├── best_skin_model.pth                # Legacy 7-class model checkpoint (~227 MB)
├── cancer_confusion_matrix.png        # Confusion matrix for binary model
├── confusion_matrix.png               # Confusion matrix for legacy 7-class model
├── requirements.txt                   # Project dependencies
├── README.md                          # Project documentation
└── venv/                              # Python Virtual Environment
```

---

## 4. Dataset: HAM10000

The **HAM10000** (*Human Against Machine with 10000 training images*) dataset is a standard medical benchmark for dermatoscopic lesion analysis containing:

- **Total Images:** ~10,015 dermatoscopic images across 2 partitions
- **Metadata:** CSV containing `image_id`, `dx` (diagnosis), demographic details, etc.

### Clinical Binary Grouping

#### Class 1: Cancer & Pre-Cancer (Malignant Target)

| Code | Disease Name | Clinical Description | Risk Level |
|------|--------------|----------------------|------------|
| `mel` | Melanoma | Highly aggressive malignant melanocytic cancer | Critical / High |
| `bcc` | Basal Cell Carcinoma | Common non-melanoma skin cancer with local invasion | High |
| `akiec` | Actinic Keratoses / Bowen's Disease | Squamous pre-cancerous / intraepidermal carcinoma | Moderate (Precancerous) |

#### Class 0: Non-Cancer (Benign)

| Code | Disease Name | Clinical Description | Risk Level |
|------|--------------|----------------------|------------|
| `nv` | Melanocytic Nevi | Common benign moles | Low |
| `bkl` | Benign Keratosis-like Lesions | Seborrheic keratoses, solar lentigines, lichen-planus-like | Low |
| `vasc` | Vascular Lesions | Cherry angiomas, angiokeratomas, pyogenic granulomas | Low |
| `df` | Dermatofibroma | Benign dermal fibrous histiocytomas | Low |

> [!NOTE]
> The dataset suffers from severe **Class Imbalance** — Benign cases account for ~80% while Malignant/Pre-cancer cases account for only ~20%. This is compensated for using balanced class weighting during training.

---

## 5. Technology Stack & Dependencies

### 5.1 Core Deep Learning

| Library | Version | Role |
|---------|---------|------|
| **PyTorch** | ≥ 2.0.0 | Core Deep Learning Framework |
| **TorchVision** | ≥ 0.15.0 | Pre-trained models (ResNet50) and image transformations |

### 5.2 Data Processing & Evaluation

| Library | Version | Role |
|---------|---------|------|
| **Pandas** | ≥ 2.0.0 | Metadata parsing and DataFrame manipulation |
| **NumPy** | ≥ 1.24.0 | Numerical operations and array manipulation |
| **Pillow (PIL)** | ≥ 9.5.0 | Image loading and preprocessing |
| **scikit-learn** | ≥ 1.3.0 | Stratified splitting, evaluation metrics, class weights |

### 5.3 Visualization & Interface

| Library | Version | Role |
|---------|---------|------|
| **Matplotlib** | ≥ 3.7.0 | Plotting confusion matrix figures |
| **Seaborn** | ≥ 0.12.0 | Heatmap styling for confusion matrix |
| **tqdm** | ≥ 4.65.0 | Real-time training progress bar |
| **Gradio** | ≥ 4.0.0 | Interactive web-based clinical demonstration interface |

### 5.4 Hardware Acceleration

- **Apple Silicon MPS (Metal Performance Shaders)**: Optimized execution on macOS Apple Silicon GPUs.
- **NVIDIA CUDA**: Supported if running on CUDA-enabled GPU hardware.
- **CPU Fallback**: Graceful fallback when no hardware accelerator is present.

---

## 6. Model Architecture

### 6.1 ResNet50 Transfer Learning (Binary Head)

The model leverages a pre-trained **ResNet50** architecture with a custom classification head:

```
📷 Input Image (224 × 224 × 3)
        │
        ▼
┌─────────────────────────────────────────┐
│         ResNet50 Backbone               │
│         (Pre-trained on ImageNet)       │
│                                         │
│  🔒 Frozen Layers (Feature Extractors): │
│     ├── conv1 + bn1                     │
│     ├── layer1                          │
│     ├── layer2                          │
│     └── layer3                          │
│                                         │
│  🔓 Trainable Layer (Fine-tuning):      │
│     └── layer4                          │
└─────────────────────┬───────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────┐
│      🧠 Custom Classifier Head          │
│                                         │
│  Linear(2048 → 512)                     │
│       ▼                                 │
│  ReLU Activation                        │
│       ▼                                 │
│  BatchNorm1d(512)                       │
│       ▼                                 │
│  Dropout(p=0.3)                         │
│       ▼                                 │
│  Linear(512 → 2)  [Binary Output]       │
└─────────────────────┬───────────────────┘
                      │
                      ▼
          Output: 2 Classes
     [0] Benign    [1] Malignant
```

### 6.2 Freezing Strategy

| Layer | Trainable? | Purpose |
|-------|------------|---------|
| `conv1`, `bn1` | ❌ Frozen | Extracts low-level visual features (edges, basic textures) |
| `layer1` | ❌ Frozen | Captures simple texture patterns |
| `layer2` | ❌ Frozen | Captures mid-level geometric representations |
| `layer3` | ❌ Frozen | Captures complex structures |
| **`layer4`** | **✅ Trainable** | **Fine-tunes high-level dermatological lesion patterns** |
| **Custom FC Head** | **✅ Trainable** | **Maps representations to binary screening probabilities** |

- **Total Parameters:** ~23.5 Million
- **Trainable Parameters:** ~16.0 Million (`layer4` + Custom FC Head)
- **Frozen Parameters:** ~8.5 Million

> [!TIP]
> Freezing early layers preserves general visual features learned from ImageNet while drastically reducing compute requirements and mitigating overfitting on small medical datasets.

---

## 7. End-to-End Pipeline

```
Step 2.1                Step 2.2              Step 2.3             Step 2.4              Step 2.5
Data Preparation   →    DataLoader      →     Training       →     Evaluation      →     Clinical Demo
                        & Augmentation         & Model               & Metrics
─────────────────────────────────────────────────────────────────────────────────────────────────
• Read CSV metadata    • Train:               • ResNet50            • Classification     • Gradio UI
• Scan image dirs        Augmentations          + Custom Head          Report               (Clinical)
• Map image_id→path    • Val/Test:            • AdamW Optimizer     • Sensitivity        • Image Upload /
• Binary Mapping         Resize+Normalize     • Weighted Loss         Specificity          Clipboard
  (Cancer vs Benign)   • PyTorch DataLoader   • Early Stopping        PPV / NPV / FNR    • Probability
• Stratified Split       batch=32             • LR Scheduler        • 2×2 Confusion        Distribution
  by Subtype (80/10/10)• workers=2              ReduceLROnPlateau     Matrix Plot        • Risk Assessment
• Class Weighting
```

---

## 8. Detailed Pipeline Steps

### 8.1 Step 2.1: Data Preparation & Preprocessing

> Code: `skin_lesion_classifier.py` (lines 188–307)

1. **Load Metadata:** Parses `HAM10000_metadata.csv`.
2. **Scan & Index Files:** Scans image partitions (`part_1` and `part_2`) to create `image_id` → file path mappings.
3. **Binary Relabeling:** Maps multi-class diagnosis strings:
   - `mel`, `bcc`, `akiec` → **Class 1 (Cancer / Pre-Cancer)**
   - `nv`, `bkl`, `vasc`, `df` → **Class 0 (Non-Cancer / Benign)**
4. **Subtype Preservation:** Stores original `dx` in `subtype` for fine-grained stratified splitting.
5. **Stratified Split:** Partitions data preserving the proportion of all 7 original subtypes across splits:
   - **Train (80%)** | **Validation (10%)** | **Test (10%)**
6. **Class Weight Computation:** Computes balanced class weights via `scikit-learn` to prioritize underrepresented cancer samples.

### 8.2 Step 2.2: Dataset & Data Augmentation

> Code: `skin_lesion_classifier.py` (lines 310–404)

#### Training Augmentations

| Augmentation | Parameters | Objective |
|--------------|------------|-----------|
| `Resize` | 224 × 224 | Standard input resolution for ResNet50 |
| `RandomHorizontalFlip` | p=0.5 | Invariance to horizontal lesion orientation |
| `RandomVerticalFlip` | p=0.5 | Invariance to vertical lesion orientation |
| `RandomRotation` | ±180° | Full rotational invariance (lesions can appear at any angle) |
| `ColorJitter` | brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05 | Robustness against lighting/dermatoscope variations |
| `RandomAffine` | translate=5%, scale=95–105% | Scale and translation invariance |
| `Normalize` | ImageNet mean/std | Standardizes feature distributions |

#### Evaluation Transform (Val/Test)

- Strict evaluation pipeline without stochastic perturbations: `Resize(224×224)` → `ToTensor()` → `Normalize()`.

#### DataLoader Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `batch_size` | 32 | Balanced GPU memory consumption and gradient stability |
| `num_workers` | 2 | Safe multi-process loading on macOS avoiding fork crashes |
| `pin_memory` | True | Accelerates host-to-device memory transfer |
| `drop_last` | True (train only) | Prevents incomplete mini-batches from corrupting BatchNorm |

### 8.3 Step 2.3: Model Building & Training Loop

> Code: `skin_lesion_classifier.py` (lines 407–642)

#### Hyperparameter Configuration

| Parameter | Value | Note |
|-----------|-------|------|
| Input Resolution | 224 × 224 | 3-channel RGB |
| Initial Learning Rate | 1e-4 | Low LR suitable for transfer learning |
| Weight Decay | 1e-2 | Decoupled L2 regularization via AdamW |
| Max Epochs | 15 | Subject to Early Stopping |
| Early Stopping Patience | 3 | Halts if validation F1 does not improve for 3 epochs |
| `NUM_CLASSES` | 2 | Binary: Non-Cancer (0) vs Cancer (1) |

#### Optimization Strategy

- **Optimizer:** `AdamW` updating only trainable parameters (`layer4` + classifier head).
- **Loss Function:** `nn.CrossEntropyLoss(weight=class_weights)`.
- **LR Scheduler:** `ReduceLROnPlateau` (mode=`max`, factor=`0.5`, patience=`2`, metric=`val_f1`).
- **Tracking:** Monitors both validation F1 and `val_cancer_recall` (Sensitivity) per epoch.
- **Checkpointing:** Saves best model state dictionary to `best_cancer_model.pth`.

### 8.4 Step 2.4: Evaluation & Clinical Safety Metrics

> Code: `skin_lesion_classifier.py` (lines 645–769)

#### Clinical Metrics Formulation

| Metric | Formula | Clinical Meaning |
|--------|---------|------------------|
| **Sensitivity (Recall)** | TP / (TP + FN) | Proportion of true cancer cases detected |
| **Specificity** | TN / (TN + FP) | Proportion of benign cases correctly ruled out |
| **PPV (Precision)** | TP / (TP + FP) | Probability that a positive test is truly malignant |
| **NPV** | TN / (TN + FN) | Probability that a negative test is truly benign |
| **FNR (False Negative Rate)** | FN / (TP + FN) | Critical metric: rate of missed malignancies |
| **Accuracy** | (TP + TN) / Total | Overall prediction correctness |
| **Weighted F1** | - | Harmonic mean of Precision and Recall |

> [!WARNING]
> In clinical cancer screening, **False Negatives (FN)** present the highest hazard (missed malignancy). Therefore, **Sensitivity** and **FNR** serve as the primary safety benchmarks.

### 8.5 Step 2.5: Gradio Clinical Web Demo

> Code: `skin_lesion_classifier.py` (lines 772–1265)

A professional, clinical-grade user interface built with Gradio Blocks:

- **Typography & Styling:** Styled with Inter/Sarabun typography and Slate neutral palette; clean medical aesthetic without casual emojis.
- **Dynamic Risk Cards:** Emits stylized HTML cards (`HIGH RISK` red card vs. `LOW RISK` green card).
- **Clinical Action Recommendations:** Contextual recommendations including urgent dermatologist referral, dermoscopy, biopsy, and ABCDE monitoring criteria.
- **Probability Distribution:** Visual comparative bar representation of predicted class likelihoods.

Access via browser at `http://127.0.0.1:7860`.

---

## 9. Summary of AI/ML Techniques

| Technique | Implementation | Clinical / ML Objective |
|-----------|----------------|--------------------------|
| **Transfer Learning** | ResNet50 pre-trained on ImageNet V2 | Leverages generalized feature representations |
| **Partial Fine-Tuning** | Unfrozen `layer4` + Custom FC head | Adapts high-level filters to skin lesions |
| **Binary Formulation** | 7 subtypes grouped into 2 clinical categories | Aligns model output with triage screening needs |
| **Stochastic Augmentation** | 6-stage transformation pipeline | Prevents overfitting to image acquisition artifacts |
| **Class Weighting** | Balanced inverse frequency weighting | Penalizes false negatives on minority cancer class |
| **Subtype Stratification** | Stratified split by all 7 granular types | Ensures identical subtype distribution across splits |
| **Early Stopping** | Monitored on validation F1 score | Prevents over-optimization on training distribution |
| **Learning Rate Decay** | Plateaux-driven learning rate halving | Enables fine convergence in loss landscape |
| **Regularization** | BatchNorm1d + Dropout(0.3) + AdamW | Controls co-adaptation and weights magnitude |

---

## 10. Installation & Usage

### Installation

```bash
# 1. Create and activate a Virtual Environment (recommended)
python3 -m venv venv
source venv/bin/activate

# 2. Install required dependencies
pip install -r requirements.txt
```

### Execution Modes

```bash
# Mode 1: Full Pipeline (Train + Evaluate + Launch Web Demo)
python skin_lesion_classifier.py

# Mode 2: Custom Hyperparameters
python skin_lesion_classifier.py --epochs 20 --batch-size 64

# Mode 3: Skip Training (Evaluate existing checkpoint + Launch Demo)
python skin_lesion_classifier.py --skip-train

# Mode 4: Launch Web Demo directly
python skin_lesion_classifier.py --demo
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--demo` | `False` | Launches Gradio web demo only (requires model checkpoint) |
| `--skip-train` | `False` | Skips training and runs test evaluation + web demo |
| `--epochs` | `15` | Total number of training epochs |
| `--batch-size` | `32` | Mini-batch size for DataLoaders |

---

## 11. Platform & Environment Configuration

### Apple Silicon / macOS Optimization

- **Device Selection:** Auto-selects `mps` when Metal Performance Shaders are available, falling back to `cuda` or `cpu`.
- **Worker Configuration:** `num_workers=2` avoids macOS spawn/fork multiprocessing deadlocks.
- **SSL Fallback:** Unverified context override handles local macOS Python certificate chain issues.

### Reproducibility

Fixed seed (`RANDOM_SEED = 42`) set across:
- `random.seed(42)`
- `np.random.seed(42)`
- `torch.manual_seed(42)`
- `torch.mps.manual_seed(42)`
- `torch.cuda.manual_seed_all(42)`

---

## 12. Experimental Results & Metrics

### Confusion Matrix (Binary Screening)

<p align="center">
  <img src="cancer_confusion_matrix.png" alt="Binary Cancer Screening Confusion Matrix" width="800"/>
</p>

### Test Set Performance Breakdown

| Metric | Count / Score | Clinical Significance |
|--------|---------------|-----------------------|
| **True Positives (TP)** | 155 | Malignant lesions correctly identified |
| **True Negatives (TN)** | 729 | Benign lesions correctly ruled out |
| **False Positives (FP)** | 77 | Benign lesions referred for review (Low safety hazard) |
| **False Negatives (FN)** | 41 | Malignant lesions missed as benign (Critical clinical hazard) |

| Clinical Performance Indicator | Value | Assessment |
|--------------------------------|-------|------------|
| **Sensitivity (Cancer Detection)** | **79.08%** (155/196) | Approaching target threshold (80.0%) |
| **Specificity (Benign Rule-Out)** | **90.45%** (729/806) | High specificity / Low false alarm rate |
| **Accuracy** | **88.22%** (884/1002) | Strong overall classification performance |
| **False Negative Rate (FNR)** | **20.92%** (41/196) | Primary area targeted for future improvement |

> [!CAUTION]
> A Sensitivity of **79.08%** implies that roughly ~21% of malignant/pre-cancerous cases were misclassified as benign in this trial. Further architectural enhancements (e.g., ensemble methods, focal loss, higher resolution) are recommended prior to clinical deployment.

---

## 13. Strengths & Limitations

### Strengths

1. **Clinically Relevant Formulation:** Focuses on actionable binary screening rather than ambiguous multi-class categorization.
2. **Comprehensive Metric Suite:** Reports medical safety metrics (Sensitivity, Specificity, PPV, NPV, FNR).
3. **Subtype-Aware Stratification:** Preserves granular disease ratios throughout dataset splits.
4. **Imbalance Mitigation:** Incorporates balanced loss weighting to safeguard sensitivity on minority classes.
5. **Polished Clinical UI:** Clean interface with automated inference and standardized ABCDE management recommendations.
6. **Platform Optimized:** High performance on Apple Silicon using native MPS acceleration.

### Limitations

1. **Residual False Negative Rate:** An FNR of ~21% requires further mitigation before practical diagnostic triage.
2. **Single Backbone Model:** Evaluated exclusively on ResNet50 without benchmark comparisons to Vision Transformers (ViT) or EfficientNet.
3. **Single Split Evaluation:** Evaluated on a single stratified split rather than multi-fold cross-validation.

---

## 14. Summary

| Step | Phase | Key Implementation Details |
|------|-------|----------------------------|
| **Step 2.1** | Data Preparation | Subtype indexing, binary grouping, stratified 80/10/10 split, class weighting |
| **Step 2.2** | Data Augmentation | 6-stage augmentation pipeline, ImageNet normalization, custom PyTorch Dataset |
| **Step 2.3** | Model & Training | ResNet50 backbone, custom binary head, AdamW, weighted loss, sensitivity tracking |
| **Step 2.4** | Evaluation | Sensitivity, Specificity, PPV, NPV, FNR calculation, 2×2 confusion matrix plot |
| **Step 2.5** | Clinical Demo | Professional Gradio web UI with risk assessment and ABCDE guidelines |

---

## CPE310 — Healthcare AI System
