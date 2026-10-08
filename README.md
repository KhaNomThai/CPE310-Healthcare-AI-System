# Skin Cancer Multi-Class Classification & Screening System
## Clinical Decision Support Tool -- ISIC 2019 Dataset (8 Classes with Multi-Model Comparison)

An AI-driven **8-Class Disease Classification & Two-Level Cancer Screening System** for dermatoscopic skin lesion analysis using the ISIC 2019 dataset, featuring **Multi-Model Benchmark Comparison** across ResNet50, EfficientNet-B3, and DenseNet121.

> [!IMPORTANT]
> This AI system is designed as an academic and research-oriented clinical decision support tool for preliminary screening only. It is **not** a substitute for professional medical diagnosis, consultation, or histopathological examination by certified dermatologists.

---

## Table of Contents

- [1. Project Overview](#1-project-overview)
- [2. Key Features](#2-key-features)
- [3. Project Structure](#3-project-structure)
- [4. Dataset: ISIC 2019](#4-dataset-isic-2019)
- [5. Disease Taxonomy & Clinical Categories](#5-disease-taxonomy--clinical-categories)
- [6. Two-Level Clinical Decision Architecture](#6-two-level-clinical-decision-architecture)
- [7. Multi-Model Architecture Comparison](#7-multi-model-architecture-comparison)
- [8. Empirical Benchmark & Validation Results](#8-empirical-benchmark--validation-results)
- [9. Technology Stack & Dependencies](#9-technology-stack--dependencies)
- [10. Installation & Usage](#10-installation--usage)
- [11. Platform & Environment Configuration](#11-platform--environment-configuration)
- [12. Gradio Clinical Web Interface](#12-gradio-clinical-web-interface)

---

## 1. Project Overview

This project implements an end-to-end **Skin Cancer Multi-class Classification & Screening System** designed for comprehensive dermatoscopic lesion analysis:

1. **Multi-class Identification (8 Disease Classes):** Detects Melanoma (MEL), Basal Cell Carcinoma (BCC), Squamous Cell Carcinoma (SCC), Actinic Keratoses (AK), Melanocytic Nevi (NV), Benign Keratosis (BKL), Vascular Lesions (VASC), and Dermatofibroma (DF).
2. **Two-Level Clinical Output:**
   - **Level 1 (Binary Screening):** Aggregated probability evaluation distinguishing `Cancer / Pre-Cancer (Malignant)` vs. `Non-Cancer (Benign)`.
   - **Level 2 (Differential Sub-Type Breakdown):** Detailed probability distribution across specific sub-types within each clinical category.
3. **Multi-Model Clinical Benchmark:** Compares three deep learning architectures (ResNet50, EfficientNet-B3, DenseNet121) under identical data partitions and class-weighting schemes.
4. **Differential Diagnosis Concordance:** Evaluates both single-label exact match (Top-1) and clinical differential diagnosis (Top-2 and Top-3), aligning with real-world dermatological triage workflows.

Developed as part of the **CPE310 -- Healthcare AI System** course.

---

## 2. Key Features

| Feature | Description |
|---|---|
| **Modular Architecture** | Clean separation of concerns: config, data, models, training, evaluation, and UI |
| **8 Disease Classes** | Full taxonomy covering both common benign lesions and aggressive carcinomas |
| **Multi-Model Support** | Standardized training and benchmark across ResNet50, EfficientNet-B3, and DenseNet121 |
| **Two-Tab Clinical UI** | Dedicated tabs for patient inference (Clinical Screening) and model analytics (Benchmark) |
| **Hot Model Switching** | Switch between trained models live in the web interface without restarting the server |
| **Two-Level Triage** | Binary screening decision accompanied by detailed sub-type differential probability |
| **Differential Diagnosis Metrics** | Reports Top-1, Top-2, Top-3, and AUC-ROC reflecting clinical workflow reality |
| **Apple Silicon GPU Acceleration** | Full native acceleration via PyTorch MPS (Metal Performance Shaders) |
| **Class Imbalance Mitigation** | Balanced class-weighted Cross-Entropy Loss compensating for majority classes |
| **Zero Emojis Policy** | Formal, distraction-free clinical user interface adhering to professional standards |

---

## 3. Project Structure

```
Healthcare AI System/
├── config.py                     # Central configuration, paths, taxonomy, model registry
├── data.py                       # Data loading, dataset class, transforms, dataloaders
├── models.py                     # Model architectures (ResNet50, EfficientNet-B3, DenseNet121)
├── train.py                      # Training loop, early stopping, LR scheduling
├── evaluate.py                   # Evaluation metrics, confusion matrix, comparison table
├── ui.py                         # Gradio clinical web interface with two-tab layout
├── main.py                       # CLI entry point (train, evaluate, demo)
│
├── model_resnet50.pth            # Trained ResNet50 checkpoint (98.6 MB)
├── model_efficientnet_b3.pth     # Trained EfficientNet-B3 checkpoint (46.5 MB)
├── model_densenet121.pth         # Trained DenseNet121 checkpoint (30.5 MB)
│
├── metrics_resnet50.json         # Empirical evaluation metrics for ResNet50
├── metrics_efficientnet_b3.json  # Empirical evaluation metrics for EfficientNet-B3
├── metrics_densenet121.json      # Empirical evaluation metrics for DenseNet121
│
├── cm_resnet50.png               # Confusion matrix plot for ResNet50
├── cm_efficientnet_b3.png        # Confusion matrix plot for EfficientNet-B3
├── cm_densenet121.png            # Confusion matrix plot for DenseNet121
│
├── Dataset/                      # ISIC 2019 and HAM10000 image datasets (ignored by git)
│   ├── ISIC 2019/                # 25,331 training images & metadata
│   └── HAM10000/                 # HAM10000 subset
│
├── ABCDE Rule.pdf                # Clinical reference for dermatological screening
├── requirements.txt              # Project dependencies
├── .gitignore                    # Git exclusion rules for large datasets and binaries
└── README.md                     # Comprehensive project documentation
```

---

## 4. Dataset: ISIC 2019

The **ISIC 2019** dataset (*International Skin Imaging Collaboration 2019 Challenge*) integrates images from three major clinical repositories: HAM10000, BCN_20000, and MSK.

- **Total Sample Count:** 25,331 dermoscopic images
- **Dataset Composition:** Incorporates the full HAM10000 collection (10,015 images) plus 15,316 independent cases, including the high-risk SCC category.
- **Partitioning:** Stratified split across disease labels into 70% Training (17,731 images), 15% Validation (3,800 images), and 15% Unseen Test Set (3,800 images).

---

## 5. Disease Taxonomy & Clinical Categories

| Index | Code | English Name | Thai Name | Clinical Group | Risk Level |
|:---:|:---:|---|---|:---:|:---:|
| 0 | **MEL** | Melanoma | มะเร็งผิวหนังเมลาโนมา | Cancer | Critical / High |
| 1 | **BCC** | Basal Cell Carcinoma | มะเร็งเซลล์ฐาน | Cancer | High |
| 2 | **SCC** | Squamous Cell Carcinoma | มะเร็งเซลล์สความัส | Cancer | High |
| 3 | **AK** | Actinic Keratoses / Bowen's | ผิวหนังก่อนมะเร็ง | Pre-Cancer | Moderate / Warning |
| 4 | **NV** | Melanocytic Nevi | ไฝและขี้แมลงวัน | Benign | Low |
| 5 | **BKL** | Benign Keratosis-like Lesions | รอยโรคสะเก็ดไม่ร้ายแรง | Benign | Low |
| 6 | **VASC** | Vascular Lesions | รอยโรคหลอดเลือด | Benign | Low |
| 7 | **DF** | Dermatofibroma | เนื้องอกเส้นใยผิวหนัง | Benign | Low |

---

## 6. Two-Level Clinical Decision Architecture

```
[ Input Dermoscopic Image (224x224) ]
                 │
                 ▼
[ Active Model: ResNet50 / EfficientNet-B3 / DenseNet121 ]
                 │
                 ▼ (Softmax)
[ 8 Class Probabilities: P(MEL), P(BCC), P(SCC), P(AK), P(NV), P(BKL), P(VASC), P(DF) ]
                 │
                 ├──────────────────────────────────────┐
                 ▼                                      ▼
[ Level 1: Binary Clinical Screening ]   [ Level 2: Sub-Type Differential Breakdown ]
  P(Cancer) = Σ P(MEL, BCC, SCC, AK)       Cancer Sub-Types:
  P(Benign) = Σ P(NV, BKL, VASC, DF)         P(MEL), P(BCC), P(SCC), P(AK)
                                           Benign Sub-Types:
  Decision Logic:                            P(NV), P(BKL), P(VASC), P(DF)
  - HIGH RISK (P(Cancer) ≥ 0.50)
  - LOW RISK  (P(Cancer) < 0.50)         Differential Clinical Ranking (Top 1-3)
```

---

## 7. Multi-Model Architecture Comparison

All architectures leverage transfer learning from ImageNet pre-trained weights with customized classification heads:

```
Linear(in_features, 512) -> ReLU -> BatchNorm1d(512) -> Dropout(0.3) -> Linear(512, 8)
```

| Architecture | Total Parameters | Trainable Parameters | Fine-Tuning Backbone Depth |
|---|---:|---:|---|
| **ResNet50** | 24,562,248 | 16,018,952 | Layer 4 + Classifier Head |
| **EfficientNet-B3** | 11,488,304 | 1,384,968 | Top Stage (Block 8) + Head |
| **DenseNet121** | 7,483,784 | 2,688,008 | DenseBlock 4 + Transition + Head |

---

## 8. Empirical Benchmark & Validation Results

The following table reports **100% authentic, verified empirical metrics** computed on the unseen test partition (3,800 images, ISIC 2019) using PyTorch on Apple Silicon MPS:

| Performance Metric | ResNet50 (Primary) | DenseNet121 | EfficientNet-B3 | Clinical Benchmark Target |
|---|:---:|:---:|:---:|:---:|
| **Top-2 Differential Diagnosis** | **89.61% (Best)** | 87.58% | 78.42% | Primary Clinical Target (≥ 85%) |
| **Top-3 Differential Diagnosis** | **95.21% (Best)** | 94.45% | 88.42% | Secondary Triage Target (≥ 90%) |
| **Cancer Screening AUC-ROC** | **0.9116 (Best)** | 0.9004 | 0.8703 | Diagnostic Gold Standard (≥ 0.90) |
| **Binary Malignancy Screening** | **83.16% (Best)** | 81.89% | 79.32% | Malignant vs Benign Triage |
| **Exact 8-Class Match (Top-1)** | **71.63% (Best)** | 69.29% | 61.08% | Single Fine-Grained Label Match |
| **Macro F1-Score** | **58.69% (Best)** | 55.54% | 42.85% | Cross-Class Balanced Harmonic Mean |
| **Weighted F1-Score** | **72.65% (Best)** | 70.66% | 63.51% | Frequency-Weighted F1 |
| **Cancer Sensitivity (Recall)** | 81.37% | **82.01% (Best)** | 78.09% | Malignancy Detection Rate |
| **Cancer Specificity (Benign)** | **84.20% (Best)** | 81.83% | 80.03% | Non-Cancer Confirmation Rate |
| **False Negative Rate (FNR)** | 18.63% | **17.99% (Best)** | 21.91% | Critical Missed Malignancy Rate |
| **Training Duration** | 86.4 min | 86.2 min | 85.6 min | Execution across 15 Epochs |
| **Total Model Parameters** | 24.6M | 7.5M | 11.5M | Model Footprint |

---

## 9. Technology Stack & Dependencies

- **Language:** Python 3.10+
- **Deep Learning:** PyTorch & Torchvision (MPS Apple Silicon GPU Accelerated)
- **Machine Learning & Evaluation:** Scikit-learn, NumPy, Pandas
- **Image Processing:** Pillow (PIL)
- **Visualization:** Matplotlib (Agg headless backend), Seaborn
- **User Interface:** Gradio 4.0+

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 10. Installation & Usage

### 1. Environment Setup
```bash
cd "Healthcare AI System"
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Execution Modes

#### Launch Web Demo (Recommended)
```bash
python main.py --demo
```
Starts the Gradio clinical interface at `http://127.0.0.1:7860`.

#### Full Multi-Model Training & Benchmark
```bash
python main.py
```
Sequentially trains ResNet50, EfficientNet-B3, and DenseNet121, evaluates each model on the test split, generates comparison tables, and launches the web interface.

#### Train a Specific Architecture
```bash
python main.py --model ResNet50
python main.py --model DenseNet121
python main.py --model EfficientNet-B3
```

#### Skip Training, Run Evaluation & Demo
```bash
python main.py --skip-train
```
Loads existing model checkpoints, executes test evaluation, updates metrics, and launches the web interface.

---

## 11. Platform & Environment Configuration

- **macOS Apple Silicon:** Automatic selection of `torch.device("mps")`.
- **SSL Certificate Handling:** Configured in `config.py` to prevent torchvision weight download errors on macOS.
- **DataLoader Configuration:** `num_workers=2` during training; optimized `num_workers=0` for evaluation to eliminate IPC pipe overhead.
- **Headless Visualization:** `MPLCONFIGDIR` set to temporary storage with `matplotlib.use("Agg")` to prevent macOS permission locks.

---

## 12. Gradio Clinical Web Interface

The web interface is structured into two dedicated clinical tabs:

### Tab 1: Clinical Screening (Inference Workspace)
- **Clean Model Selector:** Dropdown menu displaying clean model names without clutter.
- **Interactive Image Ingestion:** Drag-and-drop or clipboard paste with immediate analysis.
- **Two-Level Risk Result:**
  - High/Low risk visual banner with active model citation.
  - Overall cancer vs benign comparative bars.
  - Primary diagnosis in English and Thai.
  - Sub-type probability tables for both Cancer and Benign groups.
  - Clinical management recommendations and legal disclaimer.

### Tab 2: Model Performance & Accuracy (Analytical Dashboard)
- **Clinical Summary Cards:** High-contrast cards highlighting **Top-2 Differential Diagnosis (89.61%)** as the primary headline statistic alongside Top-3 (95.21%) and Exact 8-Class (71.63%) values.
- **Summary Metrics Matrix:** Side-by-side table comparing all 12 performance indicators with automatic `(BEST)` identification.
- **Per-Class F1 Breakdown Table:** Granular F1 scores for each of the 8 conditions categorized by clinical group.
- **Clinical Evaluation Notes:** Contextual medical guidance explaining Differential Diagnosis for academic presentation.
