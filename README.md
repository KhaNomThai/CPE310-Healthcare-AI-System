# Skin Cancer Multi-Class Classification & Screening System
## Clinical Decision Support Tool — ISIC 2019 Dataset (8 Classes with Multi-Model Comparison)

An AI-driven **8-Class Disease Classification & Two-Level Cancer Screening** system for dermatoscopic skin lesion analysis using the ISIC 2019 dataset (incorporating HAM10000), featuring **Multi-Model Comparison** across ResNet50, EfficientNet-B3, and DenseNet121.

> [!IMPORTANT]
> This AI system is designed as an academic and research-oriented clinical decision support tool for preliminary screening only. It is **not** a substitute for professional medical diagnosis, consultation, or histopathological examination by certified dermatologists.

---

## Table of Contents

- [1. Project Overview](#1-project-overview)
- [2. Key Features](#2-key-features)
- [3. Project Structure](#3-project-structure)
- [4. Dataset: ISIC 2019 (Superset of HAM10000)](#4-dataset-isic-2019-superset-of-ham10000)
- [5. Disease Taxonomy & Clinical Categories](#5-disease-taxonomy--clinical-categories)
- [6. Two-Level Classification Architecture](#6-two-level-classification-architecture)
- [7. Multi-Model Architecture Comparison](#7-multi-model-architecture-comparison)
- [8. Technology Stack & Dependencies](#8-technology-stack--dependencies)
- [9. End-to-End Pipeline](#9-end-to-end-pipeline)
- [10. Clinical Safety Evaluation](#10-clinical-safety-evaluation)
- [11. Installation & Usage](#11-installation--usage)
- [12. Platform & Environment Configuration](#12-platform--environment-configuration)
- [13. Gradio Clinical Web Interface with Model Selector](#13-gradio-clinical-web-interface-with-model-selector)

---

## 1. Project Overview

This project implements an end-to-end **Skin Cancer Multi-class Classification & Screening System** designed for comprehensive dermatoscopic lesion analysis:

1. **Multi-class Identification (8 Disease Classes):** Detects Melanoma (MEL), Basal Cell Carcinoma (BCC), Squamous Cell Carcinoma (SCC), Actinic Keratoses (AK), Melanocytic Nevi (NV), Benign Keratosis (BKL), Vascular Lesions (VASC), and Dermatofibroma (DF).
2. **Two-Level Clinical Output:**
   - **Level 1 (Binary Screening):** Aggregated probability evaluation distinguishing `Cancer / Pre-Cancer (Malignant)` vs. `Non-Cancer (Benign)`.
   - **Level 2 (Sub-Type Breakdown):** Detailed probability distribution across specific sub-types within each clinical category.
3. **Multi-Model Benchmark:** Benchmarks three deep learning architectures (ResNet50, EfficientNet-B3, DenseNet121) under identical data partitions and class-weighting schemes.

Developed as part of the **CPE310 — Healthcare AI System** course.

---

## 2. Key Features

| Feature | Description |
|---|---|
| **Modular Architecture** | Clean separation of concerns: config, data, models, training, evaluation, and UI |
| **8 Disease Classes** | Expanded coverage including SCC (Squamous Cell Carcinoma) from ISIC 2019 |
| **Multi-Model Support** | Built-in training and comparison for ResNet50, EfficientNet-B3, and DenseNet121 |
| **Interactive Model Switching** | Switch between trained models live in the Gradio web UI without restarting |
| **Comparative Metrics Table** | Side-by-side performance comparison table rendered in terminal and on web UI |
| **Two-Level Output** | Binary screening decision accompanied by detailed sub-type differential probability |
| **Apple Silicon GPU Acceleration** | Full native acceleration via PyTorch MPS (Metal Performance Shaders) |
| **Class Imbalance Mitigation** | Balanced class-weighted Cross-Entropy Loss across all 8 classes |
| **Clinical Safety Assessment** | Rigorous evaluation of Cancer Sensitivity, Specificity, PPV, NPV, and False Negative Rate |
| **Professional Web UI** | Clean, formal Gradio interface with zero casual emojis |

---

## 3. Project Structure

```
Healthcare AI System/
├── config.py                     # Central configuration, paths, taxonomy, model registry
├── data.py                       # Data loading, dataset class, transforms, dataloaders
├── models.py                     # Model architectures (ResNet50, EfficientNet-B3, DenseNet121)
├── train.py                      # Training loop, early stopping, LR scheduling
├── evaluate.py                   # Evaluation metrics, confusion matrix, comparison table
├── ui.py                         # Gradio clinical web interface with model selector
├── main.py                       # CLI entry point (train, evaluate, demo)
│
├── model_resnet50.pth            # Trained ResNet50 checkpoint
├── model_efficientnet_b3.pth     # Trained EfficientNet-B3 checkpoint
├── model_densenet121.pth         # Trained DenseNet121 checkpoint
├── metrics_resnet50.json         # Evaluation metrics for ResNet50
├── metrics_efficientnet_b3.json  # Evaluation metrics for EfficientNet-B3
├── metrics_densenet121.json      # Evaluation metrics for DenseNet121
│
├── Dataset/
│   ├── HAM10000/                 # HAM10000 subset images and metadata
│   └── ISIC 2019/
│       ├── ISIC_2019_Training_Input/         # Full 25,331 training images
│       ├── ISIC_2019_Training_GroundTruth.csv # Ground truth one-hot labels
│       └── ISIC_2019_Training_Metadata.csv    # Clinical metadata
├── requirements.txt              # Project dependencies
├── README.md                     # Project documentation
└── venv/                         # Python Virtual Environment
```

---

## 4. Dataset: ISIC 2019 (Superset of HAM10000)

The **ISIC 2019** dataset (*International Skin Imaging Collaboration 2019 Challenge*) incorporates images from three major clinical sources: HAM10000, BCN_20000, and MSK.

- **Total Images:** 25,331 dermoscopic images (handles standard and `_downsampled` filenames)
- **Relationship with HAM10000:** HAM10000 (10,015 images) is a **strict subset** of ISIC 2019. Utilizing ISIC 2019 provides complete coverage of HAM10000 while adding 15,316 independent cases and the crucial SCC class.

---

## 5. Disease Taxonomy & Clinical Categories

| Index | Code | English Name | Thai Name | Category | Risk Level |
|:---:|:---:|---|---|:---:|:---:|
| 0 | **MEL** | Melanoma | มะเร็งเมลาโนมา | Cancer | Critical / High |
| 1 | **BCC** | Basal Cell Carcinoma | มะเร็งเซลล์ฐาน | Cancer | High |
| 2 | **SCC** | Squamous Cell Carcinoma | มะเร็งเซลล์สความัส | Cancer | High |
| 3 | **AK** | Actinic Keratoses / Bowen's | โรคผิวหนังก่อนมะเร็ง | Pre-Cancer | Moderate / Warning |
| 4 | **NV** | Melanocytic Nevi | ไฝและขี้แมลงวัน | Benign | Low |
| 5 | **BKL** | Benign Keratosis-like Lesions | รอยโรคกลุ่มเคราโตซิส | Benign | Low |
| 6 | **VASC** | Vascular Lesions | รอยโรคหลอดเลือด | Benign | Low |
| 7 | **DF** | Dermatofibroma | เนื้องอกใยผิวหนัง | Benign | Low |

---

## 6. Two-Level Classification Architecture

```
[Dermoscopic Image] 
        │
        ▼
[Selected Model: ResNet50 / EfficientNet-B3 / DenseNet121]
        │
        ▼ (Softmax)
[8 Class Probabilities: P(MEL), P(BCC), P(SCC), P(AK), P(NV), P(BKL), P(VASC), P(DF)]
        │
        ├─────────────────────────────────────┐
        ▼                                     ▼
[Level 1: Binary Screening]           [Level 2: Sub-Type Breakdown]
  P(Cancer) = Σ P(MEL, BCC, SCC, AK)    Cancer Subtypes:
  P(Benign) = Σ P(NV, BKL, VASC, DF)      P(MEL), P(BCC), P(SCC), P(AK)
                                        Benign Subtypes:
  Decision:                               P(NV), P(BKL), P(VASC), P(DF)
  - HIGH RISK (P(Cancer) ≥ 0.50)
  - LOW RISK  (P(Cancer) < 0.50)        Most Likely Specific Diagnosis
```

---

## 7. Multi-Model Architecture Comparison

All models employ transfer learning from ImageNet pre-trained weights, fine-tuning top feature blocks with a shared classification head architecture:

```
Linear(in_features, 512) -> ReLU -> BatchNorm1d(512) -> Dropout(0.3) -> Linear(512, 8)
```

| Model | Total Parameters | Trainable Parameters | Backbone Feature Extraction |
|---|---:|---:|---|
| **ResNet50** | 24,562,248 | 16,018,952 | Layer 4 + Custom Head |
| **EfficientNet-B3** | 11,488,304 | 1,384,968 | Last Stage (Block 8) + Head |
| **DenseNet121** | 7,483,784 | 2,688,008 | DenseBlock 4 + Head |

---

## 8. Technology Stack & Dependencies

- **Language:** Python 3.10+
- **Deep Learning Framework:** PyTorch & Torchvision
- **Machine Learning & Evaluation:** Scikit-learn, NumPy, Pandas
- **Image Processing:** Pillow (PIL)
- **Data Visualization:** Matplotlib (Agg backend)
- **Web Interface:** Gradio
- **Hardware Acceleration:** Apple Silicon MPS / CUDA

Install requirements:
```bash
pip install -r requirements.txt
```

---

## 9. End-to-End Pipeline

```
1. Central Configuration & Device Initialization (config.py)
2. ISIC 2019 Ground Truth Parsing & Image Path Resolution (data.py)
3. Stratified Train / Val / Test Partitioning (70% / 15% / 15%)
4. Class Weight Calculation for Imbalance Compensation
5. DataLoaders Creation with Augmentation Pipelines
6. Model Instantiation via Unified Registry (models.py)
7. Training Loop with Early Stopping & LR Scheduling (train.py)
8. Multi-Model Test Set Evaluation & Metrics Persistence (evaluate.py)
9. Cross-Model Benchmark Table Generation
10. Interactive Gradio Clinical Web Demo Deployment (ui.py)
```

---

## 10. Clinical Safety Evaluation

In clinical decision support, false negatives for malignant conditions represent the highest patient risk. The evaluation pipeline computes:

- **Per-Class Metrics:** Precision, Recall, F1-Score for each of the 8 classes.
- **Binary Screening Metrics:**
  - **Cancer Sensitivity (Recall):** Percentage of malignant/pre-cancerous cases correctly flagged (target ≥ 90%).
  - **Cancer Specificity:** Percentage of benign lesions correctly classified.
  - **Positive Predictive Value (PPV) & Negative Predictive Value (NPV).**
  - **False Negative Rate (FNR):** Proportion of missed malignancies (target ≤ 10%).

---

## 11. Installation & Usage

### Setup Environment
```bash
cd "Healthcare AI System"
source venv/bin/activate
pip install -r requirements.txt
```

### Run Options

#### 1. Full Multi-Model Training & Benchmark
```bash
python main.py
```
Sequentially trains ResNet50, EfficientNet-B3, and DenseNet121, evaluates each on the test split, prints comparison table, and launches the Gradio web UI.

#### 2. Train a Specific Model
```bash
python main.py --model ResNet50
python main.py --model EfficientNet-B3
python main.py --model DenseNet121
```

#### 3. Launch Gradio Demo Directly (Using Pre-Trained Weights)
```bash
python main.py --demo
```
Loads available checkpoints and launches the interactive web demo at `http://127.0.0.1:7860`.

#### 4. Skip Training, Run Evaluation & Demo
```bash
python main.py --skip-train
```
Loads existing checkpoints, runs test set evaluation, updates comparison metrics, and launches the web demo.

---

## 12. Platform & Environment Configuration

- **macOS Apple Silicon:** Automatically uses `mps` device when available.
- **SSL Certificate Bypass:** Handled in `config.py` to prevent torchvision weight download errors on macOS.
- **DataLoader Workers:** Configured to `num_workers=2` for macOS fork safety.
- **Headless Plotting & Font Cache:** Configured with `MPLCONFIGDIR` and `matplotlib.use("Agg")` to ensure fast initialization and avoid macOS permission blocks.

---

## 13. Gradio Clinical Web Interface with Model Selector

The web demo provides a clinical-grade interface:
- **Model Selector:** Dropdown menu allowing instant switching between trained models with accuracy metrics displayed.
- **Model Benchmark Panel:** Embedded side-by-side table displaying Accuracy, Macro F1, Sensitivity, Specificity, FNR, and parameter counts.
- **Risk Status Banner:** Displays `HIGH RISK` or `LOW RISK` based on aggregate cancer probability with active model citation.
- **Screening Overview:** Dual progress bars comparing Cancer / Pre-Cancer vs Benign probabilities.
- **Most Likely Diagnosis:** Primary diagnosis with bilingual (English & Thai) descriptions.
- **Sub-Type Probability Analysis:** Differential tables for both Cancer and Benign groups.
- **Clinical Recommendations:** Evidence-based medical recommendations in English and Thai.
