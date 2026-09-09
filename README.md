# Skin Cancer Multi-Class Classification & Screening System
## Clinical Decision Support Tool — ISIC 2019 Dataset (8 Classes with Binary Grouping)

An AI-driven **8-Class Disease Classification & Two-Level Cancer Screening** system for dermatoscopic skin lesion analysis using the ISIC 2019 dataset (incorporating HAM10000), powered by Deep Learning (**ResNet50 Transfer Learning**).

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
- [7. Technology Stack & Dependencies](#7-technology-stack--dependencies)
- [8. Model Architecture](#8-model-architecture)
- [9. End-to-End Pipeline](#9-end-to-end-pipeline)
- [10. Clinical Safety Evaluation](#10-clinical-safety-evaluation)
- [11. Installation & Usage](#11-installation--usage)
- [12. Platform & Environment Configuration](#12-platform--environment-configuration)
- [13. Gradio Clinical Web Interface](#13-gradio-clinical-web-interface)

---

## 1. Project Overview

This project implements an end-to-end **Skin Cancer Multi-class Classification & Screening System** designed for comprehensive dermatoscopic lesion analysis:

1. **Multi-class Identification (8 Disease Classes):** Detects Melanoma (MEL), Basal Cell Carcinoma (BCC), Squamous Cell Carcinoma (SCC), Actinic Keratoses (AK), Melanocytic Nevi (NV), Benign Keratosis (BKL), Vascular Lesions (VASC), and Dermatofibroma (DF).
2. **Two-Level Clinical Output:**
   - **Level 1 (Binary Screening):** Aggregated probability evaluation distinguishing `Cancer / Pre-Cancer (Malignant)` vs. `Non-Cancer (Benign)`.
   - **Level 2 (Sub-Type Breakdown):** Detailed probability distribution across specific sub-types within each clinical category ("If cancer, what type? If benign, what type?").

The system utilizes **Transfer Learning** on a **ResNet50** architecture pre-trained on ImageNet V2, fine-tuned specifically for dermoscopic lesion imagery. Developed as part of the **CPE310 — Healthcare AI System** course.

---

## 2. Key Features

| Feature | Description |
|---------|-------------|
| **8 Disease Classes** | Expanded coverage including SCC (Squamous Cell Carcinoma) from ISIC 2019 |
| **Two-Level Output** | Binary screening decision accompanied by detailed sub-type differential probability |
| **Transfer Learning** | ResNet50 backbone pre-trained on ImageNet V2 with fine-tuned residual blocks |
| **Apple Silicon GPU Acceleration** | Full native acceleration via PyTorch MPS (Metal Performance Shaders) |
| **Data Augmentation** | Robust pipeline: Random Horizontal/Vertical Flip, Rotation, ColorJitter, Affine |
| **Class Imbalance Mitigation** | Computed balanced class-weighted Cross-Entropy Loss across all 8 classes |
| **Early Stopping & Scheduling** | Macro F1-score tracking with `ReduceLROnPlateau` and patience-based stopping |
| **Clinical Safety Assessment** | Rigorous evaluation of Cancer Sensitivity (Recall), Specificity, PPV, NPV, and False Negative Rate |
| **Professional Web UI** | Clean, formal Gradio interface with dual-panel layout and zero casual emojis |

---

## 3. Project Structure

```
Healthcare AI System/
├── Dataset/
│   ├── HAM10000/
│   │   ├── HAM10000_images_part_1/        # HAM10000 subset images part 1
│   │   ├── HAM10000_images_part_2/        # HAM10000 subset images part 2
│   │   └── HAM10000_metadata.csv          # Metadata (10,015 records)
│   └── ISIC 2019/
│       ├── ISIC_2019_Training_Input/         # Full 25,331 training images
│       ├── ISIC_2019_Training_GroundTruth.csv # Ground truth one-hot labels
│       └── ISIC_2019_Training_Metadata.csv    # Metadata (age, sex, site)
├── skin_lesion_classifier.py              # Single-file production pipeline
├── best_multiclass_cancer_model.pth       # Trained 8-class model checkpoint
├── multiclass_confusion_matrix.png        # 8×8 confusion matrix with group demarcation
├── requirements.txt                       # Project dependencies
├── README.md                              # Project documentation
└── venv/                                  # Python Virtual Environment
```

---

## 4. Dataset: ISIC 2019 (Superset of HAM10000)

The **ISIC 2019** dataset (*International Skin Imaging Collaboration 2019 Challenge*) incorporates images from three major clinical sources: HAM10000, BCN_20000, and MSK.

- **Total Images:** 25,331 dermoscopic images (handles standard and `_downsampled` filenames)
- **Relationship with HAM10000:** HAM10000 (10,015 images) is a **strict subset** of ISIC 2019. Utilizing ISIC 2019 provides complete coverage of HAM10000 while adding 15,316 independent cases and the crucial SCC class.

---

## 5. Disease Taxonomy & Clinical Categories

### 8-Class Clinical Distribution

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

The system uses a single unified multi-class architecture followed by structured post-processing:

```
[Dermoscopic Image] 
        │
        ▼
[ResNet50 Backbone + Custom Head]
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

## 7. Technology Stack & Dependencies

- **Language:** Python 3.10+
- **Deep Learning Framework:** PyTorch & Torchvision
- **Machine Learning & Evaluation:** Scikit-learn, NumPy, Pandas
- **Image Processing:** Pillow (PIL)
- **Data Visualization:** Matplotlib (Agg backend)
- **Web Interface:** Gradio
- **System Acceleration:** Apple Silicon Metal Performance Shaders (MPS) / CUDA

Install requirements:
```bash
pip install -r requirements.txt
```

---

## 8. Model Architecture

- **Backbone:** ResNet50 (pre-trained on ImageNet1K V2)
- **Transfer Learning Strategy:**
  - Frozen layers: `conv1`, `bn1`, `layer1`, `layer2`, `layer3`
  - Trainable layers: `layer4` (high-level feature extraction) and custom classification head
- **Classification Head:**
  ```python
  nn.Sequential(
      nn.Linear(2048, 512),
      nn.ReLU(inplace=True),
      nn.BatchNorm1d(512),
      nn.Dropout(p=0.3),
      nn.Linear(512, 8) # 8 Classes
  )
  ```

---

## 9. End-to-End Pipeline

```
1. Data Ingestion & Ground Truth Parsing (One-hot to index)
2. Image Resolution (Resolves both standard and _downsampled files)
3. Stratified Train / Val / Test Split (70% / 15% / 15%)
4. Class Weight Calculation for Imbalance Compensation
5. Data Augmentation & PyTorch DataLoader Creation (num_workers=2)
6. Model Instantiation (ResNet50 + Custom Head)
7. Training Loop (AdamW, CrossEntropyLoss with weights, ReduceLROnPlateau)
8. Early Stopping Check (Macro F1 on Validation set)
9. Test Set Evaluation & 8×8 Confusion Matrix Generation
10. Gradio Clinical Web Demo Deployment
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
# Clone or navigate to workspace
cd "Healthcare AI System"

# Activate Virtual Environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run Options

#### 1. Full Training & Evaluation
```bash
python skin_lesion_classifier.py
```
Performs complete data loading, model training, evaluation on test set, confusion matrix generation, and automatically launches the Gradio web demo.

#### 2. Launch Gradio Demo Directly (Using Pre-Trained Weights)
```bash
python skin_lesion_classifier.py --demo
```
Loads `best_multiclass_cancer_model.pth` and launches the web interface at `http://127.0.0.1:7860`.

#### 3. Skip Training, Run Evaluation & Demo
```bash
python skin_lesion_classifier.py --skip-train
```
Loads existing model checkpoint, runs full evaluation on test split, generates matrix, and launches demo.

---

## 12. Platform & Environment Configuration

- **macOS Apple Silicon:** Automatically uses `mps` device when available.
- **SSL Certificate Bypass:** Included in code to prevent torchvision pre-trained weight download failures on macOS.
- **DataLoader Workers:** Configured to `num_workers=2` to eliminate macOS fork-based multiprocessing issues.
- **Headless Plotting:** Configured with `matplotlib.use("Agg")` to prevent GUI thread conflicts on macOS.

---

## 13. Gradio Clinical Web Interface

The clinical web demo presents a clean, medical-grade interface without casual icons or emojis:
- **Risk Status Banner:** Displays `HIGH RISK` or `LOW RISK` based on aggregate cancer probability.
- **Screening Overview:** Visual comparison bars between Cancer / Pre-Cancer vs Benign groups.
- **Most Likely Diagnosis:** Displays primary diagnosis with English & Thai clinical explanations.
- **Sub-Type Probability Analysis:**
  - Group 1: Sub-type distribution across MEL, BCC, SCC, and AK.
  - Group 2: Sub-type distribution across NV, BKL, VASC, and DF.
- **Clinical Recommendations:** Tailored medical guidance in both English and Thai.
