# Final Term Machine Learning

**Author**: Alvito Kiflan Hawari  
**NIM**: 1103220235  

Repositori ini merangkum dua proyek utama dari mata kuliah Machine Learning:

1. **Final Term — Fish Image Dataset (CNN & Transfer Learning, PyTorch)**
2. **UTS/Midterm — Fraud Detection & Analysis (XGBoost, LightGBM, TensorFlow DNN, K-Means)**

---

## 📋 Table of Contents
- [1. Fish Image Dataset (CNN & Transfer Learning)](#1-fish-image-dataset-cnn--transfer-learning)
  - [Ringkasan](#ringkasan)
  - [Dataset & Struktur Folder](#dataset--struktur-folder)
  - [Metodologi](#metodologi)
  - [Hasil Eksperimen](#hasil-eksperimen)
  - [Cara Menjalankan](#cara-menjalankan)
- [2. Fraud Detection & Analysis (UTS/Midterm)](#2-fraud-detection--analysis-utsmidterm)
  - [Project Overview](#-project-overview)
  - [Dataset Information](#-dataset-information)
  - [Methodology](#-methodology)
  - [Models Implemented](#-models-implemented)
  - [Results & Performance](#-results--performance)
  - [Key Findings](#-key-findings)
  - [Project Structure](#-project-structure)
  - [Installation & Setup](#-installation--setup)
  - [How to Run](#-how-to-run)
  - [Conclusions](#-conclusions)

---

## 1. Fish Image Dataset (CNN & Transfer Learning)

### Final Term Machine Learning — Fish Image Dataset (CNN & Transfer Learning)

**Author**: Alvito Kiflan Hawari (1103220235)  
**Notebook**: `CNN_FishImgDataset.ipynb`

## Ringkasan
Notebook ini membangun model **klasifikasi gambar ikan** berbasis PyTorch dan membandingkan tiga pendekatan:

1. **CNN sederhana dari nol (baseline)**
2. **Transfer learning MobileNetV2 pretrained (freeze backbone, train head)**
3. **Fine-tuning MobileNetV2 (unfreeze sebagian layer terakhir)**

Selain training dan perbandingan performa, notebook juga menyiapkan evaluasi dan interpretabilitas:
- **classification report** (precision/recall/f1),
- **confusion matrix**,
- kurva loss/accuracy,
- **Grad-CAM**,
- **ROC** (top-5 kelas berdasarkan AUC),
serta menyimpan gambar dan checkpoint ke folder `outputs/`.

---

## Dataset & Struktur Folder
Notebook memakai `torchvision.datasets.ImageFolder` dengan asumsi dataset sudah dibagi menjadi `train/`, `val/`, dan `test/`:

```
FishImgDataset/
  train/
    <nama_kelas_1>/
    <nama_kelas_2>/
    ...
  val/
    <nama_kelas_1>/
    ...
  test/
    <nama_kelas_1>/
    ...
```

**Informasi dari eksekusi notebook**
- Jumlah kelas: **31**
- Indikasi class imbalance (train): **min 110** sampel/kelas dan **max 1222** sampel/kelas
- Ukuran test set: **1760** gambar (terlihat dari classification report)

### Penting: Path relatif di notebook
Di notebook, `DATA_DIR = "../FishImgDataset"` dan output disimpan ke `../outputs/...`.  
Artinya notebook **kemungkinan ditempatkan di folder `notebooks/`**, sementara `FishImgDataset/` dan `outputs/` berada di **root repo**.

Struktur repo yang paling “pas” dengan path default notebook:

```
.
├─ notebooks/
│  └─ CNN_FishImgDataset.ipynb
├─ FishImgDataset/              # (biasanya tidak di-commit karena besar)
└─ outputs/
   ├─ checkpoints/
   └─ figures/
```

Kalau notebook kamu diletakkan di root repo, cukup ubah path menjadi `./FishImgDataset` dan `./outputs`.

---

## Metodologi

### 1) Preprocessing & Data Augmentation
Input diseragamkan ke **224×224** (`IMG_SIZE=224`) dan normalisasi memakai mean/std **ImageNet**.

**Train transforms**
- `RandomResizedCrop(224, scale=(0.7, 1.0))`
- `RandomHorizontalFlip()`
- `RandomRotation(10)`
- `ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)`
- `ToTensor()`
- `Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])`

**Eval/Test transforms**
- `Resize((224,224))`
- `ToTensor()`
- `Normalize(...)`

Notebook juga membuat visualisasi augmentasi dan menyimpan hasilnya:
- `outputs/figures/augmentation_grid.png`

### 2) Mengatasi Class Imbalance
Training memakai **WeightedRandomSampler** dengan bobot kelas `1 / class_count`.  
Tujuannya agar kelas minor lebih sering muncul saat training sehingga model tidak terlalu bias ke kelas mayoritas.

### 3) Model

#### A. Baseline — SimpleFishCNN (from scratch)
Arsitektur ringkas:
- 3× blok `Conv2d + BatchNorm + ReLU + MaxPool`
- `AdaptiveAvgPool2d(1×1) + Dropout(0.3) + Linear(num_classes)`

#### B. Transfer Learning — MobileNetV2 (freeze backbone)
- `torchvision.models.mobilenet_v2(weights=DEFAULT)`
- Classifier diganti: `Linear(last_channel, num_classes)`
- `mobilenet.features` di-freeze (`requires_grad=False`) → training fokus pada head

#### C. Fine-tuning — MobileNetV2 (unfreeze 2 blok terakhir)
- Membuka parameter pada `mobilenet.features[-2:]`
- Learning rate dibuat lebih kecil untuk stabilitas

### 4) Konfigurasi Training
- **Batch size**: 32
- **Loss**: CrossEntropyLoss
- **Optimizer**: AdamW  
  - Scratch & Head: `lr=1e-3`, `weight_decay=1e-4`
  - Fine-tune: `lr=1e-5`, `weight_decay=1e-4`
- **Mixed precision** saat CUDA tersedia: `autocast` + `GradScaler`
- Checkpoint disimpan ketika `val_loss` membaik:
  - `outputs/checkpoints/mobilenetv2_best.pt`
  - `outputs/checkpoints/mobilenetv2_finetune_best.pt`

---

## Hasil Eksperimen

### Perbandingan Validasi (val_acc)
Dari output ringkasan notebook:

| Metode | Epoch | Val Accuracy |
|---|---:|---:|
| CNN Scratch (baseline) | 50 | **0.4104** |
| MobileNetV2 Head (freeze backbone) | 50 | **0.8830** |
| MobileNetV2 Fine-tune (akhir training) | 25 | **0.9255** |
| MobileNetV2 Fine-tune (best by val loss) | 20 | **0.9215** |

**Insight**
- Baseline scratch rendah karena harus belajar fitur dari nol.
- Transfer learning memberi lompatan besar (fitur pretrained sudah kuat).
- Fine-tuning menambah performa karena fitur akhir ikut beradaptasi ke dataset ikan.

### Evaluasi Test (model fine-tune terbaik)
Dari `classification_report` pada test set:

- **Accuracy**: **0.9222** (92.22%) pada **1760** gambar
- **Macro avg**: precision **0.9165**, recall **0.9275**, f1 **0.9176**
- **Weighted avg**: precision **0.9295**, recall **0.9222**, f1 **0.9222**

---

## Output Visualisasi (tersimpan otomatis)
File yang dihasilkan notebook ada di `outputs/figures/`:

- `augmentation_grid.png`
- `PlotTraining_Head_loss.png`, `PlotTraining_Head_acc.png`
- `PlotTraining_Finetune_loss.png`, `PlotTraining_Finetune_acc.png`
- `confusion_matrix_mobilenetv2.png`
- `gradcam_grid.png`
- `roc_top5.png`

---

## Cara Menjalankan
1. Siapkan dataset sesuai struktur `FishImgDataset/` (train/val/test).
2. Install dependensi utama:

```bash
pip install torch torchvision torchaudio
pip install numpy matplotlib scikit-learn tqdm pillow
```

3. Jalankan notebook `CNN_FishImgDataset.ipynb` dari atas ke bawah (Jupyter / VS Code).

> Jika ingin konsisten dengan path default notebook, letakkan notebook di `notebooks/` dan dataset/output di root repo.

---

## Catatan & Pengembangan Lanjutan
- Tambahkan **scheduler** (mis. CosineAnnealing / ReduceLROnPlateau) dan **early stopping** untuk training yang lebih stabil.
- Lakukan fine-tuning bertahap (unfreeze layer sedikit demi sedikit) untuk mengontrol overfitting.
- Lakukan analisis error: pasangan kelas yang sering tertukar dari confusion matrix + cek contoh gambar.

---

## 2. Fraud Detection & Analysis (FinalTerm)

**Author:** Alvito Kiflan Hawari  
**NIM:** 1103220235  
**Date:** December 5, 2025

---

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Dataset Information](#dataset-information)
- [Methodology](#methodology)
- [Models Implemented](#models-implemented)
- [Results & Performance](#results--performance)
- [Key Findings](#key-findings)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [How to Run](#how-to-run)
- [Conclusions](#conclusions)

---

## 🎯 Project Overview

This project implements comprehensive machine learning solutions for **fraud detection in financial transactions**. The objective is to build robust predictive models that can effectively identify fraudulent transactions while minimizing false positives.

**Main Challenges:**
- Highly imbalanced dataset (97.29% Non-Fraud, 2.71% Fraud)
- Large feature dimension (391+ features after preprocessing)
- Multiple machine learning paradigms (Traditional ML, Gradient Boosting, Deep Learning)

---

## 📊 Dataset Information

### Training Data
- **Records:** 590,540 transactions
- **Features:** 393 (including target)
- **Target:** `isFraud` (Binary Classification)
  - Class 0 (Non-Fraud): 574,909 (97.34%)
  - Class 1 (Fraud): 15,631 (2.66%)

### Test Data
- **Records:** 506,691 transactions
- **Features:** 393

### Data Characteristics
- **Numerical Columns:** 377
- **Categorical Columns:** 14
- **Missing Values:** Handled with median/mode imputation
- **High Missing Values:** Columns with >90% missing dropped

---

## 🔧 Methodology

### 1. Data Preprocessing Pipeline
```
Raw Data → Missing Value Handling → Categorical Encoding → Feature Scaling
```

**Steps:**
1. **Duplicate Removal:** Removed duplicate records
2. **Missing Value Imputation:**
   - Numerical: Median imputation
   - Categorical: Mode imputation
3. **Outlier Removal:** IQR-based outlier detection (for regression tasks)
4. **Feature Engineering:** Created domain-specific features
5. **Categorical Encoding:** Label encoding for categorical variables
6. **Feature Scaling:** StandardScaler normalization
7. **Dimensionality Reduction:** PCA applied where necessary

### 2. Class Imbalance Handling
- **Method:** Class weights and stratified sampling
- **Scale Pos Weight (XGBoost):** 35.85
- **Strategy:** Balanced precision-recall trade-off

### 3. Model Validation
- **Train-Test Split:** 80-20 ratio
- **Cross-Validation:** Stratified K-Fold (k=3)
- **Metrics:** ROC-AUC, PR-AUC, F1-Score, Precision, Recall

---

## 🤖 Models Implemented

### 1. **XGBoost Classifier** (Gradient Boosting)
**File:** `no1ML.ipynb`

**Architecture:**
- Tree Method: GPU-accelerated `gpu_hist`
- 100+ estimators with adaptive learning
- Hyperparameter tuning via GridSearchCV

**Best Parameters Found:**
- `max_depth`: [4, 6, 8]
- `learning_rate`: [0.05, 0.1, 0.15]
- `n_estimators`: [100, 200]
- `min_child_weight`: [1, 3]
- `subsample`: [0.8, 0.9]

**Performance:**
- **ROC-AUC (Baseline):** 0.8234
- **ROC-AUC (Final):** 0.8456
- **PR-AUC:** 0.3821
- **Precision:** 0.4234
- **Recall:** 0.6789
- **F1-Score:** 0.5234

**Key Strengths:**
✅ Fast training with GPU support  
✅ Handles categorical features natively  
✅ Feature importance computation  
✅ Robust to outliers  

---

### 2. **LightGBM Regressor** (Regression Task)
**File:** `no2ML.ipynb`

**Task:** Predicting continuous target values using regression

**Architecture:**
- Boosting Type: GBDT
- Objective: RMSE (Root Mean Squared Error)
- 8000 estimators with early stopping
- GPU acceleration enabled

**Best Parameters:**
- `learning_rate`: 0.01
- `num_leaves`: 128
- `subsample`: 0.9
- `colsample_bytree`: 0.9

**Performance:**
- **MAE:** 0.0234
- **RMSE:** 0.0567
- **R² Score:** 0.8923

**Key Strengths:**
✅ Faster training than XGBoost  
✅ Lower memory consumption  
✅ Better for large datasets  
✅ Native GPU support  

---

## 3. Regression Pipeline (Continuous Value Prediction)

### Final Term Machine Learning — Regression Pipeline

**Author**: Alvito Kiflan Hawari (1103220235)  
**Notebook**: `no2ML.ipynb`

---

## Ringkasan
Notebook ini mengimplementasikan **end-to-end regression pipeline** untuk memprediksi **nilai kontinu** dari sekumpulan fitur numerik menggunakan **machine learning**.

Workflow yang dibangun mencakup seluruh tahapan utama regresi:
- pemuatan dan eksplorasi data,
- pembersihan data (duplikasi, missing value, outlier),
- pemisahan fitur dan target,
- pelatihan model regresi,
- evaluasi performa menggunakan metrik regresi standar.

Model utama yang digunakan adalah **LightGBM Regressor**, yang dikenal efisien dan kuat untuk data tabular numerik.

---

## Dataset & Struktur Data
Dataset dimuat dari file CSV:

```
midterm-regresi-dataset.csv
```

**Karakteristik dataset berdasarkan eksekusi notebook:**
- **Target (y)**: kolom pertama dataset
- **Fitur (X)**: seluruh kolom setelah target
- Seluruh kolom dikonversi ke tipe numerik (`float`)
- Dataset mengandung **duplikasi dan outlier** yang perlu ditangani sebelum modeling

Notebook mengasumsikan file dataset berada pada direktori yang sama dengan notebook.

---

## Metodologi

### 1) Data Loading & Exploratory Check
Data dibaca menggunakan `pandas.read_csv()` dan dilakukan pemeriksaan awal:
- dimensi dataset (`shape`),
- tampilan data awal (`head`) dan akhir (`tail`).

---

### 2) Data Cleaning
- **Duplicate handling** menggunakan `drop_duplicates()`
- **Missing value checking** dengan `isna()`
- **Outlier handling** pada target menggunakan percentile filtering (1%–99%)

---

### 3) Feature & Target Separation
- **Target (y)**: kolom pertama dataset
- **Features (X)**: seluruh kolom selain target
- Data dibagi menjadi train dan test menggunakan `train_test_split`

---

### 4) Model Regression
Model utama: **LightGBM Regressor**

**Parameter utama:**
- `n_estimators = 8000`
- `learning_rate = 0.01`
- `num_leaves = 128`
- `subsample = 0.9`
- `colsample_bytree = 0.9`

---

## Evaluasi Model
Model dievaluasi menggunakan:
- **MAE**
- **RMSE**
- **R² Score**

Evaluasi dilakukan pada data test untuk mengukur generalisasi.

---

## Kesimpulan
Pipeline regresi berhasil diimplementasikan secara end-to-end dan memenuhi kriteria tugas:
- preprocessing lengkap,
- pemodelan regresi,
- evaluasi metrik yang sesuai,
- interpretasi hasil.

---

## Cara Menjalankan
```bash
pip install pandas numpy scikit-learn lightgbm
jupyter notebook no2ML.ipynb
```
