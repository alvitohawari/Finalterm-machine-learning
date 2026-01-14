# Regression Pipeline (Continuous Value Prediction)
Final Term Machine Learning — Regression Pipeline  
Author: **Alvito Kiflan Hawari (1103220235)**  
Notebook: **tugas_regresi_end_to_end (1).ipynb**

## Ringkasan
Notebook ini mengimplementasikan *end-to-end regression pipeline* untuk memprediksi nilai kontinu (target numerik) dari sekumpulan fitur numerik menggunakan machine learning.

Pipeline mencakup tahapan utama regresi:

- pemuatan dataset (unduh via Google Drive menggunakan `gdown`),
- pengecekan awal ukuran & dimensi data,
- pembersihan data (duplikasi, missing value pada target),
- *outlier handling* pada fitur menggunakan *quantile clipping*,
- pembagian data train/test,
- pemodelan dengan **XGBoost Regressor (XGBRegressor)** dalam *scikit-learn Pipeline*,
- *feature selection* (SelectKBest),
- *hyperparameter tuning* dengan RandomizedSearchCV,
- evaluasi menggunakan metrik regresi standar + visualisasi.

## Dataset & Struktur Data
Dataset diunduh dari Google Drive dan disimpan sebagai:

- `midterm-regresi-dataset.csv`

Karakteristik dataset berdasarkan implementasi notebook:

- Dataset **tidak memiliki header**.
- **Target (y)**: kolom pertama (index 0). Di notebook diperlakukan sebagai numerik (`int`) dan dicontohkan sebagai “tahun / release year”.
- **Fitur (X)**: seluruh kolom setelah target (index 1..akhir) dan dikonversi ke `float`.
- Duplikasi dihapus dengan `drop_duplicates()`.
- Missing value pada target dibuang (baris dengan `y` null), sementara missing pada fitur ditangani saat training dengan imputasi median.

> Catatan lokasi file: di notebook `file_path` ditulis `'/content/midterm-regresi-dataset.csv'` (format umum Google Colab). Jika menjalankan di lokal, ubah `file_path` ke path di komputer Anda.

## Metodologi
### 1) Data Loading & Exploratory Check
- Unduh dataset menggunakan `gdown`.
- Load CSV dengan `pandas.read_csv()`.
- Cek dimensi dataset (`df.shape`) dan ukuran file.

### 2) Data Cleaning
- **Duplicate handling**: `df.drop_duplicates()`
- **Missing target**: buang baris dengan `y` kosong (`mask = y.notna()`)

### 3) Train/Test Split
- Split data menggunakan `train_test_split(test_size=0.2, random_state=42)`.

### 4) Outlier Handling (Feature Clipping)
Outlier pada fitur ditangani dengan *quantile clipping* berbasis statistik data train:

- batas bawah: quantile 1% (`0.01`)
- batas atas : quantile 99% (`0.99`)

Clipping diterapkan ke `X_train` lalu batas yang sama digunakan untuk `X_test`.

### 5) Pipeline + Feature Selection + Model (XGBoost)
Model dibangun sebagai pipeline:

1. **SimpleImputer(strategy="median")** — imputasi missing value fitur  
2. **SelectKBest(f_regression)** — seleksi fitur terbaik (nilai `k` dituning)  
3. **XGBRegressor(objective="reg:squarederror")** — model regresi XGBoost  
   - `tree_method="hist"` dan `predictor="cpu_predictor"` (fallback CPU)

### 6) Hyperparameter Tuning
- Validasi silang: `KFold(n_splits=5, shuffle=True, random_state=42)`
- Tuning: `RandomizedSearchCV`
  - `n_iter=10`
  - `scoring="neg_root_mean_squared_error"`
  - parameter yang dituning mencakup: `select__k`, `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`, `min_child_weight`, `reg_alpha`, `reg_lambda`.

## Evaluasi Model
Evaluasi dilakukan pada *test set* dengan metrik:

- **MSE** (Mean Squared Error)
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **R² Score**

Notebook juga membuat plot:
- **Actual vs Predicted (Test)**
- **Residual Plot (Test)**

## Cara Menjalankan
### 1) Siapkan environment
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Jalankan notebook
```bash
jupyter notebook "tugas_regresi_end_to_end (1).ipynb"
```

### 3) Pastikan dataset tersedia
- Jika menjalankan di **Google Colab**, path `'/content/...'` umumnya sudah sesuai.
- Jika menjalankan di **lokal**, ubah `file_path` ke lokasi dataset di komputer Anda.

## Catatan
- Notebook mengimpor `torch` untuk kebutuhan pengecekan lingkungan (GPU/CPU) pada cell pipeline. Jika Anda ingin dependensi lebih ringan, Anda dapat menghapus `import torch` di notebook dan menghapus `torch` dari `requirements.txt`.
