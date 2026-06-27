# HR Analytics - Employee Attrition Prediction

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange)](https://www.tensorflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100-green)](https://fastapi.tiangolo.com/)

---

## 📋 Deskripsi Proyek

Proyek ini bertujuan untuk memprediksi **attrition karyawan** (apakah karyawan akan keluar dari perusahaan atau tetap) menggunakan metode **Deep Learning**. Dataset yang digunakan adalah **HR Data for Analytics** dari Kaggle.

Aplikasi ini dilengkapi dengan:
- **GUI berbasis Web** untuk input data karyawan dan prediksi
- **Dashboard Eksplorasi Data** untuk melihat statistik dataset
- **Dashboard Hasil Metode** untuk membandingkan performa model

---

## 🎯 Metode Deep Learning yang Digunakan

| Kombinasi | Arsitektur | Loss Function | Penanganan Imbalance |
|-----------|------------|---------------|---------------------|
| **MLP_NoHandling** | MLP 4-layer (256-128-64-32) + BN + Dropout | Focal Loss (γ=2.0, α=0.25) | None |
| **MLP_SMOTE_ENN** | MLP 4-layer (256-128-64-32) + BN + Dropout | Focal Loss (γ=2.0, α=0.25) | SMOTE-ENN |
| **MLP_ClassWeight** | MLP 4-layer (256-128-64-32) + BN + Dropout | Focal Loss (γ=2.0, α=0.5) | Class Weight + Focal α=0.5 |
| **TabNet_NoHandling** | TabNet (n_d=64, n_a=64, steps=5) | Cross-Entropy (default) | None |
| **TabNet_SMOTE_ENN** | TabNet (n_d=64, n_a=64, steps=5) | Cross-Entropy (default) | SMOTE-ENN |
| **TabNet_ClassWeight** | TabNet (n_d=64, n_a=64, steps=5) | Cross-Entropy (default) | Duplikasi Fraud 3x |

---

## 📊 Hasil Model Terbaik

| Ranking | Kombinasi | F1-Score | Accuracy | Precision | Recall |
|---------|-----------|----------|----------|-----------|--------|
| **#1** | **TabNet_NoHandling** | **0.9302** | **0.9775** | **0.9574** | **0.9045** |
| #2 | MLP_ClassWeight | 0.9278 | 0.9767 | 0.9524 | 0.9045 |
| #3 | MLP_NoHandling | 0.9243 | 0.9758 | 0.9620 | 0.8894 |
| #4 | MLP_SMOTE_ENN | 0.9023 | 0.9675 | 0.9000 | 0.9045 |
| #5 | TabNet_ClassWeight | 0.8916 | 0.9633 | 0.8744 | 0.9095 |
| #6 | TabNet_SMOTE_ENN | 0.8438 | 0.9442 | 0.7870 | 0.9095 |

**Model Terbaik: TabNet_NoHandling (F1-Score: 0.9302, Accuracy: 0.9775)**

---

## 🏗️ Arsitektur TabNet (TabNet_NoHandling — Model Terbaik)

```
Input Features (9 fitur)
    ↓
Feature Transformer (shared)
    ↓
Attentive Transformer  →  Sparse Feature Mask
    ↓
Feature Transformer (decision)
    ↓
  ┌──────────────────────────────────┐
  │  Aggregation (steps=5)           │
  │  n_d = 64, n_a = 64              │
  └──────────────────────────────────┘
    ↓
Output Layer (Softmax)
```

**Loss Function**: Cross-Entropy (default pytorch_tabnet)

**Parameter Kunci**:
| Parameter | Nilai |
|-----------|-------|
| n_d (decision width) | 64 |
| n_a (attention width) | 64 |
| n_steps | 5 |
| gamma (relaxation) | 1.5 |
| lambda_sparse | 1e-3 |
| Optimizer | Adam (lr=2e-2) |
| Batch Size | 256 |
| Scheduler | ReduceLROnPlateau |
| Early Stopping | patience=20

---

## 📂 Struktur Folder

```
hr-analytics-deeplearning/
│
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app + routing + database
│   ├── ml_model.py          # Load model, preprocessing, prediksi
│   ├── templates/
│   │   ├── index.html
│   │   ├── explore.html
│   │   └── results.html
│   └── static/
│       └── style.css
│
├── models/
│   ├── best_model.h5
│   ├── tabnet_best_model/
│   ├── scaler.pkl
│   ├── encoders.pkl
│   ├── feature_info.json
│   ├── comparison_results.csv
│   └── comparison_plot.png
│
├── data/
│   └── HR_comma_sep.csv
│
├── notebooks/
│   └── train_models.ipynb
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🚀 Cara Menjalankan

### 1. Clone Repository

```bash
git clone https://github.com/username/hr-analytics-deeplearning.git
cd hr-analytics-deeplearning
```

### 2. Buat Virtual Environment

```bash
# Mac/Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Library

```bash
pip install -r requirements.txt
```

### 4. Download Dataset

Download `HR_comma_sep.csv` dari [Kaggle](https://www.kaggle.com/datasets/jacksonchou/hr-data-for-analytics) dan simpan di folder `data/`.

### 5. Jalankan Training (Opsional)

```bash
jupyter notebook notebooks/train_models.ipynb
```

### 6. Jalankan Aplikasi Web

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 7. Buka Browser

```
http://127.0.0.1:8000
```

---

## 📸 Screenshot GUI

### Halaman Utama (Prediksi)

```
┌─────────────────────────────────────────────────────────────────────┐
│  🔍 HR Attrition Predictor                                         │
│  Model: TabNet_NoHandling  F1: 0.9302  Acc: 0.9775              │
├─────────────────────────────────────────────────────────────────────┤
│  📊 Statistik: Total Karyawan: 0 | Tetap: 0 | Keluar: 0 | Rate: 0%│
├─────────────────────────────────────────────────────────────────────┤
│  📝 Input Data Karyawan                                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                │
│  │ Satisfaction│  │ Last Eval   │  │ Projects    │                │
│  │ [0.8]        │  │ [0.9]       │  │ [5]         │                │
│  └─────────────┘  └─────────────┘  └─────────────┘                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                │
│  │ Hours       │  │ Years       │  │ Department  │                │
│  │ [200]       │  │ [3]         │  │ [Sales ▼]   │                │
│  └─────────────┘  └─────────────┘  └─────────────┘                │
│  [ 🚀 Prediksi Attrition ]  [ Reset ]                             │
├─────────────────────────────────────────────────────────────────────┤
│  ✅ Hasil: Karyawan TETAP (Probabilitas: 78.5%)                    │
├─────────────────────────────────────────────────────────────────────┤
│  📋 Riwayat Prediksi                                               │
│  ┌───┬──────┬───────┬──────┬──────┬───────┬──────┬────────┬──────┐│
│  │ # │ Dept │Satis. │ Eval │ Proj.│ Hours │Tenure│Accident│Salary││
│  ├───┼──────┼───────┼──────┼──────┼───────┼──────┼────────┼──────┤│
│  │ 1 │Sales │ 0.2   │ 0.5  │ 6    │ 280   │ 8    │ Tidak  │ Low  ││
│  └───┴──────┴───────┴──────┴──────┴───────┴──────┴────────┴──────┘│
└─────────────────────────────────────────────────────────────────────┘
```

### Halaman Eksplorasi Data

- Statistik dataset (total baris, kolom, missing values)
- Distribusi target
- Daftar fitur
- Statistik deskriptif
- Sample data

### Halaman Hasil Metode

- Model terbaik (winner box)
- Perbandingan 6 kombinasi model

---

## 📊 Fitur Dataset

| Fitur | Tipe | Deskripsi |
|-------|------|-----------|
| satisfaction_level | Float | Tingkat kepuasan karyawan (0-1) |
| last_evaluation | Float | Nilai evaluasi terakhir (0-1) |
| number_project | Integer | Jumlah proyek yang dikerjakan |
| average_montly_hours | Integer | Rata-rata jam kerja per bulan |
| time_spend_company | Integer | Lama bekerja di perusahaan (tahun) |
| work_accident | Integer | Pernah kecelakaan kerja (0/1) |
| **left** | **Integer (Target)** | **Status karyawan (0=Tetap, 1=Keluar)** |
| promotion_last_5years | Integer | Promosi dalam 5 tahun terakhir (0/1) |
| sales | String | Departemen tempat bekerja |
| salary | String | Tingkat gaji (low/medium/high) |

**Distribusi Target**: 76% Tetap, 24% Keluar (Imbalance Moderata)

---

## 🛠️ Tech Stack

| Lapisan | Teknologi |
|---------|-----------|
| **Backend** | FastAPI, SQLAlchemy, SQLite |
| **Frontend** | HTML + CSS (Bootstrap), JavaScript |
| **Deep Learning** | TensorFlow / Keras, TabNet |
| **Preprocessing** | Scikit-learn (StandardScaler, LabelEncoder) |
| **Imbalance Handling** | Imbalanced-learn (SMOTE-ENN) |
| **Development** | Jupyter Notebook, Uvicorn |

---

## 📋 Requirements

```
# Core
pandas
numpy
matplotlib
seaborn
scikit-learn

# Deep Learning
tensorflow
pytorch-tabnet

# Imbalance Handling
imbalanced-learn

# GUI
fastapi
uvicorn
sqlalchemy
aiosqlite
jinja2
python-multipart

# Utility
jupyter
notebook
ipykernel
```

---

## 📈 Hasil Evaluasi

### Perbandingan 6 Kombinasi (dari comparison_results.csv)

| Ranking | Kombinasi | F1-Score | Accuracy | Precision | Recall | Training (s) |
|---------|-----------|----------|----------|-----------|--------|-------------|
| **#1** | **TabNet_NoHandling** | **0.9302** | **0.9775** | **0.9574** | **0.9045** | 188.1 |
| #2 | MLP_ClassWeight | 0.9278 | 0.9767 | 0.9524 | 0.9045 | 82.7 |
| #3 | MLP_NoHandling | 0.9243 | 0.9758 | 0.9620 | 0.8894 | 88.4 |
| #4 | MLP_SMOTE_ENN | 0.9023 | 0.9675 | 0.9000 | 0.9045 | 56.8 |
| #5 | TabNet_ClassWeight | 0.8916 | 0.9633 | 0.8744 | 0.9095 | 267.2 |
| #6 | TabNet_SMOTE_ENN | 0.8438 | 0.9442 | 0.7870 | 0.9095 | 311.4 |

### Interpretasi

- **TabNet_NoHandling** memberikan F1-Score tertinggi (0.9302) — setelah outlier handling winsorizing, distribusi data lebih stabil sehingga TabNet unggul dengan *attention-based feature selection*
- **MLP_ClassWeight** dan **MLP_NoHandling** tetap kompetitif (~0.92-0.93 F1) dengan waktu training lebih cepat (56-88s vs 188s)
- **TabNet_SMOTE_ENN** memiliki Precision rendah (0.787) karena SMOTE-ENN memperkenalkan sample sintetis yang tidak sesuai dengan representasi TabNet

---

## 🔧 Troubleshooting

### Error: `ModuleNotFoundError: No module named '...'`

```bash
pip install [nama_library]
```

### Error: `zsh: command not found: uvicorn`

```bash
source venv/bin/activate
pip install uvicorn
```

### Error: Model tidak ditemukan

```bash
ls -la models/
# Pastikan best_model.h5, tabnet_best_model/, scaler.pkl, encoders.pkl, feature_info.json ada
```

### Error: Keras version incompatibility (`GlorotUniform`)

Jika muncul error `GlorotUniform.__init__() got unexpected keyword arguments 'input_axes'` saat load model:

```bash
pip install --upgrade tensorflow keras
```
Atau gunakan TensorFlow 2.13 yang kompatibel dengan model yang disimpan.

### Error: Database tidak bisa menyimpan

```bash
rm hr_prediction.db
# Database akan dibuat ulang otomatis
```

### Catatan: Tenure Blindspot

Model memiliki keterbatasan pada karyawan dengan **tenure >= 7 tahun** karena dalam dataset training tidak ada satupun karyawan dengan masa kerja >= 7 tahun yang berstatus *left* (keluar). Aplikasi mengimplementasikan **koreksi otomatis** (`_adjust_probability` di `ml_model.py`) yang meningkatkan probabilitas prediksi berdasarkan jumlah faktor risiko pada kasus tenure tinggi.

---

## 🙏 Referensi

1. [HR Data for Analytics - Kaggle](https://www.kaggle.com/datasets/jacksonchou/hr-data-for-analytics)
2. [TensorFlow Documentation](https://www.tensorflow.org/)
3. [FastAPI Documentation](https://fastapi.tiangolo.com/)
4. [TabNet: Attentive Interpretable Tabular Learning](https://arxiv.org/abs/1908.07442)
5. [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)

---
