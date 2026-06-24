# HR Analytics - Employee Attrition Prediction

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange)](https://www.tensorflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100-green)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 📋 Deskripsi Proyek

Proyek ini bertujuan untuk memprediksi **attrition karyawan** (apakah karyawan akan keluar dari perusahaan atau tetap) menggunakan metode **Deep Learning**. Dataset yang digunakan adalah **HR Data for Analytics** dari Kaggle.

Aplikasi ini dilengkapi dengan:
- **GUI berbasis Web** untuk input data karyawan dan prediksi
- **Dashboard Eksplorasi Data** untuk melihat statistik dataset
- **Dashboard Hasil Metode** untuk membandingkan performa model

---

## 🎯 Metode Deep Learning yang Digunakan

| Metode | Arsitektur | Loss Function | Penanganan Imbalance |
|--------|------------|---------------|---------------------|
| **MLP Deep + Focal Loss** | 4 Hidden Layers (256, 128, 64, 32) + BatchNorm + Dropout | Focal Loss (γ=2.0, α=0.25) | Focal Loss |
| **TabNet** | Sequential Attention + GLU | Binary Crossentropy | SMOTE-ENN |
| **MLP Modern** | 3 Hidden Layers (128, 64, 32) + BatchNorm + Dropout | Binary Crossentropy | Class Weight |

---

## 📊 Hasil Model Terbaik

| Model | F1-Score | Accuracy | Recall |
|-------|----------|----------|--------|
| **MLP Deep + Focal Loss** | **0.9302** | **0.9775** | **0.9195** |
| MLP Modern + Class Weight | 0.9082 | 0.9689 | 0.9295 |
| TabNet + No Handling | 0.8929 | 0.9639 | 0.9094 |
| TabNet + SMOTE-ENN | 0.8510 | 0.9461 | 0.9295 |

**Model Terbaik: MLP Deep + Focal Loss (F1-Score: 0.9302)**

---

## 🏗️ Arsitektur MLP Deep + Focal Loss

```
Input Layer (9 fitur)
    ↓
Dense(256, ReLU) + BatchNormalization + Dropout(0.2)
    ↓
Dense(128, ReLU) + BatchNormalization + Dropout(0.2)
    ↓
Dense(64, ReLU) + BatchNormalization + Dropout(0.2)
    ↓
Dense(32, ReLU) + BatchNormalization + Dropout(0.2)
    ↓
Output Layer (Sigmoid)
```

**Focal Loss**:
```
FL(p_t) = -α(1-p_t)^γ log(p_t)
```
dengan γ = 2.0, α = 0.25

---

## 📂 Struktur Folder

```
hr-analytics-deeplearning/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── ml_model.py
│   ├── database.py
│   ├── models.py
│   ├── schemas.py
│   ├── templates/
│   │   ├── index.html
│   │   ├── explore.html
│   │   └── results.html
│   └── static/
│       └── style.css
│
├── models/
│   ├── best_model.h5
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
│   └── train_models_final_v2.ipynb
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
cd notebooks
jupyter notebook train_models_final_v2.ipynb
# Jalankan semua cell
```

### 6. Jalankan Aplikasi GUI

```bash
cd ..
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
│  Model: MLP_Deep_FocalLoss  F1: 0.9302  Acc: 0.9775               │
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
│  ┌───┬──────────┬──────────┬──────────┬──────────┬──────────┐    │
│  │ # │ Dept     │ Satis.   │ Projects │ Hasil    │ Waktu    │    │
│  ├───┼──────────┼──────────┼──────────┼──────────┼──────────┤    │
│  │ 1 │ Sales    │ 0.8      │ 5        │ TETAP    │ 25/06/26 │    │
│  └───┴──────────┴──────────┴──────────┴──────────┴──────────┘    │
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

### Perbandingan 6 Kombinasi

| Kombinasi | F1-Score | Accuracy | Recall | Training Time (s) |
|-----------|----------|----------|--------|-------------------|
| **MLP_Deep_FocalLoss_SMOTE** | **0.9302** | **0.9775** | **0.9195** | 45.2 |
| MLP_ClassWeight | 0.9082 | 0.9689 | 0.9295 | 9.6 |
| TabNet_NoHandling | 0.8929 | 0.9639 | 0.9094 | 27.7 |
| TabNet_SMOTE_ENN | 0.8510 | 0.9461 | 0.9295 | 43.5 |
| MLP_NoHandling | 0.9272 | 0.9761 | 0.9195 | 10.3 |
| MLP_SMOTE_ENN | 0.8748 | 0.9561 | 0.9262 | 13.9 |

### Interpretasi

- **MLP Deep + Focal Loss** memberikan F1-Score tertinggi (0.9302)
- **Class Weight** efektif untuk MLP Modern
- **TabNet** lebih lambat training tapi performa stabil

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
# Pastikan best_model.h5, scaler.pkl, encoders.pkl, feature_info.json ada
```

### Error: Database tidak bisa menyimpan

```bash
rm hr_prediction.db
# Database akan dibuat ulang otomatis
```

---

## 🙏 Referensi

1. [HR Data for Analytics - Kaggle](https://www.kaggle.com/datasets/jacksonchou/hr-data-for-analytics)
2. [TensorFlow Documentation](https://www.tensorflow.org/)
3. [FastAPI Documentation](https://fastapi.tiangolo.com/)
4. [TabNet: Attentive Interpretable Tabular Learning](https://arxiv.org/abs/1908.07442)
5. [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)

---