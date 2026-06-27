# Product Requirements Document (PRD) — HR Attrition Prediction

| Metadata | |
|----------|-----------|
| **Project** | HR Analytics — Employee Attrition Prediction |
| **Status** | Final (UAS) |
| **Version** | 1.0 |
| **Last Updated** | 2026-06-27 |

---

## 1. Executive Summary

Aplikasi prediksi attrition karyawan berbasis deep learning. Menggunakan 6 kombinasi model (MLP + TabNet dengan 3 skenario imbalance handling), memilih model terbaik (TabNet_NoHandling, F1=0.93), dan menyajikannya dalam web GUI (FastAPI) dengan 3 halaman: prediksi, eksplorasi data, dan hasil metode.

---

## 2. Problem Statement

- **Turnover tinggi** merugikan perusahaan (biaya rekrutmen, training, lost productivity)
- **Deteksi dini** sulit dilakukan secara manual pada ribuan karyawan
- **Data historis** (14.999 records) tersedia tetapi belum dimanfaatkan untuk prediksi

---

## 3. Project Goals

1. **Prediksi attrition** dengan F1-score >= 0.90
2. **Web GUI** yang mudah digunakan HR tanpa coding
3. **Eksplorasi data** untuk insight distribusi dan korelasi
4. **Perbandingan metode** (MLP vs TabNet, 3 imbalance handling)

---

## 4. Scope

### In Scope
- Training 6 kombinasi model (MLP_NoHandling, MLP_SMOTE_ENN, MLP_ClassWeight, TabNet_NoHandling, TabNet_SMOTE_ENN, TabNet_ClassWeight)
- Web app: predict page, explore page, results page
- Riwayat prediksi (13 kolom) dengan SQLite

### Out of Scope
- Real-time inference API
- Multi-user authentication
- A/B testing framework
- Mobile app

---

## 5. Personas

| Persona | Role | Need |
|---------|------|------|
| **HR Manager** | Decision maker | Prediksi individu, insight tim, identifikasi risiko |
| **Data Analyst** | Technical user | Eksplorasi data, perbandingan performa model |
| **Lecturer** | Evaluator | Memvalidasi metodologi, arsitektur, hasil |

---

## 6. Functional Requirements

### FR-01: Prediksi Attrition
- Input 9 fitur via form: satisfaction_level, last_evaluation, number_project, average_montly_hours, time_spend_company, Work_accident, promotion_last_5years, department (sales), salary
- Output: TETAP/KELUAR + probabilitas (0-1)
- Koreksi otomatis untuk tenure >= 7 tahun

### FR-02: Riwayat Prediksi
- Tabel 13 kolom: No, Department, Satisfaction, Evaluation, Projects, Hours, Tenure, Accident, Salary, Predicted, Actual, Probability, Correct
- Hanya 1 baris per sesi (aktual/benar/salah diisi manual)

### FR-03: Eksplorasi Data
- Statistik dataset: total rows, columns, missing values
- Distribusi target (pie chart)
- Distribusi fitur numerik (histogram)
- Korelasi fitur (heatmap)
- 5 sample data acak

### FR-04: Hasil Metode
- Winner box: model terbaik dengan F1, Accuracy, Precision, Recall
- Tabel perbandingan 6 kombinasi
- Grafik perbandingan (F1 bar chart, heatmap, model mean, handling mean)

### FR-05: Validasi Input
- Server-side: range check per fitur (0-1 untuk satisfaction/evaluation, 1-12 untuk projects, dll)
- Client-side: min/max pada input HTML, invalid class styling
- Error banner merah dengan pesan spesifik

---

## 7. Non-Functional Requirements

### NFR-01: Performance
- Prediksi < 1 detik per karyawan (termasuk preprocessing)
- Training ~3-5 menit untuk semua 6 kombinasi

### NFR-02: Usability
- Responsive layout (CSS + Bootstrap)
- Bahasa Indonesia pada UI

### NFR-03: Reliability
- Model fallback: monkey-patch Keras GlorotUniform jika version mismatch
- Database auto-create (SQLite)

### NFR-04: Maintainability
- Kode modular: app/main.py, app/ml_model.py
- Model retrain via notebook

---

## 8. Architecture

```
[User Browser] <--> [FastAPI Server] <--> [SQLite (history)]
                        |
                   [ML Model (Keras)]
                        |
                   [Preprocessing Pipeline]
                   (scaler.pkl + encoders.pkl)
```

### Komponen
| Komponen | Teknologi |
|----------|-----------|
| Backend | FastAPI, Uvicorn |
| Database | SQLAlchemy + SQLite |
| Deep Learning | TensorFlow/Keras + PyTorch TabNet |
| Preprocessing | Scikit-learn (StandardScaler, LabelEncoder) |
| Imbalance Handling | Imbalanced-learn (SMOTE-ENN) |
| Frontend | HTML + CSS + vanilla JavaScript |

---

## 9. Data Requirements

### Dataset: HR_comma_sep.csv
- **Rows**: 14.999
- **Columns**: 9 fitur + 1 target (left)
- **Imbalance**: 76% tetap, 24% keluar
- **Missing values**: 0

### Preprocessing
| Step | Detail |
|------|--------|
| Outlier handling | Winsorizing (IQR capping) pada 5 numerik kontinu |
| Duplikat | 0 ditemukan (data asli bersih) |
| Encoding | LabelEncoder untuk biner, OneHot untuk kategorikal multi-class |
| Scaling | StandardScaler (fitur numerik kontinu) |
| Split | 80% train, 10% val, 10% test (stratified) |

---

## 10. Model Requirements

### Best Model: TabNet_NoHandling
| Parameter | Nilai |
|-----------|-------|
| n_d (decision width) | 64 |
| n_a (attention width) | 64 |
| n_steps | 5 |
| Loss | Cross-Entropy |
| Optimizer | Adam (lr=2e-2) |
| Batch Size | 256 |
| Scheduler | ReduceLROnPlateau |
| Early Stopping | patience=20 |

### All 6 Combinations
| Model | Handling | Loss |
|-------|----------|------|
| MLP_NoHandling | None | Focal Loss (=2.0, =0.25) |
| MLP_SMOTE_ENN | SMOTE-ENN | Focal Loss (=2.0, =0.25) |
| MLP_ClassWeight | Class Weight | Focal Loss (=2.0, =0.5) |
| TabNet_NoHandling | None | Cross-Entropy |
| TabNet_SMOTE_ENN | SMOTE-ENN | Cross-Entropy |
| TabNet_ClassWeight | Duplikasi Fraud 3x | Cross-Entropy |

### MLP Architecture (comparison)
```
Input(9) -> Dense(256,ReLU)+BN+Drop(0.2) ->
Dense(128,ReLU)+BN+Drop(0.2) ->
Dense(64,ReLU)+BN+Drop(0.2) ->
Dense(32,ReLU)+BN+Drop(0.2) ->
Output(Sigmoid)
```

### Tenure Blindspot Correction
- Dataset: 0 leavers pada tenure >= 7 tahun
- Model predict ~1% untuk semua input tenure >= 7
- Fix: `_adjust_probability()` menambah `risk_count * 0.08` (capped 0.50)

---

## 11. UI/UX Design

### Pages
| Page | Route | Description |
|------|-------|-------------|
| Predict | `/` | Form input 9 fitur + hasil prediksi + riwayat |
| Explore | `/explore` | Statistik dataset, distribusi, histogram, heatmap |
| Results | `/results` | Perbandingan 6 model, winner box, grafik |

### Predict Page Layout
```
Header: "HR Attrition Predictor" + model name + metrics
Stats bar: Total | Tetap | Keluar | Rate
Input form: 2 kolom, 9 fields
  Kolom 1: Satisfaction, Evaluation, Projects, Hours, Tenure
  Kolom 2: Department, Salary, Accident, Promotion
Action: [Prediksi] [Reset]
Result box: TETAP/KELUAR + probabilitas (dengan status color)
Error banner: merah jika input invalid
History table: 13 kolom, scrollable
```

### Color Scheme
| Element | Color |
|---------|-------|
| Tetap | Green (#28a745) |
| Keluar | Red (#dc3545) |
| Error | Red (#f8d7da bg) |
| Primary button | Blue (#007bff) |

---

## 12. API Specifications

### Endpoints
| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Render predict page |
| POST | `/predict` | Submit prediction, return result + update history |
| GET | `/explore` | Render explore page |
| GET | `/results` | Render results page |

### POST /predict
**Request body**: form-data (9 fields)
**Validation**: server-side range checks
**Response**: redirect with result query params (prediction, probability, error)

---

## 13. Acceptance Criteria

| ID | Criteria | Status |
|----|----------|--------|
| AC-01 | F1-score >= 0.90 pada test set | 0.93 |
| AC-02 | Web app dapat diakses via browser | OK |
| AC-03 | Prediksi < 1 detik | OK |
| AC-04 | History table menampilkan 13 kolom | OK |
| AC-05 | Error handling untuk input invalid | OK |
| AC-06 | Tenure >= 7 mendapat koreksi probabilitas | OK |
| AC-07 | Explore page menampilkan statistik & grafik | OK |
| AC-08 | Results page menampilkan perbandingan 6 model | OK |

---

## 14. Constraints

- Model format: .h5 (Keras) untuk MLP, folder untuk TabNet
- TabNet history API: History object (no .keys()), akses via []
- Keras version compatibility: monkey-patch GlorotUniform required
- Tenure blindspot: training data has zero leavers at tenure >= 7

---

## 15. Timeline

| Phase | Milestone |
|-------|-----------|
| Data understanding & preprocessing | Complete |
| Model training (6 combinations) | Complete |
| Web app development | Complete |
| PRD documentation | Current |

---

## 16. Future Enhancements

- SHAP/LIME untuk explainability per prediksi
- Batch prediction (upload CSV)
- Export riwayat ke CSV/Excel
- User authentication & multi-user history
- Retrain pipeline otomatis dengan data baru
- Deployment ke cloud (Railway, HuggingFace Spaces)

---

## 17. Glossary

| Term | Definition |
|------|------------|
| Attrition | Karyawan keluar dari perusahaan |
| F1-Score | Harmonic mean precision & recall |
| TabNet | Attentive interpretable tabular learning network (Arik & Pfister, 2019) |
| Focal Loss | Loss function yang mengurangi bobot easy samples (Lin et al., 2017) |
| Winsorizing | Capping outlier ke batas IQR, bukan menghapus |
| SMOTE-ENN | Synthetic Minority Oversampling + Edited Nearest Neighbors |
| Youden's Index | J = Sensitivity + Specificity - 1, untuk threshold tuning |
