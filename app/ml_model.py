import tensorflow as tf
import numpy as np
import pickle
import os
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "best_model.h5")
SCALER_PATH = os.path.join(BASE_DIR, "..", "models", "scaler.pkl")
ENCODERS_PATH = os.path.join(BASE_DIR, "..", "models", "encoders.pkl")
FEATURE_INFO_PATH = os.path.join(BASE_DIR, "..", "models", "feature_info.json")

model = None
scaler = None
encoders = None
feature_info = None

def load_model():
    """Memuat model dan semua preprocessing dari folder models/"""
    global model, scaler, encoders, feature_info
    
    print("=" * 50)
    print("LOADING MODEL & PREPROCESSING")
    print("=" * 50)
    
    # 1. Load model
    if model is None:
        try:
            if not os.path.exists(MODEL_PATH):
                print(f"❌ ERROR: File model tidak ditemukan: {MODEL_PATH}")
                return False
            else:
                model = tf.keras.models.load_model(MODEL_PATH)
                print(f"✅ Model berhasil dimuat dari: {MODEL_PATH}")
        except Exception as e:
            print(f"❌ Gagal load model: {e}")
            return False
    
    # 2. Load scaler
    if scaler is None:
        try:
            if os.path.exists(SCALER_PATH):
                with open(SCALER_PATH, 'rb') as f:
                    scaler = pickle.load(f)
                print(f"✅ Scaler berhasil dimuat dari: {SCALER_PATH}")
            else:
                print(f"❌ ERROR: Scaler tidak ditemukan: {SCALER_PATH}")
                return False
        except Exception as e:
            print(f"❌ Gagal load scaler: {e}")
            return False
    
    # 3. Load encoders
    if encoders is None:
        try:
            if os.path.exists(ENCODERS_PATH):
                with open(ENCODERS_PATH, 'rb') as f:
                    encoders = pickle.load(f)
                print(f"✅ Encoders berhasil dimuat dari: {ENCODERS_PATH}")
                print(f"   Encoders keys: {list(encoders.keys())}")
            else:
                print(f"❌ ERROR: Encoders tidak ditemukan: {ENCODERS_PATH}")
                return False
        except Exception as e:
            print(f"❌ Gagal load encoders: {e}")
            return False
    
    # 4. Load feature info
    if feature_info is None:
        try:
            if os.path.exists(FEATURE_INFO_PATH):
                with open(FEATURE_INFO_PATH, 'r') as f:
                    feature_info = json.load(f)
                print("✅ Feature info berhasil dimuat")
                print(f"   Best model: {feature_info.get('best_model', 'Unknown')}")
                print(f"   F1-Score: {feature_info.get('f1_score', 0):.4f}")
                print(f"   Accuracy: {feature_info.get('accuracy', 0):.4f}")
                print(f"   Feature cols: {feature_info.get('feature_cols', [])}")
            else:
                print(f"❌ ERROR: Feature info tidak ditemukan: {FEATURE_INFO_PATH}")
                return False
        except Exception as e:
            print(f"❌ Gagal load feature info: {e}")
            return False
    
    print("=" * 50)
    print("✅ SEMUA FILE BERHASIL DIMUAT!")
    print("=" * 50)
    return True

def get_model_info():
    """Mengembalikan informasi model terbaik"""
    if feature_info is None:
        load_model()
    if feature_info:
        return {
            'best_model': feature_info.get('best_model', 'Unknown'),
            'f1_score': feature_info.get('f1_score', 0),
            'accuracy': feature_info.get('accuracy', 0),
            'recall': feature_info.get('recall', 0)
        }
    return {
        'best_model': 'Unknown',
        'f1_score': 0,
        'accuracy': 0,
        'recall': 0
    }

def preprocess_input(data: dict):
    """
    Preprocessing input dari user sesuai dengan pipeline training
    """
    feature_cols = feature_info.get('feature_cols', [
        'satisfaction_level', 'last_evaluation', 'number_project', 
        'average_montly_hours', 'time_spend_company', 'work_accident',
        'promotion_last_5years', 'sales', 'salary'
    ])
    
    # 1. Map input ke dictionary
    input_dict = {
        'satisfaction_level': float(data.get('satisfaction_level', 0.5)),
        'last_evaluation': float(data.get('last_evaluation', 0.5)),
        'number_project': int(data.get('number_project', 3)),
        'average_montly_hours': int(data.get('average_montly_hours', 200)),
        'time_spend_company': int(data.get('time_spend_company', 3)),
        'work_accident': int(data.get('work_accident', 0)),
        'promotion_last_5years': int(data.get('promotion_last_5years', 0)),
        'sales': data.get('sales', 'sales'),
        'salary': data.get('salary', 'medium')
    }
    
    # 2. Encoding kategorikal
    if 'sales' in encoders:
        try:
            input_dict['sales'] = encoders['sales'].transform([data.get('sales', 'sales')])[0]
        except:
            input_dict['sales'] = 0
    else:
        input_dict['sales'] = 0
    
    if 'salary' in encoders:
        try:
            input_dict['salary'] = encoders['salary'].transform([data.get('salary', 'medium')])[0]
        except:
            input_dict['salary'] = 1
    else:
        input_dict['salary'] = 1
    
    # 3. Buat array dengan urutan yang benar
    feature_values = []
    for col in feature_cols:
        if col == 'Work_accident':
            feature_values.append(input_dict['work_accident'])
        else:
            feature_values.append(input_dict.get(col, 0))
    
    features = np.array(feature_values).reshape(1, -1)
    
    # 4. Standarisasi
    if scaler is not None:
        features_scaled = scaler.transform(features)
    else:
        features_scaled = features
    
    return features_scaled

def predict(data: dict):
    """
    Melakukan prediksi attrition
    """
    global model
    
    if model is None:
        success = load_model()
        if not success:
            return "ERROR", 0.0
    
    try:
        features = preprocess_input(data)
        prob = float(model.predict(features, verbose=0)[0][0])
        result = "KELUAR" if prob >= 0.5 else "TETAP"
        return result, prob
    except Exception as e:
        print(f"❌ Error saat prediksi: {e}")
        return "ERROR", 0.0

# Auto-load saat file di-import
print("🔄 Memuat model...")
load_model()