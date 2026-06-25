import tensorflow as tf
import numpy as np
import pickle
import os
import json

# Monkey-patch GlorotUniform to handle Keras version incompatibility
# Older Keras saved initializer config with input_axes/output_axes params
# that Keras 3.x's GlorotUniform.__init__() does not accept
try:
    from keras.src.initializers import GlorotUniform
    _orig_glorot_init = GlorotUniform.__init__
    def _patched_glorot_init(self, seed=None, input_axes=None, output_axes=None, **kwargs):
        _orig_glorot_init(self, seed=seed)
    GlorotUniform.__init__ = _patched_glorot_init
except ImportError:
    pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "best_model.h5")
MODEL_ZIP_PATH = os.path.join(BASE_DIR, "..", "models", "best_model.zip")
SCALER_PATH = os.path.join(BASE_DIR, "..", "models", "scaler.pkl")
ENCODERS_PATH = os.path.join(BASE_DIR, "..", "models", "encoders.pkl")
FEATURE_INFO_PATH = os.path.join(BASE_DIR, "..", "models", "feature_info.json")

def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        bce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        weight = alpha * tf.pow(1 - y_pred, gamma) * y_true + (1 - alpha) * tf.pow(y_pred, gamma) * (1 - y_true)
        return tf.reduce_mean(weight * bce)
    return focal_loss_fixed

model = None
scaler = None
encoders = None
feature_info = None
model_type = None

def load_model():
    """Memuat model dan semua preprocessing dari folder models/"""
    global model, scaler, encoders, feature_info, model_type
    
    print("=" * 50)
    print("LOADING MODEL & PREPROCESSING")
    print("=" * 50)
    
    # 1. Load feature info dulu untuk mengetahui tipe model
    if feature_info is None:
        try:
            if os.path.exists(FEATURE_INFO_PATH):
                with open(FEATURE_INFO_PATH, 'r') as f:
                    feature_info = json.load(f)
                print("[OK] Feature info berhasil dimuat")
                print(f"   Best model: {feature_info.get('best_model', 'Unknown')}")
                print(f"   F1-Score: {feature_info.get('f1_score', 0):.4f}")
                print(f"   Optimal Threshold: {feature_info.get('optimal_threshold', 0.3)}")
                model_type = feature_info.get('model_type', 'mlp')
                print(f"   Model Type: {model_type}")
            else:
                print(f"[ERR] ERROR: Feature info tidak ditemukan: {FEATURE_INFO_PATH}")
                return False
        except Exception as e:
            print(f"[ERR] Gagal load feature info: {e}")
            return False
    
    # 2. Load model berdasarkan tipe
    if model is None:
        try:
            if model_type == 'tabnet':
                # Load TabNet
                from pytorch_tabnet.tab_model import TabNetClassifier
                tabnet_paths = [MODEL_ZIP_PATH, MODEL_ZIP_PATH.replace('.zip', '')]
                tabnet_loaded = False
                for p in tabnet_paths:
                    if os.path.exists(p):
                        model = TabNetClassifier()
                        model.load_model(p)
                        print(f"[OK] TabNet berhasil dimuat dari: {p}")
                        tabnet_loaded = True
                        break
                if not tabnet_loaded:
                    print(f"[ERR] ERROR: File TabNet tidak ditemukan. Dicari: {tabnet_paths}")
                    return False
            else:
                # Load TensorFlow model (with custom focal loss)
                if os.path.exists(MODEL_PATH):
                    model = tf.keras.models.load_model(
                        MODEL_PATH,
                        custom_objects={'focal_loss_fixed': focal_loss()}
                    )
                    print(f"[OK] Model berhasil dimuat dari: {MODEL_PATH}")
                else:
                    print(f"[ERR] ERROR: File model tidak ditemukan: {MODEL_PATH}")
                    return False
        except Exception as e:
            print(f"[ERR] Gagal load model: {e}")
            return False
    
    # 3. Load scaler
    if scaler is None:
        try:
            if os.path.exists(SCALER_PATH):
                with open(SCALER_PATH, 'rb') as f:
                    scaler = pickle.load(f)
                print(f"[OK] Scaler berhasil dimuat dari: {SCALER_PATH}")
            else:
                print(f"[ERR] ERROR: Scaler tidak ditemukan: {SCALER_PATH}")
                return False
        except Exception as e:
            print(f"[ERR] Gagal load scaler: {e}")
            return False
    
    # 4. Load encoders
    if encoders is None:
        try:
            if os.path.exists(ENCODERS_PATH):
                with open(ENCODERS_PATH, 'rb') as f:
                    encoders = pickle.load(f)
                print(f"[OK] Encoders berhasil dimuat dari: {ENCODERS_PATH}")
                print(f"   Encoders keys: {list(encoders.keys())}")
            else:
                print(f"[ERR] ERROR: Encoders tidak ditemukan: {ENCODERS_PATH}")
                return False
        except Exception as e:
            print(f"[ERR] Gagal load encoders: {e}")
            return False
    
    print("=" * 50)
    print("[OK] SEMUA FILE BERHASIL DIMUAT!")
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
            'recall': feature_info.get('recall', 0),
            'precision': feature_info.get('precision', 0),
            'threshold': feature_info.get('optimal_threshold', 0.3)
        }
    return {
        'best_model': 'Unknown',
        'f1_score': 0,
        'accuracy': 0,
        'recall': 0,
        'precision': 0,
        'threshold': 0.3
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
    
    # 3. Buat array dengan urutan yang benar (case-insensitive matching)
    col_to_key = {k.lower(): k for k in input_dict.keys()}
    feature_values = []
    for col in feature_cols:
        key = col_to_key.get(col.lower())
        if key is not None:
            feature_values.append(input_dict[key])
        else:
            feature_values.append(0)
    
    features = np.array(feature_values).reshape(1, -1)
    
    # 4. Standarisasi
    if scaler is not None:
        features_scaled = scaler.transform(features)
    else:
        features_scaled = features
    
    return features_scaled

def _count_risk_factors(data: dict) -> int:
    count = 0
    if float(data.get('satisfaction_level', 0.5)) < 0.3:
        count += 1
    if float(data.get('last_evaluation', 0.5)) < 0.4:
        count += 1
    nproj = int(data.get('number_project', 3))
    if nproj <= 2 or nproj >= 6:
        count += 1
    hours = int(data.get('average_montly_hours', 200))
    if hours > 250:
        count += 1
    if int(data.get('work_accident', 0)) == 1:
        count += 1
    if data.get('salary', 'medium') == 'low':
        count += 1
    return count


def _adjust_probability(prob: float, data: dict) -> float:
    """
    Correct model blindspot: for tenure >= 7 the training data has ZERO leavers,
    so the model underestimates risk for high-tenure employees with risk factors.
    Uses a risk-score boost calibrated to the empirical attrition rates.
    """
    tenure = int(data.get('time_spend_company', 3))
    if tenure < 7:
        return prob

    risk_count = _count_risk_factors(data)
    if risk_count == 0:
        return prob

    corrected = prob + risk_count * 0.08
    return min(corrected, 0.50)


def predict(data: dict):
    """
    Melakukan prediksi attrition dengan optimal threshold
    """
    global model, model_type
    
    if model is None:
        success = load_model()
        if not success:
            return "ERROR", 0.0
    
    try:
        features = preprocess_input(data)
        
        # Prediksi sesuai tipe model
        if model_type == 'tabnet':
            prob = float(model.predict_proba(features)[0][1])
        else:
            prob = float(model.predict(features, verbose=0)[0][0])
        
        # Koreksi untuk blindspot model (tenure >= 7)
        prob = _adjust_probability(prob, data)
        
        # Gunakan optimal threshold dari feature_info
        optimal_threshold = feature_info.get('optimal_threshold', 0.3)
        
        result = "KELUAR" if prob >= optimal_threshold else "TETAP"
        return result, prob
    except Exception as e:
        import traceback
        print(f"[ERR] Error saat prediksi: {e}")
        traceback.print_exc()
        return "ERROR", 0.0

# Model dimuat via startup event di main.py, bukan di sini