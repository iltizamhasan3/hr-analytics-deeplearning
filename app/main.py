from fastapi import FastAPI, Request, Form, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime
import os
import pandas as pd

from app import ml_model

# ============================================
# SETUP
# ============================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# ============================================
# DATABASE
# ============================================
DB_PATH = os.path.join(BASE_DIR, '..', 'hr_prediction.db')
SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class PredictionHistory(Base):
    __tablename__ = 'predictions'
    id = Column(Integer, primary_key=True)
    satisfaction_level = Column(Float)
    last_evaluation = Column(Float)
    number_project = Column(Integer)
    average_montly_hours = Column(Integer)
    time_spend_company = Column(Integer)
    work_accident = Column(Integer)
    promotion_last_5years = Column(Integer)
    sales = Column(String(50))
    salary = Column(String(20))
    prediction_result = Column(String(20))
    probability = Column(Float)
    threshold_used = Column(Float, default=0.3)
    created_at = Column(DateTime, default=datetime.now)

Base.metadata.create_all(bind=engine)

# ============================================
# FASTAPI APP
# ============================================
app = FastAPI(title="HR Attrition Predictor", version="1.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.on_event("startup")
async def startup_event():
    ml_model.load_model()

def get_model_info():
    info = ml_model.get_model_info()
    return {
        'best_model': info.get('best_model', 'MLP_Deep_FocalLoss'),
        'f1_score': info.get('f1_score', 0.9302),
        'accuracy': info.get('accuracy', 0.9775),
        'recall': info.get('recall', 0.9195),
        'precision': info.get('precision', 0.9428),
        'threshold': info.get('threshold', 0.3)
    }

def read_template(filename):
    with open(os.path.join(TEMPLATE_DIR, filename), 'r') as f:
        return f.read()

# ============================================
# HELPER FUNCTIONS
# ============================================
def generate_prediction_html(result, probability, threshold):
    card_class = "result-success" if result == "TETAP" else "result-danger"
    badge_class = "badge-green" if result == "TETAP" else "badge-red"
    icon = "fa-circle-check" if result == "TETAP" else "fa-circle-exclamation"
    
    return f"""
    <div class="result-card {card_class}">
        <div class="result-flex">
            <div class="result-icon-box"><i class="fas {icon}"></i></div>
            <div class="result-text">
                <div class="result-title">Karyawan Diprediksi: <strong>{result}</strong></div>
                <div class="result-prob">Probabilitas: <strong>{probability}</strong></div>
                <div class="result-prob" style="font-size:11px; color:#94a3b8; margin-top:4px;">
                    <i class="fas fa-sliders-h"></i> Threshold: {threshold:.2f}
                </div>
            </div>
            <div><span class="result-badge {badge_class}">{result}</span></div>
        </div>
    </div>
    """

def generate_history_rows(history):
    if not history:
        return ""
    
    rows = ""
    for idx, record in enumerate(history, 1):
        tag_class = "tag-red" if record.prediction_result == "KELUAR" else "tag-green"
        accident_label = "Ya" if record.work_accident == 1 else "Tidak"
        rows += f"""
        <tr>
            <td>{idx}</td>
            <td>{record.sales}</td>
            <td>{record.satisfaction_level}</td>
            <td>{record.last_evaluation}</td>
            <td>{record.number_project}</td>
            <td>{record.average_montly_hours}</td>
            <td>{record.time_spend_company}</td>
            <td>{accident_label}</td>
            <td>{record.salary}</td>
            <td><span class="status-tag {tag_class}">{record.prediction_result}</span></td>
            <td>{record.probability * 100:.1f}%</td>
            <td>{record.threshold_used:.2f}</td>
            <td>{record.created_at.strftime('%d/%m/%y %H:%M')}</td>
        </tr>
        """
    return rows

# ============================================
# ROUTE HOME
# ============================================
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request, db: Session = Depends(get_db)):
    history = db.query(PredictionHistory).order_by(
        PredictionHistory.created_at.desc()
    ).limit(50).all()
    
    info = get_model_info()
    html = read_template("index.html")
    
    html = html.replace("{{ best_model }}", info['best_model'])
    html = html.replace("{{ best_f1 }}", f"{info['f1_score']:.4f}")
    html = html.replace("{{ best_acc }}", f"{info['accuracy']:.4f}")
    html = html.replace("{{ best_recall }}", f"{info['recall']:.4f}")
    html = html.replace("{{ best_precision }}", f"{info['precision']:.4f}")
    html = html.replace("{{ threshold }}", f"{info['threshold']:.2f}")
    html = html.replace("{{ prediction_html }}", "")
    html = html.replace("{{ history_count }}", str(len(history)))
    html = html.replace("{{ history_rows }}", generate_history_rows(history))
    html = html.replace("{{ history_empty }}", "" if history else '<div class="empty-state"><i class="fas fa-inbox"></i><p>Belum ada riwayat prediksi</p></div>')
    html = html.replace("{{ error_message }}", "")
    
    return HTMLResponse(content=html)

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    satisfaction_level: float = Form(...),
    last_evaluation: float = Form(...),
    number_project: int = Form(...),
    average_montly_hours: int = Form(...),
    time_spend_company: int = Form(...),
    work_accident: int = Form(...),
    promotion_last_5years: int = Form(...),
    sales: str = Form(...),
    salary: str = Form(...),
    db: Session = Depends(get_db)
):
    errors = []
    if not (0.0 <= satisfaction_level <= 1.0):
        errors.append("Satisfaction Level harus antara 0.0 - 1.0")
    if not (0.0 <= last_evaluation <= 1.0):
        errors.append("Last Evaluation harus antara 0.0 - 1.0")
    if not (1 <= number_project <= 10):
        errors.append("Number of Projects harus antara 1 - 10")
    if not (50 <= average_montly_hours <= 400):
        errors.append("Monthly Hours harus antara 50 - 400")
    if not (1 <= time_spend_company <= 20):
        errors.append("Years at Company harus antara 1 - 20")
    if work_accident not in (0, 1):
        errors.append("Work Accident tidak valid")
    if promotion_last_5years not in (0, 1):
        errors.append("Promotion tidak valid")

    if errors:
        info = get_model_info()
        html = read_template("index.html")
        error_html = '<div class="alert alert-danger" style="padding:12px 20px;border-radius:8px;background:#fef2f2;color:#b91c1c;border:1px solid #fecaca;margin-bottom:16px;"><i class="fas fa-exclamation-triangle"></i> ' + '<br>'.join(errors) + '</div>'
        html = html.replace("{{ best_model }}", info['best_model'])
        html = html.replace("{{ best_f1 }}", f"{info['f1_score']:.4f}")
        html = html.replace("{{ best_acc }}", f"{info['accuracy']:.4f}")
        html = html.replace("{{ best_recall }}", f"{info['recall']:.4f}")
        html = html.replace("{{ best_precision }}", f"{info['precision']:.4f}")
        html = html.replace("{{ threshold }}", f"{info['threshold']:.2f}")
        html = html.replace("{{ prediction_html }}", "")
        html = html.replace("{{ error_message }}", error_html)
        history = db.query(PredictionHistory).order_by(PredictionHistory.created_at.desc()).limit(50).all()
        html = html.replace("{{ history_count }}", str(len(history)))
        html = html.replace("{{ history_rows }}", generate_history_rows(history))
        html = html.replace("{{ history_empty }}", "" if history else '<div class="empty-state"><i class="fas fa-inbox"></i><p>Belum ada riwayat prediksi</p></div>')
        return HTMLResponse(content=html)

    input_data = {
        'satisfaction_level': satisfaction_level,
        'last_evaluation': last_evaluation,
        'number_project': number_project,
        'average_montly_hours': average_montly_hours,
        'time_spend_company': time_spend_company,
        'work_accident': work_accident,
        'promotion_last_5years': promotion_last_5years,
        'sales': sales,
        'salary': salary
    }

    result, prob = ml_model.predict(input_data)
    threshold = ml_model.feature_info.get('optimal_threshold', 0.3) if ml_model.feature_info else 0.3

    new_record = PredictionHistory(
        satisfaction_level=satisfaction_level,
        last_evaluation=last_evaluation,
        number_project=number_project,
        average_montly_hours=average_montly_hours,
        time_spend_company=time_spend_company,
        work_accident=work_accident,
        promotion_last_5years=promotion_last_5years,
        sales=sales,
        salary=salary,
        prediction_result=result,
        probability=prob,
        threshold_used=threshold
    )
    db.add(new_record)
    db.commit()
    db.refresh(new_record)

    history = db.query(PredictionHistory).order_by(
        PredictionHistory.created_at.desc()
    ).limit(50).all()

    info = get_model_info()
    html = read_template("index.html")
    
    prediction_html = generate_prediction_html(result, f"{prob:.2%}", threshold)
    history_rows = generate_history_rows(history)
    history_empty = "" if history else '<div class="empty-state"><i class="fas fa-inbox"></i><p>Belum ada riwayat prediksi</p></div>'
    
    html = html.replace("{{ best_model }}", info['best_model'])
    html = html.replace("{{ best_f1 }}", f"{info['f1_score']:.4f}")
    html = html.replace("{{ best_acc }}", f"{info['accuracy']:.4f}")
    html = html.replace("{{ best_recall }}", f"{info['recall']:.4f}")
    html = html.replace("{{ best_precision }}", f"{info['precision']:.4f}")
    html = html.replace("{{ threshold }}", f"{info['threshold']:.2f}")
    html = html.replace("{{ prediction_html }}", prediction_html)
    html = html.replace("{{ history_count }}", str(len(history)))
    html = html.replace("{{ history_rows }}", history_rows)
    html = html.replace("{{ history_empty }}", history_empty)
    html = html.replace("{{ error_message }}", "")
    
    return HTMLResponse(content=html)

@app.post("/clear_history")
async def clear_history(db: Session = Depends(get_db)):
    db.query(PredictionHistory).delete()
    db.commit()
    return RedirectResponse(url="/", status_code=303)

# ============================================
# ROUTE EKSPLORASI
# ============================================
@app.get("/explore", response_class=HTMLResponse)
async def explore(request: Request):
    DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "HR_comma_sep.csv")
    
    info = get_model_info()
    html = read_template("explore.html")
    
    html = html.replace("{{ best_model }}", info['best_model'])
    html = html.replace("{{ best_f1 }}", f"{info['f1_score']:.4f}")
    html = html.replace("{{ best_acc }}", f"{info['accuracy']:.4f}")
    
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        
        total_rows = f"{len(df):,}"
        total_cols = len(df.columns)
        missing = df.isnull().sum().sum()
        
        target_counts = df['left'].value_counts().to_dict()
        target_pct = (df['left'].value_counts(normalize=True) * 100).to_dict()
        
        html = html.replace("{{ total_rows }}", total_rows)
        html = html.replace("{{ total_cols }}", str(total_cols))
        html = html.replace("{{ missing }}", str(missing))
        html = html.replace("{{ target_0 }}", f"{target_counts.get(0, 0):,}")
        html = html.replace("{{ target_pct_0 }}", f"{target_pct.get(0, 0):.1f}")
        html = html.replace("{{ target_1 }}", f"{target_counts.get(1, 0):,}")
        html = html.replace("{{ target_pct_1 }}", f"{target_pct.get(1, 0):.1f}")
        
        features = df.columns.tolist()
        features_html = ' '.join([f'<span class="feature-tag"><i class="fas fa-tag"></i> {f}</span>' for f in features])
        html = html.replace("{{ feature_count }}", str(len(features)))
        html = html.replace("{{ features_html }}", features_html)
        
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        stats_df = df[numeric_cols].describe().T
        
        for col in stats_df.columns:
            if col == 'count':
                stats_df[col] = stats_df[col].apply(lambda x: f"{int(x):,}")
            else:
                stats_df[col] = stats_df[col].apply(lambda x: f"{x:.4f}")
        
        numeric_stats = stats_df.to_html(classes='table table-striped table-bordered table-hover', border=0)
        html = html.replace("{{ numeric_stats }}", numeric_stats)
        
        sample_data = df.head(10).to_html(classes='table table-striped table-bordered table-hover', border=0, float_format=lambda x: f"{x:.4f}")
        html = html.replace("{{ sample_data }}", sample_data)
        
    else:
        html = html.replace("{{ total_rows }}", "0")
        html = html.replace("{{ total_cols }}", "0")
        html = html.replace("{{ missing }}", "0")
        html = html.replace("{{ target_0 }}", "0")
        html = html.replace("{{ target_pct_0 }}", "0.0")
        html = html.replace("{{ target_1 }}", "0")
        html = html.replace("{{ target_pct_1 }}", "0.0")
        html = html.replace("{{ feature_count }}", "0")
        html = html.replace("{{ features_html }}", "<p class='text-muted'>Dataset tidak ditemukan</p>")
        html = html.replace("{{ numeric_stats }}", "<p class='text-muted'>Dataset tidak ditemukan</p>")
        html = html.replace("{{ sample_data }}", "<p class='text-muted'>Dataset tidak ditemukan</p>")
    
    return HTMLResponse(content=html)

# ============================================
# ROUTE RESULTS
# ============================================
@app.get("/results", response_class=HTMLResponse)
async def results(request: Request):
    RESULTS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "comparison_results.csv")
    
    info = get_model_info()
    html = read_template("results.html")
    
    html = html.replace("{{ best_model }}", info['best_model'])
    html = html.replace("{{ best_f1 }}", f"{info['f1_score']:.4f}")
    html = html.replace("{{ best_acc }}", f"{info['accuracy']:.4f}")
    html = html.replace("{{ best_recall }}", f"{info['recall']:.4f}")
    
    if os.path.exists(RESULTS_PATH):
        df = pd.read_csv(RESULTS_PATH)
        df_sorted = df.sort_values('F1-Score', ascending=False)
        
        df_sorted = df_sorted.rename(columns={
            'Kombinasi': 'Kombinasi',
            'Model': 'Model',
            'Imbalance': 'Imbalance',
            'Accuracy': 'Accuracy',
            'Precision': 'Precision',
            'Recall': 'Recall',
            'F1-Score': 'F1-Score',
            'AUC-ROC': 'AUC-ROC',
            'Training Time (s)': 'Training Time (s)',
            'Epochs': 'Epochs'
        })
        
        for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']:
            df_sorted[col] = df_sorted[col].apply(lambda x: f"{x:.4f}")
        df_sorted['Training Time (s)'] = df_sorted['Training Time (s)'].apply(lambda x: f"{x:.2f}")
        df_sorted.insert(0, 'Rank', range(1, len(df_sorted) + 1))
        
        table_html = df_sorted.to_html(classes='table table-striped table-bordered table-hover', index=False, border=0)
        
        table_html = table_html.replace('<td>1</td>', '<td><span class="badge-rank">#1</span></td>')
        table_html = table_html.replace('<td>2</td>', '<td><span class="badge-rank">#2</span></td>')
        table_html = table_html.replace('<td>3</td>', '<td><span class="badge-rank">#3</span></td>')
        
        html = html.replace("{{ table_html }}", table_html)
        
    else:
        html = html.replace("{{ table_html }}", '<p class="text-muted">Hasil training belum ada</p>')
    
    return HTMLResponse(content=html)

# ============================================
# RUN
# ============================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)