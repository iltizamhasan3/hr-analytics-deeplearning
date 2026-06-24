from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.sql import func
from app.database import Base

class PredictionHistory(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    
    satisfaction_level = Column(Float)
    last_evaluation = Column(Float)
    number_project = Column(Integer)
    average_montly_hours = Column(Integer)
    time_spend_company = Column(Integer)
    work_accident = Column(Integer)
    promotion_last_5years = Column(Integer)
    sales = Column(String)
    salary = Column(String)
    
    prediction_result = Column(String)
    probability = Column(Float)
    created_at = Column(DateTime(timezone=True), server_default=func.now())