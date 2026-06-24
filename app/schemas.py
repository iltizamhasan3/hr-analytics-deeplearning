from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class PredictionInput(BaseModel):
    satisfaction_level: float
    last_evaluation: float
    number_project: int
    average_montly_hours: int
    time_spend_company: int
    work_accident: int
    promotion_last_5years: int
    sales: str
    salary: str

class PredictionHistoryResponse(BaseModel):
    id: int
    satisfaction_level: float
    last_evaluation: float
    number_project: int
    average_montly_hours: int
    time_spend_company: int
    work_accident: int
    promotion_last_5years: int
    sales: str
    salary: str
    prediction_result: str
    probability: float
    created_at: datetime

    class Config:
        from_attributes = True