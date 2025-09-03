from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
from pathlib import Path
import pandas as pd

app = FastAPI(title="Churn Prediction API", version="0.2.0")

BEST_PATH = Path("models/best_model.pkl")
FALLBACK_PATH = Path("models/model.pkl")  # legacy fallback

# If no model artifacts exist, train once so the API can serve predictions
if not BEST_PATH.exists() and not FALLBACK_PATH.exists():
    from src.train import main as train_main
    train_main("data/raw/sample_telco_churn.csv")

# Prefer best_model.pkl; otherwise use legacy model.pkl
model_path = BEST_PATH if BEST_PATH.exists() else FALLBACK_PATH
pipe = joblib.load(model_path)

class Customer(BaseModel):
    gender: str = Field(..., examples=["Female"])
    SeniorCitizen: int = 0
    Partner: str = "No"
    Dependents: str = "No"
    tenure: int = 5
    PhoneService: str = "Yes"
    MultipleLines: str = "No"
    InternetService: str = "DSL"
    OnlineSecurity: str = "No"
    OnlineBackup: str = "No"
    DeviceProtection: str = "No"
    TechSupport: str = "No"
    StreamingTV: str = "No"
    StreamingMovies: str = "No"
    Contract: str = "Month-to-month"
    PaperlessBilling: str = "Yes"
    PaymentMethod: str = "Electronic check"
    MonthlyCharges: float = 29.85
    TotalCharges: float | None = None

@app.get("/")
def root():
    return {"ok": True, "message": "Churn Prediction API"}

@app.post("/predict")
def predict(cust: Customer):
    df = pd.DataFrame([cust.model_dump()])
    proba = float(pipe.predict_proba(df)[:, 1][0])
    label = int(proba >= 0.5)
    return {"churn_probability": proba, "churn": label}
