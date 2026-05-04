from fastapi import FastAPI
from pydantic import BaseModel
import joblib, pandas as pd
from sklearn.preprocessing import LabelEncoder

app = FastAPI(title="Churn Prediction API")

model    = joblib.load("model/catboost_model.pkl")
scaler   = joblib.load("model/scaler.pkl")
features = joblib.load("model/feature_columns.pkl")

class CustomerRecord(BaseModel):
    data: dict

@app.post("/predict")
def predict(customer: CustomerRecord):
    row = pd.DataFrame([customer.data])
    cat_cols = row.select_dtypes(include="object").columns
    le = LabelEncoder()
    for col in cat_cols:
        row[col] = le.fit_transform(row[col].astype(str))
    row = row.reindex(columns=features, fill_value=0)
    row = pd.DataFrame(scaler.transform(row), columns=features)

    prob = float(model.predict_proba(row)[0][1])
    return {
        "churn_probability": round(prob, 4),
        "churn_prediction": int(prob > 0.3),
        "risk_level": "High" if prob > 0.6 else "Medium" if prob > 0.3 else "Low"
    }

@app.get("/health")
def health():
    return {"status": "ok", "model": "CatBoost", "roc_auc": 0.846}