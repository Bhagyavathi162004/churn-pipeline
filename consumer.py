import json, joblib
import pandas as pd
from kafka import KafkaConsumer
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

model    = joblib.load("model/catboost_model.pkl")
scaler   = joblib.load("model/scaler.pkl")
features = joblib.load("model/feature_columns.pkl")

consumer = KafkaConsumer(
    "churn-topic",
    bootstrap_servers="localhost:9092",
    value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    auto_offset_reset="latest"
)

print("Consumer started! Waiting for records...")

for message in consumer:
    record = message.value
    try:
        row = pd.DataFrame([record])
        cat_cols = row.select_dtypes(include="object").columns
        le = LabelEncoder()
        for col in cat_cols:
            row[col] = le.fit_transform(row[col].astype(str))
        row = row.reindex(columns=features, fill_value=0)
        row = pd.DataFrame(scaler.transform(row), columns=features)

        prob = model.predict_proba(row)[0][1]
        pred = int(prob > 0.3)

        print(f"CustomerID: {record.get('CustomerID','?')} | Churn Probability: {round(prob*100, 1)}% | Prediction: {'CHURN' if pred else 'STAY'}")

    except Exception as e:
        print(f"Error: {e}")