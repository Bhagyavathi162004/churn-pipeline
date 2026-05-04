import json, time, random
import pandas as pd
from kafka import KafkaProducer

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

df = pd.read_excel("Telco_customer_churn.xlsx")
df.columns = df.columns.str.strip()
df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
df.dropna(inplace=True)

drop_cols = ["Churn Label","Churn Score","CLTV","Churn Reason","Churn Value"]
feature_df = df.drop(columns=drop_cols, errors="ignore")

print("Producer started! Sending customer records every 2 seconds...")

while True:
    record = feature_df.sample(1).iloc[0].to_dict()
    producer.send("churn-topic", value=record)
    print(f"Sent record for CustomerID: {record.get('CustomerID', 'unknown')}")
    time.sleep(2)