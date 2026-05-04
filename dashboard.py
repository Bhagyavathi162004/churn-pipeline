import streamlit as st
import pandas as pd
import requests
import time

st.set_page_config(page_title="Churn Prediction Dashboard", layout="wide")

st.title("Real-Time Churn Prediction Pipeline")
st.markdown("Live predictions powered by CatBoost — ROC-AUC: **0.846**")

col1, col2, col3 = st.columns(3)
col1.metric("Model", "CatBoost")
col2.metric("ROC-AUC Score", "0.846")
col3.metric("Pipeline Status", "Live")

st.divider()

st.subheader("Manual Churn Prediction")
st.markdown("Fill in customer details and get an instant prediction:")

c1, c2, c3 = st.columns(3)
tenure   = c1.slider("Tenure (months)", 0, 72, 12)
charges  = c2.slider("Monthly Charges ($)", 20, 120, 65)
contract = c3.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

if st.button("Predict Churn Risk"):
    try:
        payload = {"data": {
            "Tenure Months": tenure,
            "Monthly Charges": charges,
            "Contract": contract
        }}
        resp = requests.post("http://localhost:8000/predict", json=payload)
        result = resp.json()
        prob = result["churn_probability"]
        risk = result["risk_level"]

        if risk == "High":
            st.error(f"HIGH RISK — {round(prob*100,1)}% chance of churn!")
        elif risk == "Medium":
            st.warning(f"MEDIUM RISK — {round(prob*100,1)}% chance of churn")
        else:
            st.success(f"LOW RISK — {round(prob*100,1)}% chance of churn")
    except:
        st.info("API not running yet. Start api.py first!")

st.divider()
st.caption("Built with FastAPI + CatBoost + Streamlit")