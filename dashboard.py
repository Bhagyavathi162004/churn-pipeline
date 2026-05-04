import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Churn Prediction Dashboard", layout="wide")

st.title("Real-Time Churn Prediction Pipeline")
st.markdown("Live predictions powered by CatBoost — ROC-AUC: **0.846**")

col1, col2, col3 = st.columns(3)
col1.metric("Model", "CatBoost")
col2.metric("ROC-AUC Score", "0.846")
col3.metric("Pipeline Status", "Live")

st.divider()

model    = joblib.load("model/catboost_model.pkl")
scaler   = joblib.load("model/scaler.pkl")
features = joblib.load("model/feature_columns.pkl")

st.subheader("Manual Churn Prediction")
st.markdown("Fill in customer details and get an instant prediction:")

c1, c2, c3 = st.columns(3)
tenure   = c1.slider("Tenure (months)", 0, 72, 12)
charges  = c2.slider("Monthly Charges ($)", 20, 120, 65)
contract = c3.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

if st.button("Predict Churn Risk"):
    row = pd.DataFrame([{
        "Tenure Months": tenure,
        "Monthly Charges": charges,
        "Contract": contract
    }])
    cat_cols = row.select_dtypes(include="object").columns
    le = LabelEncoder()
    for col in cat_cols:
        row[col] = le.fit_transform(row[col].astype(str))
    row = row.reindex(columns=features, fill_value=0)
    row = pd.DataFrame(scaler.transform(row), columns=features)

    prob = float(model.predict_proba(row)[0][1])
    risk = "High" if prob > 0.6 else "Medium" if prob > 0.3 else "Low"

    if risk == "High":
        st.error(f"HIGH RISK — {round(prob*100, 1)}% chance of churn!")
    elif risk == "Medium":
        st.warning(f"MEDIUM RISK — {round(prob*100, 1)}% chance of churn")
    else:
        st.success(f"LOW RISK — {round(prob*100, 1)}% chance of churn")

st.divider()
st.caption("Built with CatBoost + Streamlit | Real-Time ML Pipeline")