import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Employee Attrition Predictor", page_icon="💼", layout="wide")

# load model & preprocessor
model = joblib.load("final_attrition_model.pkl")
preprocessor = joblib.load("data_preprocessor.pkl")

st.title("💼 Employee Attrition Prediction System")

# collect inputs
age = st.number_input("Age", 18, 60, 30)
monthly_income = st.number_input("Monthly Income", 1000, 20000, 5000)
overtime = st.selectbox("OverTime", ["Yes", "No"])
distance = st.slider("Distance From Home", 1, 30, 5)
job_satisfaction = st.slider("Job Satisfaction", 1, 4, 3)

# build single-row DataFrame
input_df = pd.DataFrame({
    "Age":[age],
    "MonthlyIncome":[monthly_income],
    "OverTime":[overtime],
    "DistanceFromHome":[distance],
    "JobSatisfaction":[job_satisfaction]
})

# preprocess & predict
X_transformed = preprocessor.transform(input_df)
proba = model.predict_proba(X_transformed)[0][1]
pred = "Likely to Leave" if proba > 0.40 else "Likely to Stay"

st.markdown(f"### 🔮 Prediction: **{pred}**")
st.metric("Attrition Probability", f"{proba*100:.1f} %")
