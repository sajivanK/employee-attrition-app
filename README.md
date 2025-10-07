# 💼 Employee Attrition Prediction using Machine Learning  

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-success)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📘 Overview  

This project predicts whether an employee is **likely to leave (attrition)** based on HR analytics data from IBM.  
It combines **data preprocessing, exploratory data analysis (EDA), feature engineering, model building, explainability, calibration, and Streamlit deployment** into one complete end-to-end solution.  

The goal is to help HR departments make **data-driven retention decisions**, identify high-risk employees early, and understand **why** attrition occurs.

---

## 🎯 Objectives  

- Analyze HR employee data to identify key factors driving attrition.  
- Build predictive models to estimate employee churn likelihood.  
- Apply **threshold tuning** to maximize recall (catch more leavers).  
- Use **SHAP and logistic coefficients** for explainable insights.  
- Calibrate model probabilities for trustworthy HR decision-making.  
- Deploy a **Streamlit web app** for real-time prediction.

---

## 🧩 Project Workflow  

1. **Data Understanding & Cleaning**  
   - Loaded IBM HR dataset (`WA_Fn-UseC_-HR-Employee-Attrition.csv`)  
   - Checked nulls, data types, and class distribution  

2. **Exploratory Data Analysis (EDA)**  
   - Visualized attrition patterns by Age, JobRole, Overtime, Income, etc.  
   - Discovered that **OverTime**, **Travel Frequency**, and **Low Salary** correlate with higher attrition.  

3. **Feature Engineering**  
   - Added derived features:  
     - `YearsPerCompany` = TotalWorkingYears / (NumCompaniesWorked + 1)  
     - `IncomePerYear` = MonthlyIncome / (TotalWorkingYears + 1)  
     - `ExperienceLevel` = Categorized working experience into Junior, Mid, Senior, Veteran  

4. **Model Building (Baseline)**  
   - Implemented Logistic Regression, Random Forest, and XGBoost using Scikit-Learn Pipelines  
   - Encoded categorical features using `OneHotEncoder` and scaled numeric features  

5. **Threshold Tuning & Explainability**  
   - Adjusted Logistic Regression threshold (0.40 → Recall ↑ 0.74)  
   - Extracted feature importance and visualized with SHAP values  

6. **Model Strengthening (Grid Search)**  
   - Tuned RF and XGBoost hyperparameters using `GridSearchCV`  
   - Best ROC-AUC: Logistic Regression = 0.80, RF = 0.80, XGB = 0.79  

7. **Calibration & Reliability**  
   - Applied `CalibratedClassifierCV` with isotonic regression  
   - Achieved **Brier Score = 0.1057** (excellent calibration)  

8. **Deployment (Streamlit App)**  
   - Developed an interactive UI for HR teams to input employee details and get:  
     - Prediction: *Likely to Leave / Stay*  
     - Attrition Probability (%)  
   - Integrated saved model (`final_attrition_model.pkl`) and preprocessor  

---

## 🧠 Key Insights  

- Employees who work **frequent overtime**, travel often, or have **low income** are more likely to leave.  
- **Years with current manager** and **high job satisfaction** reduce attrition risk.  
- Logistic Regression provided the **best recall (0.74)** — ideal for minimizing missed leavers.  
- Calibrated probabilities ensure predictions reflect real-world likelihoods.  

---

## 🧾 Results Summary  

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|:--|:--:|:--:|:--:|:--:|:--:|
| Logistic Regression (Threshold = 0.40) | 0.76 | 0.34 | **0.74** | 0.47 | **0.80** |
| Tuned Random Forest | 0.84 | 0.48 | 0.21 | 0.29 | 0.76 |
| Tuned XGBoost | **0.86** | **0.64** | 0.30 | 0.41 | 0.76 |

---


---




