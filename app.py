# app.py — CSV upload version
import io
import numpy as np
import pandas as pd
import streamlit as st
import cloudpickle

st.set_page_config(page_title="Employee Attrition Predictor", page_icon="💼", layout="wide")
st.title("💼 Employee Attrition Prediction — CSV Upload")

THRESHOLD = 0.40  # chosen in Phase 7

# ---------- Load artifacts ----------
@st.cache_resource(show_spinner=False)
def load_artifacts():
    with open("final_attrition_model.pkl", "rb") as f:
        model = cloudpickle.load(f)
    with open("data_preprocessor.pkl", "rb") as f:
        preprocessor = cloudpickle.load(f)
    return model, preprocessor

model, preprocessor = load_artifacts()

# Names the preprocessor expects at transform-time
EXPECTED_COLS = list(preprocessor.feature_names_in_)  # requires sklearn>=1.0
EXPECTED_SET = set(EXPECTED_COLS)

# ---------- Engineering helpers ----------
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create engineered columns used during training if they are missing.
    Safe on existing cols (won't overwrite).
    """
    # Only create if the inputs exist
    if "YearsPerCompany" not in df.columns and {"TotalWorkingYears", "NumCompaniesWorked"}.issubset(df.columns):
        df["YearsPerCompany"] = df["TotalWorkingYears"] / (df["NumCompaniesWorked"] + 1)

    if "YearsSinceLastPromotionRatio" not in df.columns and {"YearsSinceLastPromotion", "YearsAtCompany"}.issubset(df.columns):
        df["YearsSinceLastPromotionRatio"] = df["YearsSinceLastPromotion"] / (df["YearsAtCompany"] + 1)

    if "IncomePerYear" not in df.columns and {"MonthlyIncome", "TotalWorkingYears"}.issubset(df.columns):
        # Use annual income for better interpretability
        df["IncomePerYear"] = (df["MonthlyIncome"] * 12) / (df["TotalWorkingYears"] + 1)

    # ExperienceLevel from TotalWorkingYears
    if "ExperienceLevel" not in df.columns and "TotalWorkingYears" in df.columns:
        bins = [-np.inf, 5, 10, 20, np.inf]
        labels = ["Junior", "Mid", "Senior", "Veteran"]
        df["ExperienceLevel"] = pd.cut(
            df["TotalWorkingYears"], bins=bins, labels=labels, right=False, include_lowest=True
        )

    return df

def align_to_expected(user_df: pd.DataFrame, expected_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """
    Ensure user_df has all columns expected by the preprocessor.
    - Tries to add engineered columns.
    - Reports missing columns that we cannot derive.
    - Fills any still-missing with neutral defaults (0) to avoid hard failure.
      (You can switch this to a hard error if you prefer strict validation.)
    """
    df = user_df.copy()
    df = add_engineered_features(df)

    missing = [c for c in expected_cols if c not in df.columns]
    # Try to be strict by default: if important raw columns are missing, inform user.
    # You can decide which to treat as hard required; here we show a warning and fill neutrals.
    if missing:
        st.warning(
            f"{len(missing)} expected columns were missing in the upload. "
            "They will be filled with neutral defaults (0 / NaN-safe). "
            "Consider adding them to improve predictions.\n\n"
            + ", ".join(missing)
        )
        for col in missing:
            df[col] = 0  # neutral numeric default; OHE will ignore unknown categories anyway

    # Reorder to expected
    df = df[expected_cols]
    return df, missing

def predict_df(df_input: pd.DataFrame) -> pd.DataFrame:
    X_ready, missing = align_to_expected(df_input, EXPECTED_COLS)
    # Transform and predict
    X_tx = preprocessor.transform(X_ready)
    proba = model.predict_proba(X_tx)[:, 1]
    pred = (proba >= THRESHOLD).astype(int)
    out = df_input.copy()
    out["Attrition_Probability"] = np.round(proba, 4)
    out["Attrition_Prediction"] = np.where(pred == 1, "Likely to Leave", "Likely to Stay")
    out["Decision_Threshold"] = THRESHOLD
    return out

# ---------- Sidebar: template & instructions ----------
with st.sidebar:
    st.header("📄 Upload Instructions")
    st.write(
        "• Upload a CSV with **one or more employees** (one row per employee).\n"
        "• If some engineered columns are missing (e.g., `YearsPerCompany`, `ExperienceLevel`), "
        "the app will compute them if raw fields exist.\n"
        "• The model uses a decision threshold of **0.40**."
    )
    # Offer a CSV template of expected headers only
    template_csv = io.StringIO()
    pd.DataFrame(columns=EXPECTED_COLS).to_csv(template_csv, index=False)
    st.download_button("⬇️ Download Expected Columns (template.csv)", template_csv.getvalue(), file_name="template.csv", mime="text/csv")

# ---------- Main: uploader ----------
uploaded = st.file_uploader("Upload employee CSV", type=["csv"])
if uploaded is None:
    st.info("Upload a CSV to get predictions. You can start from the template in the sidebar.")
    st.stop()

try:
    df_in = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Could not read the CSV: {e}")
    st.stop()

st.subheader("👀 Preview of Uploaded Data")
st.dataframe(df_in.head(10), use_container_width=True)

# Quick check: empty file
if df_in.empty:
    st.error("The uploaded CSV has no rows.")
    st.stop()

# ---------- Predict ----------
with st.spinner("Scoring…"):
    try:
        results = predict_df(df_in)
    except Exception as e:
        st.error(f"Prediction failed. Please verify column names and types. Details: {e}")
        st.stop()

st.success("Done! See predictions below.")
st.subheader("🔮 Predictions")
st.dataframe(results.head(20), use_container_width=True)

# ---------- Download results ----------
csv_out = io.StringIO()
results.to_csv(csv_out, index=False)
st.download_button("💾 Download Predictions (CSV)", csv_out.getvalue(), file_name="attrition_predictions.csv", mime="text/csv")
