"""
NITA Academic Analytics - University Admission Predictor
========================================================
Interactive Streamlit dashboard (Cloud Computing for Data Science project).

The prediction logic is a multiple linear regression whose coefficients were
fitted locally on Admission_Predict_Ver1.1.csv (500 records). It mirrors the regression
model built with AWS SageMaker Canvas so the dashboard runs without needing an
AWS endpoint.

Run locally:   streamlit run app.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------
# Model definition (regression coefficients fitted on the full dataset)
# --------------------------------------------------------------------------
INTERCEPT = -1.275726
COEFFICIENTS = {
    "GRE Score": 0.001859,
    "TOEFL Score": 0.002778,
    "University Rating": 0.005941,
    "SOP": 0.001586,
    "LOR": 0.016859,
    "CGPA": 0.118385,
    "Research": 0.024307,
}
# Model quality reported by AWS SageMaker Canvas (Quick Build, regression)
MODEL_METRICS = {"RMSE": 0.068, "MSE": 0.005}
# Column impact (%) reported by SageMaker Canvas
FEATURE_IMPORTANCE = {
    "CGPA": 47.785, "GRE Score": 16.441, "TOEFL Score": 8.939, "LOR": 8.773,
    "Research": 8.509, "SOP": 6.630, "University Rating": 2.922,
}


def predict_admission(features: dict) -> float:
    """Return the predicted chance of admission, clipped to the range [0, 1]."""
    score = INTERCEPT + sum(COEFFICIENTS[k] * features[k] for k in COEFFICIENTS)
    return float(np.clip(score, 0.0, 1.0))


# --------------------------------------------------------------------------
# Page layout and styling
# --------------------------------------------------------------------------
st.set_page_config(page_title="NITA Admission Predictor", page_icon="🎓", layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem;}
    div[data-testid="stMetricValue"] {font-size: 2.4rem; color: #1F4E79;}
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("🎓 NITA Academic Analytics")
st.subheader("University Admission Probability Predictor")
st.caption("Regression model trained on 500 applicant records (Admission_Predict_Ver1.1).")

# --------------------------------------------------------------------------
# Sidebar - user inputs
# --------------------------------------------------------------------------
st.sidebar.header("Applicant Profile")
gre = st.sidebar.slider("GRE Score", 260, 340, 316)
toefl = st.sidebar.slider("TOEFL Score", 80, 120, 107)
cgpa = st.sidebar.slider("CGPA (out of 10)", 6.0, 10.0, 8.6, step=0.01)
sop = st.sidebar.slider("Statement of Purpose (SOP)", 1.0, 5.0, 3.5, step=0.5)
lor = st.sidebar.slider("Letter of Recommendation (LOR)", 1.0, 5.0, 3.5, step=0.5)
uni_rating = st.sidebar.number_input("University Rating", min_value=1, max_value=5, value=3, step=1)
research = st.sidebar.radio("Research Experience", ["No", "Yes"], horizontal=True)

profile = {
    "GRE Score": gre, "TOEFL Score": toefl, "University Rating": uni_rating,
    "SOP": sop, "LOR": lor, "CGPA": cgpa, "Research": 1 if research == "Yes" else 0,
}
probability = predict_admission(profile)

# --------------------------------------------------------------------------
# Main panel - prediction and charts
# --------------------------------------------------------------------------
col1, col2, col3 = st.columns(3)
col1.metric("Chance of Admit", f"{probability:.1%}")
col2.metric("Model RMSE", f"{MODEL_METRICS['RMSE']:.4f}")
col3.metric("Model MSE", f"{MODEL_METRICS['MSE']:.3f}")

if probability >= 0.80:
    st.success("High chance of admission.")
elif probability >= 0.60:
    st.info("Moderate chance of admission.")
else:
    st.warning("Low chance of admission - consider strengthening the profile.")
st.progress(probability)

left, right = st.columns(2)

with left:
    st.markdown("#### Feature importance (model)")
    fig1, ax1 = plt.subplots(figsize=(5, 3.6))
    names = list(FEATURE_IMPORTANCE)[::-1]
    values = [FEATURE_IMPORTANCE[n] for n in names]
    ax1.barh(names, values, color="#1F4E79")
    ax1.set_xlabel("Relative importance (%)")
    st.pyplot(fig1)

with right:
    st.markdown("#### Sensitivity: how CGPA changes the outcome")
    cgpa_range = np.linspace(6.0, 10.0, 41)
    curve = [predict_admission({**profile, "CGPA": c}) for c in cgpa_range]
    fig2, ax2 = plt.subplots(figsize=(5, 3.6))
    ax2.plot(cgpa_range, curve, color="#2E75B6", linewidth=2)
    ax2.scatter([cgpa], [probability], color="crimson", zorder=5, label="Your profile")
    ax2.set_xlabel("CGPA")
    ax2.set_ylabel("Chance of admit")
    ax2.set_ylim(0, 1)
    ax2.legend()
    st.pyplot(fig2)

with st.expander("Model details and your inputs"):
    st.write("Current input profile:")
    st.dataframe(pd.DataFrame([profile]), hide_index=True)
    st.write("Regression coefficients:")
    st.dataframe(pd.DataFrame(COEFFICIENTS, index=["Coefficient"]).T)