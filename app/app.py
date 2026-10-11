
from pathlib import Path

import joblib
import numpy as np
import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_DIR / "results" / "models"
ARRAY_DIR = PROJECT_DIR / "results" / "arrays"

st.set_page_config(
    page_title="Electrical Fault Detection",
    page_icon="⚡",
    layout="centered"
)

st.title("⚡ Electrical Fault Detection System")
st.write(
    "Enter the six electrical measurements below. "
    "The application will predict whether a fault is present "
    "using two trained machine-learning models."
)

try:
    logistic_data = np.load(MODEL_DIR / "logistic_parameters.npz")
    prepared_data = np.load(ARRAY_DIR / "prepared_data.npz")
    rbf_model = joblib.load(MODEL_DIR / "best_rbf_svm.joblib")
except FileNotFoundError as error:
    st.error(f"Required model or data file was not found: {error}")
    st.info(
        "Run the Task 1, Task 3, and Task 4 scripts successfully "
        "and check the results/models and results/arrays folders."
    )
    st.stop()

st.subheader("Enter Electrical Measurements")

with st.form("fault_input_form"):
    col1, col2 = st.columns(2)

    with col1:
        Ia = st.number_input("Ia (Current)", value=0.0, format="%.6f")
        Ib = st.number_input("Ib (Current)", value=0.0, format="%.6f")
        Ic = st.number_input("Ic (Current)", value=0.0, format="%.6f")

    with col2:
        Va = st.number_input("Va (Voltage)", value=0.0, format="%.6f")
        Vb = st.number_input("Vb (Voltage)", value=0.0, format="%.6f")
        Vc = st.number_input("Vc (Voltage)", value=0.0, format="%.6f")

    submitted = st.form_submit_button("Detect Fault")

if submitted:
    raw_features = np.array([[Ia, Ib, Ic, Va, Vb, Vc]])

    # Set Q: original measurements and their squares
    q_features = np.column_stack([
        raw_features,
        raw_features ** 2
    ])

    mean_q = prepared_data["train_mean_Q"]
    std_q = prepared_data["train_std_Q"]
    q_scaled = (q_features - mean_q) / std_q

    # Add the intercept required by Logistic Regression
    q_input = np.column_stack([
        np.ones(len(q_scaled)),
        q_scaled
    ])

    theta_q = logistic_data["theta_Q"]
    logistic_probability = 1 / (
        1 + np.exp(-np.clip(q_input @ theta_q, -500, 500))
    )
    logistic_prediction = int(logistic_probability[0] >= 0.5)

    # Set R: original six standardised measurements for RBF SVM
    mean_r = prepared_data["train_mean_R"]
    std_r = prepared_data["train_std_R"]
    r_input = (raw_features - mean_r) / std_r

    svm_prediction = int(rbf_model.predict(r_input)[0])

    st.subheader("Prediction Results")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Set Q Logistic Regression**")
        if logistic_prediction == 1:
            st.error("Fault Detected")
        else:
            st.success("No Fault Detected")

        st.write(
            f"Fault probability: "
            f"{logistic_probability[0] * 100:.2f}%"
        )

    with col2:
        st.markdown("**RBF Support Vector Machine**")
        if svm_prediction == 1:
            st.error("Fault Detected")
        else:
            st.success("No Fault Detected")

    if logistic_prediction == svm_prediction:
        st.info("Both models give the same prediction.")
    else:
        st.warning(
            "The models disagree. Further electrical-system "
            "checks are recommended."
        )

st.divider()
st.caption(
    "Academic project prototype. Predictions should not be used "
    "as the sole basis for real electrical protection decisions."
)
