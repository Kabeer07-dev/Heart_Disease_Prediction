"""
Heart Disease Prediction Dashboard
-----------------------------------
A Streamlit dashboard that predicts the likelihood of heart disease
using a pre-trained Random Forest model.

Run with:  streamlit run app.py

Author: Your Name
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Heart Disease Prediction Dashboard",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# GLOBAL STYLES
# ============================================================
st.markdown(
    """
    <style>
        /* ---------- General ---------- */
        .stApp {
            background-color: 233D4D;
        }
        html, body, [class*="css"] {
            font-family: 'Segoe UI', 'Inter', sans-serif;
        }

        /* ---------- Header ---------- */
        .main-header {
            padding: 1.6rem 2rem;
            background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
            border-radius: 18px;
            color: white;
            margin-bottom: 1.6rem;
            box-shadow: 0 8px 20px rgba(37, 99, 235, 0.25);
        }
        .main-header h1 {
            margin: 0;
            font-size: 2rem;
            font-weight: 800;
        }
        .main-header p {
            margin: 0.3rem 0 0 0;
            font-size: 0.95rem;
            opacity: 0.9;
        }

        /* ---------- Dashboard cards ---------- */
        .dash-card {
            background-color: #F8FAFC;
            border: 1px solid #E5E9F0;
            border-radius: 16px;
            padding: 1.3rem 1.4rem;
            margin-bottom: 1.1rem;
            box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);
        }
        .dash-card h3 {
            margin-top: 0;
            margin-bottom: 0.9rem;
            font-size: 1.05rem;
            font-weight: 700;
            color: #1E293B;
        }
        .card-icon {
            font-size: 1.2rem;
            margin-right: 0.4rem;
        }

        /* ---------- KPI mini stat cards ---------- */
        .kpi-card {
            background-color: #F1F5F9;
            border-radius: 14px;
            padding: 0.9rem 1rem;
            text-align: center;
            border: 1px solid #E2E8F0;
        }
        .kpi-value {
            font-size: 1.3rem;
            font-weight: 800;
            color: #2563EB;
        }
        .kpi-label {
            font-size: 0.75rem;
            color: #64748B;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.03em;
        }

        /* ---------- Summary rows ---------- */
        .summary-row {
            display: flex;
            justify-content: space-between;
            padding: 0.45rem 0;
            border-bottom: 1px dashed #E2E8F0;
            font-size: 0.92rem;
        }
        .summary-row span:first-child {
            color: #64748B;
            font-weight: 500;
        }
        .summary-row span:last-child {
            color: #1E293B;
            font-weight: 700;
        }

        /* ---------- Result boxes ---------- */
        .result-healthy {
            background: linear-gradient(135deg, #DCFCE7 0%, #BBF7D0 100%);
            border: 1px solid #22C55E;
            border-radius: 16px;
            padding: 1.4rem;
            text-align: center;
            color: #14532D;
        }
        .result-risk {
            background: linear-gradient(135deg, #FEE2E2 0%, #FECACA 100%);
            border: 1px solid #EF4444;
            border-radius: 16px;
            padding: 1.4rem;
            text-align: center;
            color: #7F1D1D;
        }
        .result-title {
            font-size: 1.4rem;
            font-weight: 800;
            margin-bottom: 0.2rem;
        }

        /* ---------- Probability percentage ---------- */
        .prob-percent {
            font-size: 2.2rem;
            font-weight: 800;
        }

        /* ---------- Disclaimer ---------- */
        .disclaimer-box {
            background-color: #F1F5F9;
            border-left: 4px solid #2563EB;
            border-radius: 10px;
            padding: 0.9rem 1.1rem;
            font-size: 0.85rem;
            color: #475569;
            margin-top: 1.5rem;
        }

        /* ---------- Buttons ---------- */
        div.stButton > button:first-child {
            background-color: #2563EB;
            color: white;
            font-weight: 700;
            border-radius: 12px;
            padding: 0.6rem 1rem;
            border: none;
            width: 100%;
        }
        div.stButton > button:first-child:hover {
            background-color: #1D4ED8;
            color: white;
        }

        /* Reset button - secondary style */
        div.stButton > button[kind="secondary"] {
            background-color: #F1F5F9;
            color: #1E293B;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# LOAD MODEL ARTIFACTS
# ============================================================
@st.cache_resource
def load_artifacts():
    """Load the trained model, scaler and feature column order."""
    model = joblib.load("rf_heart.pkl")
    scaler = joblib.load("scaler.pkl")
    columns = joblib.load("column.pkl")
    return model, scaler, columns


try:
    model, scaler, model_columns = load_artifacts()
    ARTIFACTS_LOADED = True
except Exception as e:
    ARTIFACTS_LOADED = False
    load_error = str(e)

# Numeric columns that were scaled during training (StandardScaler).
NUMERIC_COLUMNS = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]

# ============================================================
# SESSION STATE DEFAULTS
# ============================================================
DEFAULTS = {
    "age": 50,
    "sex": "Male",
    "cp": "ATA",
    "resting_bp": 120,
    "cholesterol": 200,
    "fasting_bs": "0",
    "resting_ecg": "Normal",
    "max_hr": 150,
    "exercise_angina": "No",
    "oldpeak": 0.0,
    "st_slope": "Up",
    "prediction_made": False,
}
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


def reset_form():
    """Reset every widget back to its default value."""
    for key, value in DEFAULTS.items():
        st.session_state[key] = value


# ============================================================
# FEATURE ENCODING (must exactly mirror the training pipeline)
# ============================================================
def build_feature_frame(inputs: dict, columns: list) -> pd.DataFrame:
    """
    Build a single-row DataFrame with one-hot encoded categorical
    values, ordered exactly as `columns` (loaded from column.pkl).
    """
    row = {col: 0 for col in columns}

    # --- Numeric / direct passthrough features ---
    row["Age"] = inputs["age"]
    row["RestingBP"] = inputs["resting_bp"]
    row["Cholesterol"] = inputs["cholesterol"]
    row["FastingBS"] = int(inputs["fasting_bs"])
    row["MaxHR"] = inputs["max_hr"]
    row["Oldpeak"] = inputs["oldpeak"]

    # --- Sex (drop_first encoding -> Sex_M) ---
    if inputs["sex"] == "Male" and "Sex_M" in row:
        row["Sex_M"] = 1

    # --- Chest Pain Type (ASY is the dropped/reference category) ---
    cp_map = {
        "ATA": "ChestPainType_ATA",
        "NAP": "ChestPainType_NAP",
        "TA": "ChestPainType_TA",
        # "ASY" -> all dummy columns remain 0
    }
    if inputs["cp"] in cp_map and cp_map[inputs["cp"]] in row:
        row[cp_map[inputs["cp"]]] = 1

    # --- Resting ECG (LVH is the dropped/reference category) ---
    ecg_map = {
        "Normal": "RestingECG_Normal",
        "ST": "RestingECG_ST",
        # "LVH" -> all dummy columns remain 0
    }
    if inputs["resting_ecg"] in ecg_map and ecg_map[inputs["resting_ecg"]] in row:
        row[ecg_map[inputs["resting_ecg"]]] = 1

    # --- Exercise Angina (N is the dropped/reference category) ---
    if inputs["exercise_angina"] == "Yes" and "ExerciseAngina_Y" in row:
        row["ExerciseAngina_Y"] = 1

    # --- ST Slope (Down is the dropped/reference category) ---
    slope_map = {
        "Flat": "ST_Slope_Flat",
        "Up": "ST_Slope_Up",
        # "Down" -> all dummy columns remain 0
    }
    if inputs["st_slope"] in slope_map and slope_map[inputs["st_slope"]] in row:
        row[slope_map[inputs["st_slope"]]] = 1

    frame = pd.DataFrame([row], columns=columns)
    return frame


def scale_numeric_features(frame: pd.DataFrame, scaler, numeric_cols: list) -> pd.DataFrame:
    """Apply the trained StandardScaler to only the numeric columns."""
    scaled = frame.copy()
    scaled[numeric_cols] = scaler.transform(frame[numeric_cols])
    return scaled


# ============================================================
# SIDEBAR - PROJECT INFORMATION
# ============================================================
with st.sidebar:
    st.markdown("## ❤️ Project Information")
    st.markdown(
        """
        <div class="dash-card">
            <h3>📊 Model Details</h3>
            <div class="summary-row"><span>Model Used</span><span>Random Forest</span></div>
            <div class="summary-row"><span>Accuracy</span><span>86.96%</span></div>
            <div class="summary-row"><span>F1 Score</span><span>88.79%</span></div>
            <div class="summary-row"><span>Dataset</span><span>Heart Disease Dataset</span></div>
            <div class="summary-row"><span>Developer</span><span>Muhammad Kabeer</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="dash-card">
            <h3>ℹ️ About</h3>
            <p style="font-size:0.85rem; color:#475569; margin:0;">
            This dashboard uses a trained Random Forest Classifier to
            estimate the likelihood of heart disease from patient
            clinical parameters. Built with Streamlit, scikit-learn
            and pandas.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 🔗 Links")
    st.markdown("- [GitHub Repository](#)")
    st.markdown("- [Dataset Source](#)")

# ============================================================
# HEADER
# ============================================================
st.markdown(
    """
    <div class="main-header">
        <h1>❤️ Heart Disease Prediction Dashboard</h1>
        <p>Predict the likelihood of heart disease using a trained Random Forest model.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not ARTIFACTS_LOADED:
    st.error(
        f"⚠️ Could not load model artifacts. Make sure `rf_heart.pkl`, "
        f"`scaler.pkl` and `column.pkl` are in the same folder as `app.py`.\n\n"
        f"Details: {load_error}"
    )
    st.stop()

# ============================================================
# TOP KPI STRIP
# ============================================================
kpi_cols = st.columns(4)
kpi_data = [
    ("🧠", "Model", "Random Forest"),
    ("🎯", "Accuracy", "86.96%"),
    ("⚖️", "F1 Score", "88.79%"),
    ("📁", "Features", f"{len(model_columns)} inputs"),
]
for col, (icon, label, value) in zip(kpi_cols, kpi_data):
    with col:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div style="font-size:1.4rem;">{icon}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-label">{label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.write("")

# ============================================================
# MAIN DASHBOARD LAYOUT
# ============================================================
left_col, right_col = st.columns([1.3, 1])

# --------------------------------------------------------
# LEFT COLUMN — Patient Information Form (grouped cards)
# --------------------------------------------------------
with left_col:

    st.markdown(
        '<div class="dash-card"><h3><span class="card-icon">🧍</span>Demographics</h3>',
        unsafe_allow_html=True,
    )
    d1, d2 = st.columns(2)
    with d1:
        age = st.number_input(
            "Age", min_value=1, max_value=120, key="age", help="Patient age in years"
        )
    with d2:
        sex = st.selectbox("Sex", ["Male", "Female"], key="sex")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        '<div class="dash-card"><h3><span class="card-icon">🩺</span>Vital Signs</h3>',
        unsafe_allow_html=True,
    )
    v1, v2 = st.columns(2)
    with v1:
        resting_bp = st.number_input(
            "Resting Blood Pressure (mm Hg)", min_value=0, max_value=250, key="resting_bp"
        )
    with v2:
        cholesterol = st.number_input(
            "Cholesterol (mg/dl)", min_value=0, max_value=700, key="cholesterol"
        )
    max_hr = st.slider(
        "Maximum Heart Rate Achieved", min_value=60, max_value=220, key="max_hr"
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        '<div class="dash-card"><h3><span class="card-icon">📈</span>Cardiac Test Results</h3>',
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    with c1:
        cp = st.selectbox(
            "Chest Pain Type", ["ATA", "NAP", "ASY", "TA"], key="cp"
        )
    with c2:
        resting_ecg = st.selectbox(
            "Resting ECG", ["Normal", "ST", "LVH"], key="resting_ecg"
        )
    e1, e2 = st.columns(2)
    with e1:
        fasting_bs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl", ["0", "1"], key="fasting_bs"
        )
    with e2:
        exercise_angina = st.selectbox(
            "Exercise-Induced Angina", ["Yes", "No"], key="exercise_angina"
        )
    o1, o2 = st.columns(2)
    with o1:
        oldpeak = st.number_input(
            "Oldpeak (ST depression)", min_value=-5.0, max_value=10.0, step=0.1, key="oldpeak"
        )
    with o2:
        st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"], key="st_slope")
    st.markdown("</div>", unsafe_allow_html=True)

    btn_col1, btn_col2 = st.columns([2, 1])
    with btn_col1:
        predict_clicked = st.button("🔍 Predict Heart Disease", use_container_width=True)
    with btn_col2:
        st.button("♻️ Reset Form", on_click=reset_form, use_container_width=True)

    with st.expander("📚 What do these features mean?"):
        st.markdown(
            """
            - **Age**: Age of the patient in years.
            - **Sex**: Biological sex of the patient.
            - **Chest Pain Type**: ATA (Atypical Angina), NAP (Non-Anginal Pain),
              ASY (Asymptomatic), TA (Typical Angina).
            - **Resting Blood Pressure**: Blood pressure measured at rest (mm Hg).
            - **Cholesterol**: Serum cholesterol level (mg/dl).
            - **Fasting Blood Sugar**: 1 if fasting blood sugar > 120 mg/dl, else 0.
            - **Resting ECG**: Resting electrocardiogram results.
            - **Maximum Heart Rate**: Highest heart rate achieved during exercise.
            - **Exercise-Induced Angina**: Whether exercise triggers chest pain.
            - **Oldpeak**: ST depression induced by exercise relative to rest.
            - **ST Slope**: Slope of the peak exercise ST segment.
            """
        )

# --------------------------------------------------------
# RIGHT COLUMN — Patient Summary + Prediction Result
# --------------------------------------------------------
with right_col:
    st.markdown(
        f"""
        <div class="dash-card">
            <h3><span class="card-icon">📋</span>Patient Summary</h3>
            <div class="summary-row"><span>Age</span><span>{st.session_state.age}</span></div>
            <div class="summary-row"><span>Sex</span><span>{st.session_state.sex}</span></div>
            <div class="summary-row"><span>Chest Pain Type</span><span>{st.session_state.cp}</span></div>
            <div class="summary-row"><span>Resting BP</span><span>{st.session_state.resting_bp} mm Hg</span></div>
            <div class="summary-row"><span>Cholesterol</span><span>{st.session_state.cholesterol} mg/dl</span></div>
            <div class="summary-row"><span>Fasting Blood Sugar</span><span>{st.session_state.fasting_bs}</span></div>
            <div class="summary-row"><span>Resting ECG</span><span>{st.session_state.resting_ecg}</span></div>
            <div class="summary-row"><span>Max Heart Rate</span><span>{st.session_state.max_hr}</span></div>
            <div class="summary-row"><span>Exercise Angina</span><span>{st.session_state.exercise_angina}</span></div>
            <div class="summary-row"><span>Oldpeak</span><span>{st.session_state.oldpeak}</span></div>
            <div class="summary-row"><span>ST Slope</span><span>{st.session_state.st_slope}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    result_placeholder = st.empty()

    if predict_clicked:
        with st.spinner("Analyzing patient data..."):
            import time

            time.sleep(0.8)  # small delay so the spinner is visible

            user_inputs = {
                "age": st.session_state.age,
                "sex": st.session_state.sex,
                "cp": st.session_state.cp,
                "resting_bp": st.session_state.resting_bp,
                "cholesterol": st.session_state.cholesterol,
                "fasting_bs": st.session_state.fasting_bs,
                "resting_ecg": st.session_state.resting_ecg,
                "max_hr": st.session_state.max_hr,
                "exercise_angina": st.session_state.exercise_angina,
                "oldpeak": st.session_state.oldpeak,
                "st_slope": st.session_state.st_slope,
            }

            # 1. Build the feature frame in the exact column order.
            feature_frame = build_feature_frame(user_inputs, model_columns)

            # 2. Scale only the numeric columns using the trained scaler.
            scaled_frame = scale_numeric_features(feature_frame, scaler, NUMERIC_COLUMNS)

            # 3. Predict class and probability.
            prediction = model.predict(scaled_frame)[0]
            probability = model.predict_proba(scaled_frame)[0]

            # Probability of the "Heart Disease" class (class label 1).
            classes = list(model.classes_)
            disease_index = classes.index(1) if 1 in classes else int(np.argmax(probability))
            disease_probability = probability[disease_index]

        with result_placeholder.container():
            if prediction == 1:
                st.markdown(
                    f"""
                    <div class="result-risk">
                        <div style="font-size:2.2rem;">⚠️</div>
                        <div class="result-title">High Risk of Heart Disease</div>
                        <p style="margin:0.3rem 0 0 0;">Please consult a cardiologist for further evaluation.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="result-healthy">
                        <div style="font-size:2.2rem;">✅</div>
                        <div class="result-title">Low Risk</div>
                        <p style="margin:0.3rem 0 0 0;">No strong indicators of heart disease were found.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.write("")
            st.markdown("**Heart Disease Probability**")
            st.progress(float(disease_probability))
            st.markdown(
                f'<div class="prob-percent">{disease_probability * 100:.1f}%</div>',
                unsafe_allow_html=True,
            )
    else:
        with result_placeholder.container():
            st.markdown(
                """
                <div class="dash-card" style="text-align:center; color:#64748B;">
                    <div style="font-size:1.8rem;">🩺</div>
                    <p style="margin:0.4rem 0 0 0;">
                    Fill in the patient information and click
                    <b>Predict Heart Disease</b> to see the result here.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ============================================================
# DISCLAIMER
# ============================================================
st.markdown(
    """
    <div class="disclaimer-box">
        ⚠️ <b>Disclaimer:</b> This prediction is generated by a Machine Learning model
        and is intended for educational purposes only. It should not be used as a
        substitute for professional medical advice.
    </div>
    """,
    unsafe_allow_html=True,
)