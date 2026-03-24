import streamlit as st
import numpy as np
import joblib
import pandas as pd

try:
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    imputer = joblib.load('imputer.pkl')
except Exception as e:
    st.error("Model files not found. Please run train_model.py first.")
    st.stop()

st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="⚕️", layout="centered")

st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    div.stButton > button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        transition: 0.3s;
    }
    div.stButton > button:hover {
        background-color: #45a049;
        transform: scale(1.02);
    }
    </style>
""", unsafe_allow_html=True)

st.title("⚕️ Diabetes Risk Predictor")
st.markdown("Enter the patient's medical metrics below to predict the likelihood of diabetes.")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    pregnancy_count = st.number_input("Pregnancy Count", min_value=0, max_value=20, value=1, step=1)
    glucose = st.number_input("Glucose Concentration", min_value=0.0, max_value=300.0, value=100.0, step=1.0)
    bp = st.number_input("Blood Pressure (mm Hg)", min_value=0.0, max_value=200.0, value=70.0, step=1.0)
    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0.0, max_value=120.0, value=20.0, step=1.0)

with col2:
    insulin = st.number_input("Insulin (mu U/ml)", min_value=0.0, max_value=1000.0, value=80.0, step=1.0)
    bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0, step=0.1)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
    age = st.number_input("Age", min_value=1, max_value=120, value=30, step=1)

if st.button("Predict 🔮", use_container_width=True):
    # Prepare input array
    features = np.array([[pregnancy_count, glucose, bp, skin_thickness, insulin, bmi, dpf, age]], dtype=float)
    
    # Process zeros to NaN for imputation (cols 1 to 5: Glucose to BMI)
    features_to_impute = features[:, 1:6]
    features_to_impute[features_to_impute == 0] = np.nan
    features[:, 1:6] = imputer.transform(features_to_impute)
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Predict
    prediction = model.predict(features_scaled)
    probability = model.predict_proba(features_scaled)[0][1]
    
    st.markdown("---")
    st.subheader("Results")
    
    if prediction[0] == 1:
        st.error(f"**High Risk of Diabetes** (Probability: {probability:.1%})")
        st.markdown("It is recommended to consult a healthcare professional right away.")
    else:
        st.success(f"**Low Risk of Diabetes** (Probability: {probability:.1%})")
        st.markdown("Continue maintaining a healthy lifestyle.")
