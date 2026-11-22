import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
st.set_page_config(page_title="Financial Distress Warning System", layout="wide")
st.title(" Financial Disterss prediction System")
st.markdown("This system utilizes an Ensemble Model (XGBoost + Random Forest) with 82.82% accuracy.")
@st.cache_resource
def load_assets():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    xgb_path = os.path.join(base_dir, 'xgb_model.pkl')
    rf_path = os.path.join(base_dir, 'rf_model.pkl')
    weights_path = os.path.join(base_dir, 'ensemble_weights.pkl')
    features_path = os.path.join(base_dir, 'model_features.pkl')
    for path in [xgb_path, rf_path, weights_path, features_path]:
        if not os.path.exists(path):
            st.error(f"File Not FOUND at: {path}")
            st.stop()
    xgb = joblib.load(xgb_path)
    rf = joblib.load(rf_path)
    weights = joblib.load(weights_path)
    features = joblib.load(features_path)        
    return xgb, rf, weights, features
try:
    xgb_model, rf_model, weights, expected_features = load_assets()
except FileNotFoundError:
    st.error("Moodel files not found! please ensure the .pkl files are in the same directory.")
    st.stop()
st.sidebar.header("Data Input")
uploaded_file = st.sidebar.file_uploader("Upload Company Data (.xlsx or .csv)", type=["xlsx", "csv"])
if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    st.write("preview of uploaded data:")
    st.dataframe(df.head())
    if st.button("Predict Status"):
        try:
            df = df[expected_features]
            p_xgb = xgb_model.predict_proba(df)[:, 1]
            p_rf = rf_model.predict_proba(df)[:, 1]
            w_xgb = weights['w_xgb']
            w_rf = weights['w_rf']
            p_ens = (p_xgb * w_xgb + p_rf * w_rf) / (w_xgb + w_rf)
            st.subheader("Prediction Results:")
            for i, prob in enumerate(p_ens):
                status = "Distress (High Risk)" if prob >= 0.20 else "Healthy"
                st.write(f"**Company {i+1}:** Distress Probability = **{prob*100:.2f}%** | Status: {status}")
        except KeyError as e:
            st.error(f"Error: Column {e} is missing in your file.please check the data format.")
        except Exception as e:
            st.error(f"Unexpected Error: {e}")
else:
    st.info("please upload a company data file from the left menu to start.")                                   