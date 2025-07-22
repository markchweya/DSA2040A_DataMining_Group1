import streamlit as st
import pandas as pd
import joblib
import os

# Page setup
st.set_page_config(page_title="Mental Health Predictor (U.S. Tech)", layout="centered")

# Load model
model_path = os.path.join(os.path.dirname(__file__), 'mental_health_model.pkl')
try:
    model = joblib.load(model_path)
except FileNotFoundError:
    st.error(" Model file not found. Please upload 'mental_health_model.pkl' to the app folder.")
    st.stop()

# App title
st.title(" Mental Health Treatment Predictor (U.S. Tech Survey)")

with st.expander("ℹ️ About this App"):
    st.markdown("""
    This app predicts whether a person is likely to seek mental health treatment, based on responses to five key questions.
    
    - Trained on U.S.-only data from the 2014 OSMI Tech Survey
    - Target variable: **Treatment sought (Yes/No)**
    
    ⚠️ This app is for demonstration only and not for clinical use.
    """)

# Form
with st.form("mh_form"):
    st.subheader(" Please answer the following:")

    age = st.slider("1️⃣ Your Age", 18, 100, 30)
    self_employed = st.selectbox("2️⃣ Are you self-employed?", ["Yes", "No"])
    family_history = st.selectbox("3️⃣ Family history of mental illness?", ["Yes", "No"])
    remote_work = st.selectbox("4️⃣ Do you work remotely?", ["Yes", "No"])
    tech_company = st.selectbox("5️⃣ Do you work in a tech company?", ["Yes", "No"])

    submitted = st.form_submit_button(" Predict")

if submitted:
    # Encode responses for model
    input_data = pd.DataFrame([{
        "Age": age,
        "self_employed": 1 if self_employed == "Yes" else 0,
        "family_history": 1 if family_history == "Yes" else 0,
        "remote_work": 1 if remote_work == "Yes" else 0,
        "tech_company": 1 if tech_company == "Yes" else 0
    }])

    st.markdown("###  Model Input")
    st.write(input_data)

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Output
    if prediction == 1:
        st.success("🧩 You are **likely** to seek mental health treatment.")
    else:
        st.info("🧩 You are **unlikely** to seek mental health treatment.")
