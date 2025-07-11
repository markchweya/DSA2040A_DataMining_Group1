# app/app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Mental Health Prediction", layout="centered")

# Load model
model = joblib.load('mental_health_model.pkl')

st.title("🧠 Mental Health Support Prediction App")
st.write("This app predicts whether a person is likely to seek mental health treatment based on their workplace and personal data.")

# === USER INPUT FORM ===
with st.form("prediction_form"):
    st.subheader("🔍 Input Details")
    
    age = st.slider("Age", 18, 100, 25)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    self_employed = st.selectbox("Are you self-employed?", ["Yes", "No", "Unknown"])
    family_history = st.selectbox("Family history of mental illness?", ["Yes", "No"])
    work_interfere = st.selectbox("Does mental health interfere with work?", ["Often", "Rarely", "Never", "Sometimes", "Don’t know"])
    no_employees = st.selectbox("Company size", ["1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000"])
    remote_work = st.selectbox("Do you work remotely?", ["Yes", "No"])
    tech_company = st.selectbox("Is it a tech company?", ["Yes", "No"])
    benefits = st.selectbox("Mental health benefits?", ["Yes", "No", "Don't know"])
    care_options = st.selectbox("Access to care options?", ["Yes", "No", "Not sure"])
    wellness_program = st.selectbox("Wellness program?", ["Yes", "No", "Don't know"])
    anonymity = st.selectbox("Is anonymity protected?", ["Yes", "No", "Don't know"])
    leave = st.selectbox("Comfort with leave?", ["Very easy", "Somewhat easy", "Somewhat difficult", "Very difficult", "Don't know"])
    mental_health_consequence = st.selectbox("Consequences for mental health issues?", ["Yes", "No", "Maybe"])
    phys_health_consequence = st.selectbox("Consequences for physical health issues?", ["Yes", "No", "Maybe"])
    coworkers = st.selectbox("Comfort talking to coworkers?", ["Yes", "No", "Some of them"])
    supervisor = st.selectbox("Comfort talking to supervisor?", ["Yes", "No", "Some of them"])
    mental_health_interview = st.selectbox("Would you discuss MH in interview?", ["Yes", "No", "Maybe"])
    phys_health_interview = st.selectbox("Would you discuss physical health in interview?", ["Yes", "No", "Maybe"])
    mental_vs_physical = st.selectbox("Is mental health as important as physical?", ["Yes", "No", "Don't know"])
    obs_consequence = st.selectbox("Observed negative consequences?", ["Yes", "No"])

    submitted = st.form_submit_button("Predict")

# === MAPPING INPUT TO NUMERIC (same as in model) ===
if submitted:
    input_dict = {
        "Age": age,
        "Gender": gender,
        "self_employed": self_employed,
        "family_history": family_history,
        "work_interfere": work_interfere,
        "no_employees": no_employees,
        "remote_work": remote_work,
        "tech_company": tech_company,
        "benefits": benefits,
        "care_options": care_options,
        "wellness_program": wellness_program,
        "anonymity": anonymity,
        "leave": leave,
        "mental_health_consequence": mental_health_consequence,
        "phys_health_consequence": phys_health_consequence,
        "coworkers": coworkers,
        "supervisor": supervisor,
        "mental_health_interview": mental_health_interview,
        "phys_health_interview": phys_health_interview,
        "mental_vs_physical": mental_vs_physical,
        "obs_consequence": obs_consequence
    }

    # Convert to dataframe
    input_df = pd.DataFrame([input_dict])

    # Encode same as training
    for col in input_df.columns:
        input_df[col] = pd.factorize(input_df[col])[0]

    # Predict
    prediction = model.predict(input_df)[0]
    result = "🔵 Will seek mental health treatment" if prediction == 1 else "🟢 Will NOT seek mental health treatment"

    st.success(result)
