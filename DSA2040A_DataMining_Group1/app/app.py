import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Mental Health Prediction", layout="centered")

# Load trained model
model = joblib.load('mental_health_model.pkl')

st.title("🧠 Mental Health Support Prediction App")
st.write("This app predicts whether a person is likely to seek mental health treatment based on workplace and personal attributes.")

with st.form("prediction_form"):
    st.subheader("🔍 Enter Your Details")

    age = st.slider("Age", 18, 100, 25)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    self_employed = st.selectbox("Are you self-employed?", ["Yes", "No", "Unknown"])
    family_history = st.selectbox("Do you have a family history of mental illness?", ["Yes", "No"])
    work_interfere = st.selectbox("Does mental health interfere with work?", ["Often", "Rarely", "Never", "Sometimes", "Don’t know"])
    no_employees = st.selectbox("Company size", ["1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000"])
    remote_work = st.selectbox("Do you work remotely?", ["Yes", "No"])
    tech_company = st.selectbox("Is it a tech company?", ["Yes", "No"])
    benefits = st.selectbox("Does your employer provide mental health benefits?", ["Yes", "No", "Don't know"])
    care_options = st.selectbox("Access to mental health care options?", ["Yes", "No", "Not sure"])
    wellness_program = st.selectbox("Does your company have a wellness program?", ["Yes", "No", "Don't know"])
    anonymity = st.selectbox("Is anonymity protected in your workplace?", ["Yes", "No", "Don't know"])
    leave = st.selectbox("How easy is it to take leave for mental health?", 
                         ["Very easy", "Somewhat easy", "Somewhat difficult", "Very difficult", "Don't know"])
    mental_health_consequence = st.selectbox("Would seeking treatment affect your career?", ["Yes", "No", "Maybe"])
    phys_health_consequence = st.selectbox("Would physical health issues affect your career?", ["Yes", "No", "Maybe"])
    coworkers = st.selectbox("Comfort talking to coworkers about mental health?", ["Yes", "No", "Some of them"])
    supervisor = st.selectbox("Comfort talking to supervisor about mental health?", ["Yes", "No", "Some of them"])
    mental_health_interview = st.selectbox("Would you discuss mental health in an interview?", ["Yes", "No", "Maybe"])
    phys_health_interview = st.selectbox("Would you discuss physical health in an interview?", ["Yes", "No", "Maybe"])
    mental_vs_physical = st.selectbox("Is mental health as important as physical health?", ["Yes", "No", "Don't know"])
    obs_consequence = st.selectbox("Have you observed negative consequences?", ["Yes", "No"])

    submitted = st.form_submit_button("Predict")

if submitted:
    input_dict = {
        "Timestamp": "2025-01-01 00:00:00",
        "Age": age,
        "Gender": gender,
        "Country": "Unknown",
        "self_employed": self_employed,
        "family_history": family_history,
        "work_interfere": work_interfere,
        "no_employees": no_employees,
        "remote_work": remote_work,
        "tech_company": tech_company,
        "benefits": benefits,
        "care_options": care_options,
        "wellness_program": wellness_program,
        "seek_help": "Don't know",
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

    input_df = pd.DataFrame([input_dict])

    # Encode all columns using factorize
    for col in input_df.columns:
        input_df[col] = pd.factorize(input_df[col])[0]

    # ✅ Must match training column order
    column_order = [
        'Timestamp', 'Age', 'Gender', 'Country', 'self_employed', 'family_history',
        'work_interfere', 'no_employees', 'remote_work', 'tech_company', 'benefits',
        'care_options', 'wellness_program', 'seek_help', 'anonymity', 'leave',
        'mental_health_consequence', 'phys_health_consequence', 'coworkers',
        'supervisor', 'mental_health_interview', 'phys_health_interview',
        'mental_vs_physical', 'obs_consequence'
    ]
    input_df = input_df[column_order]

    # Predict
    prediction = model.predict(input_df)[0]
    result = "🔵 Likely to seek mental health treatment" if prediction == 1 else "🟢 Unlikely to seek treatment"

    st.success(result)
