import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.preprocessing import LabelEncoder

# Streamlit Page Config
st.set_page_config(page_title="Mental Health Prediction", layout="centered")

# 🔹 Load Model
model_path = os.path.join(os.path.dirname(__file__), 'mental_health_model.pkl')
try:
    model = joblib.load(model_path)
except FileNotFoundError:
    st.error("❌ Model file not found. Please upload 'mental_health_model.pkl' to the app folder.")
    st.stop()

# 🔹 Title
st.title("🧠 Mental Health Support Prediction App")

# 🔹 Info
with st.expander("ℹ️ How This App Works"):
    st.markdown("""
This app predicts if a person is likely to seek mental health treatment using a machine learning model trained on tech industry survey data.

**Prediction is based on:**
- Age, gender, work situation
- Workplace mental health support
- Attitudes and policies

> This is an educational tool. It’s not a substitute for clinical evaluation.
""")

with st.expander("📊 About the Data"):
    st.markdown("""
We used the **OSMI Mental Health in Tech Survey** dataset:  
[https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey)

It includes responses from thousands of tech workers about mental health challenges and treatment patterns.
""")

# 🔹 Form
with st.form("prediction_form"):
    st.subheader("📝 Your Info")

    age = st.slider("Age", 18, 100, 25)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    self_employed = st.selectbox("Are you self-employed?", ["Yes", "No", "Unknown"])
    family_history = st.selectbox("Family history of mental illness?", ["Yes", "No"])
    work_interfere = st.selectbox("Mental health interferes with work?", ["Often", "Rarely", "Never", "Sometimes", "Don’t know"])
    no_employees = st.selectbox("Company size", ["1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000"])
    remote_work = st.selectbox("Do you work remotely?", ["Yes", "No"])
    tech_company = st.selectbox("Is your company a tech company?", ["Yes", "No"])
    benefits = st.selectbox("Employer provides mental health benefits?", ["Yes", "No", "Don't know"])
    care_options = st.selectbox("Access to mental health care options?", ["Yes", "No", "Not sure"])
    wellness_program = st.selectbox("Company has a wellness program?", ["Yes", "No", "Don't know"])
    anonymity = st.selectbox("Is your anonymity protected?", ["Yes", "No", "Don't know"])
    leave = st.selectbox("Ease of taking leave for mental health?", 
                         ["Very easy", "Somewhat easy", "Somewhat difficult", "Very difficult", "Don't know"])
    mental_health_consequence = st.selectbox("Mental health affects career?", ["Yes", "No", "Maybe"])
    phys_health_consequence = st.selectbox("Physical health affects career?", ["Yes", "No", "Maybe"])
    coworkers = st.selectbox("Comfortable talking to coworkers?", ["Yes", "No", "Some of them"])
    supervisor = st.selectbox("Comfortable talking to supervisor?", ["Yes", "No", "Some of them"])
    mental_health_interview = st.selectbox("Discuss mental health in interview?", ["Yes", "No", "Maybe"])
    phys_health_interview = st.selectbox("Discuss physical health in interview?", ["Yes", "No", "Maybe"])
    mental_vs_physical = st.selectbox("Is mental health equal to physical?", ["Yes", "No", "Don't know"])
    obs_consequence = st.selectbox("Observed consequences of mental health disclosure?", ["Yes", "No"])

    submitted = st.form_submit_button("🔮 Predict")

# 🔹 On Submit
if submitted:
    # Create a sample row
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

    # Encode with LabelEncoder per column (simulate training)
    for col in input_df.columns:
        if input_df[col].dtype == object:
            input_df[col] = LabelEncoder().fit_transform(input_df[col])

    # Column order
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

    # Show result
    if prediction == 1:
        st.success("🔵 You are **likely** to seek mental health treatment.")
    else:
        st.success("🟢 You are **unlikely** to seek mental health treatment.")

    # Debug (optional)
    with st.expander("🧪 See Encoded Input & Prediction"):
        st.write("Encoded Input:")
        st.dataframe(input_df)
        st.write("Raw Prediction:", prediction)
