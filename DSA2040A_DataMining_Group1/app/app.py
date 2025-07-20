import streamlit as st
import pandas as pd
import joblib
import os

# Streamlit Page Config
st.set_page_config(page_title="Mental Health Prediction (US Only)", layout="centered")

# Load trained model
model_path = os.path.join(os.path.dirname(__file__), 'mental_health_model.pkl')
try:
    model = joblib.load(model_path)
except FileNotFoundError:
    st.error("❌ Model file not found. Please upload 'mental_health_model.pkl' to the app folder.")
    st.stop()

# Title
st.title("🇺🇸 Mental Health Treatment Prediction - U.S. Respondents")

# Info Sections
with st.expander("ℹ️ How This App Works"):
    st.markdown("""
    This app predicts the likelihood of seeking mental health treatment based on selected personal and workplace factors.  
    It uses a machine learning model trained **only on U.S. respondents** from the OSMI 2014 mental health survey.

    > For learning purposes only — not a diagnostic tool.
    """)

with st.expander("📊 About the Data"):
    st.markdown("""
    **Dataset:** [OSMI Mental Health in Tech Survey 2014](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey)

    **Model was trained on U.S.-only data**, with features like:
    - Age, gender, self-employment
    - Company size, mental health interference
    - Access to mental health benefits, anonymity, etc.
    """)

# Form for user input
with st.form("prediction_form"):
    st.subheader("📝 Your Information")

    age = st.slider("What is your age?", 18, 100, 30)
    gender = st.selectbox("What is your gender?", ["Male", "Female", "Other"])
    self_employed = st.selectbox("Are you self-employed?", ["Yes", "No"])
    family_history = st.selectbox("Do you have a family history of mental illness?", ["Yes", "No"])
    no_employees = st.selectbox("Company size?", ["1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000"])
    work_interfere = st.selectbox("Does mental health interfere with your work?", ["Never", "Rarely", "Sometimes", "Often"])
    benefits = st.selectbox("Does your employer provide mental health benefits?", ["Yes", "No", "Don't know"])
    care_options = st.selectbox("Access to mental health care options?", ["Yes", "No", "Not sure"])
    anonymity = st.selectbox("Is your anonymity protected when discussing mental health?", ["Yes", "No", "Don't know"])

    submitted = st.form_submit_button("🔮 Predict")

# Prediction logic
if submitted:
    # User input as dataframe
    input_data = pd.DataFrame([{
        'Age': age,
        'Gender': gender,
        'self_employed': self_employed,
        'family_history': family_history,
        'no_employees': no_employees,
        'work_interfere': work_interfere,
        'benefits': benefits,
        'care_options': care_options,
        'anonymity': anonymity
    }])

    # Load training dataset to align columns
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'us_model_data', 'training_model_dataset.csv')
    try:
        full_data = pd.read_csv(data_path)
    except FileNotFoundError:
        st.error("❌ Could not find training_model_dataset.csv in /data/us_model_data/.")
        st.stop()

    # Drop target & combine with input
    full_data.drop(columns=['treatment'], inplace=True)
    combined = pd.concat([full_data, input_data], axis=0, ignore_index=True)
    combined_encoded = pd.get_dummies(combined)

    # Align with model's expected features
    model_features = model.feature_names_in_
    for col in model_features:
        if col not in combined_encoded.columns:
            combined_encoded[col] = 0

    input_encoded = combined_encoded[model_features].tail(1)

    # Prediction
    prediction = model.predict(input_encoded)[0]

    # Output
    if prediction == 1:
        st.success("🔵 Based on your responses, you are **likely** to seek mental health treatment.")
    else:
        st.success("🟢 Based on your responses, you are **unlikely** to seek mental health treatment.")

    with st.expander("🧪 Show Encoded Features"):
        st.write(input_encoded)
