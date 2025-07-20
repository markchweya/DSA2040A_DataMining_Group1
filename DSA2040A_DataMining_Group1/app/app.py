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

# Information sections
with st.expander("ℹ️ How This App Works"):
    st.markdown("""
    This app predicts the likelihood of seeking mental health treatment based on workplace factors and personal background. 
    It uses a model trained **exclusively on U.S. respondents** from the 2014 OSMI tech survey.

    > This is for educational purposes only. It’s not a diagnostic tool.
    """)

with st.expander("📊 About the Data"):
    st.markdown("""
    Data Source: [OSMI Mental Health in Tech Survey 2014](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey)

    We used cleaned and preprocessed data limited to respondents from the United States. The model was trained after:
    - Filtering to U.S.-only
    - Label encoding relevant features
    - Dropping irrelevant or empty fields (e.g., Timestamp, state)
    - Selecting "treatment" as the target variable
    """)

# Form
with st.form("prediction_form"):
    st.subheader("📝 Your Information")

    age = st.slider("Age", 18, 100, 30)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    self_employed = st.selectbox("Are you self-employed?", ["Yes", "No"])
    family_history = st.selectbox("Family history of mental illness?", ["Yes", "No"])
    no_employees = st.selectbox("Company size", ["1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000"])
    work_interfere = st.selectbox("Does mental health interfere with work?", ["Never", "Rarely", "Sometimes", "Often"])
    benefits = st.selectbox("Employer provides mental health benefits?", ["Yes", "No", "Don't know"])
    care_options = st.selectbox("Access to mental health care options?", ["Yes", "No", "Not sure"])
    anonymity = st.selectbox("Is your anonymity protected?", ["Yes", "No", "Don't know"])

    submitted = st.form_submit_button("🔮 Predict")

if submitted:
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

    # Preprocessing: one-hot encode categorical values to match model input
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'us_model_data', 'training_model_dataset.csv')
    full_feature_df = pd.read_csv(data_path)
    full_feature_df.drop(columns=['treatment'], inplace=True)
    full_feature_df = pd.concat([full_feature_df, input_data], axis=0, ignore_index=True)
    full_feature_df = pd.get_dummies(full_feature_df)
    input_encoded = full_feature_df.tail(1)

    # Align with model training columns
    model_features = model.feature_names_in_
    missing_cols = set(model_features) - set(input_encoded.columns)
    for col in missing_cols:
        input_encoded[col] = 0
    input_encoded = input_encoded[model_features]

    # Prediction
    prediction = model.predict(input_encoded)[0]

    # Result
    if prediction == 1:
        st.success("🔵 Based on your responses, you are **likely** to seek mental health treatment.")
    else:
        st.success("🟢 Based on your responses, you are **unlikely** to seek mental health treatment.")

    with st.expander("🧪 Show Processed Input"):
        st.write(input_encoded)
