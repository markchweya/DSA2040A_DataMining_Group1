import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Mental Health Prediction", layout="centered")

# Load the trained Random Forest model
model = joblib.load('mental_health_model.pkl')

# App title
st.title("🧠 Mental Health Support Prediction App")

# Section: How the model works
with st.expander("ℹ️ How This App Works"):
    st.markdown("""
This prediction tool uses a machine learning model to determine whether someone is **likely** or **unlikely** to seek mental health treatment.

### 🔍 Here's how it works:
1. **You fill in the form** with details like your age, work setup, access to mental health support, and comfort discussing issues.
2. The app prepares your input by converting your responses into numerical values (since the model only understands numbers).
3. It then passes this information to a trained **Random Forest Classifier**, which was built using real survey data from people working in the tech industry.
4. The model looks for patterns and makes a prediction:
   - **🔵 Likely** to seek mental health treatment
   - **🟢 Unlikely** to seek treatment

> This is only a predictive tool and should not replace professional advice. Always seek proper help if you’re struggling.
""")

# Section: About the data
with st.expander("📊 About the Data Source"):
    st.markdown("""
This app is powered by data from the **Mental Health in Tech Survey**, conducted by [OSMI (Open Sourcing Mental Illness)](https://osmihelp.org/).

The dataset was collected from professionals working in the tech industry and includes responses related to:
- Personal mental health history
- Work environment and company size
- Access to mental health resources
- Comfort discussing mental health issues

📁 **Dataset link on Kaggle:**  
[https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey)
""")

# User form
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

# Handle form submission
if submitted:
    # Create input dictionary
    input_dict = {
        "Timestamp": "2025-01-01 00:00:00",  # dummy timestamp
        "Age": age,
        "Gender": gender,
        "Country": "Unknown",  # default country
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

    # Encode with factorize (same as training)
    for col in input_df.columns:
        input_df[col] = pd.factorize(input_df[col])[0]

    # Match column order from training
    column_order = [
        'Timestamp', 'Age', 'Gender', 'Country', 'self_employed', 'family_history',
        'work_interfere', 'no_employees', 'remote_work', 'tech_company', 'benefits',
        'care_options', 'wellness_program', 'seek_help', 'anonymity', 'leave',
        'mental_health_consequence', 'phys_health_consequence', 'coworkers',
        'supervisor', 'mental_health_interview', 'phys_health_interview',
        'mental_vs_physical', 'obs_consequence'
    ]
    input_df = input_df[column_order]

    # Make prediction
    prediction = model.predict(input_df)[0]

    # Display result
    if prediction == 1:
        st.success("🔵 Based on your input, you are **likely to seek mental health treatment.**")
    else:
        st.success("🟢 Based on your input, you are **unlikely to seek mental health treatment.**")
