import streamlit as st
import pandas as pd
import joblib
import time
import os

# Load the model from the correct relative path
model_path = os.path.join("app", "mental_health_model.pkl")
try:
    model = joblib.load(model_path)
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'mental_health_model.pkl' is in the 'app' folder.")
    st.stop()

# Function to predict treatment
def predict_treatment(input_df):
    prediction = model.predict(input_df)[0]
    confidence = model.predict_proba(input_df)[0][1] if prediction == 1 else model.predict_proba(input_df)[0][0]
    return prediction, confidence

# Inject custom CSS for animations and styling
st.markdown("""
    <style>
    .centered {
        text-align: center;
        padding-top: 100px;
    }
    .button-style {
        display: inline-block;
        padding: 0.6em 1.4em;
        font-size: 1.1em;
        font-weight: bold;
        color: white;
        background-color: #4CAF50;
        border: none;
        border-radius: 8px;
        transition: all 0.3s ease;
        text-decoration: none;
    }
    .button-style:hover {
        background-color: #45a049;
        transform: scale(1.05);
    }
    .fade-in {
        animation: fadeIn 1.5s ease-in;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .result-box {
        padding: 20px;
        border-left: 5px solid #52c41a;
        background-color: #f6ffed;
        margin-top: 20px;
        border-radius: 10px;
        font-size: 18px;
        color: black;
    }
    </style>
""", unsafe_allow_html=True)

# Use session state to track welcome/model toggle
if "page" not in st.session_state:
    st.session_state.page = "welcome"

# WELCOME PAGE
if st.session_state.page == "welcome":
    st.balloons()
    st.markdown('<div class="centered fade-in">', unsafe_allow_html=True)
    st.markdown("<h1 class='fade-in'>Mental Health Treatment Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<p class='fade-in'>This app uses a model trained on the 2014 U.S. OSMI Tech Survey to predict if you're likely to seek mental health treatment.</p>", unsafe_allow_html=True)
    if st.button("Continue to Model", key="start"):
        st.session_state.page = "model"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# MODEL PAGE
elif st.session_state.page == "model":
    st.title("Mental Health Treatment Predictor")
    st.markdown("Please answer the following questions:")

    with st.form("mh_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("Your Age", 18, 100, 30)
            self_employed = st.selectbox("Are you self-employed?", ["Yes", "No"])
            family_history = st.selectbox("Family history of mental illness?", ["Yes", "No"])
        with col2:
            remote_work = st.selectbox("Do you work remotely?", ["Yes", "No"])
            tech_company = st.selectbox("Do you work in a tech company?", ["Yes", "No"])

        disclaimer = st.checkbox("I understand this is not medical advice and is for educational purposes only.")
        submitted = st.form_submit_button("Predict")

    if submitted:
        if not disclaimer:
            st.warning("Please check the disclaimer box to proceed.")
        else:
            with st.spinner("Analyzing your responses..."):
                time.sleep(1.5)

                # Encode inputs
                input_data = pd.DataFrame([{
                    "Age": age,
                    "self_employed": 1 if self_employed == "Yes" else 0,
                    "family_history": 1 if family_history == "Yes" else 0,
                    "remote_work": 1 if remote_work == "Yes" else 0,
                    "tech_company": 1 if tech_company == "Yes" else 0
                }])

                prediction, confidence = predict_treatment(input_data)

           
