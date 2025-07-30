import streamlit as st
import pandas as pd
import joblib
import time
import os
from io import BytesIO
from fpdf import FPDF
import base64
import numpy as np
from streamlit_lottie import st_lottie
import json

# --------------------
# STREAMLIT PAGE CONFIG
# --------------------
st.set_page_config(
    page_title="Mental Health Treatment Predictor",
    page_icon="💡",
    layout="wide"
)

# --------------------
# LOAD MODEL
# --------------------
model_path = os.path.join("app", "mental_health_model.pkl")
try:
    model = joblib.load(model_path)
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'mental_health_model.pkl' is in the 'app' folder.")
    st.stop()

# --------------------
# LOTTIE ANIMATIONS
# --------------------
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

loader_animation = load_lottiefile("app/loader_1.json")
doctor_animation = load_lottiefile("app/loader_dancing.json")

# --------------------
# PREDICTION FUNCTION
# --------------------
def predict_treatment(input_df, threshold=0.4):
    probs = model.predict_proba(input_df)[0]
    prediction = 1 if probs[1] >= threshold else 0
    confidence = probs[prediction]
    ci_range = 0.05  # ±5% for visualization
    return prediction, confidence, max(confidence-ci_range,0), min(confidence+ci_range,1)

# --------------------
# PDF GENERATOR
# --------------------
def create_pdf(age, answers, prediction, confidence):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Mental Health Prediction Report", ln=1, align="C")
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Age: {age}", ln=1)
    for q, a in answers.items():
        pdf.cell(200, 10, txt=f"{q}: {a}", ln=1)
    pdf.ln(10)
    result = "Likely to Seek Treatment" if prediction == 1 else "Unlikely to Seek Treatment"
    pdf.cell(200, 10, txt=f"Prediction: {result}", ln=1)
    pdf.cell(200, 10, txt=f"Confidence: {confidence:.2%}", ln=1)

    pdf_bytes = pdf.output(dest='S').encode('latin1')
    return BytesIO(pdf_bytes)

# --------------------
# CUSTOM CSS
# --------------------
st.markdown("""
    <style>
    .centered { text-align: center; padding-top: 60px; }
    .fade-in { animation: fadeIn 1.5s ease-in; }
    .fade-out { animation: fadeOut 1.5s ease-out; }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeOut {
        from { opacity: 1; transform: translateY(0); }
        to { opacity: 0; transform: translateY(-20px); }
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
    .loader-container {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 60vh;
    }
    </style>
""", unsafe_allow_html=True)

# --------------------
# SIDEBAR SETTINGS
# --------------------
st.sidebar.header("Settings")
theme = st.sidebar.radio("Choose Theme", ["Light Mode", "Dark Mode"], key="theme_selector")
threshold = st.sidebar.slider("Prediction Threshold", 0.1, 0.9, 0.4, 0.05, key="threshold_slider")

if theme == "Dark Mode":
    st.markdown("<style>body { background-color: #1e1e1e; color: white; }</style>", unsafe_allow_html=True)

# --------------------
# SESSION STATE
# --------------------
if "page" not in st.session_state:
    st.session_state.page = "welcome"
if "prediction_data" not in st.session_state:
    st.session_state.prediction_data = None

# --------------------
# WELCOME PAGE
# --------------------
if st.session_state.page == "welcome":
    st.markdown('<div class="centered fade-in">', unsafe_allow_html=True)
    st.markdown("<h1 class='fade-in'>Welcome to the Mental Health Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<p class='fade-in'>This app is a playful experiment to see if our ML model can guess if someone might seek mental health treatment.</p>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:14px;color:gray;'>By continuing, you acknowledge that this app is for educational purposes only and no personal data is stored.</p>", unsafe_allow_html=True)

    agree = st.checkbox("I agree to the privacy terms", key="privacy_agree")
    
    # Add a new session state to track loading
    if "loading" not in st.session_state:
        st.session_state.loading = False

    if not st.session_state.loading:
        st.button("Let's Go! ▶", disabled=not agree, key="start_btn", on_click=lambda: setattr(st.session_state, "loading", True))
    else:
        # Show loader in the center
        st.markdown('<div class="loader-container">', unsafe_allow_html=True)
        st_lottie(loader_animation, height=200, key="loader_centered")
        st.markdown('</div>', unsafe_allow_html=True)

        # After showing loader for 2.5s, go to model page
        time.sleep(2.5)
        st.session_state.page = "model"
        st.session_state.loading = False
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)
# --------------------
# MODEL PAGE
# --------------------
elif st.session_state.page == "model":
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    st.title("Mental Health Treatment Predictor")
    st.markdown("Answer a few quick questions below:")

    with st.form("mh_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("Your Age", 18, 100, 30)
            self_employed = st.selectbox("Are you self-employed?", ["Yes", "No"])
            family_history = st.selectbox("Do you have a family history of mental illness?", ["Yes", "No"])
        with col2:
            remote_work = st.selectbox("Do you usually work remotely?", ["Yes", "No"])
            tech_company = st.selectbox("Do you work in a tech company?", ["Yes", "No"])

        disclaimer = st.checkbox("I understand this is just a fun experiment and not medical advice.")
        submitted = st.form_submit_button("Make My Prediction!")

    if submitted and disclaimer:
        # Save data to session and go to results page
        input_data = pd.DataFrame([{
            "Age": age,
            "self_employed": 1 if self_employed == "Yes" else 0,
            "family_history": 1 if family_history == "Yes" else 0,
            "remote_work": 1 if remote_work == "Yes" else 0,
            "tech_company": 1 if tech_company == "Yes" else 0
        }])
        prediction, confidence, ci_low, ci_high = predict_treatment(input_data, threshold=threshold)

        # Save for results page
        st.session_state.prediction_data = {
            "age": age,
            "answers": {
                "Self-employed": self_employed,
                "Family History": family_history,
                "Remote Work": remote_work,
                "Tech Company": tech_company
            },
            "prediction": prediction,
            "confidence": confidence,
            "ci_low": ci_low,
            "ci_high": ci_high
        }

        # Show doctor animation (centered)
        st.markdown('<div class="loader-container">', unsafe_allow_html=True)
        st_lottie(doctor_animation, height=250, key="doctor_anim")
        st.markdown('</div>', unsafe_allow_html=True)
        time.sleep(2)

        st.session_state.page = "results"
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# --------------------
# RESULTS PAGE
# --------------------
elif st.session_state.page == "results":
    data = st.session_state.prediction_data
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    st.title("Your Prediction Results")

    result_text = "You seem likely to seek mental health support." if data["prediction"] == 1 else "You seem unlikely to seek mental health support."
    color = "#f6ffed" if data["prediction"] == 1 else "#fffbe6"
    border_color = "#52c41a" if data["prediction"] == 1 else "#faad14"

    st.markdown(f"""
    <div class="result-box fade-in" style="background-color:{color}; border-left-color:{border_color};">
        <b>Prediction:</b> {result_text}<br>
        <b>Confidence:</b> {data['confidence']:.2%} <br>
        <b>Approx. 95% CI:</b> {data['ci_low']:.2%} - {data['ci_high']:.2%}
    </div>
    """, unsafe_allow_html=True)

    st.progress(data['confidence'])
    st.write(f"Our model is {data['confidence']:.2%} confident in this prediction.")

    st.balloons()  # Balloons only on the final results page

    # Download PDF
    report = create_pdf(data['age'], data['answers'], data['prediction'], data['confidence'])
    b64 = base64.b64encode(report.read()).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="mental_health_report.pdf">Download Your Fun Report (PDF)</a>'
    st.markdown(href, unsafe_allow_html=True)

    # Back button
    if st.button("🔄 Start Over"):
        st.session_state.page = "welcome"
        st.rerun()
