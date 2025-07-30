import streamlit as st
import pandas as pd
import joblib
import time
import os
from io import BytesIO
from fpdf import FPDF
import base64
import numpy as np

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
# PREDICTION FUNCTION
# --------------------
def predict_treatment(input_df, threshold=0.4):
    probs = model.predict_proba(input_df)[0]
    prediction = 1 if probs[1] >= threshold else 0
    confidence = probs[prediction]
    ci_range = 0.05  # ±5% for fun visualization
    return prediction, confidence, max(confidence-ci_range,0), min(confidence+ci_range,1)

# --------------------
# FIXED PDF GENERATOR
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

    # ✅ Return as BytesIO
    pdf_bytes = pdf.output(dest='S').encode('latin-1')
    buffer = BytesIO(pdf_bytes)
    return buffer

# --------------------
# CUSTOM CSS
# --------------------
st.markdown("""
    <style>
    .centered { text-align: center; padding-top: 60px; }
    .fade-in { animation: fadeIn 1.2s ease-in; }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .slide-in { animation: slideIn 1.2s ease-in; }
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(40px); }
        to { opacity: 1; transform: translateX(0); }
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
    .dark-mode {
        background-color: #1e1e1e;
        color: white;
    }
    .cool-button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 24px;
        border-radius: 8px;
        border: none;
        font-size: 18px;
        cursor: pointer;
        transition: 0.3s;
    }
    .cool-button:hover {
        background-color: #45a049;
    }
    </style>
""", unsafe_allow_html=True)

# --------------------
# SIDEBAR SETTINGS
# --------------------
st.sidebar.header("Settings")
theme = st.sidebar.radio("Choose Theme", ["Light Mode", "Dark Mode"], key="theme_radio")
threshold = st.sidebar.slider("Prediction Threshold", 0.1, 0.9, 0.4, 0.05, key="threshold_slider")

if theme == "Dark Mode":
    st.markdown('<style>body { background-color: #1e1e1e; color: white; }</style>', unsafe_allow_html=True)

# --------------------
# SESSION STATE
# --------------------
if "page" not in st.session_state:
    st.session_state.page = "welcome"

# --------------------
# WELCOME PAGE
# --------------------
if st.session_state.page == "welcome":
    st.markdown('<div class="centered fade-in">', unsafe_allow_html=True)
    st.markdown("<h1 class='slide-in'>Welcome to the Mental Health Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<p class='fade-in'>This is a playful experiment to see if our ML model can guess if someone might seek mental health treatment.</p>", unsafe_allow_html=True)
    st.markdown("<p><b>By continuing, you acknowledge that this app is for educational and fun purposes only. No personal data is stored.</b></p>", unsafe_allow_html=True)

    privacy_accepted = st.checkbox("I agree to the Privacy Terms")
    if st.button("Let's Go! ▶", key="start", disabled=not privacy_accepted):
        st.balloons()  # ✅ Only here
        time.sleep(1.5)
        st.session_state.page = "model"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# --------------------
# MODEL PAGE
# --------------------
elif st.session_state.page == "model":
    st.title("Mental Health Treatment Predictor")
    st.markdown("Answer a few quick questions. This is just for fun!")

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
        with st.spinner("Thinking really hard about your answers..."):
            time.sleep(1.5)

            input_data = pd.DataFrame([{
                "Age": age,
                "self_employed": 1 if self_employed == "Yes" else 0,
                "family_history": 1 if family_history == "Yes" else 0,
                "remote_work": 1 if remote_work == "Yes" else 0,
                "tech_company": 1 if tech_company == "Yes" else 0
            }])

            prediction, confidence, ci_low, ci_high = predict_treatment(input_data, threshold=threshold)

        # ✅ Move to results page
        st.session_state.result = {
            "prediction": prediction,
            "confidence": confidence,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "age": age,
            "answers": {
                "Self-employed": self_employed,
                "Family History": family_history,
                "Remote Work": remote_work,
                "Tech Company": tech_company
            }
        }
        st.session_state.page = "results"
        st.rerun()

# --------------------
# RESULTS PAGE
# --------------------
elif st.session_state.page == "results":
    result = st.session_state.result
    prediction = result["prediction"]
    confidence = result["confidence"]
    ci_low = result["ci_low"]
    ci_high = result["ci_high"]
    age = result["age"]
    answers = result["answers"]

    # ❄️ Snow animation
    st.snow()

    st.title("Your Prediction Result ")

    result_text = "You seem likely to seek mental health support." if prediction == 1 else "You seem unlikely to seek mental health support."
    color = "#f6ffed" if prediction == 1 else "#fffbe6"
    border_color = "#52c41a" if prediction == 1 else "#faad14"

    st.markdown(f"""
    <div class="result-box fade-in" style="background-color:{color}; border-left-color:{border_color};">
        <b>Prediction:</b> {result_text}<br>
        <b>Confidence:</b> {confidence:.2%} <br>
        <b>Approx. 95% CI:</b> {ci_low:.2%} - {ci_high:.2%}
    </div>
    """, unsafe_allow_html=True)

    st.progress(confidence)
    st.write(f"Our model is {confidence:.2%} confident in this prediction.")

    # --------------------
    # DOWNLOAD PDF
    # --------------------
    report = create_pdf(age, answers, prediction, confidence)
    b64 = base64.b64encode(report.read()).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="mental_health_report.pdf">📄 Download Your Report (PDF)</a>'
    st.markdown(href, unsafe_allow_html=True)

    # --------------------
    # SIMPLE EXPLANATION
    # --------------------
    st.markdown("### Why might the model think this?")
    st.markdown("This is a **toy model**, but here are some playful insights:")
    if answers["Family History"] == "Yes":
        st.markdown("- Having a family history of mental illness is often linked to being more open to seeking treatment.")
    if answers["Self-employed"] == "Yes":
        st.markdown("- Being self-employed can change stress levels and flexibility, which might influence treatment decisions.")
    if answers["Remote Work"] == "Yes":
        st.markdown("- Working remotely sometimes correlates with different mental health patterns.")
    st.markdown("Remember, this is **just for fun and education**, not a medical opinion.")
