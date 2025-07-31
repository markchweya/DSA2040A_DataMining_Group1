import streamlit as st
import pandas as pd
import joblib
import time
import os
from io import BytesIO
from fpdf import FPDF
import base64

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Mental Health Treatment Predictor",
    page_icon="",
    layout="wide"
)

# -------------------- MODEL LOADING --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "mental_health_model.pkl")

st.write(f"🔍 Debug: Looking for model at `{model_path}`")  # Debug line

try:
    model = joblib.load(model_path)
except FileNotFoundError:
    st.error(f"❌ Model file not found. Expected at: `{model_path}`")
    st.stop()

# -------------------- PREDICTION FUNCTION --------------------
def predict_treatment(input_df, threshold=0.4):
    probs = model.predict_proba(input_df)[0]
    prediction = 1 if probs[1] >= threshold else 0
    confidence = probs[prediction]
    ci_range = 0.05
    return prediction, confidence, max(confidence-ci_range,0), min(confidence+ci_range,1)

# -------------------- PDF GENERATOR --------------------
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

    pdf_bytes = pdf.output(dest='S').encode('latin-1')
    return BytesIO(pdf_bytes)

# -------------------- CSS --------------------
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

# -------------------- SESSION STATE --------------------
if "page" not in st.session_state:
    st.session_state.page = "welcome"

# -------------------- SIDEBAR MENU --------------------
st.sidebar.markdown("### Navigation")
if st.sidebar.button("Home"):
    st.session_state.page = "welcome"
    st.rerun()

if st.sidebar.button("Privacy Policy"):
    st.session_state.page = "privacy"
    st.rerun()

if st.sidebar.button("Documentation"):
    st.session_state.page = "documentation"
    st.rerun()

# -------------------- HOME PAGE --------------------
if st.session_state.page == "welcome":
    st.markdown('<div class="centered fade-in">', unsafe_allow_html=True)
    st.markdown("<h1 class='slide-in'>Welcome to the Mental Health Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<p class='fade-in'>This is a playful experiment to see if our ML model can guess if someone might seek mental health treatment.</p>", unsafe_allow_html=True)
    st.markdown("<p><b>By continuing, you acknowledge that this app is for educational and fun purposes only. No personal data is stored.</b></p>", unsafe_allow_html=True)

    privacy_accepted = st.checkbox("I agree to the Privacy Terms")
    if st.button("Let's Go!", disabled=not privacy_accepted):
        st.balloons()
        time.sleep(1)
        st.session_state.page = "model"
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------- PRIVACY POLICY PAGE --------------------
elif st.session_state.page == "privacy":
    st.title("Privacy Policy")
    st.markdown("""
    We do not store any personal data.  
    This is a demonstration app for educational purposes only.  
    By using this app, you agree that your responses are processed temporarily to generate a prediction.
    """)

# -------------------- MODEL PAGE --------------------
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
        with st.spinner("Analyzing your responses..."):
            time.sleep(1.5)
            input_data = pd.DataFrame([{
                "Age": age,
                "self_employed": 1 if self_employed == "Yes" else 0,
                "family_history": 1 if family_history == "Yes" else 0,
                "remote_work": 1 if remote_work == "Yes" else 0,
                "tech_company": 1 if tech_company == "Yes" else 0
            }])

            prediction, confidence, ci_low, ci_high = predict_treatment(input_data, threshold=0.4)

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

# -------------------- RESULTS PAGE --------------------
elif st.session_state.page == "results":
    result = st.session_state.result
    prediction = result["prediction"]
    confidence = result["confidence"]
    ci_low = result["ci_low"]
    ci_high = result["ci_high"]
    age = result["age"]
    answers = result["answers"]

    st.snow()
    st.title("Your Prediction Result")

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

    report = create_pdf(age, answers, prediction, confidence)
    b64 = base64.b64encode(report.read()).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="mental_health_report.pdf">Download Your Report (PDF)</a>'
    st.markdown(href, unsafe_allow_html=True)

    st.markdown("### Why might the model think this?")
    if answers["Family History"] == "Yes":
        st.markdown("- Family history often correlates with seeking treatment.")
    if answers["Self-employed"] == "Yes":
        st.markdown("- Self-employment can affect stress and treatment decisions.")
    if answers["Remote Work"] == "Yes":
        st.markdown("- Remote work sometimes correlates with different mental health patterns.")
    st.markdown("Remember, this is **just for fun and education**, not a medical opinion.")

    if st.button("Back to Predictor"):
        st.session_state.page = "model"
        st.rerun()



# -------------------- DOCUMENTATION PAGE --------------------
elif st.session_state.page == "documentation":
    st.title("Project Documentation")

    # --- Collapsible Menu ---
    with st.expander("📄 Show Documentation Menu", expanded=True):
        menu_items = [
            "Overview",
            "Getting Started",
            "Project Structure",
            "Machine Learning Summary",
            "Key Insights",
            "Future Improvements",
            "Contributors",
            "Contact & Support"
        ]
        doc_section = st.radio("Select a section to view:", menu_items, key="doc_menu")

    # --- Content Rendering ---
    if doc_section == "Overview":
        st.header("Overview")
        st.write("""
        This project predicts the likelihood of employees seeking mental health support
        using survey data from the 2014 OSMI Mental Health in Tech dataset...
        """)

    elif doc_section == "Getting Started":
        st.header("Getting Started")
        st.markdown("""
        **Run Locally:**
        1. Clone the repo: `git clone https://github.com/markchweya/DSA2040A_DataMining_Group1.git`
        2. Install dependencies: `pip install -r requirements.txt`
        3. Launch app: `streamlit run app/app.py`
        """)

    elif doc_section == "Project Structure":
        st.header("Project Structure")
        st.code("""
        DSA2040A_DataMining_Group1/
        ├── app/
        │   └── app.py
        ├── data/
        │   └── us_model_data/training_model_dataset.csv
        ├── models/
        │   └── mental_health_model.pkl
        ├── notebooks/
        │   ├── 1_cleaning_and_filtering.ipynb
        │   ├── 2_exploratory_analysis_us.py
        │   ├── 3_classification_model.ipynb
        │   └── 4_dashboard_insights.ipynb
        ├── requirements.txt
        └── README.md
        """)

    elif doc_section == "Machine Learning Summary":
        st.header("Machine Learning Summary")
        st.markdown("""
        - **Target Variable:** treatment (Yes/No)
        - **Classifier:** Random Forest Classifier
        - **Accuracy:** ~61%
        - **F1 Score:** ~66% for treatment-seeking class
        """)

    elif doc_section == "Key Insights":
        st.header("Key Insights")
        st.write("""
        - Workers with family history and no anonymity protection were more likely to seek treatment.
        - Smaller companies lacked mental health support programs.
        - Gender and age group differences were observed.
        """)

    elif doc_section == "Future Improvements":
        st.header("Future Improvements")
        st.write("""
        - Retrain with more recent or diverse datasets
        - Add probability explanations and SHAP feature interpretations
        - Include demographic visualizations
        """)

    elif doc_section == "Contributors":
        st.header("Contributors")
        st.write("""
        - Nathan – End-to-end data pipeline
        - Nicholas – Data wrangling & encoding
        - Faith – EDA & statistical modeling
        - Merhawit – Machine learning modeling
        - Mark – Dashboard, Streamlit app & documentation
        """)

    elif doc_section == "Contact & Support":
        st.header("Contact & Support")
        st.write("""
        **Email:** chweyamark@gmail.com  
        **GitHub Repo:** [DSA2040A_DataMining_Group1](https://github.com/markchweya/DSA2040A_DataMining_Group1)
        """)
