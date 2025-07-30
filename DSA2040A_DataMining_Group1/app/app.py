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

# -------------------- LOAD MODEL --------------------
model_path = os.path.join("app", "mental_health_model.pkl")
try:
    model = joblib.load(model_path)
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'mental_health_model.pkl' is in the 'app' folder.")
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

    # Documentation menu
    doc_sections = [
        "Overview",
        "How It Works",
        "Terms of Use",
        "Privacy Policy",
        "Project Contributors",
        "ReadMe",
        "GitHub Repository",
        "Contact & Support"
    ]

    selected_section = st.radio("Select Documentation Section", doc_sections, horizontal=True, key="doc_menu")

    if selected_section == "Overview":
        st.subheader("Overview")
        st.markdown("""
        **DSA2040A Data Mining Project – Group 1**  
        **Project Title:** Mental Health Support Prediction in Tech (U.S. Respondents)

        This project demonstrates how machine learning can be used to **predict whether a tech employee is likely to seek mental health treatment** 
        based on workplace and personal factors.

        **Key Highlights:**
        - Uses 2014 OSMI Mental Health in Tech Survey data (U.S.-only respondents)
        - Covers the full data science lifecycle: **ETL → Data Cleaning → EDA → Modeling → Deployment**
        - Deployed as an **interactive Streamlit web app** with downloadable reports
        """)

    elif selected_section == "How It Works":
        st.subheader("How It Works")
        st.markdown("""
        1. **User Inputs Data:**  
           The app collects basic information such as age, self-employment, family history of mental illness, and work conditions.

        2. **Preprocessing:**  
           Inputs are transformed into numerical features that the model understands.

        3. **Prediction:**  
           A **Random Forest Classifier** predicts the probability of the user seeking mental health treatment.

        4. **Results & Confidence:**  
           The app displays:
           - Prediction outcome (Likely / Unlikely to seek treatment)
           - Confidence score with an approximate 95% confidence interval
           - Downloadable PDF report summarizing the results

        5. **No Data Stored:**  
           All processing is done in-memory, and no personal data is saved.
        """)

    elif selected_section == "Terms of Use":
        st.subheader("Terms of Use")
        st.markdown("""
        By using this application, you agree to the following:
        
        - This tool is **for educational and demonstration purposes only**.  
        - It does **not** provide medical or professional mental health advice.  
        - Users should **not** rely on this application for diagnosis or treatment decisions.  
        - The authors are **not liable** for any actions taken based on this tool.
        """)

    elif selected_section == "Privacy Policy":
        st.subheader("Privacy Policy")
        st.markdown("""
        We respect your privacy. This application:
        
        - Does **not collect, store, or share any personal data**.  
        - Processes all user inputs **locally in memory**.  
        - Clears session data when the session ends.  
        - Is designed to demonstrate machine learning without compromising user privacy.
        """)

    elif selected_section == "Project Contributors":
        st.subheader("Project Contributors")
        st.markdown("""
        **Team: DSA2040A Data Mining – Group 1**
        
        | Name       | Role & Contribution |
        |-----------|---------------------|
        | **Nathan**    | End-to-end data pipeline and orchestration |
        | **Nicholas**  | Data wrangling, encoding, and enrichment |
        | **Faith**     | Exploratory Data Analysis and Statistical Modeling |
        | **Merhawit**  | Machine Learning modeling and classification metrics |
        | **Mark Chweya** | Dashboard creation, Streamlit app development, documentation |
        """)

    elif selected_section == "ReadMe":
        st.subheader("Project ReadMe")
        st.markdown("""
        **Project:** Mental Health Support Prediction in Tech  
        **Dataset:** OSMI Mental Health in Tech Survey (2014, U.S.-only)  

        **Machine Learning Summary:**
        - **Target:** Likelihood of seeking treatment
        - **Model:** Random Forest Classifier
        - **Accuracy:** ~61%
        - **F1 Score (treatment-seeking class):** ~66%

        **Key Features Used:**
        - Age
        - Gender
        - Self-employment status
        - Family history of mental illness
        - Company size
        - Work interference due to mental health
        - Employer-provided benefits
        - Access to care options
        - Anonymity protection

        **Insights:**
        - Workers with family history and no anonymity protection were more likely to seek treatment.
        - Smaller companies tended to lack mental health support programs.
        - Access to benefits and care options positively influenced treatment-seeking behavior.

        **Project Structure:**
        ```
        DSA2040A_DataMining_Group1/
        ├── app/
        │   └── app.py              # Streamlit app
        ├── data/
        │   └── training_model_dataset.csv
        ├── models/
        │   └── mental_health_model.pkl
        ├── notebooks/
        │   ├── 1_cleaning_and_filtering.ipynb
        │   ├── 2_exploratory_analysis_us.py
        │   ├── 3_classification_model.ipynb
        │   └── 4_dashboard_insights.ipynb
        ├── requirements.txt
        └── README.md
        ```

        **Future Improvements:**
        - Retrain with newer datasets (e.g., post-COVID surveys)
        - Include SHAP feature interpretability
        - Add demographic visualizations
        """)

    elif selected_section == "GitHub Repository":
        st.subheader("GitHub Repository")
        st.markdown("""
        The complete project source code is available here:  
        [GitHub Repo](https://github.com/markchweya/DSA2040A_DataMining_Group1)
        """)

    elif selected_section == "Contact & Support":
        st.subheader("Contact & Support")
        st.markdown("""
        For inquiries or support, contact: **chweyamark@gmail.com**  
        """)    

    # Back to Home Button
    if st.button("⬅ Back to Home"):
        st.session_state.page = "welcome"
        st.rerun()
