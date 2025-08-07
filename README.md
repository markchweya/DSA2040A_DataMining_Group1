DSA2040A Data Mining Project – Group 1
Project Title: Mental Health Support Prediction in Tech (U.S. Respondents)
This project uses 2014 survey data from the Open Sourcing Mental Illness (OSMI) initiative to predict whether a tech employee is likely to seek mental health treatment based on personal, workplace, and organizational factors. We demonstrate the full data science lifecycle: ETL, data cleaning, EDA, feature engineering, model building, evaluation, insights generation, and deployment.

Live App: Try the Streamlit App


Team Members & Contributions
Name	Role & Contribution
Nathan	End-to-end data pipeline and orchestration
Nicholas	Data wrangling, encoding, and enrichment
Faith	Exploratory Data Analysis and Statistical Modeling
Merhawit	Machine Learning modeling and classification metrics
Mark	Dashboard creation, Streamlit app development, documentation

Dataset
Source: OSMI Mental Health in Tech Survey (2014)

Size: ~1,200 responses

Focus: U.S.-only respondents for increased regional reliability

Project Structure
sql
Copy
Edit
DSA2040A_DataMining_Group1/
│
├── app/
│   └── app.py              ← Streamlit front-end app
│
├── data/
│   └── us_model_data/
│       └── training_model_dataset.csv  ← Preprocessed training dataset (U.S. only)
│
├── models/
│   └── mental_health_model.pkl        ← Final trained Random Forest model
│
├── notebooks/
│   ├── 1_cleaning_and_filtering.ipynb
│   ├── 2_exploratory_analysis_us.py
│   ├── 3_classification_model.ipynb
│   └── 4_dashboard_insights.ipynb
│
├── requirements.txt        ← Streamlit app dependencies
├── README.md               ← This file
Machine Learning Summary
Target: treatment (Yes/No)

Classifier: Random Forest Classifier

Training subset: U.S.-only data

Train-test split: 80/20

Accuracy: ~61%

F1 Score: ~66% for treatment-seeking class

Key Features Used
The model was trained on the following variables:

Age

Gender

Self-employed status

Family history of mental illness

Company size

Work interference due to mental health

Employer-provided benefits

Access to care options

Protection of anonymity

Insights From the Data
Workers with family history and no anonymity protection were more likely to seek treatment.

Smaller companies tended to lack mental health support programs.

There were gender and age group differences in treatment-seeking patterns.

Access to benefits and care options positively influenced mental health treatment.

How to Run Locally
1. Clone the Repo
bash
Copy
Edit
git clone https://github.com/[your-username]/DSA2040A_DataMining_Group1.git
cd DSA2040A_DataMining_Group1
2. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
3. Run the Streamlit App
bash
Copy
Edit
streamlit run app/app.py
Ensure the mental_health_model.pkl and the preprocessed dataset are in their expected folders.

Notes
The app supports only U.S. respondents as per the model training scope.

This is not a diagnostic tool, but a demonstration of data mining techniques on real-world survey data.

Model bias may exist due to limited demographics (2014 data, tech sector only).

Future Improvements
Retrain with more recent or diverse datasets (e.g., post-COVID surveys).

Add confidence intervals or probability estimates.

Include demographic visualizations on app homepage.

Incorporate SHAP for feature interpretability.

