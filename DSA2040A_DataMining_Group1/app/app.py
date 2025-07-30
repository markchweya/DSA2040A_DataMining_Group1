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

