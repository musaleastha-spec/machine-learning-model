import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.title("🎙️ Voice Classifier")

# Load model
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

try:
    model = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
