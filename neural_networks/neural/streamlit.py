import streamlit as st
import numpy as np
import pandas as pd
from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.model import train_model
from src.predict import predict_single

st.title("🎓 Grad School Admission Chance Predictor")

df = load_data("Data/raw/Admission(in).csv")
(X_train, X_test, y_train, y_test), scaler = preprocess_data(df)
model = train_model(X_train, y_train)

st.sidebar.header("Input Student Profile")
gre = st.sidebar.slider("GRE Score", 260, 340, 320)
toefl = st.sidebar.slider("TOEFL Score", 80, 120, 110)
rating = st.sidebar.slider("University Rating", 1, 5, 3)
sop = st.sidebar.slider("SOP Strength", 1.0, 5.0, 3.0)
lor = st.sidebar.slider("LOR Strength", 1.0, 5.0, 3.0)
cgpa = st.sidebar.slider("CGPA", 6.0, 10.0, 8.5)
research = st.sidebar.selectbox("Research Experience", [0, 1])

input_data = [gre, toefl, rating, sop, lor, cgpa, research]

if st.button("Predict Admission Chance"):
    chance = predict_single(model, scaler, input_data)
    st.success(f"🎯 Predicted Admission Chance: {chance:.2f}")
