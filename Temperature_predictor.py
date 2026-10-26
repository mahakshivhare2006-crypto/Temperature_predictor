import streamlit as st
import pandas as pd
import numpy as np
import joblib

# App ka Title
st.title("🎓 Student Marks Prediction App")
st.write("Apne Study Hours aur Attendance darj karein taaki predicted marks pata chal sakein!")

# 1. Model Load karein
@st.cache_resource
def load_model():
    return joblib.load("student_marks_model.pkl")

model = load_model()

# 2. User Input ke liye Sliders banayein
st.header("Student Details Input:")
study_hours = st.slider("Study Hours (Daily)", min_value=1.0, max_value=12.0, value=5.0, step=0.5)
attendance = st.slider("Attendance (%)", min_value=10.0, max_value=100.0, value=75.0, step=1.0)

# 3. Predict Button
if st.button("Predict Marks"):
    # Input data ko DataFrame me convert karein
    input_data = pd.DataFrame({
        "Study_Hours": [study_hours],
        "Attendance": [attendance]
    })
    
    # Prediction karein
    prediction = model.predict(input_data)
    predicted_marks = round(prediction[0], 2)
    
    # Result show karein
    st.success(f"🎉 Predicted Marks: **{predicted_marks}**")
