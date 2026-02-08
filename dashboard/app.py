import streamlit as st
import pickle
import numpy as np

MODEL_PATH = "best_model.pkl"

st.title("Student Dropout Risk Prediction")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

attendance = st.number_input("Attendance Percentage", 0.0, 100.0)
assignment = st.number_input("Assignment Score", 0.0, 100.0)
midsem = st.number_input("Mid-Sem Exam Score", 0.0, 100.0)
semester = st.number_input("Semester", 1, 8)
difficulty = st.number_input("Course Difficulty", 1, 5)

if st.button("Predict"):
    X = np.array([[attendance, assignment, midsem, semester, difficulty]])
    prediction = model.predict(X)[0]

    if prediction == 1:
        st.error("Student is AT RISK of dropout")
    else:
        st.success("Student is NOT at risk of dropout")
