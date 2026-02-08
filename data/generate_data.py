import pandas as pd
import numpy as np
import os

def generate_student_data(n_samples=1000, random_state=42):
    np.random.seed(random_state)

    attendance = np.random.uniform(40, 100, n_samples)
    assignment_score = np.random.uniform(35, 100, n_samples)
    midsem_score = np.random.uniform(30, 100, n_samples)
    semester = np.random.randint(1, 9, n_samples)
    course_difficulty = np.random.randint(1, 6, n_samples)

    dropout_risk = []

    for i in range(n_samples):
        risk_score = 0

        if attendance[i] < 65:
            risk_score += 1
        if assignment_score[i] < 50:
            risk_score += 1
        if midsem_score[i] < 45:
            risk_score += 1
        if course_difficulty[i] >= 4:
            risk_score += 1

        if risk_score >= 2:
            dropout_risk.append(1)  # At Risk
        else:
            dropout_risk.append(0)  # Not At Risk

    data = pd.DataFrame({
        "attendance_percentage": attendance,
        "assignment_score": assignment_score,
        "midsem_score": midsem_score,
        "semester": semester,
        "course_difficulty": course_difficulty,
        "dropout_risk": dropout_risk
    })

    return data

if __name__ == "__main__":
    os.makedirs("data/output", exist_ok=True)

    df = generate_student_data(n_samples=1000)
    df.to_csv("data/output/student_data.csv", index=False)

    print("Student dataset generated successfully.")
