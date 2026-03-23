"""
generate_data.py
----------------
Responsible ONLY for generating synthetic student data and saving it as a CSV.
Run this first to produce the raw dataset before running the ETL pipeline.

Usage:
    python data/generate_data.py
"""

import pandas as pd
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Output path (relative to project root)
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "student_data.csv")


def generate_student_data(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic student dataset with realistic dropout risk labels.

    Features
    --------
    attendance_percentage : float  [40, 100]
    assignment_score      : float  [35, 100]
    midsem_score          : float  [30, 100]
    semester              : int    [1, 8]
    course_difficulty     : int    [1, 5]

    Label
    -----
    dropout_risk : 0 (Not At Risk) | 1 (At Risk)
        A student is marked At Risk when ≥2 of the following conditions hold:
            - attendance  < 65 %
            - assignment  < 50
            - midsem      < 45
            - difficulty  ≥  4
    """
    np.random.seed(random_state)

    attendance        = np.random.uniform(40, 100, n_samples)
    assignment_score  = np.random.uniform(35, 100, n_samples)
    midsem_score      = np.random.uniform(30, 100, n_samples)
    semester          = np.random.randint(1, 9,  n_samples)
    course_difficulty = np.random.randint(1, 6,  n_samples)

    dropout_risk = []
    for i in range(n_samples):
        risk_score = sum([
            attendance[i]        < 65,
            assignment_score[i]  < 50,
            midsem_score[i]      < 45,
            course_difficulty[i] >= 4,
        ])
        dropout_risk.append(1 if risk_score >= 2 else 0)

    df = pd.DataFrame({
        "attendance_percentage": attendance,
        "assignment_score":      assignment_score,
        "midsem_score":          midsem_score,
        "semester":              semester,
        "course_difficulty":     course_difficulty,
        "dropout_risk":          dropout_risk,
    })

    logger.info(f"Generated {n_samples} records | At-Risk: {sum(dropout_risk)} ({sum(dropout_risk)/n_samples*100:.1f}%)")
    return df


def save_data(df: pd.DataFrame, path: str = OUTPUT_CSV) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Dataset saved → {path}")


if __name__ == "__main__":
    df = generate_student_data(n_samples=1000)
    save_data(df)
    print("\nSample (first 5 rows):")
    print(df.head())
    print(f"\nClass distribution:\n{df['dropout_risk'].value_counts().rename({0: 'Not At Risk', 1: 'At Risk'})}")
