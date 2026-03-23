"""
evaluate_model.py
-----------------
Loads the best saved model and the SQLite database, evaluates
performance on a fresh 20 % test split, and prints:
  - Accuracy, Precision, Recall, F1-Score
  - Full Classification Report

Usage:
    python models/evaluate_model.py
"""

import os
import pickle
import sqlite3
import sys

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from config import DB_PATH, MODEL_PATH


def load_data() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql("SELECT * FROM students", conn)
    finally:
        conn.close()
    return df


def evaluate_model() -> None:
    print("Loading data …")
    df = load_data()

    X = df.drop(columns=["student_id", "dropout_risk"])
    y = df["dropout_risk"]

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Loading model from {MODEL_PATH} …")
    with open(MODEL_PATH, "rb") as fh:
        model = pickle.load(fh)

    y_pred = model.predict(X_test)

    print("\n" + "=" * 45)
    print("       MODEL EVALUATION RESULTS")
    print("=" * 45)
    print(f"  Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Precision : {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"  Recall    : {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"  F1 Score  : {f1_score(y_test, y_pred, zero_division=0):.4f}")
    print("=" * 45)

    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Not At Risk", "At Risk"]))


if __name__ == "__main__":
    evaluate_model()
