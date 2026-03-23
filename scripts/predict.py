"""
predict.py
----------
Command-line utility for single-student dropout risk prediction.
Loads the saved best model and returns a prediction + probability.

Usage:
    python models/predict.py
  or import and call predict_dropout() from other scripts.
"""

import os
import pickle
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from config import MODEL_PATH


def predict_dropout(
    attendance: float,
    assignment: float,
    midsem: float,
    semester: int,
    difficulty: int,
) -> dict:
    """
    Predict dropout risk for a single student profile.

    Returns
    -------
    dict with keys:
        label       : "At Risk" | "Not At Risk"
        probability : float (0–100 %, confidence of At Risk)
    """
    with open(MODEL_PATH, "rb") as fh:
        model = pickle.load(fh)

    X = np.array([[attendance, assignment, midsem, semester, difficulty]])
    label = "At Risk" if model.predict(X)[0] == 1 else "Not At Risk"

    try:
        prob = model.predict_proba(X)[0][1] * 100
    except AttributeError:
        prob = 95.0 if label == "At Risk" else 5.0

    return {"label": label, "probability": round(prob, 2)}


if __name__ == "__main__":
    # Demo prediction
    profile = dict(attendance=60, assignment=45, midsem=40, semester=3, difficulty=4)
    result  = predict_dropout(**profile)

    print("\n── Prediction ─────────────────────────────")
    print(f"  Input   : {profile}")
    print(f"  Label   : {result['label']}")
    print(f"  Risk %  : {result['probability']:.1f}%")
    print("────────────────────────────────────────────")
