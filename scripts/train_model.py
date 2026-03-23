"""
train_model.py
--------------
Trains multiple classifiers on the student data stored in SQLite,
evaluates each on a held-out test set, saves per-model metrics
(Accuracy, Precision, Recall, F1-Score) and full Classification
Reports to JSON, then persists the best model (by F1) as a pickle.

Run AFTER etl_pipeline.py.

Usage:
    python models/train_model.py
"""

import json
import logging
import os
import pickle
import sqlite3
import sys
from typing import Dict, Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# ── resolve project root ──────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
from config import DB_PATH, MODEL_PATH, METRICS_PATH, CLF_REPORT_PATH, MODELS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    logger.info(f"Loading data from {DB_PATH} …")
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql("SELECT * FROM students", conn)
    finally:
        conn.close()
    logger.info(f"Loaded {len(df):,} records")
    return df


def build_pipeline(classifier) -> Pipeline:
    """Wrap any sklearn classifier in a StandardScaler pipeline."""
    return Pipeline([
        ("scaler",     StandardScaler()),
        ("classifier", classifier),
    ])


# ── main training function ────────────────────────────────────────────────────
def train_and_evaluate() -> None:
    logger.info("=" * 60)
    logger.info("       ML TRAINING PIPELINE — START")
    logger.info("=" * 60)

    # 1. Load
    try:
        df = load_data()
    except Exception as exc:
        logger.error(f"Failed to load data: {exc}")
        return

    if df.empty:
        logger.error("Database is empty. Run etl_pipeline.py first.")
        return

    # 2. Split
    X = df.drop(columns=["student_id", "dropout_risk"])
    y = df["dropout_risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    # 3. Define classifiers
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
        "Decision Tree":       DecisionTreeClassifier(random_state=42),
        "Support Vector":      SVC(kernel="rbf", probability=True, random_state=42),
        "K-Neighbors":         KNeighborsClassifier(n_neighbors=5),
    }

    best_pipeline  = None
    best_f1        = 0.0
    all_metrics:   Dict[str, Dict[str, float]] = {}
    all_clf_reports: Dict[str, Any]            = {}

    logger.info("\n--- Training Models ---")

    for name, clf in classifiers.items():
        logger.info(f"  ▶ Training {name} …")
        pipeline = build_pipeline(clf)
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1   = f1_score(y_test, y_pred, zero_division=0)

        all_metrics[name] = {
            "Accuracy":  round(acc,  4),
            "Precision": round(prec, 4),
            "Recall":    round(rec,  4),
            "F1 Score":  round(f1,   4),
        }

        # Full classification report (as dict for JSON serialisation)
        all_clf_reports[name] = classification_report(
            y_test, y_pred,
            target_names=["Not At Risk", "At Risk"],
            output_dict=True,
        )

        logger.info(
            f"    Acc: {acc:.3f} | Prec: {prec:.3f} | "
            f"Rec: {rec:.3f} | F1: {f1:.3f}"
        )

        if f1 > best_f1:
            best_f1      = f1
            best_pipeline = pipeline

    # 4. Save artefacts
    os.makedirs(MODELS_DIR, exist_ok=True)

    logger.info(f"\nSaving metrics  → {METRICS_PATH}")
    with open(METRICS_PATH, "w") as fh:
        json.dump(all_metrics, fh, indent=4)

    logger.info(f"Saving clf reports → {CLF_REPORT_PATH}")
    with open(CLF_REPORT_PATH, "w") as fh:
        json.dump(all_clf_reports, fh, indent=4)

    logger.info(f"Saving best model (F1={best_f1:.3f}) → {MODEL_PATH}")
    with open(MODEL_PATH, "wb") as fh:
        pickle.dump(best_pipeline, fh)

    logger.info("=" * 60)
    logger.info("       ML TRAINING PIPELINE — COMPLETE ✔")
    logger.info("=" * 60)

    # 5. Print summary table
    print("\n\n📊  MODEL PERFORMANCE SUMMARY")
    print("-" * 62)
    print(f"{'Model':<22} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}")
    print("-" * 62)
    for name, m in all_metrics.items():
        print(
            f"{name:<22} "
            f"{m['Accuracy']:>7.3f} "
            f"{m['Precision']:>7.3f} "
            f"{m['Recall']:>7.3f} "
            f"{m['F1 Score']:>7.3f}"
        )
    print("-" * 62)


if __name__ == "__main__":
    train_and_evaluate()
