import sqlite3
import pandas as pd
import pickle
import json
import logging
from typing import Dict, Any

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

try:
    from config import DB_PATH, MODEL_PATH, METRICS_PATH
except ImportError:
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from config import DB_PATH, MODEL_PATH, METRICS_PATH

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data() -> pd.DataFrame:
    """Load preprocessed student data from SQLite."""
    logger.info(f"Connecting to database at {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql("SELECT * FROM students", conn)
    finally:
        conn.close()
    return df

def build_pipeline(classifier) -> Pipeline:
    """Build a standard ML pipeline including scaling."""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', classifier)
    ])

def train_and_evaluate():
    """Train multiple models and save the best one along with metrics for all."""
    logger.info("Loading data...")
    try:
        df = load_data()
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    if df.empty:
        logger.error("Database is empty. Please run the ETL pipeline first.")
        return

    # Features and Target
    X = df.drop(columns=["student_id", "dropout_risk"])
    y = df["dropout_risk"]

    logger.info("Splitting dataset into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Support Vector": SVC(probability=True, random_state=42),
        "K-Neighbors": KNeighborsClassifier(),
    }

    best_model = None
    best_f1 = 0.0
    all_metrics: Dict[str, Dict[str, float]] = {}

    logger.info("--- Starting Model Training ---")
    for name, clf in models.items():
        logger.info(f"Training {name}...")
        pipeline = build_pipeline(clf)
        pipeline.fit(X_train, y_train)
        
        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        all_metrics[name] = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1
        }
        logger.info(f"[{name}] Acc: {acc:.3f} | Prec: {prec:.3f} | Rec: {rec:.3f} | F1: {f1:.3f}")

        if f1 > best_f1:
            best_f1 = f1
            best_model = pipeline

    logger.info("Saving model metrics to JSON...")
    with open(METRICS_PATH, "w") as f:
        json.dump(all_metrics, f, indent=4)

    logger.info(f"Saving best model (F1: {best_f1:.3f}) to {MODEL_PATH}...")
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best_model, f)
        
    logger.info("Training pipeline completed successfully.")

if __name__ == "__main__":
    train_and_evaluate()
