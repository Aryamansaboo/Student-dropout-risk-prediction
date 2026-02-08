import sqlite3
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

DB_PATH = "database/student_data.db"
MODEL_PATH = "best_model.pkl"


def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM students", conn)
    conn.close()
    return df


def evaluate_model():
    df = load_data()

    X = df.drop(columns=["student_id", "dropout_risk"])
    y = df["dropout_risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    y_pred = model.predict(X_test)

    print("Model Evaluation Results")
    print("-------------------------")
    print(f"Accuracy  : {accuracy_score(y_test, y_pred):.2f}")
    print(f"Precision : {precision_score(y_test, y_pred):.2f}")
    print(f"Recall    : {recall_score(y_test, y_pred):.2f}")
    print(f"F1 Score  : {f1_score(y_test, y_pred):.2f}")

    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    evaluate_model()
