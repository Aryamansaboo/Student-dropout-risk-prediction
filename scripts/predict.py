import pickle
import numpy as np

MODEL_PATH = "best_model.pkl"

def predict_dropout(attendance, assignment, midsem, semester, difficulty):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    X = np.array([[attendance, assignment, midsem, semester, difficulty]])
    prediction = model.predict(X)[0]

    return "At Risk" if prediction == 1 else "Not At Risk"


if __name__ == "__main__":
    result = predict_dropout(
        attendance=60,
        assignment=45,
        midsem=40,
        semester=3,
        difficulty=4
    )
    print("Prediction:", result)
