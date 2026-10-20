
from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)

BASE_DIR = Path(__file__).resolve().parent.parent

TEST_PATH = BASE_DIR / "dataset" / "test.csv"
MODEL_DIR = BASE_DIR / "models"

df = pd.read_csv(TEST_PATH)

df = df.dropna(subset=["cleaned_text", "severity"])
df = df[df["cleaned_text"].astype(str).str.strip().ne("")]

results = []

for filename in ["optimized_model.pkl", "improved_model.pkl"]:

    bundle = joblib.load(MODEL_DIR / filename)

    model = bundle["model"]
    vectorizer = bundle["vectorizer"]
    encoder = bundle["encoder"]

    X = vectorizer.transform(df["cleaned_text"].astype(str))
    y_true = df["severity"].astype(str)

    predictions = model.predict(X)
    y_pred = encoder.inverse_transform(predictions)

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")

    print(f"\nMODEL: {filename}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    results.append({
        "Model": filename,
        "Accuracy": accuracy,
        "Macro F1": macro_f1,
        "Weighted F1": weighted_f1
    })

print("\nFINAL MODEL COMPARISON")
print(pd.DataFrame(results).to_string(index=False))
