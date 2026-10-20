
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

test = pd.read_csv(TEST_PATH)
test = test.dropna(subset=["cleaned_text", "severity"])

X_test = test["cleaned_text"].astype(str)
y_test = test["severity"].astype(str)

models = {
    "Original Optimized SVM": MODEL_DIR / "optimized_model.pkl",
    "Advanced SVM": MODEL_DIR / "advanced_model.pkl"
}

results = []

print("\nBUGSENSE AI - FINAL MODEL EVALUATION\n")

for name, path in models.items():

    bundle = joblib.load(path)

    model = bundle["model"]
    encoder = bundle["encoder"]

    y_true = encoder.transform(y_test)

    if bundle.get("pipeline_includes_vectorizer", False):
        predictions = model.predict(X_test)
    else:
        vectorizer = bundle["vectorizer"]
        features = vectorizer.transform(X_test)
        predictions = model.predict(features)

    accuracy = accuracy_score(y_true, predictions)

    macro_f1 = f1_score(
        y_true,
        predictions,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_true,
        predictions,
        average="weighted",
        zero_division=0
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Macro F1": macro_f1,
        "Weighted F1": weighted_f1
    })

    print(f"\n{name}")
    print("Accuracy:", round(accuracy, 4))
    print("Macro F1:", round(macro_f1, 4))
    print("Weighted F1:", round(weighted_f1, 4))

    print("\nClassification Report:")
    print(classification_report(
        y_true,
        predictions,
        labels=range(len(encoder.classes_)),
        target_names=encoder.classes_,
        zero_division=0
    ))

results_df = pd.DataFrame(results)

print("\nFINAL COMPARISON")
print(results_df.to_string(index=False))

results_dir = BASE_DIR / "results"
results_dir.mkdir(exist_ok=True)

results_df.to_csv(
    results_dir / "advanced_final_comparison.csv",
    index=False
)

print("\nEvaluation completed!")
