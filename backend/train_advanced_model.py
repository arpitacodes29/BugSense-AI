
from pathlib import Path
import time
import json

import joblib
import pandas as pd

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

print("\nBUGSENSE AI - ADVANCED MODEL TRAINING\n")

# Load training data only
df = pd.read_csv(DATA_DIR / "train.csv")
df = df.dropna(subset=["cleaned_text", "severity"])
df = df[df["cleaned_text"].astype(str).str.strip().ne("")]

X = df["cleaned_text"].astype(str)
y = df["severity"].astype(str)

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train)
y_val_encoded = encoder.transform(y_val)

def make_features():
    return FeatureUnion([
        ("word", TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            max_features=20000,
            min_df=2,
            sublinear_tf=True
        )),
        ("char", TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            max_features=30000,
            min_df=2,
            sublinear_tf=True
        ))
    ])

models = {
    "Advanced SVM": LinearSVC(
        C=0.5,
        class_weight="balanced",
        max_iter=5000,
        random_state=42
    ),
    "Advanced Logistic Regression": LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=2000,
        solver="lbfgs"
    )
}

results = []
best_score = -1
best_name = None

for name, classifier in models.items():
    print(f"\nTraining {name}...")
    start = time.time()

    pipeline = Pipeline([
        ("features", make_features()),
        ("classifier", classifier)
    ])

    pipeline.fit(X_train, y_train_encoded)
    predictions = pipeline.predict(X_val)

    accuracy = accuracy_score(y_val_encoded, predictions)
    macro_f1 = f1_score(
        y_val_encoded, predictions,
        average="macro",
        zero_division=0
    )
    weighted_f1 = f1_score(
        y_val_encoded, predictions,
        average="weighted",
        zero_division=0
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Macro F1": macro_f1,
        "Weighted F1": weighted_f1,
        "Training Time": round(time.time() - start, 2)
    })

    print(classification_report(
        y_val_encoded,
        predictions,
        target_names=encoder.classes_,
        zero_division=0
    ))

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    if macro_f1 > best_score:
        best_score = macro_f1
        best_name = name

# Save comparison
results_df = pd.DataFrame(results)
results_df.to_csv(
    RESULTS_DIR / "advanced_model_comparison.csv",
    index=False
)

print("\nADVANCED MODEL COMPARISON")
print(results_df.to_string(index=False))

print("\nBest model:", best_name)

# Refit the winner on the entire training dataset
final_pipeline = Pipeline([
    ("features", make_features()),
    ("classifier", models[best_name])
])

final_pipeline.fit(
    X,
    encoder.transform(y)
)

joblib.dump({
    "model": final_pipeline,
    "encoder": encoder,
    "model_name": best_name,
    "input_format": "cleaned_text",
    "pipeline_includes_vectorizer": True
}, MODEL_DIR / "advanced_model.pkl")

print("\nSaved: models/advanced_model.pkl")
print("Training completed successfully!")
