
from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Use training data only for optimization
df = pd.read_csv(DATA_DIR / "train.csv")
df = df.dropna(subset=["cleaned_text", "severity"])

encoder = LabelEncoder()
y = encoder.fit_transform(df["severity"])
X = df["cleaned_text"].astype(str)

X_train_text, X_val_text, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

print("Training samples:", len(X_train_text))
print("Validation samples:", len(X_val_text))

vectorizer = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1, 2),
    min_df=3,
    sublinear_tf=True,
    dtype=np.float32
)

X_train = vectorizer.fit_transform(X_train_text)
X_val = vectorizer.transform(X_val_text)

# Calculate class weights for XGBoost
class_counts = np.bincount(y_train)
class_weights = len(y_train) / (
    len(class_counts) * class_counts
)
sample_weights = class_weights[y_train]

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    ),

    "SVM": LinearSVC(
        C=0.5,
        class_weight="balanced",
        max_iter=5000,
        random_state=42
    ),

    "XGBoost": XGBClassifier(
        n_estimators=250,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        tree_method="hist",
        eval_metric="mlogloss",
        n_jobs=4,
        random_state=42
    )
}

results = []
trained_models = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    start = time.time()

    if name == "XGBoost":
        model.fit(
            X_train,
            y_train,
            sample_weight=sample_weights
        )
    else:
        model.fit(X_train, y_train)

    predictions = model.predict(X_val)

    accuracy = accuracy_score(y_val, predictions)
    macro_f1 = f1_score(
        y_val, predictions, average="macro"
    )
    weighted_f1 = f1_score(
        y_val, predictions, average="weighted"
    )

    results.append({
        "Model": name,
        "Accuracy": round(accuracy, 4),
        "Macro F1": round(macro_f1, 4),
        "Weighted F1": round(weighted_f1, 4),
        "Time (s)": round(time.time() - start, 2)
    })

    trained_models[name] = model

    print(classification_report(
        y_val,
        predictions,
        target_names=encoder.classes_,
        zero_division=0
    ))

results_df = pd.DataFrame(results)
results_df.to_csv(
    MODEL_DIR / "optimized_comparison.csv",
    index=False
)

print("\nOPTIMIZED MODEL COMPARISON")
print(results_df.to_string(index=False))

best_name = results_df.sort_values(
    "Macro F1",
    ascending=False
).iloc[0]["Model"]

best_model = trained_models[best_name]

joblib.dump(
    {
        "model": best_model,
        "vectorizer": vectorizer,
        "encoder": encoder,
        "model_name": best_name
    },
    MODEL_DIR / "optimized_model.pkl"
)

print("\nBest optimized model:", best_name)
print("Saved to models/optimized_model.pkl")
