
from pathlib import Path
import json
import time

import joblib
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")

X_train_text = train["cleaned_text"].fillna("")
X_test_text = test["cleaned_text"].fillna("")

encoder = LabelEncoder()
y_train = encoder.fit_transform(train["severity"])
y_test = encoder.transform(test["severity"])

print("Training samples:", len(train))
print("Testing samples:", len(test))
print("Classes:", encoder.classes_)

# Fit only on training data
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=3,
    sublinear_tf=True,
    dtype=np.float32
)

X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

print("TF-IDF shape:", X_train.shape)

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=25,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    ),
    "SVM": LinearSVC(
        class_weight="balanced",
        max_iter=3000,
        random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.1,
        tree_method="hist",
        objective="multi:softprob",
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

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    precision, recall, macro_f1, _ = (
        precision_recall_fscore_support(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        )
    )

    weighted_f1 = precision_recall_fscore_support(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )[2]

    elapsed = time.time() - start

    results.append({
        "Model": name,
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "Macro F1": round(macro_f1, 4),
        "Weighted F1": round(weighted_f1, 4),
        "Time (s)": round(elapsed, 2)
    })

    trained_models[name] = model

    print(classification_report(
        y_test,
        predictions,
        target_names=encoder.classes_,
        zero_division=0
    ))

    cm = confusion_matrix(y_test, predictions)
    pd.DataFrame(
        cm,
        index=encoder.classes_,
        columns=encoder.classes_
    ).to_csv(
        MODEL_DIR / (
            name.lower().replace(" ", "_")
            + "_confusion_matrix.csv"
        )
    )

results_df = pd.DataFrame(results)
results_df.to_csv(
    MODEL_DIR / "model_comparison.csv",
    index=False
)

print("\nMODEL COMPARISON")
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
    MODEL_DIR / "best_model.pkl"
)

with open(
    MODEL_DIR / "training_summary.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        {
            "selected_model": best_name,
            "classes": encoder.classes_.tolist(),
            "results": results
        },
        f,
        indent=2
    )

print("\nBest model:", best_name)
print("Model saved to models/best_model.pkl")
print("Training complete!")
