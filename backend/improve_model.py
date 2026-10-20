
from pathlib import Path
import json
import time

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_PATH = BASE_DIR / "dataset" / "train.csv"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


def main():
    print("\nBUGSENSE AI - SVM IMPROVEMENT\n")

    df = pd.read_csv(TRAIN_PATH)

    required = {"cleaned_text", "severity"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.dropna(subset=["cleaned_text", "severity"])
    df = df[df["cleaned_text"].str.strip().ne("")]

    # Validation split from training data only
    X_train, X_val, y_train, y_val = train_test_split(
        df["cleaned_text"],
        df["severity"],
        test_size=0.2,
        random_state=42,
        stratify=df["severity"],
    )

    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)
    y_val_encoded = encoder.transform(y_val)

    configurations = [
        {
            "name": "SVM Balanced",
            "max_features": 10000,
            "ngram_range": (1, 2),
            "C": 1.0,
            "class_weight": "balanced",
        },
        {
            "name": "SVM Balanced C0.5",
            "max_features": 20000,
            "ngram_range": (1, 2),
            "C": 0.5,
            "class_weight": "balanced",
        },
        {
            "name": "SVM Balanced C2",
            "max_features": 20000,
            "ngram_range": (1, 2),
            "C": 2.0,
            "class_weight": "balanced",
        },
        {
            "name": "SVM Unweighted",
            "max_features": 20000,
            "ngram_range": (1, 2),
            "C": 1.0,
            "class_weight": None,
        },
    ]

    results = []
    best_score = -1
    best_config = None

    for config in configurations:
        print(f"\nTraining {config['name']}...")

        start = time.perf_counter()

        vectorizer = TfidfVectorizer(
            max_features=config["max_features"],
            ngram_range=config["ngram_range"],
            sublinear_tf=True,
            min_df=2,
        )

        X_train_vec = vectorizer.fit_transform(X_train)
        X_val_vec = vectorizer.transform(X_val)

        model = LinearSVC(
            C=config["C"],
            class_weight=config["class_weight"],
            max_iter=5000,
            random_state=42,
        )

        model.fit(X_train_vec, y_train_encoded)
        predictions = model.predict(X_val_vec)

        accuracy = accuracy_score(
            y_val_encoded, predictions
        )

        macro_f1 = f1_score(
            y_val_encoded,
            predictions,
            average="macro",
            zero_division=0,
        )

        weighted_f1 = f1_score(
            y_val_encoded,
            predictions,
            average="weighted",
            zero_division=0,
        )

        elapsed = time.perf_counter() - start

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro F1: {macro_f1:.4f}")
        print(f"Weighted F1: {weighted_f1:.4f}")

        print(
            classification_report(
                y_val_encoded,
                predictions,
                labels=list(range(len(encoder.classes_))),
                target_names=encoder.classes_,
                zero_division=0,
            )
        )

        results.append({
            "model": config["name"],
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "training_time_seconds": elapsed,
        })

        if macro_f1 > best_score:
            best_score = macro_f1
            best_config = config.copy()

    # Refit the selected configuration on all training data
    print("\nRefitting best configuration on full training set...")

    final_vectorizer = TfidfVectorizer(
        max_features=best_config["max_features"],
        ngram_range=best_config["ngram_range"],
        sublinear_tf=True,
        min_df=2,
    )

    X_full = final_vectorizer.fit_transform(
        df["cleaned_text"]
    )
    y_full = encoder.transform(df["severity"])

    final_model = LinearSVC(
        C=best_config["C"],
        class_weight=best_config["class_weight"],
        max_iter=5000,
        random_state=42,
    )

    final_model.fit(X_full, y_full)

    bundle = {
        "model": final_model,
        "vectorizer": final_vectorizer,
        "encoder": encoder,
        "model_name": best_config["name"],
    }

    model_path = MODEL_DIR / "improved_model.pkl"
    joblib.dump(bundle, model_path)

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        RESULTS_DIR / "svm_improvement_comparison.csv",
        index=False,
    )

    with open(
        RESULTS_DIR / "svm_best_configuration.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "best_configuration": best_config,
                "validation_macro_f1": best_score,
            },
            file,
            indent=4,
        )

    print("\nIMPROVEMENT COMPLETE")
    print("\nValidation comparison:")
    print(results_df.to_string(index=False))
    print("\nBest configuration:", best_config["name"])
    print("Best validation Macro F1:", round(best_score, 4))
    print("Saved model:", model_path)


if __name__ == "__main__":
    main()
