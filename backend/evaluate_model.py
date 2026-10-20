
from pathlib import Path
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "optimized_model.pkl"


def clean_text(text):
    import re

    text = str(text).lower()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^a-z0-9_./+-]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def find_test_data():
    test_path = BASE_DIR / "dataset" / "test.csv"

    if not test_path.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {test_path}"
        )

    return test_path


def main():
    print("\nBUGSENSE AI - MODEL EVALUATION\n")

    bundle = joblib.load(MODEL_PATH)

    model = bundle["model"]
    vectorizer = bundle["vectorizer"]
    encoder = bundle["encoder"]

    print("Loaded model:", bundle["model_name"])

    test_path = find_test_data()
    test_df = pd.read_csv(test_path)

    print("Test dataset:", test_path.name)
    print("Columns:", list(test_df.columns))

    # Match these names to the preprocessing output.
    text_candidates = [
        "clean_text", "text", "description",
        "Short Description"
    ]
    label_candidates = [
        "severity", "label", "Severity Label"
    ]

    text_col = next(
        (c for c in text_candidates if c in test_df.columns),
        None
    )
    label_col = next(
        (c for c in label_candidates if c in test_df.columns),
        None
    )

    if text_col is None or label_col is None:
        raise ValueError(
            "Could not identify text/severity columns. "
            "Update text_candidates and label_candidates."
        )

    test_df = test_df.dropna(
        subset=[text_col, label_col]
    ).copy()

    texts = test_df[text_col].astype(str).map(clean_text)
    labels = (
        test_df[label_col]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    valid = (
        texts.str.len().gt(0)
        & labels.isin(encoder.classes_)
    )

    texts = texts[valid]
    labels = labels[valid]

    if len(texts) == 0:
        raise ValueError("No valid test records found.")

    X_test = vectorizer.transform(texts)
    y_true = encoder.transform(labels)

    print("Evaluation samples:", len(y_true))

    y_pred = model.predict(X_test)

    metrics = {
        "model": bundle["model_name"],
        "test_samples": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(
            y_true, y_pred,
            average="macro",
            zero_division=0
        ),
        "recall_macro": recall_score(
            y_true, y_pred,
            average="macro",
            zero_division=0
        ),
        "macro_f1": f1_score(
            y_true, y_pred,
            average="macro",
            zero_division=0
        ),
        "weighted_f1": f1_score(
            y_true, y_pred,
            average="weighted",
            zero_division=0
        )
    }

    print("\nEVALUATION RESULTS")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    labels_encoded = np.arange(len(encoder.classes_))

    report = classification_report(
        y_true,
        y_pred,
        labels=labels_encoded,
        target_names=encoder.classes_,
        zero_division=0
    )

    print("\nCLASSIFICATION REPORT\n")
    print(report)

    # Save metrics and classification report.
    pd.DataFrame([metrics]).to_csv(
        RESULTS_DIR / "evaluation_metrics.csv",
        index=False
    )

    (RESULTS_DIR / "classification_report.txt").write_text(
        report,
        encoding="utf-8"
    )

    # Confusion matrix.
    cm = confusion_matrix(
        y_true, y_pred,
        labels=labels_encoded
    )

    fig, ax = plt.subplots(figsize=(9, 7))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=encoder.classes_
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        colorbar=False
    )

    ax.set_title("BugSense AI - Confusion Matrix")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "confusion_matrix.png",
        dpi=300
    )
    plt.close(fig)

    # Evaluation metrics chart.
    chart_metrics = {
        "Accuracy": metrics["accuracy"],
        "Precision": metrics["precision_macro"],
        "Recall": metrics["recall_macro"],
        "Macro F1": metrics["macro_f1"],
        "Weighted F1": metrics["weighted_f1"]
    }

    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.bar(
        list(chart_metrics.keys()),
        list(chart_metrics.values())
    )

    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("BugSense AI - Model Evaluation")

    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{bar.get_height():.3f}",
            ha="center"
        )

    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "evaluation_chart.png",
        dpi=300
    )
    plt.close(fig)

    print("\nEvaluation completed!")
    print("Results saved to:", RESULTS_DIR)


if __name__ == "__main__":
    main()
