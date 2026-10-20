
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
MODEL_PATH = MODEL_DIR / "advanced_model.pkl"

SIMILARITY_THRESHOLD = 0.90


def load_data():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    for name, df in [("train", train), ("test", test)]:
        required = {"cleaned_text", "severity"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"{name} missing columns: {missing}"
            )

    train = train.dropna(
        subset=["cleaned_text", "severity"]
    ).copy()

    test = test.dropna(
        subset=["cleaned_text", "severity"]
    ).copy()

    for df in (train, test):
        df["cleaned_text"] = (
            df["cleaned_text"].astype(str).str.strip()
        )
        df["severity"] = (
            df["severity"].astype(str).str.strip().str.lower()
        )

    train = train[train["cleaned_text"] != ""]
    test = test[test["cleaned_text"] != ""]

    return train, test


def filter_near_duplicates(train, test):
    print("\nChecking train/test similarity...")

    # Fit similarity detector on training text only.
    detector = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=1,
        max_features=50000,
        dtype=np.float32,
    )

    train_vectors = detector.fit_transform(
        train["cleaned_text"]
    )
    test_vectors = detector.transform(
        test["cleaned_text"]
    )

    neighbors = NearestNeighbors(
        n_neighbors=1,
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    )
    neighbors.fit(train_vectors)

    # Batch queries to limit temporary memory usage.
    max_similarity = np.empty(len(test))

    batch_size = 128

    for start in range(0, len(test), batch_size):
        end = min(start + batch_size, len(test))

        distances, _ = neighbors.kneighbors(
            test_vectors[start:end]
        )

        max_similarity[start:end] = (
            1.0 - distances[:, 0]
        )

        print(
            f"\rSimilarity checked: {end}/{len(test)}",
            end="",
            flush=True,
        )

    print()

    keep_mask = max_similarity < SIMILARITY_THRESHOLD

    clean_test = test.loc[keep_mask].copy()
    clean_test["max_train_similarity"] = (
        max_similarity[keep_mask]
    )

    excluded = test.loc[~keep_mask].copy()
    excluded["max_train_similarity"] = (
        max_similarity[~keep_mask]
    )

    clean_test.to_csv(
        RESULTS_DIR / "advanced_clean_test.csv",
        index=False,
    )

    excluded.to_csv(
        RESULTS_DIR / "advanced_excluded_duplicates.csv",
        index=False,
    )

    return clean_test, excluded


def evaluate():
    print("=" * 60)
    print("BUGSENSE AI - ADVANCED SVM CLEAN EVALUATION")
    print("=" * 60)

    train, test = load_data()

    print("Training samples:", len(train))
    print("Original test samples:", len(test))

    bundle = joblib.load(MODEL_PATH)

    model = bundle["model"]
    encoder = bundle["encoder"]

    model_name = bundle.get("model_name", "Advanced SVM")

    print("Loaded model:", model_name)
    print("Pipeline steps:", list(model.named_steps.keys()))

    model_name = bundle.get(
        "model_name", "Advanced SVM"
    )

    print("Loaded model:", model_name)

    clean_test, excluded = filter_near_duplicates(
        train, test
    )

    print("\nExcluded near-duplicates:", len(excluded))
    print("Clean test samples:", len(clean_test))

    if clean_test.empty:
        raise ValueError("No clean test samples remain.")

    # The saved pipeline performs TF-IDF internally
    X = clean_test["cleaned_text"].astype(str)

    y_true = encoder.transform(
        clean_test["severity"]
    )

    y_pred = model.predict(X)

    labels = np.arange(len(encoder.classes_))
    class_names = list(encoder.classes_)

    metrics = {
        "model": model_name,
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "original_test_samples": len(test),
        "excluded_near_duplicates": len(excluded),
        "clean_test_samples": len(clean_test),
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "macro_precision": float(
            precision_recall_fscore_support(
                y_true, y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )[0]
        ),
        "macro_recall": float(
            precision_recall_fscore_support(
                y_true, y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )[1]
        ),
        "macro_f1": float(
            f1_score(
                y_true, y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true, y_pred,
                labels=labels,
                average="weighted",
                zero_division=0,
            )
        ),
    }

    print("\nFINAL CLEAN TEST METRICS")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).transpose()

    report_df.to_csv(
        RESULTS_DIR / "advanced_clean_classification_report.csv"
    )

    print("\nCLASSIFICATION REPORT")
    print(report_df.to_string())

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    pd.DataFrame(
        cm,
        index=class_names,
        columns=class_names,
    ).to_csv(
        RESULTS_DIR / "advanced_clean_confusion_matrix.csv"
    )

    fig, ax = plt.subplots(figsize=(10, 8))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names,
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        colorbar=False,
    )

    ax.set_title(
        "BugSense AI - Advanced SVM Clean Test Confusion Matrix"
    )

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "advanced_clean_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Per-class F1 chart
    per_class_f1 = [
        report[name]["f1-score"]
        for name in class_names
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(class_names, per_class_f1)

    ax.set_ylim(0, 1)
    ax.set_ylabel("F1-score")
    ax.set_title(
        "Advanced SVM - Clean Test Per-Class F1"
    )

    for bar, score in zip(bars, per_class_f1):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            score + 0.015,
            f"{score:.3f}",
            ha="center",
        )

    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "advanced_clean_per_class_f1.png",
        dpi=300,
    )
    plt.close(fig)

    # Overall metrics chart
    metric_names = [
        "Accuracy",
        "Macro Precision",
        "Macro Recall",
        "Macro F1",
        "Weighted F1",
    ]

    metric_values = [
        metrics["accuracy"],
        metrics["macro_precision"],
        metrics["macro_recall"],
        metrics["macro_f1"],
        metrics["weighted_f1"],
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(metric_names, metric_values)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("BugSense AI - Final Clean Test Metrics")

    for bar, value in zip(bars, metric_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.015,
            f"{value:.3f}",
            ha="center",
        )

    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "advanced_clean_metrics.png",
        dpi=300,
    )
    plt.close(fig)

    with open(
        RESULTS_DIR / "advanced_clean_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, indent=4)

    print("\nEvaluation completed!")
    print("Results saved to:", RESULTS_DIR)


if __name__ == "__main__":
    evaluate()
