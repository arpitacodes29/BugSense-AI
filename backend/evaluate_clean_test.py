
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "dataset"
MODEL_PATH = BASE_DIR / "models" / "optimized_model.pkl"

train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")

bundle = joblib.load(MODEL_PATH)

model = bundle["model"]
vectorizer = bundle["vectorizer"]
encoder = bundle["encoder"]

train_text = train["cleaned_text"].fillna("").astype(str)
test_text = test["cleaned_text"].fillna("").astype(str)

# Separate vectorizer used only for similarity screening
similarity_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    min_df=2,
    max_features=30000
)

X_train_sim = similarity_vectorizer.fit_transform(train_text)
X_test_sim = similarity_vectorizer.transform(test_text)

threshold = 0.90
batch_size = 100

keep_mask = np.ones(len(test), dtype=bool)

for start in range(0, len(test), batch_size):
    end = min(start + batch_size, len(test))

    # Sparse multiplication avoids a large dense matrix
    similarities = (
        X_test_sim[start:end] @ X_train_sim.T
    ).tocsr()

    max_scores = np.asarray(
        similarities.max(axis=1).toarray()
    ).ravel()

    keep_mask[start:end] = max_scores < threshold

clean_test = test.loc[keep_mask].copy()

print("\nBUGSENSE AI - CLEAN TEST EVALUATION\n")
print("Original test samples:", len(test))
print("Excluded near-duplicates:", (~keep_mask).sum())
print("Clean test samples:", len(clean_test))

if clean_test.empty:
    raise ValueError("No test samples remain.")

X_clean = vectorizer.transform(
    clean_test["cleaned_text"].astype(str)
)

predictions = model.predict(X_clean)

# Use the encoder saved with the trained model
y_true = encoder.transform(
    clean_test["severity"].astype(str)
)

print("\nCLEAN TEST RESULTS")
print("Accuracy:", round(
    accuracy_score(y_true, predictions), 4
))
print("Macro F1:", round(
    f1_score(
        y_true, predictions,
        average="macro",
        zero_division=0
    ), 4
))
print("Weighted F1:", round(
    f1_score(
        y_true, predictions,
        average="weighted",
        zero_division=0
    ), 4
))

print("\nCLASSIFICATION REPORT")
print(classification_report(
    y_true,
    predictions,
    labels=np.arange(len(encoder.classes_)),
    target_names=encoder.classes_,
    zero_division=0
))
