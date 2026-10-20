from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dataset"

train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")

train_text = train["cleaned_text"].fillna("").astype(str)
test_text = test["cleaned_text"].fillna("").astype(str)

print("\nBUGSENSE AI - NEAR DUPLICATE CHECK\n")

vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    min_df=2,
    max_features=30000
)

X_train = vectorizer.fit_transform(train_text)
X_test = vectorizer.transform(test_text)

threshold = 0.90
batch_size = 100

near_duplicates = 0
max_similarity = 0.0

for start in range(0, X_test.shape[0], batch_size):

    end = min(start + batch_size, X_test.shape[0])

    similarities = cosine_similarity(
        X_test[start:end],
        X_train
    )

    best_scores = similarities.max(axis=1)

    near_duplicates += np.sum(
        best_scores >= threshold
    )

    max_similarity = max(
        max_similarity,
        float(best_scores.max())
    )

    print(f"Checked {end}/{X_test.shape[0]} test reports")

print("\nRESULTS")
print("Near-duplicate test reports:", near_duplicates)
print("Maximum similarity:", round(max_similarity, 4))

print(
    "Affected test percentage:",
    round(100 * near_duplicates / len(test), 2),
    "%"
)