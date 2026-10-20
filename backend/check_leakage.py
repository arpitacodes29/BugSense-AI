
from pathlib import Path
import re
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dataset"
MODEL_PATH = BASE_DIR / "models" / "optimized_model.pkl"

train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")

print("=" * 60)
print("BUGSENSE AI - DATA LEAKAGE CHECK")
print("=" * 60)

print("\nTRAIN / TEST DATA")
print("Training records:", len(train))
print("Testing records:", len(test))
print("Train columns:", train.columns.tolist())
print("Test columns:", test.columns.tolist())

# Normalize descriptions for duplicate detection
def normalize(text):
    if pd.isna(text):
        return ""
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text

text_col = "cleaned_text"
label_col = "severity"

if text_col not in train or text_col not in test:
    raise ValueError("cleaned_text column missing")

train["normalized_text"] = train[text_col].apply(normalize)
test["normalized_text"] = test[text_col].apply(normalize)

# 1. Exact duplicates across train and test
train_texts = set(
    train.loc[train["normalized_text"] != "", "normalized_text"]
)
test_texts = set(
    test.loc[test["normalized_text"] != "", "normalized_text"]
)

overlap = train_texts.intersection(test_texts)

affected_test = test["normalized_text"].isin(train_texts).sum()

print("\n1. EXACT TEXT OVERLAP")
print("Overlapping unique descriptions:", len(overlap))
print("Affected test records:", affected_test)
print("Affected test percentage:",
      round(affected_test / max(len(test), 1) * 100, 2), "%")

# 2. Duplicate bug IDs
if "bug_id" in train.columns and "bug_id" in test.columns:
    train_ids = set(train["bug_id"].dropna().astype(str))
    test_ids = set(test["bug_id"].dropna().astype(str))

    shared_ids = train_ids.intersection(test_ids)

    print("\n2. BUG ID OVERLAP")
    print("Shared bug IDs:", len(shared_ids))
else:
    print("\n2. BUG ID OVERLAP: columns not found")

# 3. Conflicting severity labels for identical descriptions
combined = pd.concat([train, test], ignore_index=True)

if label_col in combined.columns:
    valid = combined[combined["normalized_text"] != ""]

    conflicts = valid.groupby("normalized_text")[label_col].nunique()
    conflicting_texts = conflicts[conflicts > 1]

    print("\n3. CONFLICTING LABELS")
    print("Descriptions with multiple severity labels:",
          len(conflicting_texts))

# 4. Duplicates within each dataset
print("\n4. INTERNAL DUPLICATES")
print("Duplicate train descriptions:",
      train["normalized_text"].duplicated().sum())
print("Duplicate test descriptions:",
      test["normalized_text"].duplicated().sum())

# 5. Check saved vectorizer vocabulary
print("\n5. TF-IDF VOCABULARY CHECK")

bundle = joblib.load(MODEL_PATH)
vectorizer = bundle["vectorizer"]

print("Saved vocabulary size:",
      len(vectorizer.vocabulary_))

# A saved vocabulary alone cannot prove whether
# TF-IDF was fitted using test data.
print("NOTE: Inspect preprocess.py and training scripts")
print("to confirm TF-IDF was fitted on training data only.")

# 6. Save overlap examples
if overlap:
    examples = test[
        test["normalized_text"].isin(overlap)
    ].copy()

    output = BASE_DIR / "results"
    output.mkdir(exist_ok=True)

    examples.head(100).to_csv(
        output / "leakage_examples.csv",
        index=False
    )

    print("\nOverlap examples saved to:")
    print(output / "leakage_examples.csv")

print("\n" + "=" * 60)
print("LEAKAGE CHECK COMPLETED")
print("=" * 60)
