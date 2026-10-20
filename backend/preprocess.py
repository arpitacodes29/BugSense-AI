
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dataset"

INPUT_FILE = DATA_DIR / "bugs.csv"
OUTPUT_FILE = DATA_DIR / "processed_bugs.csv"


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^a-z0-9_./+-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print("Available columns:")
    print(df.columns.tolist())

    # Dataset-specific column names
    df = df.rename(columns={
        "Short Description": "description",
        "Severity Label": "severity",
        "Bug ID": "bug_id"
    })

    required = ["description", "severity"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    print("Original records:", len(df))

    df = df.dropna(subset=required).copy()

    df["cleaned_text"] = df["description"].apply(clean_text)
    df["severity"] = (
        df["severity"].astype(str).str.strip().str.lower()
    )

    df = df[df["cleaned_text"].str.len() > 5]

    # Remove repeated descriptions to reduce leakage
    df = df.drop_duplicates(subset=["cleaned_text"])

    print("\nSeverity distribution:")
    print(df["severity"].value_counts())

    if df["severity"].nunique() < 2:
        raise ValueError("At least two classes are required.")

    counts = df["severity"].value_counts()

    if counts.min() < 2:
        raise ValueError(
            "Each severity class needs at least 2 records."
        )

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["severity"]
    )

    train_df.to_csv(DATA_DIR / "train.csv", index=False)
    test_df.to_csv(DATA_DIR / "test.csv", index=False)
    df.to_csv(OUTPUT_FILE, index=False)

    print("\nPreprocessing complete!")
    print("Total records:", len(df))
    print("Training records:", len(train_df))
    print("Testing records:", len(test_df))


if __name__ == "__main__":
    main()
