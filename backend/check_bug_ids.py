
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

train = pd.read_csv(BASE_DIR / "dataset/train.csv")
test = pd.read_csv(BASE_DIR / "dataset/test.csv")

# Compare IDs with their project names
train["bug_key"] = (
    train["Project"].astype(str).str.strip()
    + "::"
    + train["bug_id"].astype(str).str.strip()
)

test["bug_key"] = (
    test["Project"].astype(str).str.strip()
    + "::"
    + test["bug_id"].astype(str).str.strip()
)

shared = set(train["bug_key"]) & set(test["bug_key"])

print("Shared Bug IDs within the same project:", len(shared))

if shared:
    print("\nExamples of overlapping bugs:")

    examples = pd.concat([
        train[train["bug_key"].isin(shared)].assign(split="train"),
        test[test["bug_key"].isin(shared)].assign(split="test")
    ])

    print(
        examples[
            ["split", "Project", "bug_id", "severity", "description"]
        ].head(20).to_string(index=False)
    )

    output = BASE_DIR / "results"
    output.mkdir(exist_ok=True)

    examples.to_csv(
        output / "shared_bug_ids.csv",
        index=False
    )

    print("\nSaved to results/shared_bug_ids.csv")
else:
    print("No project-specific Bug ID overlap detected.")
