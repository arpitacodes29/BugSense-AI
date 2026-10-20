
import pandas as pd
from pathlib import Path

path = Path(__file__).resolve().parent.parent / "dataset" / "bugs.csv"

df = pd.read_csv(path)

print("Dataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 records:")
print(df.head().to_string(index=False))
