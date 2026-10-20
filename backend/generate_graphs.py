
from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Actual measured values from your SVM evaluation
metrics = {
    "Accuracy": 0.7636291285316354,
    "Macro Precision": 0.3716880590243616,
    "Macro Recall": 0.4051313505532002,
    "Macro F1": 0.3864118664341818,
    "Weighted F1": 0.7700275210456166,
}

severity_f1 = {
    "Blocker": 0.31,
    "Critical": 0.66,
    "Major": 0.17,
    "Minor": 0.12,
    "Normal": 0.87,
    "Trivial": 0.18,
}

# Graph 1: Overall evaluation metrics
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(metrics.keys(), metrics.values())
ax.set_title("BugSense AI - SVM Evaluation Metrics")
ax.set_ylabel("Score")
ax.set_ylim(0, 1)
ax.tick_params(axis="x", rotation=25)
ax.grid(axis="y", alpha=0.2)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "svm_metrics.png", dpi=300)
plt.close(fig)

# Graph 2: Severity-wise F1-score
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(severity_f1.keys(), severity_f1.values())
ax.set_title("SVM Performance Across Bug Severity Classes")
ax.set_ylabel("F1-score")
ax.set_ylim(0, 1)
ax.grid(axis="y", alpha=0.2)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "severity_f1.png", dpi=300)
plt.close(fig)

# Graph 3: Test dataset severity distribution
severity_counts = {
    "Blocker": 159,
    "Critical": 1166,
    "Major": 907,
    "Minor": 623,
    "Normal": 14324,
    "Trivial": 412,
}

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(severity_counts.keys(), severity_counts.values())
ax.set_title("Distribution of Bug Severity Labels in Test Data")
ax.set_ylabel("Number of Bug Reports")
ax.grid(axis="y", alpha=0.2)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "severity_distribution.png", dpi=300)
plt.close(fig)

print("Evaluation graphs generated successfully!")
print("Saved to:", RESULTS_DIR)
