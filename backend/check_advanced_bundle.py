
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "advanced_model.pkl"

bundle = joblib.load(MODEL_PATH)

print("\nBUGSENSE AI - ADVANCED MODEL STRUCTURE")
print("=" * 50)

print("Bundle type:", type(bundle))

if isinstance(bundle, dict):
    print("\nAvailable keys:")
    for key, value in bundle.items():
        print(f"{key}: {type(value).__name__}")
else:
    print("\nSaved object:", type(bundle).__name__)

print("\nInspection completed.")
