
from pathlib import Path
import joblib
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "advanced_model.pkl"

def main():
    print("=" * 55)
    print("BUGSENSE AI - MODEL CONSISTENCY AUDIT")
    print("=" * 55)

    if not MODEL_PATH.exists():
        print("FAIL: Advanced model file not found.")
        return

    bundle = joblib.load(MODEL_PATH)

    print("\nModel file:", MODEL_PATH)
    print("Model name:", bundle.get("model_name"))
    print("Input format:", bundle.get("input_format"))
    print("Pipeline includes vectorizer:",
          bundle.get("pipeline_includes_vectorizer"))

    model = bundle["model"]
    encoder = bundle["encoder"]

    print("\nPipeline steps:")

    if isinstance(model, Pipeline):
        for name, step in model.steps:
            print(f"  {name}: {type(step).__name__}")
    else:
        print("WARNING: Model is not a Pipeline.")

    classifier = (
        model.named_steps.get("classifier")
        if isinstance(model, Pipeline)
        else model
    )

    print("\nClassifier:", type(classifier).__name__)

    if hasattr(classifier, "get_params"):
        params = classifier.get_params()
        for key in ["C", "class_weight", "max_iter"]:
            if key in params:
                print(f"{key}: {params[key]}")

    print("\nSeverity classes:")
    print(list(encoder.classes_))

    print("\nMODEL AUDIT COMPLETED")
    print("Next: Compare these values with Flask configuration.")

if __name__ == "__main__":
    main()
