
from pathlib import Path
import logging
import re

import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

from log_analyzer import analyze_error_log
from code_analyzer import analyze_source_code


# ==========================================
# FLASK APPLICATION
# ==========================================

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==========================================
# PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

ADVANCED_MODEL_PATH = MODEL_DIR / "advanced_model.pkl"


# ==========================================
# LOAD ADVANCED MODEL
# ==========================================

if not ADVANCED_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Advanced model not found: {ADVANCED_MODEL_PATH}. "
        "Run backend/train_advanced_model.py first."
    )

bundle = joblib.load(ADVANCED_MODEL_PATH)

model = bundle["model"]
encoder = bundle["encoder"]
model_name = bundle.get("model_name", "Advanced SVM")

logger.info("BugSense AI backend initialized")
logger.info("Loaded model: %s", model_name)


# ==========================================
# TEXT PREPROCESSING
# ==========================================

def clean_text(text):
    """
    Apply the same text preprocessing used
    during model training.
    """

    text = str(text).lower()

    text = re.sub(
        r"https?://\S+",
        " ",
        text
    )

    text = re.sub(
        r"[^a-z0-9_./+-]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ==========================================
# BUG REPORT PREDICTION
# ==========================================

def analyze_bug_report(text):
    """
    Predict software bug severity using
    the trained Advanced SVM pipeline.
    """

    cleaned = clean_text(text)

    if not cleaned:
        raise ValueError(
            "No usable text found after preprocessing."
        )

    # The advanced model already contains
    # both TF-IDF feature extractors.
    prediction = model.predict([cleaned])[0]

    severity = encoder.inverse_transform(
        [prediction]
    )[0]

    return {
        "status": "success",
        "input_type": "bug_report",
        "model": model_name,
        "severity": str(severity),
        "category": None,
        "priority": None,
        "analysis_method": (
            "Word + Character TF-IDF with Linear SVM"
        ),
        "explanation": (
            "Bug severity predicted using a "
            "machine learning model trained "
            "on historical software bug reports."
        ),
        "message": "Bug severity prediction completed."
    }


# ==========================================
# HOME ROUTE
# ==========================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "project": "BugSense AI",
        "status": "running",
        "model": model_name,
        "version": "2.0",
        "message": "BugSense AI backend is working!",
        "features": [
            "Bug severity prediction",
            "Source code static analysis",
            "Error log analysis"
        ]
    })


# ==========================================
# HEALTH CHECK
# ==========================================

@app.route("/api/health", methods=["GET"])
def health():

    return jsonify({
        "status": "healthy",
        "model_loaded": True,
        "model": model_name,
        "supported_input_types": [
            "bug_report",
            "source_code",
            "error_log"
        ],
        "supported_languages": [
            "python",
            "javascript"
        ]
    })


# ==========================================
# MAIN ANALYSIS ENDPOINT
# ==========================================

@app.route("/api/analyze", methods=["POST"])
def analyze():

    data = request.get_json(silent=True)

    # Validate JSON
    if not isinstance(data, dict):
        return jsonify({
            "status": "error",
            "error": "A valid JSON object is required."
        }), 400

    text = data.get("text")
    input_type = data.get(
        "input_type",
        "bug_report"
    )

    # Validate input text
    if not isinstance(text, str) or not text.strip():
        return jsonify({
            "status": "error",
            "error": "Please provide valid input text."
        }), 400

    # Limit input size
    if len(text) > 20000:
        return jsonify({
            "status": "error",
            "error": "Input exceeds 20,000 characters."
        }), 400

    # Validate input type
    supported_types = {
        "bug_report",
        "source_code",
        "error_log"
    }

    if input_type not in supported_types:
        return jsonify({
            "status": "error",
            "error": "Unsupported input type."
        }), 400

    try:

        # ----------------------------------
        # BUG REPORT ANALYSIS
        # ----------------------------------

        if input_type == "bug_report":

            result = analyze_bug_report(text)

            return jsonify(result), 200

        # ----------------------------------
        # SOURCE CODE ANALYSIS
        # ----------------------------------

        if input_type == "source_code":

            language = data.get(
                "language",
                "python"
            )

            if not isinstance(language, str):
                return jsonify({
                    "status": "error",
                    "error": "Language must be a string."
                }), 400

            language = language.strip().lower()

            if language not in {
                "python",
                "javascript"
            }:
                return jsonify({
                    "status": "error",
                    "error": (
                        "Only Python and JavaScript "
                        "are supported."
                    )
                }), 400

            result = analyze_source_code(
                text,
                language
            )

            return jsonify(result), 200

        # ----------------------------------
        # ERROR LOG ANALYSIS
        # ----------------------------------

        if input_type == "error_log":

            result = analyze_error_log(text)

            return jsonify(result), 200

    except ValueError as error:

        return jsonify({
            "status": "error",
            "error": str(error)
        }), 400

    except Exception:

        logger.exception("Analysis failed")

        return jsonify({
            "status": "error",
            "error": (
                "An internal analysis error occurred."
            )
        }), 500


# ==========================================
# ERROR HANDLERS
# ==========================================

@app.errorhandler(404)
def not_found(error):

    return jsonify({
        "status": "error",
        "error": "Endpoint not found."
    }), 404


@app.errorhandler(405)
def method_not_allowed(error):

    return jsonify({
        "status": "error",
        "error": "HTTP method not allowed."
    }), 405


# ==========================================
# START SERVER
# ==========================================

if __name__ == "__main__":

    print("\n================================")
    print("BUGSENSE AI - BACKEND SERVER")
    print("================================")
    print("Model:", model_name)
    print("Bug reports: Advanced SVM")
    print("Source code: Static analysis")
    print("Error logs: Rule-based analysis")
    print("================================\n")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
