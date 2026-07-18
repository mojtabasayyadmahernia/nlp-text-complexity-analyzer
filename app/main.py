"""
main.py
-------
FastAPI application for the SFL Text Complexity Analyzer.

This API accepts a raw text string and returns a complexity classification
(Elementary, Intermediate, or Advanced) along with the five SFL feature
scores that drove the prediction.

Endpoints:
    GET  /           — Health check
    POST /analyze    — Analyze text complexity
"""

import os
import sys
import json
import joblib
import boto3
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Add the project root to the path so we can import from src/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features import extract_all_features

# --- App Initialization ---

app = FastAPI(
    title="SFL Text Complexity Analyzer",
    description=(
        "An NLP API that classifies text complexity using features grounded "
        "in Systemic Functional Linguistics (SFL) theory. "
        "Developed by Mojtaba Sayyad Mahernia."
    ),
    version="1.0.0",
)

# --- Configuration ---

# S3 settings — these match the bucket you created on Day 2
S3_BUCKET = "mojtaba-nlp-models-stockholm"
S3_MODEL_KEY = "models/complexity_model.pkl"
S3_FEATURES_KEY = "models/feature_cols.json"

# Local paths where the model will be cached after downloading from S3
LOCAL_MODEL_PATH = "models/complexity_model.pkl"
LOCAL_FEATURES_PATH = "models/feature_cols.json"

# Label mapping
LABEL_MAP = {0: "Elementary", 1: "Intermediate", 2: "Advanced"}

# --- Model Loading ---

def load_model_from_s3():
    """
    Download the model and feature column list from S3 if not already cached locally.
    This function runs once when the server starts.
    """
    os.makedirs("models", exist_ok=True)

    # Only download if the file doesn't already exist locally
    if not os.path.exists(LOCAL_MODEL_PATH):
        print(f"Downloading model from S3: s3://{S3_BUCKET}/{S3_MODEL_KEY}")
        s3 = boto3.client("s3", region_name="eu-north-1")
        s3.download_file(S3_BUCKET, S3_MODEL_KEY, LOCAL_MODEL_PATH)
        print("Model downloaded successfully.")
    else:
        print("Model already cached locally. Skipping S3 download.")

    if not os.path.exists(LOCAL_FEATURES_PATH):
        print(f"Downloading feature columns from S3: s3://{S3_BUCKET}/{S3_FEATURES_KEY}")
        s3 = boto3.client("s3", region_name="eu-north-1")
        s3.download_file(S3_BUCKET, S3_FEATURES_KEY, LOCAL_FEATURES_PATH)

    model = joblib.load(LOCAL_MODEL_PATH)
    with open(LOCAL_FEATURES_PATH, "r") as f:
        feature_cols = json.load(f)

    return model, feature_cols


# Load the model when the server starts
print("Loading model...")
model, feature_cols = load_model_from_s3()
print("Model ready.")


# --- Request and Response Schemas ---

class TextInput(BaseModel):
    """Schema for the incoming request body."""
    text: str = Field(
        ...,
        min_length=50,
        description="The text to analyze. Must be at least 50 characters long.",
        example=(
            "The operationalization of theoretical constructs within the "
            "framework of systemic functional linguistics necessitates a "
            "reconceptualization of the relationship between grammatical "
            "metaphor and ideational meaning."
        )
    )


class ComplexityResult(BaseModel):
    """Schema for the outgoing response body."""
    predicted_level: str
    predicted_label: int
    confidence: float
    sfl_features: dict
    interpretation: str


# --- API Endpoints ---

@app.get("/", summary="Health Check")
def root():
    """
    Health check endpoint. Returns a simple status message to confirm
    the API is running correctly.
    """
    return {
        "status": "ok",
        "message": "SFL Text Complexity Analyzer is running.",
        "version": "1.0.0"
    }


@app.post("/analyze", response_model=ComplexityResult, summary="Analyze Text Complexity")
def analyze_text(input_data: TextInput):
    """
    Analyze the complexity of a given text using SFL-informed features.

    The text is processed through five linguistic feature extractors grounded
    in Systemic Functional Linguistics (SFL) theory, then classified by a
    machine learning model trained on the OneStopEnglish corpus.

    Returns the predicted complexity level (Elementary, Intermediate, or
    Advanced), the confidence score, and the individual SFL feature values.
    """
    try:
        # Step 1: Extract SFL features from the input text
        features = extract_all_features(input_data.text)
    except ValueError as e:
        # Return a clear error if the text is too short
        raise HTTPException(status_code=422, detail=str(e))

    # Step 2: Build the feature vector in the correct column order
    feature_vector = pd.DataFrame([features])[feature_cols]

    # Step 3: Get the model prediction and confidence score
    predicted_label = int(model.predict(feature_vector)[0])
    probabilities = model.predict_proba(feature_vector)[0]
    confidence = round(float(probabilities[predicted_label]), 4)
    predicted_level = LABEL_MAP[predicted_label]

    # Step 4: Build a human-readable interpretation
    interpretation = _build_interpretation(predicted_level, features)

    return ComplexityResult(
        predicted_level=predicted_level,
        predicted_label=predicted_label,
        confidence=confidence,
        sfl_features=features,
        interpretation=interpretation
    )


def _build_interpretation(level: str, features: dict) -> str:
    """
    Generate a plain-English explanation of why the model made its prediction.
    This is what makes the API genuinely useful rather than just a black box.
    """
    nom = features.get("nominalization_density", 0)
    ld = features.get("lexical_density", 0)
    mlc = features.get("mean_length_of_clause", 0)
    passive = features.get("passive_voice_ratio", 0)

    parts = [f"This text was classified as {level}."]

    if nom > 10:
        parts.append(
            f"It has a high nominalization density ({nom:.1f} per 100 tokens), "
            "indicating heavy use of grammatical metaphor typical of academic writing."
        )
    elif nom > 5:
        parts.append(
            f"It has a moderate nominalization density ({nom:.1f} per 100 tokens)."
        )
    else:
        parts.append(
            f"It has a low nominalization density ({nom:.1f} per 100 tokens), "
            "suggesting a more conversational or simple register."
        )

    if mlc > 15:
        parts.append(
            f"The mean clause length of {mlc:.1f} tokens indicates dense, "
            "information-packed clauses."
        )

    if passive > 0.3:
        parts.append(
            f"A passive voice ratio of {passive:.0%} suggests a formal, "
            "impersonal register."
        )

    return " ".join(parts)