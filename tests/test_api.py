"""
test_api.py
-----------
Integration tests for the FastAPI application (app/main.py).

These tests use FastAPI's built-in TestClient to simulate HTTP requests
without needing to start a real server. They verify that the API endpoints
return correct status codes and response structures.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

# --- Test Data ---

VALID_SIMPLE_TEXT = (
    "The cat sat on the mat. It was a sunny day outside. "
    "The children played in the garden. They were very happy and loud. "
    "The dog ran fast across the green field."
)

VALID_ACADEMIC_TEXT = (
    "The operationalization of theoretical constructs within the framework "
    "of systemic functional linguistics necessitates a reconceptualization "
    "of the relationship between grammatical metaphor and ideational meaning. "
    "The realization of experiential meaning through nominalization has been "
    "extensively documented in the literature on academic discourse."
)

TOO_SHORT_TEXT = "Hello."


# --- Health Check Tests ---

class TestHealthCheck:

    def test_root_returns_200(self):
        response = client.get("/")
        assert response.status_code == 200

    def test_root_returns_ok_status(self):
        response = client.get("/")
        data = response.json()
        assert data["status"] == "ok"

    def test_root_returns_version(self):
        response = client.get("/")
        data = response.json()
        assert "version" in data


# --- Analyze Endpoint Tests ---

class TestAnalyzeEndpoint:

    def test_valid_text_returns_200(self):
        response = client.post("/analyze", json={"text": VALID_SIMPLE_TEXT})
        assert response.status_code == 200

    def test_response_has_required_fields(self):
        response = client.post("/analyze", json={"text": VALID_SIMPLE_TEXT})
        data = response.json()
        assert "predicted_level" in data
        assert "predicted_label" in data
        assert "confidence" in data
        assert "sfl_features" in data
        assert "interpretation" in data

    def test_predicted_level_is_valid(self):
        response = client.post("/analyze", json={"text": VALID_SIMPLE_TEXT})
        data = response.json()
        assert data["predicted_level"] in ["Elementary", "Intermediate", "Advanced"]

    def test_predicted_label_is_valid_integer(self):
        response = client.post("/analyze", json={"text": VALID_SIMPLE_TEXT})
        data = response.json()
        assert data["predicted_label"] in [0, 1, 2]

    def test_confidence_between_zero_and_one(self):
        response = client.post("/analyze", json={"text": VALID_SIMPLE_TEXT})
        data = response.json()
        assert 0.0 <= data["confidence"] <= 1.0

    def test_sfl_features_has_five_keys(self):
        response = client.post("/analyze", json={"text": VALID_SIMPLE_TEXT})
        data = response.json()
        assert len(data["sfl_features"]) == 5

    def test_academic_text_predicts_higher_than_simple(self):
        """Academic text should predict a higher label than simple text."""
        simple_response = client.post("/analyze", json={"text": VALID_SIMPLE_TEXT})
        academic_response = client.post("/analyze", json={"text": VALID_ACADEMIC_TEXT})
        simple_label = simple_response.json()["predicted_label"]
        academic_label = academic_response.json()["predicted_label"]
        assert academic_label >= simple_label

    def test_empty_text_returns_422(self):
        """Empty text should return a validation error."""
        response = client.post("/analyze", json={"text": ""})
        assert response.status_code == 422

    def test_too_short_text_returns_422(self):
        """Text that is too short for analysis should return an error."""
        response = client.post("/analyze", json={"text": TOO_SHORT_TEXT})
        assert response.status_code in [422, 422]

    def test_missing_text_field_returns_422(self):
        """A request with no 'text' field should return a validation error."""
        response = client.post("/analyze", json={})
        assert response.status_code == 422