"""
test_features.py
----------------
Unit tests for the SFL feature extraction module (src/features.py).

These tests verify that each feature function returns values within
expected ranges and that the overall extractor handles edge cases correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from src.features import (
    extract_all_features,
    lexical_density,
    nominalization_density,
    passive_voice_ratio,
)
import spacy

nlp = spacy.load("en_core_web_sm")

# --- Test Data ---

SIMPLE_TEXT = (
    "The cat sat on the mat. It was a sunny day. "
    "The children played outside. They were very happy."
)

ACADEMIC_TEXT = (
    "The operationalization of theoretical constructs within the framework "
    "of systemic functional linguistics necessitates a reconceptualization "
    "of the relationship between grammatical metaphor and ideational meaning. "
    "The realization of experiential meaning through nominalization has been "
    "extensively documented in the literature on academic discourse."
)

SHORT_TEXT = "Hello world."


# --- Tests for lexical_density ---

class TestLexicalDensity:

    def test_returns_float(self):
        doc = nlp(SIMPLE_TEXT)
        result = lexical_density(doc)
        assert isinstance(result, float)

    def test_value_between_zero_and_one(self):
        doc = nlp(SIMPLE_TEXT)
        result = lexical_density(doc)
        assert 0.0 <= result <= 1.0

    def test_academic_text_higher_than_simple(self):
        """Academic text should have higher lexical density than simple text."""
        simple_ld = lexical_density(nlp(SIMPLE_TEXT))
        academic_ld = lexical_density(nlp(ACADEMIC_TEXT))
        assert academic_ld > simple_ld

    def test_empty_doc_returns_zero(self):
        doc = nlp("")
        result = lexical_density(doc)
        assert result == 0.0


# --- Tests for nominalization_density ---

class TestNominalizationDensity:

    def test_returns_float(self):
        doc = nlp(SIMPLE_TEXT)
        result = nominalization_density(doc)
        assert isinstance(result, float)

    def test_academic_text_higher_than_simple(self):
        """Academic text should have far more nominalizations than simple text."""
        simple_nd = nominalization_density(nlp(SIMPLE_TEXT))
        academic_nd = nominalization_density(nlp(ACADEMIC_TEXT))
        assert academic_nd > simple_nd

    def test_simple_text_low_nominalization(self):
        """Simple conversational text should have low nominalization density."""
        doc = nlp(SIMPLE_TEXT)
        result = nominalization_density(doc)
        assert result < 5.0  # less than 5 per 100 tokens


# --- Tests for passive_voice_ratio ---

class TestPassiveVoiceRatio:

    def test_returns_float(self):
        doc = nlp(SIMPLE_TEXT)
        result = passive_voice_ratio(doc)
        assert isinstance(result, float)

    def test_value_between_zero_and_one(self):
        doc = nlp(ACADEMIC_TEXT)
        result = passive_voice_ratio(doc)
        assert 0.0 <= result <= 1.0

    def test_passive_sentence_detected(self):
        """A sentence with clear passive voice should be detected."""
        passive_text = (
            "The experiment was conducted by the researchers. "
            "The results were analyzed carefully. "
            "The findings were published in a peer-reviewed journal."
        )
        doc = nlp(passive_text)
        result = passive_voice_ratio(doc)
        assert result > 0.0


# --- Tests for extract_all_features ---

class TestExtractAllFeatures:

    def test_returns_dict(self):
        result = extract_all_features(SIMPLE_TEXT)
        assert isinstance(result, dict)

    def test_returns_all_five_features(self):
        result = extract_all_features(SIMPLE_TEXT)
        expected_keys = {
            "lexical_density",
            "nominalization_density",
            "clausal_complexity",
            "mean_length_of_clause",
            "passive_voice_ratio",
        }
        assert set(result.keys()) == expected_keys

    def test_all_values_are_numeric(self):
        result = extract_all_features(SIMPLE_TEXT)
        for key, value in result.items():
            assert isinstance(value, float), f"{key} is not a float: {type(value)}"

    def test_raises_on_empty_string(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            extract_all_features("")

    def test_raises_on_short_text(self):
        with pytest.raises(ValueError, match="too short"):
            extract_all_features(SHORT_TEXT)

    def test_academic_text_higher_complexity_than_simple(self):
        """Academic text should score higher on complexity features."""
        simple = extract_all_features(SIMPLE_TEXT)
        academic = extract_all_features(ACADEMIC_TEXT)
        assert academic["nominalization_density"] > simple["nominalization_density"]
        assert academic["mean_length_of_clause"] > simple["mean_length_of_clause"]