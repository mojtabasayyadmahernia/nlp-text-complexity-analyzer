"""
features.py
-----------
SFL-informed feature extraction for text complexity analysis.

This module extracts linguistically grounded features from raw text using
Systemic Functional Linguistics (SFL) theory. Unlike surface-level readability
formulas (e.g., Flesch-Kincaid), these features capture grammatical complexity
at the level of clause structure, lexical choice, and grammatical metaphor.

Features extracted:
    1. Lexical Density         — ratio of content words to total words
    2. Nominalization Density  — frequency of nominalized forms (grammatical metaphor)
    3. Clausal Complexity      — ratio of subordinate clauses to total clauses
    4. Mean Length of Clause   — average number of tokens per clause
    5. Passive Voice Ratio     — frequency of passive constructions per sentence

Author: Mojtaba Sayyad Mahernia
"""

import spacy
from typing import Dict

# Load the English spaCy model.
# Run 'python -m spacy download en_core_web_sm' if not already installed.
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise OSError(
        "spaCy model 'en_core_web_sm' not found. "
        "Run: python -m spacy download en_core_web_sm"
    )

# --- Constants ---

# Part-of-speech tags that count as "content words" in SFL.
CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV"}

# Nominal suffixes that are strong indicators of nominalization
# (i.e., grammatical metaphor in SFL terms).
NOMINALIZATION_SUFFIXES = (
    "tion", "sion", "ment", "ness", "ity", "ance", "ence",
    "ing", "al", "ure", "age", "ism", "ist"
)

# Dependency labels that identify subordinate clauses in spaCy's parser.
SUBORDINATE_CLAUSE_DEPS = {"advcl", "relcl", "ccomp", "xcomp", "acl"}


# --- Core Feature Functions ---

def lexical_density(doc: spacy.tokens.Doc) -> float:
    """
    Calculate the lexical density of a text.

    Lexical density is the proportion of content words (nouns, main verbs,
    adjectives, adverbs) to the total number of tokens. In SFL, higher lexical
    density is associated with written, formal, and complex registers.

    Args:
        doc: A spaCy Doc object.

    Returns:
        A float between 0 and 1. Higher values indicate greater complexity.
    """
    tokens = [t for t in doc if not t.is_punct and not t.is_space]
    if not tokens:
        return 0.0
    content_words = [t for t in tokens if t.pos_ in CONTENT_POS]
    return round(len(content_words) / len(tokens), 4)


def nominalization_density(doc: spacy.tokens.Doc) -> float:
    """
    Calculate the nominalization density of a text.

    Nominalization (e.g., "analysis" from "analyze", "development" from
    "develop") is a key marker of grammatical metaphor in SFL. It is a
    primary indicator of academic and bureaucratic registers.

    Args:
        doc: A spaCy Doc object.

    Returns:
        A float representing nominalizations per 100 tokens.
    """
    tokens = [t for t in doc if not t.is_punct and not t.is_space]
    if not tokens:
        return 0.0
    nominalizations = [
        t for t in tokens
        if t.pos_ == "NOUN" and t.text.lower().endswith(NOMINALIZATION_SUFFIXES)
    ]
    return round((len(nominalizations) / len(tokens)) * 100, 4)


def clausal_complexity(doc: spacy.tokens.Doc) -> float:
    """
    Calculate the clausal complexity of a text.

    This measures the ratio of subordinate clauses to total sentences.
    A higher ratio indicates more deeply embedded grammatical structures,
    which is a marker of syntactic complexity in SFL.

    Args:
        doc: A spaCy Doc object.

    Returns:
        A float representing the average number of subordinate clauses
        per sentence. Higher values indicate greater complexity.
    """
    sentences = list(doc.sents)
    if not sentences:
        return 0.0
    subordinate_clauses = [
        t for t in doc if t.dep_ in SUBORDINATE_CLAUSE_DEPS
    ]
    return round(len(subordinate_clauses) / len(sentences), 4)


def mean_length_of_clause(doc: spacy.tokens.Doc) -> float:
    """
    Calculate the Mean Length of Clause (MLC).

    MLC is a more granular measure of syntactic complexity than mean
    sentence length. It is calculated as the total number of tokens
    divided by the total number of clausal units (sentences + subordinate
    clauses).

    Args:
        doc: A spaCy Doc object.

    Returns:
        A float representing the average number of tokens per clause.
    """
    tokens = [t for t in doc if not t.is_punct and not t.is_space]
    sentences = list(doc.sents)
    subordinate_clauses = [
        t for t in doc if t.dep_ in SUBORDINATE_CLAUSE_DEPS
    ]
    total_clauses = len(sentences) + len(subordinate_clauses)
    if total_clauses == 0:
        return 0.0
    return round(len(tokens) / total_clauses, 4)


def passive_voice_ratio(doc: spacy.tokens.Doc) -> float:
    """
    Calculate the passive voice ratio of a text.

    Passive constructions are associated with formal and impersonal registers
    in SFL. This function detects passive voice by identifying auxiliary
    verbs combined with past participial verb forms.

    Args:
        doc: A spaCy Doc object.

    Returns:
        A float representing the proportion of sentences containing at
        least one passive construction. Higher values indicate greater
        formality and complexity.
    """
    sentences = list(doc.sents)
    if not sentences:
        return 0.0

    passive_sentences = 0
    for sent in sentences:
        # A passive construction in spaCy is identified by the dependency
        # label 'nsubjpass' (nominal subject of a passive verb) or by
        # the combination of an auxiliary 'be' verb and a past participle.
        has_passive = any(
            t.dep_ == "nsubjpass" or
            (t.dep_ == "auxpass")
            for t in sent
        )
        if has_passive:
            passive_sentences += 1

    return round(passive_sentences / len(sentences), 4)


# --- Main Extraction Function ---

def extract_all_features(text: str) -> Dict[str, float]:
    """
    Extract all SFL-informed complexity features from a raw text string.

    This is the primary entry point for the feature extraction pipeline.
    It validates the input, processes it with spaCy, and returns a
    dictionary of all five complexity features.

    Args:
        text: A raw string of text to analyze.

    Returns:
        A dictionary with the following keys:
            - lexical_density (float, 0–1)
            - nominalization_density (float, per 100 tokens)
            - clausal_complexity (float, subordinate clauses per sentence)
            - mean_length_of_clause (float, tokens per clause)
            - passive_voice_ratio (float, 0–1)

    Raises:
        ValueError: If the input text is empty or contains fewer than
                    10 tokens (insufficient for meaningful analysis).
    """
    if not text or not text.strip():
        raise ValueError("Input text cannot be empty.")

    doc = nlp(text)

    # Validate that the text is long enough for meaningful analysis.
    meaningful_tokens = [t for t in doc if not t.is_punct and not t.is_space]
    if len(meaningful_tokens) < 10:
        raise ValueError(
            f"Input text is too short ({len(meaningful_tokens)} tokens). "
            "Please provide at least 10 words for a reliable analysis."
        )

    features = {
        "lexical_density": lexical_density(doc),
        "nominalization_density": nominalization_density(doc),
        "clausal_complexity": clausal_complexity(doc),
        "mean_length_of_clause": mean_length_of_clause(doc),
        "passive_voice_ratio": passive_voice_ratio(doc),
    }

    return features


# --- Quick Test (run this file directly to verify it works) ---

if __name__ == "__main__":
    sample_texts = {
        "Simple": (
            "The cat sat on the mat. It was a sunny day. "
            "The children played outside. They were happy."
        ),
        "Academic": (
            "The operationalization of theoretical constructs within the "
            "framework of systemic functional linguistics necessitates a "
            "reconceptualization of the relationship between grammatical "
            "metaphor and ideational meaning. The realization of experiential "
            "meaning through nominalization has been extensively documented "
            "in the literature on academic discourse."
        ),
    }

    print("=" * 60)
    print("SFL Feature Extraction — Test Run")
    print("=" * 60)

    for label, text in sample_texts.items():
        print(f"\n[{label} Text]")
        print(f"Text: {text[:80]}...")
        features = extract_all_features(text)
        for feature_name, value in features.items():
            print(f"  {feature_name:<30} {value}")

    print("\n" + "=" * 60)
    print("All features extracted successfully.")
