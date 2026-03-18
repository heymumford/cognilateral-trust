"""Tests for extract_confidence() from the trust wrapper examples.

Validates confidence extraction across multiple input patterns:
plain text, structured labels, percentage expressions, JSON, and
absent signals.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the examples directory importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))

from openai_trust_wrapper import extract_confidence


class TestExplicitConfidenceLabel:
    """Confidence expressed as 'Confidence: <float>'."""

    def test_standard_label(self) -> None:
        assert extract_confidence("Some answer.\nConfidence: 0.85") == pytest.approx(0.85)

    def test_uppercase_label(self) -> None:
        assert extract_confidence("Confidence: 0.42") == pytest.approx(0.42)

    def test_label_with_no_leading_zero(self) -> None:
        assert extract_confidence("Confidence: .75") == pytest.approx(0.75)

    def test_label_integer_one(self) -> None:
        assert extract_confidence("Confidence: 1.0") == pytest.approx(1.0)

    def test_label_zero(self) -> None:
        assert extract_confidence("Confidence: 0.0") == pytest.approx(0.0)


class TestConfidenceScoreLabel:
    """Confidence expressed as 'confidence_score: <float>'."""

    def test_underscore_variant(self) -> None:
        assert extract_confidence("confidence_score: 0.7") == pytest.approx(0.7)

    def test_mixed_text_around(self) -> None:
        text = "Here is my analysis.\nconfidence_score: 0.65\nThank you."
        assert extract_confidence(text) == pytest.approx(0.65)


class TestPercentagePattern:
    """Confidence expressed as 'N% confident'."""

    def test_eighty_percent(self) -> None:
        assert extract_confidence("I'm about 80% confident in this answer.") == pytest.approx(0.80)

    def test_low_percentage(self) -> None:
        assert extract_confidence("I'm only 15% confident here.") == pytest.approx(0.15)

    def test_hundred_percent(self) -> None:
        assert extract_confidence("I am 100% confident.") == pytest.approx(1.0)


class TestJsonExtraction:
    """Confidence embedded in a JSON object."""

    def test_json_confidence_key(self) -> None:
        assert extract_confidence('{"confidence": 0.9}') == pytest.approx(0.9)

    def test_json_confidence_score_key(self) -> None:
        assert extract_confidence('{"confidence_score": 0.6}') == pytest.approx(0.6)

    def test_json_certainty_key(self) -> None:
        assert extract_confidence('{"certainty": 0.78}') == pytest.approx(0.78)

    def test_json_percentage_value(self) -> None:
        # Values >1.0 are interpreted as percentages and divided by 100
        assert extract_confidence('{"confidence": 92}') == pytest.approx(0.92)


class TestNoConfidenceSignal:
    """When no recognizable confidence pattern is present, return 0.5."""

    def test_plain_text(self) -> None:
        assert extract_confidence("The capital of France is Paris.") == pytest.approx(0.5)

    def test_empty_string(self) -> None:
        assert extract_confidence("") == pytest.approx(0.5)

    def test_unrelated_numbers(self) -> None:
        assert extract_confidence("There are 50 states in the US.") == pytest.approx(0.5)
