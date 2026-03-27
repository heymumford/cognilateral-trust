"""Tests for nutrition label (D6)."""

from __future__ import annotations

from cognilateral_trust import NutritionLabel, not_evaluated_label, nutrition_label


class TestNutritionLabel:
    def test_generates_disclosure_string(self):
        label = nutrition_label(0.7)
        assert "Trust evaluated" in label.disclosure
        assert "0.70" in label.disclosure
        assert "ACT" in label.disclosure or "ESCALATE" in label.disclosure

    def test_includes_calibration_when_provided(self):
        label = nutrition_label(0.7, calibration_accuracy=0.625)
        assert "62.5%" in label.disclosure

    def test_omits_calibration_when_none(self):
        label = nutrition_label(0.7)
        assert "Calibration" not in label.disclosure

    def test_str_returns_disclosure(self):
        label = nutrition_label(0.5)
        assert str(label) == label.disclosure

    def test_frozen_dataclass(self):
        label = nutrition_label(0.7)
        assert isinstance(label, NutritionLabel)
        assert label.evaluated is True

    def test_high_confidence_irreversible_escalates(self):
        label = nutrition_label(0.9, is_reversible=False, touches_external=True)
        assert label.verdict == "ESCALATE"

    def test_low_confidence_acts(self):
        label = nutrition_label(0.2)
        assert label.verdict == "ACT"

    def test_not_evaluated_label(self):
        label = not_evaluated_label()
        assert label.evaluated is False
        assert label.verdict == "NOT_EVALUATED"
        assert "Not trust-evaluated" in label.disclosure

    def test_confidence_in_range(self):
        for c in [0.0, 0.1, 0.5, 0.9, 1.0]:
            label = nutrition_label(c)
            assert label.confidence == c
