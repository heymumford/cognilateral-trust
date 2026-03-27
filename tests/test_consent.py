"""Tests for open consent (D5)."""

from __future__ import annotations

from cognilateral_trust import ConsentProfile, ConsentResult, evaluate_with_consent


class TestOpenConsent:
    def test_default_profile_does_not_override(self):
        """Default profile passes through base evaluation."""
        result = evaluate_with_consent(0.7)
        assert result.consent_override is False
        assert result.verdict == result.original_verdict

    def test_min_confidence_escalates(self):
        """User min_confidence causes escalation when confidence is below."""
        profile = ConsentProfile(min_confidence=0.8)
        result = evaluate_with_consent(0.7, profile)
        assert result.verdict == "ESCALATE"
        assert result.consent_override is True
        assert "minimum confidence" in result.override_reason

    def test_min_confidence_passes_when_met(self):
        """Confidence above min_confidence proceeds."""
        profile = ConsentProfile(min_confidence=0.5)
        result = evaluate_with_consent(0.7, profile)
        assert result.verdict == "ACT"
        assert result.consent_override is False

    def test_always_escalate_irreversible(self):
        """User requires escalation for all irreversible actions."""
        profile = ConsentProfile(always_escalate_irreversible=True)
        # Use low confidence (base ACTs) but irreversible
        result = evaluate_with_consent(0.3, profile, is_reversible=False)
        assert result.verdict == "ESCALATE"
        assert result.consent_override is True
        assert "irreversible" in result.override_reason

    def test_always_escalate_external(self):
        """User requires escalation for external actions."""
        profile = ConsentProfile(always_escalate_external=True)
        # Use low confidence (base ACTs) but external
        result = evaluate_with_consent(0.3, profile, touches_external=True)
        assert result.verdict == "ESCALATE"
        assert result.consent_override is True

    def test_require_calibration_without_data(self):
        """User requires calibration but none provided."""
        profile = ConsentProfile(require_calibration=True)
        result = evaluate_with_consent(0.7, profile)
        assert result.verdict == "ESCALATE"
        assert result.consent_override is True
        assert "calibration" in result.override_reason

    def test_require_calibration_with_data(self):
        """Calibration provided satisfies the requirement."""
        profile = ConsentProfile(require_calibration=True)
        result = evaluate_with_consent(0.7, profile, calibration_accuracy=0.8)
        assert result.verdict == "ACT"

    def test_consent_cannot_weaken_base_escalation(self):
        """If base says ESCALATE, consent cannot override to ACT."""
        profile = ConsentProfile(min_confidence=0.0)  # maximally permissive
        result = evaluate_with_consent(0.9, profile, is_reversible=False, touches_external=True)
        # Base evaluation escalates due to sovereignty gate
        assert result.verdict == "ESCALATE"
        assert result.consent_override is False  # not a consent override, base decided

    def test_returns_consent_result_type(self):
        result = evaluate_with_consent(0.5)
        assert isinstance(result, ConsentResult)
        assert isinstance(result.profile, ConsentProfile)

    def test_original_verdict_preserved(self):
        """Original verdict is always recorded even when overridden."""
        profile = ConsentProfile(min_confidence=0.9)
        result = evaluate_with_consent(0.7, profile)
        assert result.original_verdict == "ACT"
        assert result.verdict == "ESCALATE"
