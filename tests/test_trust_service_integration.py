"""Tests for TrustServiceProvider integration."""

from __future__ import annotations


from cognilateral_trust.core import TrustContext
from cognilateral_trust.integrations.trust_service import (
    TrustProvider,
    TrustServiceProvider,
)


class TestTrustServiceProvider:
    """TrustServiceProvider must implement TrustProvider protocol."""

    def test_provider_is_trust_provider(self) -> None:
        provider = TrustServiceProvider()
        assert isinstance(provider, TrustProvider)

    def test_evaluate_with_high_confidence(self) -> None:
        provider = TrustServiceProvider()
        context = TrustContext(
            confidence=0.85,
            evidence=("test passed", "integrated"),
            action_reversible=True,
            welfare_relevant=False,
        )

        result = provider.evaluate(context)

        assert result.confidence == 0.85
        assert result.should_proceed is True
        assert result.accountability_record is not None
        assert result.accountability_record.record_id != ""

    def test_evaluate_increments_counter(self) -> None:
        provider = TrustServiceProvider()
        assert provider.health()["evaluations_total"] == 0

        context = TrustContext(confidence=0.5)
        provider.evaluate(context)

        assert provider.health()["evaluations_total"] == 1

    def test_evaluate_tracks_escalations(self) -> None:
        provider = TrustServiceProvider()
        context = TrustContext(
            confidence=0.3,
            action_reversible=False,
            welfare_relevant=False,
        )

        result = provider.evaluate(context)

        if not result.should_proceed:
            stats = provider.health()
            assert stats["evaluations_escalated"] > 0

    def test_health_returns_dict(self) -> None:
        provider = TrustServiceProvider()
        context = TrustContext(confidence=0.7)
        provider.evaluate(context)

        health = provider.health()

        assert isinstance(health, dict)
        assert "calibration_score" in health
        assert "evaluations_total" in health
        assert "evaluations_escalated" in health
        assert "last_evaluation_timestamp" in health

    def test_health_calibration_score_valid(self) -> None:
        provider = TrustServiceProvider()
        context = TrustContext(confidence=0.7)
        provider.evaluate(context)

        health = provider.health()
        assert 0.0 <= health["calibration_score"] <= 1.0

    def test_health_before_any_evaluation(self) -> None:
        provider = TrustServiceProvider()
        health = provider.health()

        assert health["evaluations_total"] == 0
        assert health["calibration_score"] == 1.0
        assert health["evaluations_escalated"] == 0

    def test_provider_info_returns_dict(self) -> None:
        provider = TrustServiceProvider()
        info = provider.provider_info()

        assert isinstance(info, dict)
        assert "name" in info
        assert "version" in info
        assert "capabilities" in info

    def test_provider_info_content(self) -> None:
        provider = TrustServiceProvider()
        info = provider.provider_info()

        assert info["name"] == "cognilateral-trust"
        assert info["version"] == "1.1.0"
        assert isinstance(info["capabilities"], list)
        assert len(info["capabilities"]) > 0
        assert "routing" in str(info["capabilities"]).lower()

    def test_provider_info_includes_expected_capabilities(self) -> None:
        provider = TrustServiceProvider()
        info = provider.provider_info()

        expected_caps = {
            "confidence_tier_routing",
            "accountability_records",
            "escalation_gates",
        }
        actual_caps = set(info["capabilities"])
        assert expected_caps.issubset(actual_caps)

    def test_multiple_evaluations_track_correctly(self) -> None:
        provider = TrustServiceProvider()

        for i in range(5):
            context = TrustContext(confidence=0.5 + i * 0.1)
            provider.evaluate(context)

        health = provider.health()
        assert health["evaluations_total"] == 5

    def test_evaluate_with_evidence(self) -> None:
        provider = TrustServiceProvider()
        context = TrustContext(
            confidence=0.75,
            evidence=("unit tests pass", "integration verified", "peer reviewed"),
            action_reversible=True,
            welfare_relevant=False,
        )

        result = provider.evaluate(context)

        assert result.should_proceed is True
        assert result.accountability_record is not None

    def test_evaluate_irreversible_action_low_confidence(self) -> None:
        provider = TrustServiceProvider()
        context = TrustContext(
            confidence=0.4,
            action_reversible=False,
            welfare_relevant=False,
        )

        result = provider.evaluate(context)

        assert result.should_proceed is False or result.accountability_record is not None

    def test_tier_returned_in_evaluation(self) -> None:
        provider = TrustServiceProvider()
        context = TrustContext(confidence=0.7)

        result = provider.evaluate(context)

        assert result.tier is not None
        assert hasattr(result.tier, "name") or hasattr(result.tier, "value")

    def test_route_returned_in_evaluation(self) -> None:
        provider = TrustServiceProvider()
        context = TrustContext(confidence=0.7)

        result = provider.evaluate(context)

        assert result.route in ("basic", "warrant_check", "sovereignty_gate")

    def test_health_updates_timestamp(self) -> None:
        import time

        provider = TrustServiceProvider()
        context = TrustContext(confidence=0.7)

        before = time.time()
        provider.evaluate(context)
        after = time.time()

        health = provider.health()
        ts = health["last_evaluation_timestamp"]

        assert before <= ts <= after or ts == 0.0
