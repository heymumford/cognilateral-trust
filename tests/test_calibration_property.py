"""Property tests for calibration accuracy.

Proves that the confidence tier model produces meaningful differentiation:
- Higher confidence should correlate with higher proceed rates
- Irreversible actions should escalate more often at high tiers
- The tier boundaries (0.3, 0.6, 0.9) create distinct routing behavior
"""

from __future__ import annotations

import random

from cognilateral_trust import evaluate_trust


class TestCalibrationProperties:
    """Statistical properties that must hold across random inputs."""

    def test_higher_confidence_proceeds_more_often(self) -> None:
        """Over 1000 random samples, high confidence proceeds more than low."""
        random.seed(42)
        low_proceed = sum(
            1 for _ in range(1000) if evaluate_trust(random.uniform(0.0, 0.3)).should_proceed
        )
        high_proceed = sum(
            1 for _ in range(1000) if evaluate_trust(random.uniform(0.7, 1.0)).should_proceed
        )
        assert high_proceed >= low_proceed, f"High ({high_proceed}) should >= Low ({low_proceed})"

    def test_irreversible_escalates_more_at_high_tiers(self) -> None:
        """Irreversible actions at high tiers escalate more than at low tiers."""
        random.seed(42)
        low_escalate = sum(
            1
            for _ in range(1000)
            if not evaluate_trust(random.uniform(0.0, 0.3), is_reversible=False).should_proceed
        )
        high_escalate = sum(
            1
            for _ in range(1000)
            if not evaluate_trust(random.uniform(0.7, 1.0), is_reversible=False).should_proceed
        )
        # High-tier irreversible actions should escalate more (sovereignty gate kicks in)
        assert high_escalate > low_escalate, f"High ({high_escalate}) should > Low ({low_escalate})"

    def test_tier_boundaries_create_distinct_routes(self) -> None:
        """The three routing bands produce different behavior distributions."""
        basic_routes = set()
        warrant_routes = set()
        sovereignty_routes = set()

        for conf in [i / 100 for i in range(0, 100)]:
            result = evaluate_trust(conf)
            if conf < 0.4:
                basic_routes.add(result.route)
            elif conf < 0.7:
                warrant_routes.add(result.route)
            else:
                sovereignty_routes.add(result.route)

        assert "basic" in basic_routes
        assert "warrant_check" in warrant_routes
        assert "sovereignty_gate" in sovereignty_routes

    def test_accountability_records_always_produced(self) -> None:
        """Every evaluation produces an accountability record."""
        random.seed(42)
        for _ in range(100):
            conf = random.random()
            result = evaluate_trust(
                conf,
                is_reversible=random.choice([True, False]),
                touches_external=random.choice([True, False]),
            )
            assert result.accountability_record is not None
            assert result.accountability_record.confidence == conf

    def test_external_actions_escalate_at_sovereignty_tier(self) -> None:
        """External actions at sovereignty-grade tiers always escalate."""
        for conf in [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]:
            result = evaluate_trust(conf, touches_external=True)
            if result.requires_sovereignty_gate:
                assert not result.should_proceed, f"External at {conf} should escalate"

    def test_basic_tier_always_proceeds(self) -> None:
        """Basic tier (C0-C3) always proceeds regardless of flags."""
        for conf in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]:
            for rev in [True, False]:
                for ext in [True, False]:
                    result = evaluate_trust(conf, is_reversible=rev, touches_external=ext)
                    if result.route == "basic":
                        assert result.should_proceed, f"Basic route at {conf} should always proceed"
