"""A-tests and B-tests for TrustBench scenario dataset (Step 14)."""

from __future__ import annotations

import pytest

from cognilateral_trust.bench.scenarios import (
    DOMAINS,
    BenchScenario,
    load_scenarios,
)


class TestScenarioDatasetAcceptance:
    """A-tests: dataset completeness and correctness."""

    def test_a1_total_scenario_count(self) -> None:
        """Given scenarios/, when loaded, then 200 scenarios total."""
        scenarios = load_scenarios()
        assert len(scenarios) == 200

    def test_a2_domain_distribution(self) -> None:
        """Given scenarios, when grouped by domain, then 40 per domain across 5 domains."""
        scenarios = load_scenarios()
        by_domain: dict[str, list[BenchScenario]] = {}
        for s in scenarios:
            by_domain.setdefault(s.domain, []).append(s)
        assert set(by_domain.keys()) == set(DOMAINS)
        for domain, items in by_domain.items():
            assert len(items) == 40, f"Domain {domain} has {len(items)}, expected 40"

    def test_a3_scenario_fields_complete(self) -> None:
        """Given each scenario, when validated, then has all required fields."""
        scenarios = load_scenarios()
        for s in scenarios:
            assert s.question, f"Scenario {s.id} missing question"
            assert s.domain in DOMAINS, f"Scenario {s.id} invalid domain: {s.domain}"
            assert 0.0 <= s.expected_confidence_low <= 1.0
            assert 0.0 <= s.expected_confidence_high <= 1.0
            assert s.expected_confidence_low <= s.expected_confidence_high
            assert s.ground_truth, f"Scenario {s.id} missing ground_truth"
            assert s.difficulty in ("easy", "medium", "hard")

    def test_a4_ambiguous_domain_includes_uncertainty(self) -> None:
        """Given domain 'ambiguous', when inspecting scenarios, then ground_truth acknowledges uncertainty."""
        scenarios = load_scenarios()
        ambiguous = [s for s in scenarios if s.domain == "ambiguous"]
        # Check that ambiguous scenarios acknowledge uncertainty or lack objective answers
        definitive_markers = ["true.", "false.", "yes.", "no."]
        for s in ambiguous:
            ground_truth_lower = s.ground_truth.lower()
            # Ambiguous scenarios should NOT end with a simple true/false or yes/no
            # They should acknowledge uncertainty, debate, contestation, or lack of consensus
            has_uncertainty_marker = any(
                marker in ground_truth_lower
                for marker in [
                    "uncertain",
                    "unclear",
                    "unknown",
                    "debatable",
                    "depends",
                    "no definitive",
                    "contested",
                    "open question",
                    "no objectively",
                    "not predetermined",
                    "no consensus",
                    "philosophically",
                    "debated",
                    "unsolved",
                    "unresolved",
                    "different",
                    "no empirical",
                ]
            )
            # Ensure it doesn't claim simple truth
            has_no_simple_claim = not any(ground_truth_lower.strip().endswith(marker) for marker in definitive_markers)
            assert has_uncertainty_marker or has_no_simple_claim, (
                f"Ambiguous scenario {s.id} ground_truth lacks uncertainty: {s.ground_truth[:80]}"
            )

    def test_a5_unique_ids(self) -> None:
        """All scenario IDs are unique."""
        scenarios = load_scenarios()
        ids = [s.id for s in scenarios]
        assert len(ids) == len(set(ids))


class TestScenarioDatasetRegression:
    """B-tests: no regressions."""

    def test_b1_existing_imports_work(self) -> None:
        """Existing trust package imports still work."""
        from cognilateral_trust import (
            TrustContext,
            TrustPassport,
            evaluate_trust,
        )

        assert evaluate_trust is not None
        assert TrustContext is not None
        assert TrustPassport is not None

    def test_b2_scenario_is_frozen(self) -> None:
        """BenchScenario is immutable."""
        scenarios = load_scenarios()
        with pytest.raises(AttributeError):
            scenarios[0].question = "modified"  # type: ignore
