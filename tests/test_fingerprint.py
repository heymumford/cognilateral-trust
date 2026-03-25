"""A-tests and B-tests for calibration fingerprint generator (Step 16)."""

from __future__ import annotations

import pytest

from cognilateral_trust.bench.fingerprint import (
    CalibrationFingerprint,
    FingerprintSpoke,
    fingerprint_to_dict,
    generate_fingerprint,
)
from cognilateral_trust.bench.scoring import BenchScore, DomainScore


class TestGenerateFingerprint:
    """A-tests: Fingerprint generation from BenchScore."""

    def test_a1_fingerprint_has_spokes_per_domain(self) -> None:
        """Given BenchScore with 3 domains, fingerprint has 3 spokes."""
        domain_scores = (
            DomainScore("factual", 0.1, 40),
            DomainScore("reasoning", 0.2, 40),
            DomainScore("ambiguous", 0.15, 40),
        )
        score = BenchScore("TestModel", 0.15, domain_scores)
        fp = generate_fingerprint(score)

        assert len(fp.spokes) == 3
        assert fp.model == "TestModel"

    def test_a2_gap_is_one_minus_calibration_error(self) -> None:
        """Gap = 1.0 - calibration_error."""
        domain_scores = (DomainScore("test_domain", 0.3, 10),)
        score = BenchScore("CalcModel", 0.3, domain_scores)
        fp = generate_fingerprint(score)

        spoke = fp.spokes[0]
        expected_gap = 1.0 - 0.3
        assert spoke.gap == expected_gap
        assert spoke.gap == 0.7

    def test_a3_perfect_calibration_gives_large_gap(self) -> None:
        """When ECE is near 0 (perfect calibration), gap is near 1.0."""
        domain_scores = (
            DomainScore("domain1", 0.01, 50),
            DomainScore("domain2", 0.02, 50),
        )
        score = BenchScore("PerfectModel", 0.015, domain_scores)
        fp = generate_fingerprint(score)

        for spoke in fp.spokes:
            assert spoke.gap > 0.95

    def test_a4_poor_calibration_gives_small_gap(self) -> None:
        """When ECE is high (poor calibration), gap is near 0.0."""
        domain_scores = (DomainScore("bad_domain", 0.95, 50),)
        score = BenchScore("BadModel", 0.95, domain_scores)
        fp = generate_fingerprint(score)

        spoke = fp.spokes[0]
        assert spoke.gap < 0.1

    def test_a5_spokes_match_domain_order(self) -> None:
        """Fingerprint spokes correspond to input domain order."""
        domain_scores = (
            DomainScore("domain_a", 0.1, 20),
            DomainScore("domain_b", 0.2, 20),
            DomainScore("domain_c", 0.3, 20),
        )
        score = BenchScore("OrderModel", 0.2, domain_scores)
        fp = generate_fingerprint(score)

        assert [s.domain for s in fp.spokes] == ["domain_a", "domain_b", "domain_c"]
        assert [s.gap for s in fp.spokes] == [0.9, 0.8, 0.7]


class TestFingerprintToDict:
    """B-tests: JSON serialization."""

    def test_b1_dict_has_model_and_spokes(self) -> None:
        """fingerprint_to_dict returns dict with model and spokes."""
        domain_scores = (DomainScore("test", 0.2, 10),)
        score = BenchScore("TestModel", 0.2, domain_scores)
        fp = generate_fingerprint(score)

        result = fingerprint_to_dict(fp)

        assert "model" in result
        assert "spokes" in result
        assert result["model"] == "TestModel"

    def test_b2_spokes_is_list_of_dicts(self) -> None:
        """Spokes are rendered as list of dicts with domain and gap."""
        domain_scores = (
            DomainScore("domain1", 0.1, 10),
            DomainScore("domain2", 0.2, 10),
        )
        score = BenchScore("TestModel", 0.15, domain_scores)
        fp = generate_fingerprint(score)

        result = fingerprint_to_dict(fp)
        spokes = result["spokes"]

        assert isinstance(spokes, list)
        assert len(spokes) == 2
        assert spokes[0] == {"domain": "domain1", "gap": 0.9}
        assert spokes[1] == {"domain": "domain2", "gap": 0.8}

    def test_b3_round_trip_preserves_data(self) -> None:
        """dict -> BenchScore -> fingerprint -> dict preserves values."""
        original_domain_scores = (
            DomainScore("domain1", 0.1, 30),
            DomainScore("domain2", 0.2, 30),
        )
        original_score = BenchScore("RoundTripModel", 0.15, original_domain_scores)
        fp = generate_fingerprint(original_score)
        fp_dict = fingerprint_to_dict(fp)

        # Reconstruct and verify
        assert fp_dict["model"] == "RoundTripModel"
        assert len(fp_dict["spokes"]) == 2
        assert fp_dict["spokes"][0]["gap"] == 0.9
        assert fp_dict["spokes"][1]["gap"] == 0.8

    def test_b4_dict_is_json_serializable(self) -> None:
        """fingerprint_to_dict result can be JSON-encoded."""
        import json

        domain_scores = (DomainScore("test", 0.25, 10),)
        score = BenchScore("JsonModel", 0.25, domain_scores)
        fp = generate_fingerprint(score)

        fp_dict = fingerprint_to_dict(fp)
        json_str = json.dumps(fp_dict)

        assert isinstance(json_str, str)
        reconstructed = json.loads(json_str)
        assert reconstructed["model"] == "JsonModel"


class TestFingerprintImmutability:
    """C-tests: Immutability."""

    def test_c1_fingerprint_spoke_frozen(self) -> None:
        """FingerprintSpoke is immutable."""
        spoke = FingerprintSpoke("test_domain", 0.85)

        with pytest.raises(AttributeError):
            spoke.domain = "modified"  # type: ignore

    def test_c2_fingerprint_frozen(self) -> None:
        """CalibrationFingerprint is immutable."""
        spokes = (FingerprintSpoke("domain1", 0.9),)
        fp = CalibrationFingerprint("TestModel", spokes)

        with pytest.raises(AttributeError):
            fp.model = "modified"  # type: ignore

        with pytest.raises(AttributeError):
            fp.spokes = ()  # type: ignore
