"""Tests for trust-weighted consensus — Bayesian composition of multi-agent verdicts.

TDD — tests written before implementation. Covers:
- WeightedVerdict: immutable verdict from a single agent
- weighted_consensus: Bayesian trust-weighted consensus (NOT majority vote)
- Weight formula: weight = calibration_score * confidence
- High-calibration ESCALATE outweighs low-calibration ACT
- Equal calibration scores → equivalent to majority vote
- Trust-weighted outperforms naive majority on adversarial cases (FF-TC-01)
"""

from __future__ import annotations

import pytest

from cognilateral_trust.network.consensus import ConsensusResult, WeightedVerdict, weighted_consensus


class TestWeightedVerdict:
    """Immutable verdict from a single agent."""

    def test_create(self) -> None:
        """WeightedVerdict stores agent, verdict, and calibration."""
        verdict = WeightedVerdict(
            agent_id="agent-a",
            verdict="ACT",
            confidence=0.9,
            calibration_score=0.85,
        )
        assert verdict.agent_id == "agent-a"
        assert verdict.verdict == "ACT"
        assert verdict.confidence == 0.9
        assert verdict.calibration_score == 0.85

    def test_frozen(self) -> None:
        """WeightedVerdict cannot be mutated."""
        verdict = WeightedVerdict(
            agent_id="agent-a",
            verdict="ACT",
            confidence=0.9,
            calibration_score=0.85,
        )
        with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
            verdict.calibration_score = 0.5  # type: ignore

    def test_verdict_literal(self) -> None:
        """Verdict must be ACT or ESCALATE."""
        # Valid
        v1 = WeightedVerdict(
            agent_id="a1",
            verdict="ACT",
            confidence=0.9,
            calibration_score=0.8,
        )
        v2 = WeightedVerdict(
            agent_id="a2",
            verdict="ESCALATE",
            confidence=0.9,
            calibration_score=0.8,
        )
        assert v1.verdict == "ACT"
        assert v2.verdict == "ESCALATE"


class TestWeightedConsensus:
    """Bayesian trust-weighted consensus."""

    def test_single_verdict(self) -> None:
        """Single verdict is the consensus."""
        verdicts = [
            WeightedVerdict(
                agent_id="a1",
                verdict="ACT",
                confidence=0.9,
                calibration_score=0.8,
            ),
        ]
        result = weighted_consensus(verdicts)
        assert result.consensus_verdict == "ACT"

    def test_equal_calibration_acts_as_majority(self) -> None:
        """Equal calibration scores → equivalent to majority vote.

        2 ACT (cal=0.5) vs 1 ESCALATE (cal=0.5) → ACT wins by weight.
        """
        verdicts = [
            WeightedVerdict(
                agent_id="a1",
                verdict="ACT",
                confidence=0.9,
                calibration_score=0.5,
            ),
            WeightedVerdict(
                agent_id="a2",
                verdict="ACT",
                confidence=0.9,
                calibration_score=0.5,
            ),
            WeightedVerdict(
                agent_id="a3",
                verdict="ESCALATE",
                confidence=0.9,
                calibration_score=0.5,
            ),
        ]
        result = weighted_consensus(verdicts)
        assert result.consensus_verdict == "ACT"

    def test_high_calibration_escalate_outweighs_low_calibration_act(self) -> None:
        """High-calibration ESCALATE outweighs low-calibration ACT.

        ACT (cal=0.3) + ACT (cal=0.3) vs ESCALATE (cal=0.8) → ESCALATE wins.
        """
        verdicts = [
            WeightedVerdict(
                agent_id="a1",
                verdict="ACT",
                confidence=0.9,
                calibration_score=0.3,
            ),
            WeightedVerdict(
                agent_id="a2",
                verdict="ACT",
                confidence=0.9,
                calibration_score=0.3,
            ),
            WeightedVerdict(
                agent_id="a3",
                verdict="ESCALATE",
                confidence=0.9,
                calibration_score=0.8,
            ),
        ]
        result = weighted_consensus(verdicts)
        assert result.consensus_verdict == "ESCALATE"

    def test_weight_formula_calibration_times_confidence(self) -> None:
        """Weight = calibration_score * confidence.

        ACT with weight 0.9*0.8=0.72 vs ESCALATE with weight 0.7*0.85=0.595
        → ACT wins.
        """
        verdicts = [
            WeightedVerdict(
                agent_id="a1",
                verdict="ACT",
                confidence=0.9,
                calibration_score=0.8,
            ),
            WeightedVerdict(
                agent_id="a2",
                verdict="ESCALATE",
                confidence=0.7,
                calibration_score=0.85,
            ),
        ]
        result = weighted_consensus(verdicts)
        assert result.consensus_verdict == "ACT"

    def test_returns_consensus_result(self) -> None:
        """Result includes verdict, weights, and reasoning."""
        verdicts = [
            WeightedVerdict(
                agent_id="a1",
                verdict="ACT",
                confidence=0.9,
                calibration_score=0.8,
            ),
            WeightedVerdict(
                agent_id="a2",
                verdict="ESCALATE",
                confidence=0.7,
                calibration_score=0.85,
            ),
        ]
        result = weighted_consensus(verdicts)
        assert isinstance(result, ConsensusResult)
        assert result.consensus_verdict in ["ACT", "ESCALATE"]
        assert len(result.weights) == 2
        assert "a1" in result.weights
        assert "a2" in result.weights

    def test_empty_verdicts_raises(self) -> None:
        """Empty verdict list raises ValueError."""
        with pytest.raises(ValueError):
            weighted_consensus([])

    def test_calibration_weighted_outperforms_majority(self) -> None:
        """Calibration-weighted consensus should outperform naive majority.

        Setup: 3 low-calibration agents vote ACT (cal=0.1 each) vs
               1 high-calibration agent votes ESCALATE (cal=0.95).

        Naive majority: ACT (3 votes vs 1).
        Weighted: ESCALATE (0.95 > 0.1+0.1+0.1 = 0.3).

        Weighted approach is more defensible when expert (high-cal) disagrees.
        """
        verdicts = [
            WeightedVerdict(
                agent_id="novice1",
                verdict="ACT",
                confidence=0.8,
                calibration_score=0.1,
            ),
            WeightedVerdict(
                agent_id="novice2",
                verdict="ACT",
                confidence=0.8,
                calibration_score=0.1,
            ),
            WeightedVerdict(
                agent_id="novice3",
                verdict="ACT",
                confidence=0.8,
                calibration_score=0.1,
            ),
            WeightedVerdict(
                agent_id="expert",
                verdict="ESCALATE",
                confidence=0.9,
                calibration_score=0.95,
            ),
        ]
        result = weighted_consensus(verdicts)
        # Weighted should choose ESCALATE, not ACT (FF-TC-01)
        assert result.consensus_verdict == "ESCALATE"
