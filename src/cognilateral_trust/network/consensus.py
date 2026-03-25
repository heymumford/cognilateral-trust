"""Trust-weighted consensus — Bayesian composition of multi-agent verdicts.

Combines verdicts from multiple agents using calibration scores as weights.
Not a majority vote — high-calibration agents dominate low-calibration consensus.

Weight formula: weight = calibration_score * confidence
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence


@dataclass(frozen=True)
class WeightedVerdict:
    """Immutable verdict from a single agent."""

    agent_id: str
    verdict: Literal["ACT", "ESCALATE"]
    confidence: float
    calibration_score: float


@dataclass(frozen=True)
class ConsensusResult:
    """Result of weighted consensus."""

    consensus_verdict: Literal["ACT", "ESCALATE"]
    weights: dict[str, float]  # agent_id -> absolute weight (calibration_score * confidence)
    total_weight_act: float
    total_weight_escalate: float


def weighted_consensus(verdicts: Sequence[WeightedVerdict]) -> ConsensusResult:
    """Bayesian trust-weighted consensus. NOT majority vote.

    Weight formula: weight = calibration_score * confidence

    Higher-calibration agents have more influence. A single
    high-calibration ESCALATE can override multiple low-calibration ACTs.

    Args:
        verdicts: Sequence of weighted verdicts from agents

    Returns:
        ConsensusResult with final verdict and normalized weights

    Raises:
        ValueError: If verdicts sequence is empty
    """
    if not verdicts:
        msg = "verdicts sequence must not be empty"
        raise ValueError(msg)

    weights: dict[str, float] = {}
    act_weight = 0.0
    escalate_weight = 0.0

    # Calculate weights
    for verdict in verdicts:
        weight = verdict.calibration_score * verdict.confidence
        weights[verdict.agent_id] = weight

        if verdict.verdict == "ACT":
            act_weight += weight
        elif verdict.verdict == "ESCALATE":
            escalate_weight += weight

    # Determine consensus
    if escalate_weight > act_weight:
        consensus_verdict: Literal["ACT", "ESCALATE"] = "ESCALATE"
    elif act_weight > escalate_weight:
        consensus_verdict = "ACT"
    else:
        # Tie: default to safer option (ESCALATE)
        consensus_verdict = "ESCALATE"

    return ConsensusResult(
        consensus_verdict=consensus_verdict,
        weights=weights,
        total_weight_act=act_weight,
        total_weight_escalate=escalate_weight,
    )
