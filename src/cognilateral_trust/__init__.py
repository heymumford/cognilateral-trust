"""Cognilateral Trust Engine — epistemic confidence tiers and routing.

AI that tells you when it's guessing.

Maps epistemic confidence levels (C0-C9) to pipeline routing decisions:
- C0-C3 (basic): straight-through relay, no additional checks
- C4-C6 (warrant_check): require warrant freshness verification
- C7-C9 (sovereignty_gate): require sovereignty gate ACT approval

Usage:
    from cognilateral_trust import evaluate_trust

    result = evaluate_trust(0.7)
    if result.should_proceed:
        # Safe to act
        ...
    else:
        # Escalate to human
        print(result.accountability_record.reasons)
"""

from __future__ import annotations

__all__ = [
    "ConfidenceTier",
    "ROUTE_BASIC",
    "ROUTE_SOVEREIGNTY_GATE",
    "ROUTE_WARRANT_CHECK",
    "TierRoutingResult",
    "PredictionStore",
    "TrustEvaluation",
    "evaluate_tier_routing",
    "evaluate_trust",
    "route_by_tier",
]

from cognilateral_trust.core import (
    ROUTE_BASIC,
    ROUTE_SOVEREIGNTY_GATE,
    ROUTE_WARRANT_CHECK,
    ConfidenceTier,
    TierRoutingResult,
    evaluate_tier_routing,
    route_by_tier,
)
from cognilateral_trust.evaluate import TrustEvaluation, evaluate_trust
from cognilateral_trust.prediction_store import PredictionStore
