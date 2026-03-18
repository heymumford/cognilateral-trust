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
    "CalibratedTrustEngine",
    "evaluate_trust",
    "route_by_tier",
    # S1: confidence extraction
    "extract_confidence",
    "extract_confidence_from_anthropic_response",
    "extract_confidence_from_openai_response",
    "extract_confidence_from_text",
    # S2: warrants
    "Warrant",
    "WarrantStore",
    "evaluate_trust_with_warrant",
    "evaluate_warrant",
    # S3: middleware
    "TrustEscalation",
    "TrustMiddleware",
    "async_evaluate_trust",
    "trust_gate",
    # S4: JSONL persistence
    "JSONLAccountabilityStore",
    "JSONLPredictionStore",
    # S6: claim extraction
    "Claim",
    "ClaimSet",
    "extract_claims",
    # S7: fidelity verification
    "FidelityResult",
    "verify_fidelity",
    "verify_fidelity_batch",
    # S8: epistemic firewall
    "EpistemicFirewall",
    "EpistemicLevel",
    "MismatchResult",
    "check_epistemic_mismatch",
    # S9: sovereignty gate
    "DEFAULT_POLICY",
    "SovereigntyDecision",
    "SovereigntyError",
    "SovereigntyPolicy",
    "evaluate_sovereignty",
    "sovereignty_gate",
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
from cognilateral_trust.calibrated import CalibratedTrustEngine
from cognilateral_trust.extractors import (
    extract_confidence,
    extract_confidence_from_anthropic_response,
    extract_confidence_from_openai_response,
    extract_confidence_from_text,
)
from cognilateral_trust.warrants import (
    Warrant,
    WarrantStore,
    evaluate_trust_with_warrant,
    evaluate_warrant,
)
from cognilateral_trust.middleware import (
    TrustEscalation,
    TrustMiddleware,
    async_evaluate_trust,
    trust_gate,
)
from cognilateral_trust.persistence import (
    JSONLAccountabilityStore,
    JSONLPredictionStore,
)
from cognilateral_trust.claims import (
    Claim,
    ClaimSet,
    extract_claims,
)
from cognilateral_trust.fidelity import (
    FidelityResult,
    verify_fidelity,
    verify_fidelity_batch,
)
from cognilateral_trust.firewall import (
    EpistemicFirewall,
    EpistemicLevel,
    MismatchResult,
    check_epistemic_mismatch,
)
from cognilateral_trust.sovereignty import (
    DEFAULT_POLICY,
    SovereigntyDecision,
    SovereigntyError,
    SovereigntyPolicy,
    evaluate_sovereignty,
    sovereignty_gate,
)
