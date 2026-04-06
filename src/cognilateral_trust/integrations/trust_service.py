"""Trust service provider — trust evaluation service primitive.

Provides declarative routing by confidence tier and accountability for
trust decisions in AI orchestration platforms.

Usage:
    from cognilateral_trust.integrations.trust_service import TrustServiceProvider
    from cognilateral_trust import TrustContext

    provider = TrustServiceProvider()

    context = TrustContext(
        confidence=0.85,
        evidence=("test1 passed", "integration verified"),
        action_reversible=True,
        welfare_relevant=False,
    )

    verdict = provider.evaluate(context)
    if verdict.should_proceed:
        perform_action()
    else:
        escalate(verdict.accountability_record.reasons)

    # Check provider health
    stats = provider.health()
    print(f"Provider calibration: {stats['calibration_score']}")

Zero dependencies beyond cognilateral-trust itself.
"""

from __future__ import annotations

import time
from typing import Any, Protocol, runtime_checkable

from cognilateral_trust.core import TrustContext
from cognilateral_trust.evaluate import TrustEvaluation, evaluate_trust


@runtime_checkable
class TrustProvider(Protocol):
    """Protocol for trust evaluation providers.

    Any object implementing this protocol can serve as a trust evaluator
    in agent orchestration systems. Implementations must support evaluation,
    health reporting, and provider metadata.
    """

    def evaluate(self, context: TrustContext) -> TrustEvaluation:
        """Evaluate trust for a decision point."""
        ...

    def health(self) -> dict[str, Any]:
        """Return health and calibration statistics."""
        ...

    def provider_info(self) -> dict[str, Any]:
        """Return provider metadata and capabilities."""
        ...


class TrustServiceProvider:
    """Trust service provider — epistemic routing for AI agents.

    Wraps the core evaluate_trust function and tracks calibration metrics.
    Implements the TrustProvider Protocol for framework integration.
    """

    def __init__(self) -> None:
        """Initialize trust service provider."""
        self._total_evaluations = 0
        self._total_escalated = 0
        self._last_evaluation_ts = 0.0

    def evaluate(self, context: TrustContext) -> TrustEvaluation:
        """Evaluate trust for a decision point.

        Args:
            context: Epistemic context (confidence, evidence, reversibility, welfare).

        Returns:
            TrustEvaluation with verdict, tier, routing, and accountability.
        """
        self._last_evaluation_ts = time.time()
        self._total_evaluations += 1

        result = evaluate_trust(
            context.confidence,
            is_reversible=context.action_reversible,
            touches_external=getattr(context, "touches_external", False),
            context={"evidence": context.evidence},
        )

        if not result.should_proceed:
            self._total_escalated += 1

        return result

    def health(self) -> dict[str, Any]:
        """Return health and calibration statistics."""
        if self._total_evaluations == 0:
            calibration_score = 1.0
        else:
            acted = self._total_evaluations - self._total_escalated
            calibration_score = max(0.0, min(1.0, acted / self._total_evaluations))

        return {
            "calibration_score": calibration_score,
            "evaluations_total": self._total_evaluations,
            "evaluations_escalated": self._total_escalated,
            "last_evaluation_timestamp": self._last_evaluation_ts,
        }

    def provider_info(self) -> dict[str, Any]:
        """Return provider metadata."""
        return {
            "name": "cognilateral-trust",
            "version": "1.1.0",
            "capabilities": [
                "confidence_tier_routing",
                "accountability_records",
                "escalation_gates",
                "calibration_tracking",
            ],
        }
