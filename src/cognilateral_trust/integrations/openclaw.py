"""OpenClaw TrustProvider — trust evaluation service primitive.

OpenClaw is the trust primitive for AI orchestration platforms, providing
declarative routing by confidence tier and accountability for trust decisions.

Usage:
    from cognilateral_trust.integrations.openclaw import OpenClawTrustProvider
    from cognilateral_trust import TrustContext

    provider = OpenClawTrustProvider()

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
        """Evaluate trust for a decision point.

        Args:
            context: Epistemic context including confidence, evidence, and constraints.

        Returns:
            TrustEvaluation with verdict, tier, routing, and accountability record.
        """
        ...

    def health(self) -> dict[str, Any]:
        """Return health and calibration statistics.

        Returns dict with keys like:
            - calibration_score (0.0-1.0): How well-calibrated this provider is
            - evaluations_total (int): Total evaluations processed
            - evaluations_escalated (int): How many were escalated
            - last_evaluation_timestamp (float): ISO 8601 or Unix timestamp
        """
        ...

    def provider_info(self) -> dict[str, Any]:
        """Return provider metadata and capabilities.

        Returns dict with keys like:
            - name (str): Provider identifier (e.g., "openclaw-trust-v1")
            - version (str): Semantic version
            - capabilities (list[str]): Feature list (e.g., ["routing", "warrants"])
        """
        ...


class OpenClawTrustProvider:
    """OpenClaw trust provider — epistemic routing for AI agents.

    Wraps the core evaluate_trust function and tracks calibration metrics.
    Implements the TrustProvider Protocol for framework integration.
    """

    def __init__(self) -> None:
        """Initialize OpenClaw trust provider."""
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
        """Return health and calibration statistics.

        Returns:
            Dict with calibration_score, evaluation counts, and last evaluation time.
        """
        # Simple calibration: ratio of ACTs to ESCALATEs
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
        """Return provider metadata.

        Returns:
            Dict with name, version, and capabilities.
        """
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
