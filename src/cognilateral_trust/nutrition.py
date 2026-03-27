"""Nutrition label — standard disclosure for AI trust evaluation (D6).

Generates a human-readable disclosure that can be attached to any
AI-generated output. The label tells the end user:

1. Whether this response was trust-evaluated
2. What confidence the system had
3. What the calibration accuracy is
4. Whether the system recommended acting or escalating

Usage:
    from cognilateral_trust import nutrition_label

    label = nutrition_label(confidence=0.7, calibration_accuracy=0.625)
    # "Trust evaluated. Confidence: 0.70 (C7). Calibration: 62.5%. Verdict: ACT."

The label is designed to be appended to API responses, UI footers,
or metadata fields. It is the minimum viable disclosure — the person
downstream deserves to know.
"""

from __future__ import annotations

from dataclasses import dataclass

from cognilateral_trust.evaluate import evaluate_trust


@dataclass(frozen=True)
class NutritionLabel:
    """Structured trust disclosure for an AI response."""

    evaluated: bool
    confidence: float
    tier: str
    route: str
    verdict: str
    calibration_accuracy: float | None
    disclosure: str

    def __str__(self) -> str:
        return self.disclosure


def nutrition_label(
    confidence: float,
    *,
    is_reversible: bool = True,
    touches_external: bool = False,
    calibration_accuracy: float | None = None,
) -> NutritionLabel:
    """Generate a nutrition label for an AI trust evaluation.

    Args:
        confidence: Agent confidence (0.0-1.0)
        is_reversible: Whether the action can be undone
        touches_external: Whether the action affects external systems
        calibration_accuracy: Historical calibration accuracy (0.0-1.0), if known

    Returns:
        NutritionLabel with human-readable disclosure string
    """
    result = evaluate_trust(
        confidence,
        is_reversible=is_reversible,
        touches_external=touches_external,
    )

    verdict = "ACT" if result.should_proceed else "ESCALATE"
    tier_name = result.tier.name if hasattr(result.tier, "name") else str(result.tier)

    parts = [
        "Trust evaluated.",
        f"Confidence: {confidence:.2f} ({tier_name}).",
    ]

    if calibration_accuracy is not None:
        parts.append(f"Calibration: {calibration_accuracy * 100:.1f}%.")

    parts.append(f"Verdict: {verdict}.")

    disclosure = " ".join(parts)

    return NutritionLabel(
        evaluated=True,
        confidence=confidence,
        tier=tier_name,
        route=result.route,
        verdict=verdict,
        calibration_accuracy=calibration_accuracy,
        disclosure=disclosure,
    )


def not_evaluated_label() -> NutritionLabel:
    """Label for responses that were NOT trust-evaluated."""
    return NutritionLabel(
        evaluated=False,
        confidence=0.0,
        tier="unknown",
        route="unknown",
        verdict="NOT_EVALUATED",
        calibration_accuracy=None,
        disclosure="Not trust-evaluated. No confidence assessment was performed on this response.",
    )
