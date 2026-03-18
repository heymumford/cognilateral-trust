"""S8: Epistemic Firewall — demanded vs supplied level mismatch detection.

Zero external dependencies. Maintains a registry of required epistemic levels
per action type and checks each action before execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


# ---------------------------------------------------------------------------
# EpistemicLevel
# ---------------------------------------------------------------------------


class EpistemicLevel(IntEnum):
    """Epistemic confidence levels in ascending order of rigor.

    Higher values indicate stronger epistemic justification.
    Matches the D-07 sovereignty framework in the cognilateral heuristics.
    """

    RAW = 0  # Unprocessed input; no verification
    OBSERVED = 1  # Directly observed but not measured
    MEASURED = 2  # Measured with an instrument or metric
    TESTED = 3  # Covered by automated tests
    VALIDATED = 4  # Validated against acceptance criteria
    FALSIFIABLE = 5  # Expressed as falsifiable hypothesis
    GOVERNANCE = 6  # Auditable, governance-controlled


# ---------------------------------------------------------------------------
# MismatchResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MismatchResult:
    """Result of an epistemic level check."""

    demanded: EpistemicLevel
    supplied: EpistemicLevel
    gap: int
    should_escalate: bool
    reason: str


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------


def check_epistemic_mismatch(
    demanded: int | EpistemicLevel,
    supplied: int | EpistemicLevel,
    max_gap: int = 2,
) -> MismatchResult:
    """Check whether the supplied epistemic level satisfies the demanded level.

    Args:
        demanded: The minimum epistemic level required for the action.
        supplied: The epistemic level the caller claims to hold.
        max_gap: Maximum allowed gap before escalation. Default 2 (two tiers).
                 A gap of 0 means any shortfall triggers escalation.

    Returns:
        MismatchResult with gap, should_escalate, and human-readable reason.
    """
    demanded_level = EpistemicLevel(int(demanded))
    supplied_level = EpistemicLevel(int(supplied))

    if supplied_level >= demanded_level:
        return MismatchResult(
            demanded=demanded_level,
            supplied=supplied_level,
            gap=0,
            should_escalate=False,
            reason=f"Supplied level {supplied_level.name} satisfies demanded {demanded_level.name}",
        )

    gap = int(demanded_level) - int(supplied_level)

    if gap <= max_gap:
        return MismatchResult(
            demanded=demanded_level,
            supplied=supplied_level,
            gap=gap,
            should_escalate=False,
            reason=(
                f"Gap of {gap} tier(s) between {demanded_level.name} (demanded) "
                f"and {supplied_level.name} (supplied) is within tolerance (max_gap={max_gap})"
            ),
        )

    return MismatchResult(
        demanded=demanded_level,
        supplied=supplied_level,
        gap=gap,
        should_escalate=True,
        reason=(
            f"Epistemic gap of {gap} tier(s) exceeds max_gap={max_gap}: "
            f"demanded {demanded_level.name} ({int(demanded_level)}) but "
            f"supplied only {supplied_level.name} ({int(supplied_level)}). "
            "Escalate for human review or obtain stronger evidence."
        ),
    )


# ---------------------------------------------------------------------------
# EpistemicFirewall — per-action registry
# ---------------------------------------------------------------------------


class EpistemicFirewall:
    """Registry of required epistemic levels per action type.

    Checks incoming actions against registered requirements. Unknown action
    types default to GOVERNANCE (highest level) for maximum safety.
    """

    def __init__(self) -> None:
        self._requirements: dict[str, EpistemicLevel] = {}

    def require(self, action: str, level: EpistemicLevel) -> None:
        """Register a minimum epistemic level for an action type.

        Args:
            action: The action identifier (e.g., "deploy", "publish").
            level: The minimum EpistemicLevel required.
        """
        self._requirements[action] = level

    def check(
        self,
        action: str,
        supplied: int | EpistemicLevel,
        max_gap: int = 2,
    ) -> MismatchResult:
        """Check whether the supplied level satisfies the action requirement.

        Unknown action types default to GOVERNANCE (strictest).

        Args:
            action: The action identifier to look up.
            supplied: The epistemic level the caller holds.
            max_gap: Maximum allowed gap before escalation.

        Returns:
            MismatchResult.
        """
        demanded = self._requirements.get(action, EpistemicLevel.GOVERNANCE)
        return check_epistemic_mismatch(demanded, supplied, max_gap=max_gap)
