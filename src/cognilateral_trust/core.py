"""Core confidence tier routing — zero external dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

__all__ = [
    "ConfidenceTier",
    "ROUTE_BASIC",
    "ROUTE_SOVEREIGNTY_GATE",
    "ROUTE_WARRANT_CHECK",
    "TierRoutingResult",
    "evaluate_tier_routing",
    "route_by_tier",
]


class ConfidenceTier(IntEnum):
    """Epistemic confidence tiers (C0-C9)."""

    C0 = 0  # Unverified
    C1 = 1  # Anecdotal
    C2 = 2  # Observed
    C3 = 3  # Measured
    C4 = 4  # Tested
    C5 = 5  # Validated
    C6 = 6  # Falsifiable
    C7 = 7  # Governance-grade
    C8 = 8  # Audited
    C9 = 9  # Resilient


ROUTE_BASIC = "basic"
ROUTE_WARRANT_CHECK = "warrant_check"
ROUTE_SOVEREIGNTY_GATE = "sovereignty_gate"

_TIER_MIN = 0
_TIER_MAX = 9


def route_by_tier(tier: int | ConfidenceTier) -> str:
    """Return the routing decision string for a given confidence tier.

    Raises ValueError for tiers outside the valid 0-9 range.
    """
    tier_val = int(tier)
    if tier_val < _TIER_MIN or tier_val > _TIER_MAX:
        msg = f"Confidence tier must be {_TIER_MIN}-{_TIER_MAX}, got {tier_val}"
        raise ValueError(msg)
    if tier_val <= 3:
        return ROUTE_BASIC
    if tier_val <= 6:
        return ROUTE_WARRANT_CHECK
    return ROUTE_SOVEREIGNTY_GATE


@dataclass(frozen=True)
class TierRoutingResult:
    """Immutable result of tier routing evaluation."""

    tier: int
    route: str
    requires_warrant_check: bool
    requires_sovereignty_gate: bool
    tier_name: str


def evaluate_tier_routing(tier: int) -> TierRoutingResult:
    """Evaluate routing for a confidence tier, returning a full result object."""
    route = route_by_tier(tier)
    try:
        name = ConfidenceTier(tier).name
    except ValueError:
        name = f"C{tier}"
    return TierRoutingResult(
        tier=tier,
        route=route,
        requires_warrant_check=tier >= 4,
        requires_sovereignty_gate=tier >= 7,
        tier_name=name,
    )
