"""Core confidence tier routing and protocol types — zero external dependencies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import IntEnum
from typing import Any, Literal

__all__ = [
    "ConfidenceTier",
    "PassportStamp",
    "ROUTE_BASIC",
    "ROUTE_SOVEREIGNTY_GATE",
    "ROUTE_WARRANT_CHECK",
    "TierRoutingResult",
    "TrustContext",
    "TrustPassport",
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


# ---------------------------------------------------------------------------
# Trust Protocol Types — frozen atoms consumed by all swimlanes
# ---------------------------------------------------------------------------

StampAction = Literal["spawn", "handoff", "evaluate", "terminate"]


@dataclass(frozen=True)
class TrustContext:
    """Epistemic context for a trust evaluation.

    Captures the confidence level, supporting evidence, reversibility,
    and welfare relevance of an action under evaluation.
    """

    confidence: float
    evidence: tuple[str, ...] = ()
    action_reversible: bool = True
    welfare_relevant: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            msg = f"confidence must be 0.0-1.0, got {self.confidence}"
            raise ValueError(msg)

    def serialize(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return asdict(self)

    @classmethod
    def restore(cls, data: dict[str, Any]) -> TrustContext:
        """Restore from a serialized dict."""
        return cls(
            confidence=data["confidence"],
            evidence=tuple(data.get("evidence", ())),
            action_reversible=data.get("action_reversible", True),
            welfare_relevant=data.get("welfare_relevant", False),
        )


@dataclass(frozen=True)
class PassportStamp:
    """Record of a trust-relevant action taken by an agent.

    Each stamp captures the agent, action type, confidence at the time,
    timestamp, and any evidence contributed during that action.
    """

    agent_id: str
    action: StampAction
    confidence_at_action: float
    timestamp: str  # ISO 8601
    evidence_added: tuple[str, ...] = ()

    def serialize(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return asdict(self)

    @classmethod
    def restore(cls, data: dict[str, Any]) -> PassportStamp:
        """Restore from a serialized dict."""
        return cls(
            agent_id=data["agent_id"],
            action=data["action"],
            confidence_at_action=data["confidence_at_action"],
            timestamp=data["timestamp"],
            evidence_added=tuple(data.get("evidence_added", ())),
        )


@dataclass(frozen=True)
class TrustPassport:
    """Accumulated trust context that travels with agent control transfer.

    Collects stamps from each trust-relevant action (spawn, handoff,
    evaluate, terminate). Immutable — add_stamp() returns a new passport.
    """

    context: TrustContext
    stamps: tuple[PassportStamp, ...] = ()
    warrant_chain: tuple[str, ...] = ()
    created_at: str = ""  # ISO 8601
    agent_id: str = ""

    def add_stamp(self, stamp: PassportStamp) -> TrustPassport:
        """Return a NEW passport with the stamp appended. Does not mutate."""
        return TrustPassport(
            context=self.context,
            stamps=(*self.stamps, stamp),
            warrant_chain=self.warrant_chain,
            created_at=self.created_at,
            agent_id=self.agent_id,
        )

    def serialize(self) -> dict[str, Any]:
        """Serialize the full passport to a JSON-compatible dict."""
        return {
            "context": self.context.serialize(),
            "stamps": [s.serialize() for s in self.stamps],
            "warrant_chain": list(self.warrant_chain),
            "created_at": self.created_at,
            "agent_id": self.agent_id,
        }

    @classmethod
    def restore(cls, data: dict[str, Any]) -> TrustPassport:
        """Restore a passport from a serialized dict."""
        return cls(
            context=TrustContext.restore(data["context"]),
            stamps=tuple(PassportStamp.restore(s) for s in data.get("stamps", ())),
            warrant_chain=tuple(data.get("warrant_chain", ())),
            created_at=data.get("created_at", ""),
            agent_id=data.get("agent_id", ""),
        )
