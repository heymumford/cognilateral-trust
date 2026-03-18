"""Warrant system — evidence-backed confidence that decays over time.

A Warrant wraps a confidence value with provenance (evidence_source) and
a TTL. The effective confidence degrades linearly as the warrant ages,
reaching zero at expiry regardless of the original value.

Public API:
    Warrant                         — frozen dataclass
    evaluate_warrant(w)             — current effective confidence
    WarrantStore                    — in-memory store with auto-expiry
    evaluate_trust_with_warrant(w)  — trust evaluation using warrant confidence
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from cognilateral_trust.evaluate import TrustEvaluation, evaluate_trust

__all__ = [
    "Warrant",
    "WarrantStore",
    "evaluate_trust_with_warrant",
    "evaluate_warrant",
]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Warrant dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Warrant:
    """Evidence-backed confidence value with TTL and optional decay.

    Attributes:
        evidence_source: Identifier for the evidence backing this warrant
                         (e.g. "web-search", "database-lookup", "peer-review").
        issued_at:       When the warrant was created (UTC-aware).
        expires_at:      When the warrant becomes void (UTC-aware).
        confidence:      Original confidence value in [0.0, 1.0].
        decay_rate:      How rapidly confidence decays as a fraction of TTL.
                         0.0 = no decay (flat until expiry),
                         1.0 = linear decay to 0.0 at expiry,
                         >1.0 = accelerated decay (clips to 0.0).
    """

    evidence_source: str
    issued_at: datetime
    expires_at: datetime
    confidence: float
    decay_rate: float = 0.0


# ---------------------------------------------------------------------------
# evaluate_warrant — effective confidence after decay
# ---------------------------------------------------------------------------


def evaluate_warrant(warrant: Warrant, *, at: datetime | None = None) -> float:
    """Return the current effective confidence of a warrant.

    Expired warrants return 0.0.
    Active warrants apply linear decay scaled by elapsed_fraction and decay_rate:

        effective = confidence * max(0.0, 1.0 - decay_rate * elapsed_fraction)

    where elapsed_fraction = elapsed / ttl ∈ [0.0, 1.0].

    Args:
        warrant: The warrant to evaluate.
        at:      The reference time (defaults to utcnow).

    Returns float in [0.0, original confidence].
    """
    now = at or _utcnow()

    if now >= warrant.expires_at:
        return 0.0

    ttl_seconds = (warrant.expires_at - warrant.issued_at).total_seconds()
    if ttl_seconds <= 0:
        return 0.0

    elapsed_seconds = (now - warrant.issued_at).total_seconds()
    elapsed_fraction = max(0.0, min(1.0, elapsed_seconds / ttl_seconds))

    decay_factor = max(0.0, 1.0 - warrant.decay_rate * elapsed_fraction)
    return max(0.0, min(warrant.confidence, warrant.confidence * decay_factor))


# ---------------------------------------------------------------------------
# WarrantStore — in-memory store with auto-expiry
# ---------------------------------------------------------------------------


@dataclass
class WarrantStore:
    """In-memory store for active warrants.

    Warrants are keyed by a UUID string. Expired warrants are lazily
    removed on access or eagerly removed via expire_stale().
    """

    _warrants: dict[str, Warrant] = field(default_factory=dict)

    def add(self, warrant: Warrant) -> str:
        """Add a warrant and return its unique ID."""
        wid = str(uuid.uuid4())
        self._warrants[wid] = warrant
        return wid

    def get(self, warrant_id: str) -> Warrant | None:
        """Return a warrant by ID, or None if not found or expired."""
        warrant = self._warrants.get(warrant_id)
        if warrant is None:
            return None
        if _utcnow() >= warrant.expires_at:
            del self._warrants[warrant_id]
            return None
        return warrant

    def revoke(self, warrant_id: str) -> None:
        """Remove a warrant immediately. No-op if not found."""
        self._warrants.pop(warrant_id, None)

    def expire_stale(self) -> int:
        """Remove all expired warrants. Returns count removed."""
        now = _utcnow()
        stale = [wid for wid, w in self._warrants.items() if now >= w.expires_at]
        for wid in stale:
            del self._warrants[wid]
        return len(stale)

    def active_count(self) -> int:
        """Return count of non-expired warrants without mutating state."""
        now = _utcnow()
        return sum(1 for w in self._warrants.values() if now < w.expires_at)


# ---------------------------------------------------------------------------
# evaluate_trust_with_warrant
# ---------------------------------------------------------------------------


def evaluate_trust_with_warrant(
    warrant: Warrant,
    *,
    is_reversible: bool = True,
    touches_external: bool = False,
    context: dict[str, Any] | None = None,
    at: datetime | None = None,
) -> TrustEvaluation:
    """Evaluate trust using the warrant's current effective confidence.

    Computes the decayed confidence via evaluate_warrant(), then delegates
    to evaluate_trust() with that value. Expired warrants evaluate as 0.0
    and will typically result in ESCALATE routing.

    Args:
        warrant:          The warrant backing this evaluation.
        is_reversible:    Whether the action can be undone (default True).
        touches_external: Whether the action affects external systems.
        context:          Optional metadata attached to the accountability record.
        at:               Reference time for decay calculation (defaults to utcnow).

    Returns TrustEvaluation with effective_confidence as the confidence value.
    """
    now = at or _utcnow()
    effective_confidence = evaluate_warrant(warrant, at=now)
    is_expired = now >= warrant.expires_at

    ctx = dict(context or {})
    ctx.setdefault("warrant_source", warrant.evidence_source)
    ctx.setdefault("warrant_issued_at", warrant.issued_at.isoformat())
    ctx.setdefault("warrant_expires_at", warrant.expires_at.isoformat())
    ctx.setdefault("warrant_original_confidence", warrant.confidence)
    ctx.setdefault("warrant_expired", is_expired)

    result = evaluate_trust(
        effective_confidence,
        is_reversible=is_reversible,
        touches_external=touches_external,
        context=ctx,
    )

    # Expired warrants must always escalate — the evidence basis is void.
    if is_expired and result.should_proceed:
        from cognilateral_trust.accountability import create_accountability_record
        from cognilateral_trust.evaluate import TrustEvaluation

        record = create_accountability_record(
            verdict="ESCALATE",
            reasons=("warrant expired — evidence basis is void",),
            context=ctx,
            confidence=effective_confidence,
            confidence_tier=result.tier.value,
        )
        return TrustEvaluation(
            confidence=result.confidence,
            tier=result.tier,
            route=result.route,
            requires_warrant_check=result.requires_warrant_check,
            requires_sovereignty_gate=result.requires_sovereignty_gate,
            should_proceed=False,
            accountability_record=record,
        )

    return result
