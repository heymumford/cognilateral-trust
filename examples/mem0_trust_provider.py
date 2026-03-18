"""Mem0 TrustProvider — wraps cognilateral-trust for memory confidence scoring.

Mem0 stores agent memories. cognilateral-trust scores how much to trust
each memory based on age, source confidence, and verification status.

Usage:
    pip install cognilateral-trust mem0ai

    from mem0_trust_provider import score_memory_trust

    trust = score_memory_trust(
        confidence=0.8,
        age_hours=48,
        verified=False,
    )
    print(trust["should_use"])  # True/False
    print(trust["tier"])        # "C6"

This is an example integration. Adapt to your memory pipeline.
"""

from __future__ import annotations

from typing import Any

from cognilateral_trust import evaluate_trust
from cognilateral_trust.core import ConfidenceTier


def _age_decay(confidence: float, age_hours: float, half_life_hours: float = 168.0) -> float:
    """Apply exponential decay to confidence based on memory age.

    Default half-life: 168 hours (1 week). A memory that was 0.9 confident
    a week ago is now 0.45 confident — it needs re-verification.
    """
    import math

    if age_hours <= 0:
        return confidence
    decay_factor = math.exp(-0.693 * age_hours / half_life_hours)
    return confidence * decay_factor


def score_memory_trust(
    confidence: float,
    age_hours: float = 0.0,
    verified: bool = False,
    is_reversible: bool = True,
    half_life_hours: float = 168.0,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score trust for a Mem0 memory.

    Combines raw confidence with age decay and verification status.
    Verified memories get a confidence boost; old memories decay.

    Returns dict with trust evaluation + memory-specific metadata.
    """
    # Start with raw confidence
    effective = confidence

    # Apply age decay
    effective = _age_decay(effective, age_hours, half_life_hours)

    # Verification boost: verified memories get +0.1 (capped at 1.0)
    if verified:
        effective = min(1.0, effective + 0.1)

    result = evaluate_trust(
        effective,
        is_reversible=is_reversible,
        context={
            **(context or {}),
            "source": "mem0",
            "age_hours": age_hours,
            "verified": verified,
            "raw_confidence": confidence,
        },
    )

    return {
        "should_use": result.should_proceed,
        "tier": result.tier.name,
        "route": result.route,
        "effective_confidence": round(effective, 4),
        "raw_confidence": confidence,
        "age_hours": age_hours,
        "verified": verified,
        "decay_applied": age_hours > 0,
        "record_id": result.accountability_record.record_id if result.accountability_record else "",
    }


if __name__ == "__main__":
    # Fresh, high-confidence, verified memory
    fresh = score_memory_trust(confidence=0.9, age_hours=1, verified=True)
    print(f"Fresh verified: tier={fresh['tier']}, use={fresh['should_use']}, conf={fresh['effective_confidence']}")

    # Week-old, unverified memory
    stale = score_memory_trust(confidence=0.9, age_hours=168, verified=False)
    print(f"Week-old unverified: tier={stale['tier']}, use={stale['should_use']}, conf={stale['effective_confidence']}")

    # Month-old memory
    ancient = score_memory_trust(confidence=0.9, age_hours=720, verified=False)
    print(f"Month-old: tier={ancient['tier']}, use={ancient['should_use']}, conf={ancient['effective_confidence']}")
