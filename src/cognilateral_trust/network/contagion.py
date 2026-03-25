"""Contagion tracker — detect low-trust claim propagation through agent networks.

In-memory graph tracking how low-confidence claims spread across agent networks.
Flags propagation when confidence falls below threshold within specified hop limit.

Re-verification resets the hop counter, allowing legitimate correction chains.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class PropagationEntry:
    """Immutable record of a claim propagation event."""

    claim_id: str
    agent_id: str
    confidence: float
    hop_number: int
    timestamp: str  # ISO 8601
    re_verified: bool


@dataclass(frozen=True)
class ContagionAlert:
    """Alert indicating potential low-confidence claim contagion."""

    claim_id: str
    min_confidence: float
    max_hops_detected: int
    threshold: float
    message: str


class ContagionTracker:
    """In-memory graph tracking low-confidence claim propagation."""

    def __init__(self) -> None:
        """Initialize empty tracker."""
        self._propagations: dict[str, list[PropagationEntry]] = {}

    def record_propagation(
        self,
        claim_id: str,
        agent_id: str,
        confidence: float,
        re_verified: bool = False,
    ) -> None:
        """Record a propagation event.

        Args:
            claim_id: Unique claim identifier
            agent_id: Agent receiving or re-verifying the claim
            confidence: Confidence at this hop
            re_verified: Whether this hop re-verified the claim (resets hop counter)
        """
        if claim_id not in self._propagations:
            self._propagations[claim_id] = []

        chain = self._propagations[claim_id]

        # Generate ISO 8601 timestamp
        timestamp = datetime.now(timezone.utc).isoformat()

        # If re-verified, reset hop counter
        if re_verified:
            hop_number = 0
        else:
            # Count hops since last re-verification
            hop_number = 0
            for entry in reversed(chain):
                if entry.re_verified:
                    break
                hop_number += 1

        entry = PropagationEntry(
            claim_id=claim_id,
            agent_id=agent_id,
            confidence=confidence,
            hop_number=hop_number,
            timestamp=timestamp,
            re_verified=re_verified,
        )
        chain.append(entry)

    def check_contagion(
        self,
        claim_id: str,
        threshold: float = 0.3,
        max_hops: int = 2,
    ) -> ContagionAlert | None:
        """Check if claim exhibits contagion pattern.

        A claim is flagged if:
        1. Any propagation event has confidence < threshold
        2. The low-confidence hop is within max_hops of the start

        Args:
            claim_id: Claim to check
            threshold: Minimum acceptable confidence
            max_hops: Maximum hops before low-confidence is flagged

        Returns:
            ContagionAlert if contagion detected, else None
        """
        if claim_id not in self._propagations:
            return None

        chain = self._propagations[claim_id]
        if not chain:
            return None

        min_confidence = min(e.confidence for e in chain)
        if min_confidence >= threshold:
            return None

        # Find the hop number of the lowest confidence entry
        # Reset hops after re-verification
        effective_hops = 0
        max_hops_at_low_point = 0
        for entry in chain:
            if entry.re_verified:
                effective_hops = 0
            else:
                effective_hops = entry.hop_number

            if entry.confidence < threshold:
                max_hops_at_low_point = max(max_hops_at_low_point, effective_hops)

        if max_hops_at_low_point <= max_hops:
            msg = (
                f"Claim {claim_id} shows low-confidence propagation "
                f"(min: {min_confidence:.2f}) at or before {max_hops} hops (detected at hop {max_hops_at_low_point})"
            )
            return ContagionAlert(
                claim_id=claim_id,
                min_confidence=min_confidence,
                max_hops_detected=max_hops_at_low_point,
                threshold=threshold,
                message=msg,
            )

        return None

    def get_propagation_chain(self, claim_id: str) -> tuple[PropagationEntry, ...]:
        """Retrieve full propagation chain for a claim.

        Args:
            claim_id: Claim to retrieve

        Returns:
            Tuple of propagation entries in chronological order
        """
        if claim_id not in self._propagations:
            return ()
        return tuple(self._propagations[claim_id])
