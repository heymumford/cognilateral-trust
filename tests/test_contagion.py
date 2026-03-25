"""Tests for contagion tracker — low-trust claim propagation detection.

TDD — tests written before implementation. Covers:
- PropagationEntry: immutable record of claim propagation
- ContagionTracker: in-memory graph tracking low-confidence claims
- record_propagation: register agent receiving claim
- check_contagion: detect low-confidence propagation within threshold
- get_propagation_chain: retrieve full chain for a claim
- Low-confidence claim through 3 agents → flagged as contagion
- High-confidence claim through 3 agents → NOT flagged
- Hop counter resets on re-verification (FF-CD-01)
"""

from __future__ import annotations

import pytest

from cognilateral_trust.network.contagion import ContagionTracker, PropagationEntry


class TestPropagationEntry:
    """Immutable propagation record."""

    def test_create(self) -> None:
        """PropagationEntry is frozen."""
        entry = PropagationEntry(
            claim_id="claim-1",
            agent_id="agent-a",
            confidence=0.5,
            hop_number=1,
            timestamp="2026-03-25T10:00:00Z",
            re_verified=False,
        )
        assert entry.claim_id == "claim-1"
        assert entry.agent_id == "agent-a"
        assert entry.confidence == 0.5
        assert entry.hop_number == 1

    def test_frozen(self) -> None:
        """PropagationEntry cannot be mutated."""
        entry = PropagationEntry(
            claim_id="claim-1",
            agent_id="agent-a",
            confidence=0.5,
            hop_number=1,
            timestamp="2026-03-25T10:00:00Z",
            re_verified=False,
        )
        with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
            entry.confidence = 0.7  # type: ignore


class TestContagionTracker:
    """Track low-confidence claim propagation through agent networks."""

    def test_create(self) -> None:
        """ContagionTracker initializes empty."""
        tracker = ContagionTracker()
        assert len(tracker.get_propagation_chain("claim-1")) == 0

    def test_record_propagation(self) -> None:
        """Record a propagation event."""
        tracker = ContagionTracker()
        tracker.record_propagation("claim-1", "agent-a", 0.8, re_verified=False)
        chain = tracker.get_propagation_chain("claim-1")
        assert len(chain) == 1
        assert chain[0].claim_id == "claim-1"
        assert chain[0].agent_id == "agent-a"
        assert chain[0].confidence == 0.8
        assert chain[0].hop_number == 0

    def test_multi_hop_chain(self) -> None:
        """Record multiple hops for a single claim."""
        tracker = ContagionTracker()
        tracker.record_propagation("claim-1", "agent-a", 0.8, re_verified=False)
        tracker.record_propagation("claim-1", "agent-b", 0.6, re_verified=False)
        tracker.record_propagation("claim-1", "agent-c", 0.4, re_verified=False)
        chain = tracker.get_propagation_chain("claim-1")
        assert len(chain) == 3
        assert chain[0].hop_number == 0
        assert chain[1].hop_number == 1
        assert chain[2].hop_number == 2

    def test_low_confidence_flagged_within_threshold(self) -> None:
        """Claim with confidence 0.2 through 3 agents → flagged as contagion (FF-CD-01)."""
        tracker = ContagionTracker()
        tracker.record_propagation("claim-1", "agent-a", 0.2, re_verified=False)
        tracker.record_propagation("claim-1", "agent-b", 0.2, re_verified=False)
        tracker.record_propagation("claim-1", "agent-c", 0.2, re_verified=False)

        alert = tracker.check_contagion("claim-1", threshold=0.3, max_hops=2)
        assert alert is not None
        assert alert.claim_id == "claim-1"
        assert alert.min_confidence == 0.2
        assert alert.max_hops_detected >= 2

    def test_high_confidence_not_flagged(self) -> None:
        """Claim with confidence 0.9 through 3 agents → NOT flagged."""
        tracker = ContagionTracker()
        tracker.record_propagation("claim-1", "agent-a", 0.9, re_verified=False)
        tracker.record_propagation("claim-1", "agent-b", 0.9, re_verified=False)
        tracker.record_propagation("claim-1", "agent-c", 0.9, re_verified=False)

        alert = tracker.check_contagion("claim-1", threshold=0.3, max_hops=2)
        assert alert is None

    def test_re_verification_resets_hop_counter(self) -> None:
        """Re-verified claim resets hop counter for contagion detection."""
        tracker = ContagionTracker()
        tracker.record_propagation("claim-1", "agent-a", 0.2, re_verified=False)
        tracker.record_propagation("claim-1", "agent-b", 0.2, re_verified=False)
        # Re-verify at agent-b
        tracker.record_propagation("claim-1", "agent-b-revised", 0.8, re_verified=True)
        tracker.record_propagation("claim-1", "agent-c", 0.8, re_verified=False)

        # With max_hops=2, should NOT flag (hops reset after re-verify)
        alert = tracker.check_contagion("claim-1", threshold=0.3, max_hops=2)
        assert alert is None or alert.max_hops_detected < 2

    def test_mixed_confidence_levels(self) -> None:
        """Chain with mixed confidence levels — alert if any below threshold."""
        tracker = ContagionTracker()
        tracker.record_propagation("claim-1", "agent-a", 0.9, re_verified=False)
        tracker.record_propagation("claim-1", "agent-b", 0.2, re_verified=False)  # Low
        tracker.record_propagation("claim-1", "agent-c", 0.8, re_verified=False)

        alert = tracker.check_contagion("claim-1", threshold=0.3, max_hops=2)
        assert alert is not None
        assert alert.min_confidence == 0.2

    def test_nonexistent_claim_returns_none(self) -> None:
        """Checking a nonexistent claim returns None."""
        tracker = ContagionTracker()
        alert = tracker.check_contagion("nonexistent", threshold=0.3)
        assert alert is None

    def test_separate_claims_tracked_independently(self) -> None:
        """Multiple claims tracked separately."""
        tracker = ContagionTracker()
        tracker.record_propagation("claim-1", "agent-a", 0.2, re_verified=False)
        tracker.record_propagation("claim-2", "agent-x", 0.9, re_verified=False)

        chain1 = tracker.get_propagation_chain("claim-1")
        chain2 = tracker.get_propagation_chain("claim-2")

        assert len(chain1) == 1
        assert len(chain2) == 1
        assert chain1[0].agent_id == "agent-a"
        assert chain2[0].agent_id == "agent-x"
