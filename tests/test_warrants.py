"""Sprint S2: Tests for cognilateral_trust.warrants module.

Covers Warrant lifecycle, decay calculation, WarrantStore add/get/expire,
and trust evaluation with warrant freshness.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from cognilateral_trust.warrants import (
    Warrant,
    WarrantStore,
    evaluate_trust_with_warrant,
    evaluate_warrant,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _make_warrant(
    confidence: float = 0.85,
    *,
    age_seconds: float = 0.0,
    ttl_seconds: float = 3600.0,
    decay_rate: float = 0.0,
    evidence_source: str = "test-source",
) -> Warrant:
    """Helper — build a Warrant with controlled timestamps."""
    issued_at = _now() - timedelta(seconds=age_seconds)
    expires_at = issued_at + timedelta(seconds=ttl_seconds)
    return Warrant(
        evidence_source=evidence_source,
        issued_at=issued_at,
        expires_at=expires_at,
        confidence=confidence,
        decay_rate=decay_rate,
    )


# ---------------------------------------------------------------------------
# Warrant dataclass
# ---------------------------------------------------------------------------


class TestWarrantDataclass:
    def test_is_frozen(self) -> None:
        w = _make_warrant()
        with pytest.raises((AttributeError, TypeError)):
            w.confidence = 0.5  # type: ignore[misc]

    def test_fields_present(self) -> None:
        w = _make_warrant(confidence=0.7, evidence_source="unit-test")
        assert w.confidence == pytest.approx(0.7)
        assert w.evidence_source == "unit-test"
        assert isinstance(w.issued_at, datetime)
        assert isinstance(w.expires_at, datetime)
        assert w.expires_at > w.issued_at

    def test_decay_rate_default_zero(self) -> None:
        w = _make_warrant()
        assert w.decay_rate == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# evaluate_warrant — confidence decay over time
# ---------------------------------------------------------------------------


class TestEvaluateWarrant:
    def test_fresh_warrant_returns_full_confidence(self) -> None:
        w = _make_warrant(confidence=0.9, age_seconds=0.0, ttl_seconds=3600.0)
        result = evaluate_warrant(w)
        assert result == pytest.approx(0.9, abs=0.01)

    def test_expired_warrant_returns_zero(self) -> None:
        w = _make_warrant(confidence=0.9, age_seconds=7201.0, ttl_seconds=3600.0)
        assert evaluate_warrant(w) == pytest.approx(0.0)

    def test_warrant_at_exact_expiry_returns_zero(self) -> None:
        # expires_at == now (within tolerance)
        now = _now()
        w = Warrant(
            evidence_source="test",
            issued_at=now - timedelta(hours=1),
            expires_at=now - timedelta(milliseconds=1),
            confidence=0.8,
            decay_rate=0.0,
        )
        assert evaluate_warrant(w) == pytest.approx(0.0)

    def test_no_decay_rate_returns_original_confidence_when_valid(self) -> None:
        w = _make_warrant(confidence=0.75, decay_rate=0.0, age_seconds=1800.0, ttl_seconds=3600.0)
        result = evaluate_warrant(w)
        assert result == pytest.approx(0.75, abs=0.01)

    def test_positive_decay_reduces_confidence_over_time(self) -> None:
        """decay_rate > 0 means confidence degrades linearly with elapsed fraction."""
        w = _make_warrant(confidence=0.8, decay_rate=0.5, age_seconds=1800.0, ttl_seconds=3600.0)
        result = evaluate_warrant(w)
        # 50% through TTL with decay_rate=0.5 → 0.8 * (1 - 0.5*0.5) = 0.8 * 0.75 = 0.60
        assert result < 0.8
        assert result > 0.0

    def test_full_decay_rate_at_half_life_drops_confidence(self) -> None:
        """decay_rate=1.0 means at 100% elapsed the confidence hits 0."""
        w = _make_warrant(confidence=0.9, decay_rate=1.0, age_seconds=3600.0, ttl_seconds=3600.0)
        # At expiry with full decay, result should be 0 (or very close)
        result = evaluate_warrant(w)
        assert result == pytest.approx(0.0, abs=0.01)

    def test_result_always_non_negative(self) -> None:
        w = _make_warrant(confidence=0.5, decay_rate=2.0, age_seconds=3500.0, ttl_seconds=3600.0)
        assert evaluate_warrant(w) >= 0.0

    def test_result_never_exceeds_original_confidence(self) -> None:
        w = _make_warrant(confidence=0.6, decay_rate=0.1, age_seconds=100.0, ttl_seconds=3600.0)
        assert evaluate_warrant(w) <= 0.6 + 1e-9


# ---------------------------------------------------------------------------
# WarrantStore — add / get / expire
# ---------------------------------------------------------------------------


class TestWarrantStore:
    def test_add_and_retrieve(self) -> None:
        store = WarrantStore()
        w = _make_warrant()
        wid = store.add(w)
        assert isinstance(wid, str)
        assert len(wid) > 0
        retrieved = store.get(wid)
        assert retrieved is not None
        assert retrieved.confidence == w.confidence

    def test_get_missing_returns_none(self) -> None:
        store = WarrantStore()
        assert store.get("nonexistent-id") is None

    def test_expired_warrant_auto_removed_on_get(self) -> None:
        store = WarrantStore()
        expired = _make_warrant(age_seconds=7200.0, ttl_seconds=3600.0)
        wid = store.add(expired)
        # Expired warrant should not be returned
        result = store.get(wid)
        assert result is None

    def test_active_warrant_returned_on_get(self) -> None:
        store = WarrantStore()
        fresh = _make_warrant(age_seconds=10.0, ttl_seconds=3600.0)
        wid = store.add(fresh)
        result = store.get(wid)
        assert result is not None

    def test_expire_all_stale(self) -> None:
        store = WarrantStore()
        expired1 = _make_warrant(age_seconds=7200.0, ttl_seconds=3600.0)
        expired2 = _make_warrant(age_seconds=9000.0, ttl_seconds=3600.0)
        fresh = _make_warrant(age_seconds=5.0, ttl_seconds=3600.0)
        store.add(expired1)
        store.add(expired2)
        wid_fresh = store.add(fresh)
        store.expire_stale()
        assert store.get(wid_fresh) is not None
        assert store.active_count() == 1

    def test_active_count_empty(self) -> None:
        store = WarrantStore()
        assert store.active_count() == 0

    def test_active_count_after_additions(self) -> None:
        store = WarrantStore()
        store.add(_make_warrant())
        store.add(_make_warrant())
        assert store.active_count() == 2

    def test_active_count_excludes_expired(self) -> None:
        store = WarrantStore()
        store.add(_make_warrant(age_seconds=7200.0, ttl_seconds=3600.0))
        store.add(_make_warrant(age_seconds=10.0, ttl_seconds=3600.0))
        store.expire_stale()
        assert store.active_count() == 1

    def test_add_multiple_warrants_distinct_ids(self) -> None:
        store = WarrantStore()
        ids = {store.add(_make_warrant()) for _ in range(5)}
        assert len(ids) == 5

    def test_revoke_removes_warrant(self) -> None:
        store = WarrantStore()
        wid = store.add(_make_warrant())
        store.revoke(wid)
        assert store.get(wid) is None

    def test_revoke_nonexistent_is_noop(self) -> None:
        store = WarrantStore()
        store.revoke("ghost-id")  # should not raise


# ---------------------------------------------------------------------------
# evaluate_trust_with_warrant
# ---------------------------------------------------------------------------


class TestEvaluateTrustWithWarrant:
    def test_fresh_high_confidence_warrant_proceeds(self) -> None:
        w = _make_warrant(confidence=0.9)
        result = evaluate_trust_with_warrant(w)
        assert result.should_proceed is True

    def test_expired_warrant_escalates(self) -> None:
        w = _make_warrant(confidence=0.9, age_seconds=7200.0, ttl_seconds=3600.0)
        result = evaluate_trust_with_warrant(w)
        assert result.should_proceed is False

    def test_low_decayed_confidence_routes_lower_tier(self) -> None:
        """Heavy decay collapses confidence → lower tier routing."""
        w = _make_warrant(confidence=0.95, decay_rate=1.0, age_seconds=3500.0, ttl_seconds=3600.0)
        result = evaluate_trust_with_warrant(w)
        # Near expiry with full decay → effective confidence near zero → low tier
        from cognilateral_trust.core import ConfidenceTier

        assert result.tier.value <= ConfidenceTier.C3.value

    def test_warrant_confidence_used_not_raw_confidence(self) -> None:
        """The effective decayed confidence drives routing, not the warrant's original."""
        no_decay = _make_warrant(confidence=0.5, decay_rate=0.0)
        heavy_decay = _make_warrant(confidence=0.95, decay_rate=1.0, age_seconds=3500.0, ttl_seconds=3600.0)
        result_no_decay = evaluate_trust_with_warrant(no_decay)
        result_heavy = evaluate_trust_with_warrant(heavy_decay)
        # No decay 0.5 should have higher effective confidence than decayed 0.95-near-zero
        assert result_no_decay.confidence >= result_heavy.confidence

    def test_result_has_accountability_record(self) -> None:
        w = _make_warrant(confidence=0.8)
        result = evaluate_trust_with_warrant(w)
        assert result.accountability_record is not None
        assert result.accountability_record.record_id

    def test_kwargs_forwarded_to_evaluate_trust(self) -> None:
        """is_reversible=False on sovereignty-grade tier should escalate."""
        w = _make_warrant(confidence=0.95)
        result = evaluate_trust_with_warrant(w, is_reversible=False)
        # C9 tier, irreversible → should escalate
        assert result.should_proceed is False

    def test_context_forwarded(self) -> None:
        w = _make_warrant(confidence=0.7)
        result = evaluate_trust_with_warrant(w, context={"task": "deploy"})
        assert result.accountability_record is not None
