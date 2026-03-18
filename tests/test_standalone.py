"""Standalone tests for cognilateral-trust package.

These tests verify the package works independently of the main cognilateral repo.
"""

from __future__ import annotations

import pytest

from cognilateral_trust import (
    ROUTE_BASIC,
    ROUTE_SOVEREIGNTY_GATE,
    ROUTE_WARRANT_CHECK,
    ConfidenceTier,
    TrustEvaluation,
    evaluate_tier_routing,
    evaluate_trust,
    route_by_tier,
)
from cognilateral_trust.accountability import (
    AccountabilityRecord,
    AccountabilityStore,
    create_accountability_record,
)


class TestCoreRouting:
    def test_c0_basic(self) -> None:
        assert route_by_tier(0) == ROUTE_BASIC

    def test_c5_warrant(self) -> None:
        assert route_by_tier(5) == ROUTE_WARRANT_CHECK

    def test_c9_sovereignty(self) -> None:
        assert route_by_tier(9) == ROUTE_SOVEREIGNTY_GATE

    def test_invalid_raises(self) -> None:
        with pytest.raises(ValueError):
            route_by_tier(10)

    def test_evaluate_returns_result(self) -> None:
        result = evaluate_tier_routing(5)
        assert result.tier == 5
        assert result.tier_name == "C5"


class TestEvaluateTrust:
    def test_high_confidence_proceeds(self) -> None:
        result = evaluate_trust(0.9)
        assert result.should_proceed is True
        assert result.tier == ConfidenceTier.C9

    def test_irreversible_escalates(self) -> None:
        result = evaluate_trust(0.9, is_reversible=False)
        assert result.should_proceed is False

    def test_record_created(self) -> None:
        result = evaluate_trust(0.5)
        assert result.accountability_record is not None
        assert isinstance(result.accountability_record, AccountabilityRecord)


class TestAccountability:
    def test_record_frozen(self) -> None:
        record = create_accountability_record(verdict="ACT", confidence=0.8)
        with pytest.raises(AttributeError):
            record.verdict = "ESCALATE"  # type: ignore[misc]

    def test_store_crud(self) -> None:
        store = AccountabilityStore()
        record = create_accountability_record(verdict="ACT", confidence=0.9)
        store.append(record)
        assert store.get(record.record_id) is not None
        assert len(store.list_recent()) == 1
