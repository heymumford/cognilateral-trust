"""Tests for S9: Sovereignty Gate — D-07 with welfare constraint.

TDD — tests written before implementation.

Critical invariant: welfare_affected=True + policy.welfare_critical=True
ALWAYS produces ESCALATE regardless of confidence or other gates.
This is the D-05 hard constraint — it must never be bypassed.
"""

from __future__ import annotations

import pytest

from cognilateral_trust.sovereignty import (
    DEFAULT_POLICY,
    SovereigntyPolicy,
    evaluate_sovereignty,
    sovereignty_gate,
)


# ---------------------------------------------------------------------------
# Happy path — ACT
# ---------------------------------------------------------------------------


def test_high_confidence_reversible_tests_pass_gives_act() -> None:
    result = evaluate_sovereignty(
        confidence=0.9,
        is_reversible=True,
        tests_passing=True,
        touches_external=False,
        welfare_affected=False,
    )
    assert result.verdict == "ACT"


def test_confidence_at_min_threshold_gives_act() -> None:
    # DEFAULT_POLICY.min_confidence must be <= 0.5 per spec
    result = evaluate_sovereignty(
        confidence=DEFAULT_POLICY.min_confidence,
        is_reversible=True,
        tests_passing=True,
        touches_external=False,
        welfare_affected=False,
    )
    assert result.verdict == "ACT"


# ---------------------------------------------------------------------------
# Confidence gate
# ---------------------------------------------------------------------------


def test_low_confidence_escalates() -> None:
    result = evaluate_sovereignty(
        confidence=0.1,
        is_reversible=True,
        tests_passing=True,
        touches_external=False,
        welfare_affected=False,
    )
    assert result.verdict == "ESCALATE"


def test_confidence_below_threshold_escalates() -> None:
    policy = SovereigntyPolicy(
        min_confidence=0.7,
        require_reversible_for_auto=True,
        require_tests_pass=True,
        welfare_critical=False,
        max_external_impact=0,
    )
    result = evaluate_sovereignty(
        confidence=0.5,
        policy=policy,
        is_reversible=True,
        tests_passing=True,
        touches_external=False,
        welfare_affected=False,
    )
    assert result.verdict == "ESCALATE"


# ---------------------------------------------------------------------------
# Reversibility gate
# ---------------------------------------------------------------------------


def test_irreversible_action_escalates_when_policy_requires() -> None:
    policy = SovereigntyPolicy(
        min_confidence=0.5,
        require_reversible_for_auto=True,
        require_tests_pass=True,
        welfare_critical=False,
        max_external_impact=0,
    )
    result = evaluate_sovereignty(
        confidence=0.9,
        policy=policy,
        is_reversible=False,
        tests_passing=True,
        touches_external=False,
        welfare_affected=False,
    )
    assert result.verdict == "ESCALATE"


def test_irreversible_allowed_when_policy_does_not_require_reversible() -> None:
    policy = SovereigntyPolicy(
        min_confidence=0.5,
        require_reversible_for_auto=False,
        require_tests_pass=True,
        welfare_critical=False,
        max_external_impact=0,
    )
    result = evaluate_sovereignty(
        confidence=0.9,
        policy=policy,
        is_reversible=False,
        tests_passing=True,
        touches_external=False,
        welfare_affected=False,
    )
    assert result.verdict == "ACT"


# ---------------------------------------------------------------------------
# Tests gate
# ---------------------------------------------------------------------------


def test_failing_tests_escalate() -> None:
    result = evaluate_sovereignty(
        confidence=0.9,
        is_reversible=True,
        tests_passing=False,
        touches_external=False,
        welfare_affected=False,
    )
    assert result.verdict == "ESCALATE"


def test_failing_tests_allowed_when_policy_does_not_require() -> None:
    policy = SovereigntyPolicy(
        min_confidence=0.5,
        require_reversible_for_auto=True,
        require_tests_pass=False,
        welfare_critical=False,
        max_external_impact=0,
    )
    result = evaluate_sovereignty(
        confidence=0.9,
        policy=policy,
        is_reversible=True,
        tests_passing=False,
        touches_external=False,
        welfare_affected=False,
    )
    assert result.verdict == "ACT"


# ---------------------------------------------------------------------------
# D-05 HARD GATE — welfare constraint (IMMUTABLE)
# ---------------------------------------------------------------------------


def test_welfare_affected_with_welfare_critical_policy_always_escalates() -> None:
    """Even at confidence=1.0, welfare gate MUST escalate. This is D-05."""
    policy = SovereigntyPolicy(
        min_confidence=0.5,
        require_reversible_for_auto=True,
        require_tests_pass=True,
        welfare_critical=True,  # D-05 active
        max_external_impact=0,
    )
    result = evaluate_sovereignty(
        confidence=1.0,  # Maximum possible confidence
        policy=policy,
        is_reversible=True,
        tests_passing=True,
        touches_external=False,
        welfare_affected=True,  # Welfare is affected
    )
    assert result.verdict == "ESCALATE"


def test_welfare_affected_with_non_critical_policy_can_act() -> None:
    policy = SovereigntyPolicy(
        min_confidence=0.5,
        require_reversible_for_auto=True,
        require_tests_pass=True,
        welfare_critical=False,  # D-05 not active
        max_external_impact=0,
    )
    result = evaluate_sovereignty(
        confidence=0.9,
        policy=policy,
        is_reversible=True,
        tests_passing=True,
        touches_external=False,
        welfare_affected=True,
    )
    assert result.verdict == "ACT"


def test_welfare_not_affected_does_not_trigger_welfare_gate() -> None:
    policy = SovereigntyPolicy(
        min_confidence=0.5,
        require_reversible_for_auto=True,
        require_tests_pass=True,
        welfare_critical=True,
        max_external_impact=0,
    )
    result = evaluate_sovereignty(
        confidence=0.9,
        policy=policy,
        is_reversible=True,
        tests_passing=True,
        touches_external=False,
        welfare_affected=False,  # No welfare impact
    )
    assert result.verdict == "ACT"


# ---------------------------------------------------------------------------
# External impact gate
# ---------------------------------------------------------------------------


def test_touches_external_escalates_when_max_zero() -> None:
    result = evaluate_sovereignty(
        confidence=0.9,
        is_reversible=True,
        tests_passing=True,
        touches_external=True,
        welfare_affected=False,
    )
    # DEFAULT_POLICY.max_external_impact should disallow external touches
    assert result.verdict == "ESCALATE"


# ---------------------------------------------------------------------------
# Decision structure
# ---------------------------------------------------------------------------


def test_decision_contains_all_check_results() -> None:
    result = evaluate_sovereignty(
        confidence=0.9,
        is_reversible=True,
        tests_passing=True,
        touches_external=False,
        welfare_affected=False,
    )
    assert "confidence" in result.checks_passed
    assert "reversible" in result.checks_passed
    assert "tests_passing" in result.checks_passed
    assert "welfare" in result.checks_passed
    assert "external_impact" in result.checks_passed


def test_decision_is_frozen() -> None:
    result = evaluate_sovereignty(0.9)
    with pytest.raises((AttributeError, TypeError)):
        result.verdict = "ACT"  # type: ignore[misc]


def test_reasons_is_tuple() -> None:
    result = evaluate_sovereignty(0.1)
    assert isinstance(result.reasons, tuple)


def test_default_policy_is_reasonable() -> None:
    # Reasonable defaults: not too restrictive (mid confidence should ACT on clean ops)
    result = evaluate_sovereignty(
        confidence=0.8,
        is_reversible=True,
        tests_passing=True,
        touches_external=False,
        welfare_affected=False,
    )
    assert result.verdict == "ACT"


# ---------------------------------------------------------------------------
# sovereignty_gate decorator
# ---------------------------------------------------------------------------


def test_decorator_allows_execution_on_act() -> None:
    call_log: list[str] = []

    @sovereignty_gate(confidence=0.9, is_reversible=True)
    def my_action() -> str:
        call_log.append("executed")
        return "done"

    result = my_action()
    assert result == "done"
    assert call_log == ["executed"]


def test_decorator_blocks_execution_on_escalate() -> None:
    from cognilateral_trust.sovereignty import SovereigntyError

    call_log: list[str] = []

    @sovereignty_gate(confidence=0.1)
    def dangerous_action() -> str:
        call_log.append("executed")
        return "done"

    with pytest.raises(SovereigntyError):
        dangerous_action()

    assert call_log == []
