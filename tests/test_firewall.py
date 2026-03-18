"""Tests for S8: Epistemic Firewall (demanded vs supplied level mismatch).

TDD — tests written before implementation.
"""

from __future__ import annotations

import pytest

from cognilateral_trust.firewall import (
    EpistemicFirewall,
    EpistemicLevel,
    check_epistemic_mismatch,
)


# ---------------------------------------------------------------------------
# EpistemicLevel enum values
# ---------------------------------------------------------------------------


def test_epistemic_level_ordering() -> None:
    assert EpistemicLevel.RAW < EpistemicLevel.OBSERVED
    assert EpistemicLevel.OBSERVED < EpistemicLevel.MEASURED
    assert EpistemicLevel.MEASURED < EpistemicLevel.TESTED
    assert EpistemicLevel.TESTED < EpistemicLevel.VALIDATED
    assert EpistemicLevel.VALIDATED < EpistemicLevel.FALSIFIABLE
    assert EpistemicLevel.FALSIFIABLE < EpistemicLevel.GOVERNANCE


def test_epistemic_level_int_values() -> None:
    assert EpistemicLevel.RAW == 0
    assert EpistemicLevel.GOVERNANCE == 6


# ---------------------------------------------------------------------------
# No mismatch — supplied >= demanded
# ---------------------------------------------------------------------------


def test_no_mismatch_when_supplied_equals_demanded() -> None:
    result = check_epistemic_mismatch(EpistemicLevel.TESTED, EpistemicLevel.TESTED)
    assert result.should_escalate is False
    assert result.gap == 0


def test_no_mismatch_when_supplied_exceeds_demanded() -> None:
    result = check_epistemic_mismatch(EpistemicLevel.MEASURED, EpistemicLevel.VALIDATED)
    assert result.should_escalate is False
    assert result.gap == 0


# ---------------------------------------------------------------------------
# Small gap — proceed with warning
# ---------------------------------------------------------------------------


def test_gap_of_one_does_not_escalate_by_default() -> None:
    result = check_epistemic_mismatch(EpistemicLevel.TESTED, EpistemicLevel.MEASURED)
    # gap = 3 - 2 = 1, max_gap default = 2
    assert result.gap == 1
    assert result.should_escalate is False


def test_gap_of_two_does_not_escalate_by_default() -> None:
    result = check_epistemic_mismatch(EpistemicLevel.VALIDATED, EpistemicLevel.MEASURED)
    # gap = 4 - 2 = 2, max_gap default = 2
    assert result.gap == 2
    assert result.should_escalate is False


# ---------------------------------------------------------------------------
# Large gap — escalate
# ---------------------------------------------------------------------------


def test_gap_of_three_escalates() -> None:
    result = check_epistemic_mismatch(EpistemicLevel.GOVERNANCE, EpistemicLevel.OBSERVED)
    # gap = 6 - 1 = 5 > 2
    assert result.should_escalate is True
    assert result.gap == 5


def test_gap_of_three_exact_escalates() -> None:
    result = check_epistemic_mismatch(EpistemicLevel.TESTED, EpistemicLevel.RAW)
    # gap = 3 - 0 = 3 > max_gap=2
    assert result.should_escalate is True
    assert result.gap == 3


def test_escalation_reason_is_non_empty() -> None:
    result = check_epistemic_mismatch(EpistemicLevel.GOVERNANCE, EpistemicLevel.RAW)
    assert result.should_escalate is True
    assert len(result.reason) > 0


# ---------------------------------------------------------------------------
# Custom max_gap
# ---------------------------------------------------------------------------


def test_custom_max_gap_zero_escalates_on_any_gap() -> None:
    result = check_epistemic_mismatch(EpistemicLevel.TESTED, EpistemicLevel.MEASURED, max_gap=0)
    # gap = 1 > 0
    assert result.should_escalate is True


def test_custom_max_gap_large_allows_big_gaps() -> None:
    result = check_epistemic_mismatch(EpistemicLevel.GOVERNANCE, EpistemicLevel.RAW, max_gap=6)
    # gap = 6 <= 6
    assert result.should_escalate is False


# ---------------------------------------------------------------------------
# Int inputs
# ---------------------------------------------------------------------------


def test_accepts_int_demanded() -> None:
    result = check_epistemic_mismatch(3, EpistemicLevel.MEASURED)
    assert result.demanded == EpistemicLevel.TESTED
    assert result.gap == 1


def test_accepts_int_supplied() -> None:
    result = check_epistemic_mismatch(EpistemicLevel.TESTED, 2)
    assert result.supplied == EpistemicLevel.MEASURED
    assert result.gap == 1


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------


def test_result_demanded_and_supplied_are_epistemic_levels() -> None:
    result = check_epistemic_mismatch(EpistemicLevel.VALIDATED, EpistemicLevel.OBSERVED)
    assert isinstance(result.demanded, EpistemicLevel)
    assert isinstance(result.supplied, EpistemicLevel)


def test_result_is_frozen() -> None:
    result = check_epistemic_mismatch(EpistemicLevel.TESTED, EpistemicLevel.TESTED)
    with pytest.raises((AttributeError, TypeError)):
        result.gap = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# EpistemicFirewall — registry
# ---------------------------------------------------------------------------


def test_firewall_registers_action_requirement() -> None:
    fw = EpistemicFirewall()
    fw.require("deploy", EpistemicLevel.TESTED)
    result = fw.check("deploy", EpistemicLevel.TESTED)
    assert result.should_escalate is False


def test_firewall_escalates_when_level_below_required() -> None:
    fw = EpistemicFirewall()
    fw.require("deploy", EpistemicLevel.VALIDATED)
    result = fw.check("deploy", EpistemicLevel.RAW)
    assert result.should_escalate is True


def test_firewall_unknown_action_defaults_to_highest_level() -> None:
    fw = EpistemicFirewall()
    # No registration for "unknown_action" — defaults to GOVERNANCE (6)
    result = fw.check("unknown_action", EpistemicLevel.OBSERVED)
    # GOVERNANCE (6) demanded vs OBSERVED (1) supplied — gap = 5 > 2
    assert result.should_escalate is True


def test_firewall_allows_when_supplied_exceeds_requirement() -> None:
    fw = EpistemicFirewall()
    fw.require("publish", EpistemicLevel.MEASURED)
    result = fw.check("publish", EpistemicLevel.GOVERNANCE)
    assert result.should_escalate is False
