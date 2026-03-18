"""Tests for S7: Fidelity Verification (zero deps, word-overlap heuristic).

TDD — tests written before implementation.
"""

from __future__ import annotations

import pytest

from cognilateral_trust.fidelity import (
    FidelityResult,
    verify_fidelity,
    verify_fidelity_batch,
)


# ---------------------------------------------------------------------------
# Score bounds
# ---------------------------------------------------------------------------


def test_identical_text_scores_one() -> None:
    text = "The server response time increased by 200 milliseconds."
    result = verify_fidelity(text, text)
    assert result.score == pytest.approx(1.0, abs=1e-6)


def test_completely_unrelated_scores_near_zero() -> None:
    claim = "Quantum entanglement enables faster-than-light communication."
    source = "The weather forecast predicts heavy rainfall this weekend."
    result = verify_fidelity(claim, source)
    assert result.score < 0.15


def test_paraphrase_with_shared_keywords_scores_medium() -> None:
    claim = "The system latency increased significantly under load."
    source = "Under heavy traffic load, system latency grew significantly worse."
    result = verify_fidelity(claim, source)
    assert 0.3 < result.score < 0.9


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_claim_returns_zero() -> None:
    result = verify_fidelity("", "The source has meaningful content here.")
    assert result.score == 0.0


def test_empty_source_returns_zero() -> None:
    result = verify_fidelity("The claim is valid.", "")
    assert result.score == 0.0


def test_both_empty_returns_zero() -> None:
    result = verify_fidelity("", "")
    assert result.score == 0.0


def test_whitespace_only_treated_as_empty() -> None:
    result = verify_fidelity("   ", "   ")
    assert result.score == 0.0


# ---------------------------------------------------------------------------
# Stopword exclusion
# ---------------------------------------------------------------------------


def test_stopwords_correctly_excluded() -> None:
    # "the a an is are was were be been of in to for and or but not" — all stopwords
    # Two texts that share only stopwords should score near zero
    claim = "The a an is are."
    source = "The a an is are."
    # Same text but all stopwords — after exclusion token sets are empty
    result = verify_fidelity(claim, source)
    # When both token sets are empty after stopword removal, score = 0.0
    assert result.score == 0.0


def test_score_cap_at_one() -> None:
    text = "Performance improved by thirty percent in load testing."
    result = verify_fidelity(text, text)
    assert result.score <= 1.0


# ---------------------------------------------------------------------------
# Case insensitivity
# ---------------------------------------------------------------------------


def test_case_insensitive() -> None:
    claim = "Revenue INCREASED by 30 percent."
    source = "revenue increased by 30 percent."
    result = verify_fidelity(claim, source)
    assert result.score > 0.8


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------


def test_result_method_is_word_overlap() -> None:
    result = verify_fidelity("claim text here", "claim text here and more")
    assert result.method == "word_overlap"


def test_result_contains_claim_and_source_text() -> None:
    claim = "The API is fast."
    source = "The API performs well."
    result = verify_fidelity(claim, source)
    assert result.claim_text == claim
    assert result.source_text == source


def test_result_is_frozen() -> None:
    result = verify_fidelity("text", "text")
    with pytest.raises((AttributeError, TypeError)):
        result.score = 0.5  # type: ignore[misc]


def test_result_details_is_string() -> None:
    result = verify_fidelity("claim", "source")
    assert isinstance(result.details, str)


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------


def test_batch_processes_all_claims() -> None:
    source = "The system improved performance by 40% and reduced latency."
    claims = [
        "Performance improved by 40%.",
        "Latency was reduced.",
        "The weather is sunny today.",
    ]
    results = verify_fidelity_batch(claims, source)
    assert len(results) == 3


def test_batch_returns_fidelity_results() -> None:
    results = verify_fidelity_batch(["a", "b"], "a b c d")
    assert all(isinstance(r, FidelityResult) for r in results)


def test_batch_preserves_order() -> None:
    source = "alpha beta gamma delta"
    claims = ["alpha", "gamma", "beta"]
    results = verify_fidelity_batch(claims, source)
    assert results[0].claim_text == "alpha"
    assert results[1].claim_text == "gamma"
    assert results[2].claim_text == "beta"


def test_very_long_texts_do_not_crash() -> None:
    claim = "performance " * 1000
    source = "system performance improved " * 1000
    result = verify_fidelity(claim.strip(), source.strip())
    assert isinstance(result, FidelityResult)
    assert 0.0 <= result.score <= 1.0
