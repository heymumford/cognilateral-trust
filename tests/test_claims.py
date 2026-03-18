"""Tests for S6: Claim Extraction (zero deps).

TDD — tests written before implementation.
All cases use pure regex/rule-based extraction; no external dependencies.
"""

from __future__ import annotations

import pytest

from cognilateral_trust.claims import (
    ClaimSet,
    extract_claims,
)


# ---------------------------------------------------------------------------
# Quantitative claim extraction
# ---------------------------------------------------------------------------


def test_extracts_percentage_claims() -> None:
    text = "Revenue increased by 30% over the previous quarter."
    claims = extract_claims(text)
    assert len(claims) >= 1
    assert any("30%" in c.text for c in claims)


def test_extracts_dollar_amount_claims() -> None:
    text = "The project costs $5M and will be delivered next year."
    claims = extract_claims(text)
    assert any("$5M" in c.text or "5M" in c.text for c in claims)


def test_quantitative_claim_type() -> None:
    text = "Performance improved by 45 percent."
    claims = extract_claims(text)
    quant = [c for c in claims if c.claim_type == "quantitative"]
    assert len(quant) >= 1


# ---------------------------------------------------------------------------
# Causal claim extraction
# ---------------------------------------------------------------------------


def test_extracts_because_causal() -> None:
    text = "The system failed because the memory limit was exceeded."
    claims = extract_claims(text)
    causal = [c for c in claims if c.claim_type == "causal"]
    assert len(causal) >= 1


def test_extracts_therefore_causal() -> None:
    text = "The tests all passed; therefore the build is safe to deploy."
    claims = extract_claims(text)
    causal = [c for c in claims if c.claim_type == "causal"]
    assert len(causal) >= 1


def test_extracts_leads_to_causal() -> None:
    text = "High latency leads to poor user experience."
    claims = extract_claims(text)
    causal = [c for c in claims if c.claim_type == "causal"]
    assert len(causal) >= 1


# ---------------------------------------------------------------------------
# Comparative claim extraction
# ---------------------------------------------------------------------------


def test_extracts_better_than_comparative() -> None:
    text = "The new algorithm is better than the previous one."
    claims = extract_claims(text)
    comparative = [c for c in claims if c.claim_type == "comparative"]
    assert len(comparative) >= 1


def test_extracts_faster_than_comparative() -> None:
    text = "This approach is faster than using a database lookup."
    claims = extract_claims(text)
    comparative = [c for c in claims if c.claim_type == "comparative"]
    assert len(comparative) >= 1


# ---------------------------------------------------------------------------
# Attribution claim extraction
# ---------------------------------------------------------------------------


def test_extracts_according_to_attribution() -> None:
    text = "According to the report, adoption rates have doubled."
    claims = extract_claims(text)
    assert len(claims) >= 1


def test_extracts_research_shows_attribution() -> None:
    text = "Research shows that sleep deprivation impairs judgment."
    claims = extract_claims(text)
    assert len(claims) >= 1


# ---------------------------------------------------------------------------
# Hedging — NOT extracted as claims
# ---------------------------------------------------------------------------


def test_hedged_language_not_extracted() -> None:
    text = "It might be possible that this could work under certain conditions."
    claims = extract_claims(text)
    # Fully hedged — should extract zero or only the weakest signal
    # Key invariant: no claim with claim_type "factual" from pure hedges
    factual = [c for c in claims if c.claim_type == "factual"]
    assert len(factual) == 0


def test_could_language_not_factual() -> None:
    text = "This could potentially lead to improvements in some cases."
    claims = extract_claims(text)
    factual = [c for c in claims if c.claim_type == "factual"]
    assert len(factual) == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_text_returns_empty_list() -> None:
    claims = extract_claims("")
    assert claims == []


def test_whitespace_only_returns_empty_list() -> None:
    claims = extract_claims("   \n  ")
    assert claims == []


def test_very_long_text_does_not_crash() -> None:
    # 10,000-character stress test — must complete without error
    long_text = ("The system increased throughput by 15%. " * 250) + ("Because of this, performance improved. " * 250)
    claims = extract_claims(long_text)
    assert isinstance(claims, list)


def test_claims_preserve_source_position() -> None:
    text = "Revenue increased by 30% this quarter."
    claims = extract_claims(text)
    assert len(claims) >= 1
    for c in claims:
        assert c.start_pos >= 0
        assert c.end_pos > c.start_pos
        assert c.end_pos <= len(text)
        # The claim text must match the span
        assert text[c.start_pos : c.end_pos] == c.text


# ---------------------------------------------------------------------------
# ClaimSet
# ---------------------------------------------------------------------------


def test_claim_set_wraps_claims() -> None:
    text = "The API is faster than its predecessor by 40%."
    result = extract_claims(text)
    claim_set = ClaimSet(claims=tuple(result), source_text=text, extraction_method="heuristic")
    assert claim_set.extraction_method == "heuristic"
    assert claim_set.source_text == text
    assert len(claim_set.claims) == len(result)


def test_claim_is_frozen() -> None:
    text = "Revenue increased by 30%."
    claims = extract_claims(text)
    assert len(claims) >= 1
    with pytest.raises((AttributeError, TypeError)):
        claims[0].text = "mutated"  # type: ignore[misc]


def test_claim_set_is_frozen() -> None:
    cs = ClaimSet(claims=(), source_text="x", extraction_method="heuristic")
    with pytest.raises((AttributeError, TypeError)):
        cs.extraction_method = "mutated"  # type: ignore[misc]
