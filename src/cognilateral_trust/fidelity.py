"""S7: Fidelity Verification — zero external dependencies.

Verifies that a claim is supported by a source text using a word-overlap
heuristic (Jaccard/overlap coefficient). Optional NLI integration belongs
in the [verify] extra but is not required for this module.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

# ---------------------------------------------------------------------------
# Stopwords — small hardcoded set per spec
# ---------------------------------------------------------------------------

_STOPWORDS: frozenset[str] = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "of",
        "in",
        "to",
        "for",
        "and",
        "or",
        "but",
        "not",
        "it",
        "its",
        "by",
        "at",
        "on",
        "with",
        "this",
        "that",
        "from",
    }
)

# Tokenize: split on whitespace and punctuation, keep word characters only
_TOKEN_RE = re.compile(r"\w+")

FidelityMethod = Literal["word_overlap", "nli"]


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FidelityResult:
    """Result of a fidelity check between a claim and a source text."""

    score: float
    method: FidelityMethod
    claim_text: str
    source_text: str
    details: str


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> frozenset[str]:
    """Lowercase tokens with stopwords removed."""
    tokens = {t.lower() for t in _TOKEN_RE.findall(text)}
    return frozenset(tokens - _STOPWORDS)


def _overlap_coefficient(a: frozenset[str], b: frozenset[str]) -> float:
    """Overlap coefficient: |A ∩ B| / min(|A|, |B|).

    Returns 0.0 when either set is empty to avoid ZeroDivisionError.
    Capped at 1.0.
    """
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    denominator = min(len(a), len(b))
    return min(1.0, intersection / denominator)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def verify_fidelity(claim: str, source: str) -> FidelityResult:
    """Verify that a claim is supported by the source text.

    Uses a word-overlap (overlap coefficient) heuristic. Stopwords are
    excluded before comparison. Comparison is case-insensitive.

    Args:
        claim: The claim text to verify.
        source: The source text to verify against.

    Returns:
        FidelityResult with score in [0.0, 1.0], method="word_overlap".
    """
    if not claim or not claim.strip() or not source or not source.strip():
        return FidelityResult(
            score=0.0,
            method="word_overlap",
            claim_text=claim,
            source_text=source,
            details="Empty input — score is 0.0",
        )

    claim_tokens = _tokenize(claim)
    source_tokens = _tokenize(source)

    if not claim_tokens or not source_tokens:
        return FidelityResult(
            score=0.0,
            method="word_overlap",
            claim_text=claim,
            source_text=source,
            details="All tokens are stopwords — score is 0.0",
        )

    score = _overlap_coefficient(claim_tokens, source_tokens)
    shared = claim_tokens & source_tokens
    details = (
        f"claim_tokens={len(claim_tokens)}, source_tokens={len(source_tokens)}, "
        f"shared={len(shared)}, overlap_coefficient={score:.4f}"
    )

    return FidelityResult(
        score=score,
        method="word_overlap",
        claim_text=claim,
        source_text=source,
        details=details,
    )


def verify_fidelity_batch(claims: list[str], source: str) -> list[FidelityResult]:
    """Verify fidelity for multiple claims against a single source text.

    Preserves claim order. Each claim is verified independently.

    Args:
        claims: List of claim texts.
        source: The source text to verify against.

    Returns:
        List of FidelityResult in the same order as claims.
    """
    return [verify_fidelity(claim, source) for claim in claims]
