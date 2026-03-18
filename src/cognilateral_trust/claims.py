"""S6: Claim Extraction — zero external dependencies.

Extracts typed claims from free-form text using regex and rule-based heuristics.
No NLP libraries required; optional spaCy/NLI integration belongs in the [verify] extra.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

ClaimType = Literal["factual", "causal", "comparative", "quantitative"]


@dataclass(frozen=True)
class Claim:
    """A single extracted claim with provenance back to the source span."""

    text: str
    source_text: str
    start_pos: int
    end_pos: int
    claim_type: ClaimType


@dataclass(frozen=True)
class ClaimSet:
    """Collection of claims extracted from a single source text."""

    claims: tuple[Claim, ...]
    source_text: str
    extraction_method: str


# ---------------------------------------------------------------------------
# Hedge patterns — sentences containing these signals are NOT factual claims
# ---------------------------------------------------------------------------

_HEDGE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bmight\b",
        r"\bcould\b",
        r"\bpossibly\b",
        r"\bperhaps\b",
        r"\bpotentially\b",
        r"\bmaybe\b",
        r"\bseems?\b",
        r"\bappears?\b",
        r"\bsuggests?\b",
        r"\bsome\s+cases\b",
        r"\bunder\s+certain\s+conditions\b",
    ]
)

# ---------------------------------------------------------------------------
# Causal markers
# ---------------------------------------------------------------------------

_CAUSAL_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bbecause\b",
        r"\btherefore\b",
        r"\bthus\b",
        r"\bhence\b",
        r"\bas\s+a\s+result\b",
        r"\bleads?\s+to\b",
        r"\bcauses?\b",
        r"\bresults?\s+in\b",
        r"\bdue\s+to\b",
        r"\bconsequently\b",
    ]
)

# ---------------------------------------------------------------------------
# Comparative markers
# ---------------------------------------------------------------------------

_COMPARATIVE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bbetter\s+than\b",
        r"\bfaster\s+than\b",
        r"\bslower\s+than\b",
        r"\bmore\s+\w+\s+than\b",
        r"\bless\s+\w+\s+than\b",
        r"\bworse\s+than\b",
        r"\blarger\s+than\b",
        r"\bsmaller\s+than\b",
        r"\bhigher\s+than\b",
        r"\blower\s+than\b",
        r"\bsuperior\s+to\b",
        r"\binferior\s+to\b",
    ]
)

# ---------------------------------------------------------------------------
# Quantitative markers
# ---------------------------------------------------------------------------

_QUANTITATIVE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\d+\s*%",
        r"\d+\s*percent\b",
        r"\$\s*\d+[\d,\.]*\s*[KMBTkbt]?\b",
        r"\b\d+[\d,\.]*\s*[KMBTkbt]\b(?!\s*%)",  # e.g., "5M", "2.3B"
        r"\b\d+\s+(?:million|billion|thousand|hundred)\b",
        r"\b(?:doubled|tripled|quadrupled)\b",
        r"\b\d+x\s+(?:faster|slower|larger|smaller|more|less)\b",
    ]
)

# ---------------------------------------------------------------------------
# Attribution markers
# ---------------------------------------------------------------------------

_ATTRIBUTION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\baccording\s+to\b",
        r"\bresearch\s+shows?\b",
        r"\bstudies?\s+show\b",
        r"\bdata\s+shows?\b",
        r"\breports?\s+(?:show|indicate|suggest)\b",
        r"\b\w+\s+said\b",
        r"\b\w+\s+stated\b",
        r"\b\w+\s+found\b",
        r"\banalysis\s+(?:shows?|indicates?)\b",
    ]
)

# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

_SENTENCE_RE = re.compile(r"(?<=[.!?;])\s+")


def _split_sentences(text: str) -> list[tuple[str, int]]:
    """Split text into (sentence, start_offset) pairs."""
    if not text.strip():
        return []
    sentences: list[tuple[str, int]] = []
    pos = 0
    for part in _SENTENCE_RE.split(text):
        start = text.index(part, pos)
        sentences.append((part.strip(), start))
        pos = start + len(part)
    return sentences


def _is_hedged(sentence: str) -> bool:
    return any(p.search(sentence) for p in _HEDGE_PATTERNS)


def _classify_sentence(sentence: str) -> ClaimType | None:
    """Return the dominant claim type for a sentence, or None if not a claim."""
    if _is_hedged(sentence):
        return None

    # Priority order: quantitative > causal > comparative > attribution > factual
    if any(p.search(sentence) for p in _QUANTITATIVE_PATTERNS):
        return "quantitative"
    if any(p.search(sentence) for p in _CAUSAL_PATTERNS):
        return "causal"
    if any(p.search(sentence) for p in _COMPARATIVE_PATTERNS):
        return "comparative"
    if any(p.search(sentence) for p in _ATTRIBUTION_PATTERNS):
        # Attribution implies a factual-ish claim but label it separately — use "factual"
        # because the spec uses 4 types; attribution is expressed as factual + definite language.
        return "factual"

    # Factual: definite "is/are/was/were" without hedges
    if re.search(r"\b(?:is|are|was|were|has|have|had|will|does|do)\b", sentence, re.IGNORECASE):
        return "factual"

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_claims(text: str) -> list[Claim]:
    """Extract typed claims from free-form text.

    Uses zero external dependencies. Applies sentence segmentation then
    classifies each sentence by its dominant claim signal. Hedged sentences
    are excluded from factual claims regardless of other signals.

    Args:
        text: Raw input text.

    Returns:
        List of Claim objects sorted by start_pos. Empty when text is blank.
    """
    if not text or not text.strip():
        return []

    claims: list[Claim] = []

    for sentence, offset in _split_sentences(text):
        if not sentence:
            continue
        claim_type = _classify_sentence(sentence)
        if claim_type is None:
            continue

        # Find exact position of this sentence in original text
        start = text.find(sentence, offset)
        if start == -1:
            start = offset
        end = start + len(sentence)

        claims.append(
            Claim(
                text=sentence,
                source_text=text,
                start_pos=start,
                end_pos=end,
                claim_type=claim_type,
            )
        )

    return sorted(claims, key=lambda c: c.start_pos)
