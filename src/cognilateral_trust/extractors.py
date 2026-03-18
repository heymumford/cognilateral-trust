"""Confidence extraction from LLM responses — zero dependencies.

Parses confidence signals from:
- Plain text (regex patterns, verbal tiers)
- OpenAI-format response dicts (content text + logprobs)
- Anthropic Messages API response dicts (content blocks)

All functions return float | None — None means "no signal found".
Callers decide the fallback (e.g. 0.5 neutral, or raise, or skip).

Public API:
    extract_confidence_from_text(text) -> float | None
    extract_confidence_from_openai_response(response) -> float | None
    extract_confidence_from_anthropic_response(response) -> float | None
    extract_confidence(response, *, source="auto") -> float | None
"""

from __future__ import annotations

import math
import re
from typing import Any

__all__ = [
    "extract_confidence",
    "extract_confidence_from_anthropic_response",
    "extract_confidence_from_openai_response",
    "extract_confidence_from_text",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Verbal confidence tiers → midpoint float values
_VERBAL_MAP: dict[str, float] = {
    "very high": 0.90,
    "very confident": 0.90,
    "highly confident": 0.90,
    "quite confident": 0.85,
    "fairly confident": 0.75,
    "high": 0.80,
    "moderate": 0.55,
    "medium": 0.55,
    "somewhat confident": 0.55,
    "low": 0.25,
    "low confidence": 0.25,
    "not confident": 0.20,
    "uncertain": 0.25,
    "quite uncertain": 0.20,
    "very uncertain": 0.15,
    "unsure": 0.25,
    "guess": 0.20,
    "guessing": 0.20,
}

# Regex: explicit numeric label — "Confidence: 0.85", "certainty: .91", "confidence_score: 0.7"
_NUMERIC_LABEL_RE = re.compile(
    r"(?i)(?:confidence[_\s]*(?:score)?|certainty)\s*[:\s]\s*(-?[0-9]*\.?[0-9]+)",
    re.IGNORECASE,
)

# Regex: percentage + confidence word — "85% confident", "about 70% certain"
_PCT_CONFIDENT_RE = re.compile(
    r"(\d{1,3}(?:\.\d+)?)%\s*(?:confident|certain|sure)",
    re.IGNORECASE,
)

# Regex: "I'm confident" / "high confidence" verbal phrase
_VERBAL_PHRASE_RE = re.compile(
    r"(?i)\b(" + "|".join(re.escape(k) for k in sorted(_VERBAL_MAP, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

# Truncation penalty for Anthropic stop_reason="max_tokens"
_MAX_TOKENS_PENALTY = 0.10


# ---------------------------------------------------------------------------
# Core: text extraction
# ---------------------------------------------------------------------------


def _clip(value: float) -> float:
    """Clip a confidence value to [0.0, 1.0]."""
    return max(0.0, min(1.0, value))


def extract_confidence_from_text(text: str) -> float | None:
    """Extract a confidence value from plain text.

    Recognises:
    - Numeric labels: "Confidence: 0.85", "certainty: .91", "confidence_score: 0.7"
    - Percentage + keyword: "85% confident", "70% certain"
    - Verbal tiers: "high confidence", "I'm very confident", "uncertain"

    Returns None when no signal is found.
    Values are clipped to [0.0, 1.0].
    """
    if not text:
        return None

    # 1. Numeric label (highest specificity)
    m = _NUMERIC_LABEL_RE.search(text)
    if m:
        val = float(m.group(1))
        # Values like 85 (without %) after label are treated as decimals, not percent
        return _clip(val)

    # 2. Percentage + confidence keyword
    m = _PCT_CONFIDENT_RE.search(text)
    if m:
        val = float(m.group(1)) / 100.0
        return _clip(val)

    # 3. Verbal tiers — scan for longest matching phrase
    for phrase_match in _VERBAL_PHRASE_RE.finditer(text.lower()):
        phrase = phrase_match.group(1).lower()
        if phrase in _VERBAL_MAP:
            return _VERBAL_MAP[phrase]

    return None


# ---------------------------------------------------------------------------
# OpenAI-format response dict
# ---------------------------------------------------------------------------


def _extract_from_logprobs(logprobs_content: list[dict[str, Any]]) -> float | None:
    """Convert OpenAI logprob entries to a confidence estimate.

    Takes the mean of exp(logprob) across all token entries.
    Returns None if the list is empty.
    """
    if not logprobs_content:
        return None
    probs = []
    for entry in logprobs_content:
        lp = entry.get("logprob")
        if lp is not None:
            try:
                probs.append(math.exp(float(lp)))
            except (ValueError, OverflowError):
                pass
    if not probs:
        return None
    return _clip(sum(probs) / len(probs))


def extract_confidence_from_openai_response(response: dict[str, Any]) -> float | None:
    """Extract confidence from an OpenAI chat completion response dict.

    Priority:
    1. Logprobs (mean token probability across all tokens)
    2. Text patterns in message content

    Accepts the raw dict format returned by the OpenAI API (not the SDK object).
    Returns None if no confidence signal is found.
    """
    choices = response.get("choices")
    if not choices:
        return None

    first = choices[0]

    # 1. Logprobs — if present, most reliable signal
    logprobs_data = first.get("logprobs") or {}
    logprobs_content = logprobs_data.get("content") if isinstance(logprobs_data, dict) else None
    if logprobs_content:
        lp_conf = _extract_from_logprobs(logprobs_content)
        if lp_conf is not None:
            return lp_conf

    # 2. Fall back to text extraction
    message = first.get("message") or {}
    content = message.get("content") or ""
    if content:
        return extract_confidence_from_text(content)

    return None


# ---------------------------------------------------------------------------
# Anthropic-format response dict
# ---------------------------------------------------------------------------


def extract_confidence_from_anthropic_response(response: dict[str, Any]) -> float | None:
    """Extract confidence from an Anthropic Messages API response dict.

    Reads the first text content block and applies text extraction.
    Applies a small penalty when stop_reason is "max_tokens" (truncated response).

    Accepts the raw dict format; does not depend on the anthropic SDK.
    Returns None if no confidence signal is found.
    """
    content_blocks = response.get("content")
    if not content_blocks:
        return None

    # Find first text block
    text = None
    for block in content_blocks:
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text") or ""
            break

    if not text:
        return None

    conf = extract_confidence_from_text(text)
    if conf is None:
        return None

    # Truncation penalty
    stop_reason = response.get("stop_reason", "")
    if stop_reason == "max_tokens":
        conf = _clip(conf - _MAX_TOKENS_PENALTY)

    return conf


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------


def extract_confidence(response: Any, *, source: str = "auto") -> float | None:
    """Unified confidence extraction — auto-detects response format.

    Args:
        response: str, OpenAI response dict, or Anthropic response dict.
        source: "auto" | "text" | "openai" | "anthropic"
                "auto" inspects the response shape to choose the extractor.

    Returns float in [0.0, 1.0] or None if no signal is found.

    Usage:
        # Plain text
        extract_confidence("I'm 85% confident.")          # → 0.85

        # OpenAI response dict (no SDK import needed)
        extract_confidence(openai_response_dict)          # → auto-detected

        # Anthropic response dict
        extract_confidence(anthropic_response_dict)       # → auto-detected

        # Explicit source
        extract_confidence(resp, source="anthropic")
    """
    if response is None:
        return None

    if source == "text":
        if not isinstance(response, str):
            return None
        return extract_confidence_from_text(response)

    if source == "openai":
        if not isinstance(response, dict):
            return None
        return extract_confidence_from_openai_response(response)

    if source == "anthropic":
        if not isinstance(response, dict):
            return None
        return extract_confidence_from_anthropic_response(response)

    # source == "auto"
    if isinstance(response, str):
        return extract_confidence_from_text(response)

    if isinstance(response, dict):
        # OpenAI: has "choices" key with "object": "chat.completion"
        if "choices" in response:
            return extract_confidence_from_openai_response(response)
        # Anthropic: has "content" list and "type": "message"
        if "content" in response and (response.get("type") == "message" or isinstance(response.get("content"), list)):
            return extract_confidence_from_anthropic_response(response)

    return None
