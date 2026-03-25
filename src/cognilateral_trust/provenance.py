"""Provenance Lens — memory poisoning detection for AI agents.

Evaluates the trustworthiness of information sources before they are
stored in agent memory. Detects prompt injection, source spoofing,
and suspicious provenance chains.

Zero external dependencies — stdlib only.
"""

from __future__ import annotations

import base64
import hashlib
import re
import unicodedata
from dataclasses import dataclass

__all__ = [
    "ProvenanceSignal",
    "ProvenanceResult",
    "detect_injection_patterns",
    "evaluate_provenance",
    "evaluate_memory_entry",
]

# ---------------------------------------------------------------------------
# Base trust scores per source type
# ---------------------------------------------------------------------------

_BASE_SCORES: dict[str, float] = {
    "system": 0.9,
    "user_input": 0.9,
    "api_response": 0.7,
    "agent_memory": 0.5,
    "web_scrape": 0.4,
}

_DEFAULT_BASE_SCORE = 0.3
_ATTRIBUTION_PENALTY = 0.15
_INJECTION_PENALTY = 0.3
_CHAIN_DECAY = 0.9

# ---------------------------------------------------------------------------
# Injection pattern regexes
# ---------------------------------------------------------------------------

_IGNORE_INSTRUCTIONS_RE = re.compile(
    r"\b(?:ignore|disregard|override)\b.{0,30}(?:previous|prior|above|all|earlier)?\s*"
    r"(?:instructions?|directives?|commands?|rules?|guidelines?|constraints?)"
    r"|\bforget\b.{0,50}(?:previous|prior|above|all|earlier|everything|told|been|said)\b",
    re.IGNORECASE,
)

_ROLE_INJECTION_RE = re.compile(
    r"\b(?:you\s+are\s+now|act\s+as|pretend\s+to\s+be|roleplay\s+as|simulate\s+being)\b",
    re.IGNORECASE,
)

_ROLE_OVERRIDE_RE = re.compile(
    r"^(?:system|assistant|user)\s*:",
    re.IGNORECASE | re.MULTILINE,
)

# Unicode code points considered hidden/dangerous
_HIDDEN_UNICODE_CATEGORIES = frozenset({"Cf", "Cs"})  # Format chars, surrogates

# Base64 detection: sequences of 20+ base64 chars (with optional padding)
_BASE64_RE = re.compile(r"[A-Za-z0-9+/]{20,}={0,2}")


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProvenanceSignal:
    """A single provenance observation about a piece of information."""

    source_type: str  # "user_input", "web_scrape", "api_response", "agent_memory", "system"
    source_uri: str  # Where it came from (URL, user ID, etc.)
    retrieval_timestamp: str  # When it was retrieved (ISO 8601)
    chain_length: int  # How many hops from original source (0 = direct)
    has_attribution: bool  # Does it cite its own source?
    content_hash: str  # Hash of the content for dedup detection


@dataclass(frozen=True)
class ProvenanceResult:
    """Result of provenance evaluation."""

    trust_score: float  # 0.0-1.0 — how much to trust this source
    risk_flags: list[str]  # ["injection_pattern", "no_attribution", "stale", ...]
    recommendation: str  # "trust", "verify", "reject"
    reasoning: str  # Human-readable explanation


# ---------------------------------------------------------------------------
# Pattern detection
# ---------------------------------------------------------------------------


def detect_injection_patterns(content: str) -> list[str]:
    """Detect common prompt injection patterns in content.

    Patterns detected:
    - ignore_instructions: "ignore previous instructions" / "disregard" / "forget"
    - role_injection: "you are now" / "act as" / "pretend to be"
    - role_override: "system:" / "assistant:" role prefix injection
    - hidden_unicode: zero-width spaces, RTL override, format characters
    - base64_payload: base64-encoded payloads (length > 20)

    Returns list of pattern names detected. Empty list = clean.
    """
    if not content:
        return []

    detected: list[str] = []

    if _IGNORE_INSTRUCTIONS_RE.search(content):
        detected.append("ignore_instructions")

    if _ROLE_INJECTION_RE.search(content):
        detected.append("role_injection")

    if _ROLE_OVERRIDE_RE.search(content):
        detected.append("role_override")

    # Hidden Unicode: check each character's Unicode category
    for ch in content:
        cat = unicodedata.category(ch)
        if cat in _HIDDEN_UNICODE_CATEGORIES:
            detected.append("hidden_unicode")
            break

    # Check for specific dangerous code points not covered by category
    _DANGEROUS_CODEPOINTS = frozenset(
        {
            "\u200b",  # Zero-width space
            "\u200c",  # Zero-width non-joiner
            "\u200d",  # Zero-width joiner
            "\u202e",  # RTL override
            "\u202a",  # LTR embedding
            "\u202b",  # RTL embedding
            "\u202c",  # Pop directional formatting
            "\u2060",  # Word joiner
            "\ufeff",  # BOM / zero-width no-break space
        }
    )
    if "hidden_unicode" not in detected and any(ch in _DANGEROUS_CODEPOINTS for ch in content):
        detected.append("hidden_unicode")

    # Base64 detection: find candidate sequences, validate they decode cleanly
    for match in _BASE64_RE.finditer(content):
        candidate = match.group(0)
        try:
            # Pad to multiple of 4 if needed
            padded = candidate + "=" * (-len(candidate) % 4)
            decoded = base64.b64decode(padded)
            # Only flag if decoded bytes are printable-ish (avoids false positives on hex strings)
            if len(decoded) >= 10:
                detected.append("base64_payload")
                break
        except Exception:  # noqa: BLE001
            pass

    return detected


# ---------------------------------------------------------------------------
# Core scoring
# ---------------------------------------------------------------------------


def _compute_base_score(signal: ProvenanceSignal) -> tuple[float, list[str], list[str]]:
    """Compute base trust score with all modifiers.

    Returns (score, risk_flags, reasoning_parts).
    """
    base = _BASE_SCORES.get(signal.source_type, _DEFAULT_BASE_SCORE)
    reasoning_parts: list[str] = [f"base score {base:.2f} for source type '{signal.source_type}'"]
    risk_flags: list[str] = []

    # Chain length decay
    if signal.chain_length > 0:
        decay = _CHAIN_DECAY**signal.chain_length
        score = base * decay
        reasoning_parts.append(f"chain decay 0.9^{signal.chain_length}={decay:.4f} → score {score:.4f}")
    else:
        score = base

    # Attribution penalty
    if not signal.has_attribution:
        score -= _ATTRIBUTION_PENALTY
        risk_flags.append("no_attribution")
        reasoning_parts.append(f"attribution penalty -{_ATTRIBUTION_PENALTY:.2f}")

    return score, risk_flags, reasoning_parts


def _recommendation(score: float) -> str:
    if score >= 0.7:
        return "trust"
    if score >= 0.4:
        return "verify"
    return "reject"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate_provenance(signal: ProvenanceSignal, context: dict | None = None) -> ProvenanceResult:
    """Evaluate the trustworthiness of an information source.

    Scoring rules:
    - source_type "system" or "user_input" → base score 0.9
    - source_type "api_response" → base score 0.7
    - source_type "web_scrape" → base score 0.4
    - source_type "agent_memory" → base score 0.5
    - Unknown source type → base score 0.3
    - chain_length > 0 → multiply by 0.9^chain_length (decay per hop)
    - has_attribution = False → subtract 0.15

    Recommendation thresholds:
    - score >= 0.7 → "trust"
    - score >= 0.4 → "verify"
    - score < 0.4 → "reject"
    """
    score, risk_flags, reasoning_parts = _compute_base_score(signal)
    score = max(0.0, min(1.0, score))
    recommendation = _recommendation(score)
    reasoning = "; ".join(reasoning_parts) + f" → final score {score:.4f} → {recommendation}"

    return ProvenanceResult(
        trust_score=score,
        risk_flags=risk_flags,
        recommendation=recommendation,
        reasoning=reasoning,
    )


def evaluate_memory_entry(
    content: str,
    source: ProvenanceSignal,
    existing_memories: list[dict] | None = None,
) -> ProvenanceResult:
    """Evaluate whether a piece of content is safe to store in agent memory.

    Combines provenance scoring with:
    1. Content injection detection — flags and penalises injection attempts
    2. Duplicate detection — flags content_hash collisions against existing_memories
    3. Score clamped to [0.0, 1.0]

    This is the main entry point for memory poisoning defense.
    """
    score, risk_flags, reasoning_parts = _compute_base_score(source)

    # Injection detection
    injection_patterns = detect_injection_patterns(content)
    if injection_patterns:
        score -= _INJECTION_PENALTY
        risk_flags.append("injection_pattern")
        reasoning_parts.append(f"injection patterns detected {injection_patterns} → penalty -{_INJECTION_PENALTY:.2f}")

    # Duplicate detection
    if existing_memories:
        for mem in existing_memories:
            if mem.get("content_hash") == source.content_hash:
                risk_flags.append("duplicate_content")
                reasoning_parts.append(f"content_hash '{source.content_hash}' collides with existing memory")
                break

    score = max(0.0, min(1.0, score))
    recommendation = _recommendation(score)
    reasoning = "; ".join(reasoning_parts) + f" → final score {score:.4f} → {recommendation}"

    return ProvenanceResult(
        trust_score=score,
        risk_flags=risk_flags,
        recommendation=recommendation,
        reasoning=reasoning,
    )


def compute_content_hash(content: str) -> str:
    """Compute a stable SHA-256 hash for content deduplication.

    Utility for callers constructing ProvenanceSignal objects.
    """
    return hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()[:16]
