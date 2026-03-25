"""MCP tool implementations — wrappers around the trust package public API."""

from __future__ import annotations

import time
from typing import Any


_SERVER_START_TIME = time.monotonic()


def trust_evaluate(
    confidence: float,
    evidence: list[str] | None = None,
    action_reversible: bool = True,
    touches_external: bool = False,
    welfare_relevant: bool = False,
) -> dict[str, Any]:
    """Evaluate trust for a decision point.

    Returns verdict (ACT/ESCALATE), confidence tier, routing, and accountability record.
    """
    from cognilateral_trust.evaluate import evaluate_trust

    if not isinstance(confidence, (int, float)):
        return {"error": "confidence must be a number between 0.0 and 1.0"}
    if confidence < 0.0 or confidence > 1.0:
        return {"error": f"confidence must be 0.0-1.0, got {confidence}"}

    context: dict[str, Any] = {}
    if evidence:
        context["evidence"] = evidence
    if welfare_relevant:
        context["welfare_relevant"] = True

    result = evaluate_trust(
        confidence,
        is_reversible=action_reversible,
        touches_external=touches_external,
        context=context,
    )

    record = result.accountability_record
    record_dict: dict[str, Any] | None = None
    if record is not None:
        record_dict = {
            "record_id": record.record_id,
            "timestamp": record.timestamp,
            "verdict": record.verdict,
            "reasons": list(record.reasons),
            "confidence": record.confidence,
            "confidence_tier": record.confidence_tier,
        }

    return {
        "confidence": result.confidence,
        "tier": result.tier.value,
        "tier_name": result.tier.name,
        "route": result.route,
        "requires_warrant_check": result.requires_warrant_check,
        "requires_sovereignty_gate": result.requires_sovereignty_gate,
        "should_proceed": result.should_proceed,
        "accountability_record": record_dict,
    }


def trust_extract_confidence(
    response: Any,
    source: str = "auto",
) -> dict[str, Any]:
    """Extract confidence signal from an LLM response.

    Accepts plain text, OpenAI-format dicts, or Anthropic-format dicts.
    Returns the extracted confidence float or null if no signal found.
    """
    from cognilateral_trust.extractors import extract_confidence

    valid_sources = ("auto", "text", "openai", "anthropic")
    if source not in valid_sources:
        return {"error": f"source must be one of {valid_sources}, got '{source}'"}

    try:
        value = extract_confidence(response, source=source)
    except Exception as exc:
        return {"confidence": None, "found": False, "error": str(exc)}

    return {
        "confidence": value,
        "found": value is not None,
        "source": source,
    }


def trust_health() -> dict[str, Any]:
    """Health check — version, uptime, response time."""
    start = time.monotonic()

    try:
        import cognilateral_trust

        version = getattr(cognilateral_trust, "__version__", "unknown")
    except Exception:
        version = "unknown"

    elapsed_ms = (time.monotonic() - start) * 1000
    uptime_s = time.monotonic() - _SERVER_START_TIME

    return {
        "status": "healthy",
        "version": version,
        "uptime_seconds": round(uptime_s, 1),
        "response_time_ms": round(elapsed_ms, 2),
    }


# Tool schemas for MCP tools/list response
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "trust_evaluate",
        "description": (
            "Evaluate trust for a decision point. Maps confidence (0.0-1.0) to an epistemic tier (C0-C9), "
            "routing decision, and ACT/ESCALATE verdict. Returns an accountability record."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Confidence level (0.0 = no confidence, 1.0 = certain)",
                },
                "evidence": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Evidence supporting the confidence level",
                },
                "action_reversible": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether the action can be undone",
                },
                "touches_external": {
                    "type": "boolean",
                    "default": False,
                    "description": "Whether the action affects external systems (independent of reversibility)",
                },
                "welfare_relevant": {
                    "type": "boolean",
                    "default": False,
                    "description": "Whether the action affects human welfare (D-05 hard constraint)",
                },
            },
            "required": ["confidence"],
        },
    },
    {
        "name": "trust_extract_confidence",
        "description": (
            "Extract a confidence signal from an LLM response. "
            "Supports plain text, OpenAI-format dicts, and Anthropic-format dicts."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "response": {
                    "description": "The LLM response to extract confidence from (string or dict)",
                },
                "source": {
                    "type": "string",
                    "enum": ["auto", "text", "openai", "anthropic"],
                    "default": "auto",
                    "description": "Response format hint",
                },
            },
            "required": ["response"],
        },
    },
    {
        "name": "trust_health",
        "description": "Health check — returns server version, uptime, and response time.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
]

# Dispatch table: tool name → callable
TOOL_DISPATCH: dict[str, Any] = {
    "trust_evaluate": trust_evaluate,
    "trust_extract_confidence": trust_extract_confidence,
    "trust_health": trust_health,
}
