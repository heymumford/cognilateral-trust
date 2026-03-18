"""Anthropic Claude trust wrapper — add confidence scoring to any Claude API call.

Wraps your existing Claude API calls with trust evaluation. The LLM provides
a self-assessed confidence score, and cognilateral-trust decides whether
the response should be used or escalated for human review.

Usage:
    pip install cognilateral-trust anthropic

    from anthropic_trust_wrapper import trusted_message

    result = trusted_message(
        model="claude-sonnet-4-20250514",
        prompt="What is the capital of France?",
    )
    if result["trusted"]:
        print(result["answer"])
    else:
        print(f"Low confidence ({result['tier']}): needs human review")

Works with any LLM that can self-report confidence. No API required
from cognilateral — runs entirely locally.
"""

from __future__ import annotations

import json
import re
from typing import Any

from cognilateral_trust import evaluate_trust


def extract_confidence(response_text: str) -> float:
    """Extract a confidence score from LLM response text.

    Looks for patterns like:
    - "Confidence: 0.85"
    - "confidence_score: 0.7"
    - "I'm about 80% confident"
    - JSON with "confidence" key

    Returns 0.5 (neutral) if no confidence signal found.
    """
    # Try JSON extraction first
    try:
        data = json.loads(response_text)
        if isinstance(data, dict):
            for key in ("confidence", "confidence_score", "certainty"):
                if key in data:
                    val = float(data[key])
                    return val if val <= 1.0 else val / 100.0
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Pattern: "Confidence: 0.85" or "confidence_score: 0.7"
    match = re.search(r"[Cc]onfidence[_\s]*(?:score)?[:\s]*([0-9]*\.?[0-9]+)", response_text)
    if match:
        val = float(match.group(1))
        return val if val <= 1.0 else val / 100.0

    # Pattern: "85% confident" or "about 70%"
    match = re.search(r"(\d{1,3})%\s*confident", response_text)
    if match:
        return float(match.group(1)) / 100.0

    return 0.5  # neutral default


def trusted_message(
    model: str = "claude-sonnet-4-20250514",
    prompt: str = "",
    *,
    system_prompt: str | None = None,
    is_reversible: bool = True,
    touches_external: bool = False,
    confidence_override: float | None = None,
    _mock_response: str | None = None,
) -> dict[str, Any]:
    """Make a Claude API call with trust evaluation.

    If confidence_override is provided, uses that directly.
    Otherwise, asks the LLM to self-assess and extracts confidence.

    Set _mock_response to skip the actual API call (for testing).
    """
    # Build the confidence-aware system prompt
    trust_system = (
        "After your answer, on a new line, provide your confidence as: "
        "Confidence: <float between 0.0 and 1.0>. "
        "Be honest — say 0.3 if you're guessing, 0.9 if you're certain."
    )
    full_system = f"{system_prompt}\n\n{trust_system}" if system_prompt else trust_system

    # Get LLM response (or use mock)
    if _mock_response is not None:
        response_text = _mock_response
    else:
        try:
            import anthropic

            client = anthropic.Anthropic()
            response = client.messages.create(
                model=model,
                max_tokens=1024,
                system=full_system,
                messages=[
                    {"role": "user", "content": prompt},
                ],
            )
            response_text = response.content[0].text if response.content else ""
        except ImportError:
            return {
                "error": "anthropic package not installed. pip install anthropic",
                "trusted": False,
            }

    # Extract confidence
    confidence = confidence_override if confidence_override is not None else extract_confidence(response_text)

    # Evaluate trust
    result = evaluate_trust(
        confidence,
        is_reversible=is_reversible,
        touches_external=touches_external,
        context={"model": model, "prompt": prompt[:100]},
    )

    return {
        "answer": response_text,
        "confidence": confidence,
        "tier": result.tier.name,
        "route": result.route,
        "trusted": result.should_proceed,
        "verdict": "ACT" if result.should_proceed else "ESCALATE",
        "reasons": list(result.accountability_record.reasons) if result.accountability_record else [],
        "record_id": result.accountability_record.record_id if result.accountability_record else "",
    }


if __name__ == "__main__":
    # Demo with mock responses (no API key needed)
    print("=== High confidence response ===")
    r1 = trusted_message(
        prompt="What is 2+2?",
        _mock_response="The answer is 4.\nConfidence: 0.99",
    )
    print(f"Answer: {r1['answer']}")
    print(f"Trusted: {r1['trusted']} | Tier: {r1['tier']} | Confidence: {r1['confidence']}")

    print("\n=== Low confidence response ===")
    r2 = trusted_message(
        prompt="What will the stock market do tomorrow?",
        _mock_response="Based on current trends, the market may go up.\nConfidence: 0.25",
    )
    print(f"Answer: {r2['answer']}")
    print(f"Trusted: {r2['trusted']} | Tier: {r2['tier']} | Confidence: {r2['confidence']}")

    print("\n=== Irreversible high-confidence action ===")
    r3 = trusted_message(
        prompt="Should I deploy to production?",
        _mock_response="Yes, all tests pass. Deploy recommended.\nConfidence: 0.92",
        is_reversible=False,
    )
    print(f"Answer: {r3['answer']}")
    print(f"Trusted: {r3['trusted']} | Tier: {r3['tier']} | Verdict: {r3['verdict']}")
    if r3["reasons"]:
        print(f"Reasons: {r3['reasons']}")
