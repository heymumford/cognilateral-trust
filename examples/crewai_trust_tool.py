"""CrewAI TrustTool — wraps cognilateral-trust for CrewAI agents.

Usage:
    pip install cognilateral-trust crewai

    from crewai_trust_tool import TrustTool, trust_check

    # As a CrewAI tool
    tool = TrustTool()
    result = tool._run(confidence=0.7, action="deploy")

    # Or as a plain function
    result = trust_check(confidence=0.7, action="deploy")

This is an example integration. Adapt to your agent's tool schema.
"""

from __future__ import annotations

from typing import Any

from cognilateral_trust import evaluate_trust

# CrewAI is optional
try:
    from crewai.tools import BaseTool
    HAS_CREWAI = True
except ImportError:
    HAS_CREWAI = False


def trust_check(
    confidence: float,
    action: str = "unknown",
    is_reversible: bool = True,
    touches_external: bool = False,
) -> dict[str, Any]:
    """Evaluate trust for an action. Works without any framework."""
    result = evaluate_trust(
        confidence,
        is_reversible=is_reversible,
        touches_external=touches_external,
        context={"action": action},
    )
    return {
        "should_proceed": result.should_proceed,
        "decision": "proceed" if result.should_proceed else "escalate",
        "tier": result.tier.name,
        "route": result.route,
        "confidence": result.confidence,
        "reasons": list(result.accountability_record.reasons) if result.accountability_record else [],
        "record_id": result.accountability_record.record_id if result.accountability_record else "",
    }


if HAS_CREWAI:
    class TrustTool(BaseTool):
        """CrewAI tool that evaluates epistemic trust before agent actions."""

        name: str = "trust_check"
        description: str = (
            "Evaluate whether an AI action should proceed based on epistemic confidence. "
            "Returns 'proceed' or 'escalate' with accountability record."
        )

        def _run(
            self,
            confidence: float = 0.5,
            action: str = "unknown",
            is_reversible: bool = True,
            touches_external: bool = False,
        ) -> str:
            result = trust_check(confidence, action, is_reversible, touches_external)
            if result["should_proceed"]:
                return f"PROCEED (tier {result['tier']}, route {result['route']})"
            return f"ESCALATE: {', '.join(result['reasons'])}"


if __name__ == "__main__":
    result = trust_check(confidence=0.3, action="send_email", touches_external=True)
    print(f"Decision: {result['decision']}")
    print(f"Tier: {result['tier']}, Route: {result['route']}")
    if result["reasons"]:
        print(f"Reasons: {result['reasons']}")
