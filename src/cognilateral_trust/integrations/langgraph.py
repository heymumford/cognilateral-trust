"""LangGraph TrustNode with TrustPassport support."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from cognilateral_trust import evaluate_trust
from cognilateral_trust.core import PassportStamp, TrustPassport


def trust_gate_node(state: dict[str, Any], passport: TrustPassport | None = None) -> dict[str, Any]:
    """LangGraph node that evaluates trust before an agent acts."""
    confidence = state.get("confidence", 0.5)
    is_reversible = state.get("is_reversible", True)
    touches_external = state.get("touches_external", False)
    passport = passport or state.get("passport")

    result = evaluate_trust(
        confidence,
        is_reversible=is_reversible,
        touches_external=touches_external,
    )

    verdict = "ACT" if result.should_proceed else "ESCALATE"
    record = result.accountability_record

    output = {
        "trust_verdict": verdict,
        "trust_should_proceed": result.should_proceed,
        "trust_tier": result.tier.name,
        "trust_route": result.route,
        "trust_record_id": record.record_id if record else "",
        "trust_reasons": list(record.reasons) if record else [],
    }

    if isinstance(passport, TrustPassport):
        stamp = PassportStamp(
            agent_id="langgraph_node",
            action="evaluate",
            confidence_at_action=confidence,
            timestamp=datetime.now(timezone.utc).isoformat(),
            evidence_added=tuple(state.get("evidence", [])),
        )
        updated_passport = passport.add_stamp(stamp)
        output["passport"] = updated_passport

    return output


def should_proceed(state: dict[str, Any]) -> str:
    """Conditional edge router: returns "execute" or "escalate"."""
    if state.get("trust_should_proceed", False):
        return "execute"
    return "escalate"
