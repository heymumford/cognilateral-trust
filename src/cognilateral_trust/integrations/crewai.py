"""CrewAI TrustTool with TrustPassport support."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from cognilateral_trust import evaluate_trust
from cognilateral_trust.core import PassportStamp, TrustPassport


class TrustEvaluationTool:
    """CrewAI-compatible tool for trust evaluation with optional TrustPassport support."""

    name: str = "trust_evaluation"
    description: str = (
        "Evaluate whether an AI agent should proceed with an action based on "
        "confidence level. Input: confidence (0.0-1.0), is_reversible (bool), "
        "touches_external (bool), passport (optional TrustPassport). "
        "Returns ACT or ESCALATE verdict with accountability record."
    )

    def _run(
        self,
        confidence: float = 0.5,
        is_reversible: bool = True,
        touches_external: bool = False,
        passport: TrustPassport | None = None,
    ) -> dict[str, Any]:
        """Execute trust evaluation. Called by CrewAI when an agent uses this tool."""
        result = evaluate_trust(
            confidence,
            is_reversible=is_reversible,
            touches_external=touches_external,
        )

        verdict = "ACT" if result.should_proceed else "ESCALATE"
        record = result.accountability_record

        output: dict[str, Any] = {
            "verdict": verdict,
            "should_proceed": result.should_proceed,
            "confidence": confidence,
            "tier": result.tier.name,
            "route": result.route,
            "record_id": record.record_id if record else "",
            "reasons": list(record.reasons) if record else [],
        }

        if isinstance(passport, TrustPassport):
            stamp = PassportStamp(
                agent_id="crewai_tool",
                action="evaluate",
                confidence_at_action=confidence,
                timestamp=datetime.now(timezone.utc).isoformat(),
                evidence_added=(),
            )
            updated_passport = passport.add_stamp(stamp)
            output["passport"] = updated_passport

        return output

    def run(self, **kwargs: Any) -> dict[str, Any]:
        """Public entry point — delegates to _run."""
        return self._run(**kwargs)
