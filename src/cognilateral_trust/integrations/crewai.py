"""CrewAI TrustTool — trust evaluation as a CrewAI tool.

Usage:
    from cognilateral_trust.integrations.crewai import TrustEvaluationTool

    trust_tool = TrustEvaluationTool()

    agent = Agent(
        role="Decision Maker",
        tools=[trust_tool],
        ...
    )

The tool accepts a confidence level and returns an ACT/ESCALATE verdict
with accountability record. Agents can call it before taking any action
to verify whether they should proceed.

Zero dependencies beyond cognilateral-trust itself. CrewAI is only
needed at runtime when this module is imported.
"""

from __future__ import annotations

from typing import Any

from cognilateral_trust import evaluate_trust


class TrustEvaluationTool:
    """CrewAI-compatible tool for trust evaluation.

    Implements the CrewAI Tool protocol (name, description, _run).
    Can be used directly or subclassed for framework-specific base classes.
    """

    name: str = "trust_evaluation"
    description: str = (
        "Evaluate whether an AI agent should proceed with an action based on "
        "confidence level. Input: confidence (0.0-1.0), is_reversible (bool), "
        "touches_external (bool). Returns ACT or ESCALATE verdict with "
        "accountability record."
    )

    def _run(
        self,
        confidence: float = 0.5,
        is_reversible: bool = True,
        touches_external: bool = False,
    ) -> dict[str, Any]:
        """Execute trust evaluation. Called by CrewAI when an agent uses this tool."""
        result = evaluate_trust(
            confidence,
            is_reversible=is_reversible,
            touches_external=touches_external,
        )

        verdict = "ACT" if result.should_proceed else "ESCALATE"
        record = result.accountability_record

        return {
            "verdict": verdict,
            "should_proceed": result.should_proceed,
            "confidence": confidence,
            "tier": result.tier.name,
            "route": result.route,
            "record_id": record.record_id if record else "",
            "reasons": list(record.reasons) if record else [],
        }

    def run(self, **kwargs: Any) -> dict[str, Any]:
        """Public entry point — delegates to _run."""
        return self._run(**kwargs)
