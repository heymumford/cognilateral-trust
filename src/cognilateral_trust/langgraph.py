"""LangGraph integration — TrustNode for agentic workflows.

Drop a TrustNode into any LangGraph StateGraph to add trust evaluation
at decision points. Three lines to add trust to your agent:

    from cognilateral_trust.langgraph import trust_gate_node, trust_should_proceed

    graph.add_node("trust_gate", trust_gate_node)
    graph.add_conditional_edges("trust_gate", trust_should_proceed,
                                {"proceed": "act", "escalate": "human_review"})

The node reads `confidence` from state, evaluates trust, and writes
`trust_verdict`, `trust_tier`, `should_proceed`, and `nutrition_label`
back to state. The conditional edge function routes based on `should_proceed`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class TrustNode:
    """LangGraph-compatible trust evaluation node.

    Reads confidence from state, evaluates trust, writes verdict back.
    Configurable state key names for flexibility.
    """

    confidence_key: str = "confidence"
    reversible_key: str = "is_reversible"
    external_key: str = "touches_external"
    verdict_key: str = "trust_verdict"
    tier_key: str = "trust_tier"
    proceed_key: str = "should_proceed"
    label_key: str = "nutrition_label"

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        """Evaluate trust and return state updates."""
        from cognilateral_trust.evaluate import evaluate_trust

        confidence = state.get(self.confidence_key)

        # Missing confidence → escalate
        if confidence is None:
            return {
                self.verdict_key: "ESCALATE",
                self.tier_key: "UNKNOWN",
                self.proceed_key: False,
                self.label_key: {"evaluated": False, "reason": "no confidence provided"},
            }

        # Validate confidence is numeric
        try:
            conf = float(confidence)
        except (TypeError, ValueError):
            return {
                self.verdict_key: "ESCALATE",
                self.tier_key: "UNKNOWN",
                self.proceed_key: False,
                self.label_key: {"evaluated": False, "reason": f"invalid confidence: {confidence!r}"},
            }

        is_reversible = bool(state.get(self.reversible_key, True))
        touches_external = bool(state.get(self.external_key, False))

        result = evaluate_trust(
            conf,
            is_reversible=is_reversible,
            touches_external=touches_external,
        )
        verdict = "ACT" if result.should_proceed else "ESCALATE"

        # Build nutrition label using the actual API
        label: dict[str, Any] = {"evaluated": True, "confidence": conf, "tier": result.tier.name}
        try:
            from cognilateral_trust.nutrition import nutrition_label

            nl = nutrition_label(
                conf,
                is_reversible=is_reversible,
                touches_external=touches_external,
            )
            label = asdict(nl)
        except (ImportError, Exception):
            pass

        return {
            self.verdict_key: verdict,
            self.tier_key: result.tier.name,
            self.proceed_key: result.should_proceed,
            self.label_key: label,
        }


# Module-level convenience instance
_default_node = TrustNode()


def trust_gate_node(state: dict[str, Any]) -> dict[str, Any]:
    """Evaluate trust — use as a LangGraph node function.

    Reads state["confidence"], writes trust_verdict, trust_tier,
    should_proceed, nutrition_label.
    """
    return _default_node(state)


def trust_should_proceed(state: dict[str, Any]) -> str:
    """Conditional edge function — returns "proceed" or "escalate".

    Uses the default TrustNode's proceed_key for consistency.

    Use with graph.add_conditional_edges():
        graph.add_conditional_edges("trust_gate", trust_should_proceed,
                                    {"proceed": "act", "escalate": "review"})
    """
    if state.get(_default_node.proceed_key, False):
        return "proceed"
    return "escalate"
