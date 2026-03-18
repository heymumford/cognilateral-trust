"""LangGraph TrustNode — wraps cognilateral-trust for LangGraph pipelines.

Usage:
    pip install cognilateral-trust langgraph

    from langgraph_trust_node import build_trust_graph

    graph = build_trust_graph()
    result = graph.invoke({"confidence": 0.7, "action": "deploy"})
    print(result["decision"])  # "proceed" or "escalate"

This is an example integration. Adapt to your agent's state schema.
"""

from __future__ import annotations

from typing import Any, TypedDict

# cognilateral-trust is the only required dependency
from cognilateral_trust import evaluate_trust

# LangGraph is optional — this example shows the integration pattern
try:
    from langgraph.graph import StateGraph, END

    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False


class TrustState(TypedDict, total=False):
    """State flowing through the trust-augmented graph."""

    confidence: float
    action: str
    is_reversible: bool
    touches_external: bool
    decision: str
    tier: str
    route: str
    reasons: list[str]
    record_id: str


def trust_gate(state: TrustState) -> TrustState:
    """Evaluate trust and add decision to state."""
    result = evaluate_trust(
        state.get("confidence", 0.5),
        is_reversible=state.get("is_reversible", True),
        touches_external=state.get("touches_external", False),
        context={"action": state.get("action", "unknown")},
    )
    return {
        **state,
        "decision": "proceed" if result.should_proceed else "escalate",
        "tier": result.tier.name,
        "route": result.route,
        "reasons": list(result.accountability_record.reasons) if result.accountability_record else [],
        "record_id": result.accountability_record.record_id if result.accountability_record else "",
    }


def should_proceed(state: TrustState) -> str:
    """Route based on trust decision."""
    return "execute" if state.get("decision") == "proceed" else "escalate"


def execute_action(state: TrustState) -> TrustState:
    """Placeholder: execute the trusted action."""
    return {**state, "decision": "executed"}


def escalate_action(state: TrustState) -> TrustState:
    """Placeholder: escalate to human review."""
    return {**state, "decision": "escalated"}


def build_trust_graph() -> Any:
    """Build a LangGraph with trust gate.

    Returns a compiled graph if langgraph is installed, otherwise None.
    """
    if not HAS_LANGGRAPH:
        return None

    builder = StateGraph(TrustState)
    builder.add_node("trust_gate", trust_gate)
    builder.add_node("execute", execute_action)
    builder.add_node("escalate", escalate_action)

    builder.set_entry_point("trust_gate")
    builder.add_conditional_edges(
        "trust_gate",
        should_proceed,
        {
            "execute": "execute",
            "escalate": "escalate",
        },
    )
    builder.add_edge("execute", END)
    builder.add_edge("escalate", END)

    return builder.compile()


# Standalone usage without LangGraph
if __name__ == "__main__":
    # Works without LangGraph installed
    state: TrustState = {
        "confidence": 0.7,
        "action": "deploy",
        "is_reversible": True,
        "touches_external": False,
    }
    result = trust_gate(state)
    print(f"Decision: {result['decision']}")
    print(f"Tier: {result['tier']}, Route: {result['route']}")
    if result["reasons"]:
        print(f"Reasons: {result['reasons']}")
