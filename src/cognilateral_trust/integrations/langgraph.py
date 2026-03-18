"""LangGraph TrustNode — drop-in trust evaluation for StateGraph workflows.

Usage:
    from cognilateral_trust.integrations.langgraph import trust_gate_node, should_proceed
    from langgraph.graph import StateGraph, START, END

    workflow = StateGraph(AgentState)
    workflow.add_node("evaluate_trust", trust_gate_node)
    workflow.add_node("execute_action", my_action_node)
    workflow.add_node("escalate", my_escalation_node)

    workflow.add_edge(START, "evaluate_trust")
    workflow.add_conditional_edges(
        "evaluate_trust",
        should_proceed,
        {"execute": "execute_action", "escalate": "escalate"},
    )
    workflow.add_edge("execute_action", END)
    workflow.add_edge("escalate", END)

    agent = workflow.compile()
    result = agent.invoke({"confidence": 0.85, "messages": [...]})

The TrustNode reads `confidence` from state, evaluates trust, and writes
the verdict back into state. The `should_proceed` router returns "execute"
or "escalate" based on the verdict.

Zero dependencies beyond cognilateral-trust itself. LangGraph is only
needed at runtime when this module is imported.
"""

from __future__ import annotations

from typing import Any

from cognilateral_trust import evaluate_trust


def trust_gate_node(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node that evaluates trust before an agent acts.

    Reads from state:
        confidence (float): 0.0-1.0 confidence level (required)
        is_reversible (bool): whether the action can be undone (default True)
        touches_external (bool): whether the action affects external systems (default False)

    Writes to state:
        trust_verdict (str): "ACT" or "ESCALATE"
        trust_should_proceed (bool): whether the agent should proceed
        trust_tier (str): confidence tier name (e.g., "C8")
        trust_route (str): routing decision (e.g., "sovereignty_gate")
        trust_record_id (str): accountability record ID
        trust_reasons (list[str]): reasons for the decision
    """
    confidence = state.get("confidence", 0.5)
    is_reversible = state.get("is_reversible", True)
    touches_external = state.get("touches_external", False)

    result = evaluate_trust(
        confidence,
        is_reversible=is_reversible,
        touches_external=touches_external,
    )

    verdict = "ACT" if result.should_proceed else "ESCALATE"
    record = result.accountability_record

    return {
        "trust_verdict": verdict,
        "trust_should_proceed": result.should_proceed,
        "trust_tier": result.tier.name,
        "trust_route": result.route,
        "trust_record_id": record.record_id if record else "",
        "trust_reasons": list(record.reasons) if record else [],
    }


def should_proceed(state: dict[str, Any]) -> str:
    """Conditional edge router: returns "execute" or "escalate".

    Use with `add_conditional_edges`:
        workflow.add_conditional_edges(
            "evaluate_trust",
            should_proceed,
            {"execute": "execute_action", "escalate": "escalate"},
        )
    """
    if state.get("trust_should_proceed", False):
        return "execute"
    return "escalate"
