"""LangGraph Trust Agent — complete example of trust-gated autonomous action.

This example shows how to build a LangGraph agent that evaluates trust
before taking any action. The trust gate prevents the agent from acting
on low-confidence decisions, routing them to a human escalation handler
instead.

Requirements:
    pip install cognilateral-trust langgraph langchain-core

Usage:
    python langgraph_trust_agent.py
"""

from __future__ import annotations

from typing import Any, TypedDict

# -- Trust integration (zero additional deps) --
from cognilateral_trust import evaluate_trust
from cognilateral_trust.integrations.langgraph import should_proceed, trust_gate_node


# -- Define agent state --
class AgentState(TypedDict, total=False):
    """State flowing through the trust-gated agent graph."""

    task: str
    confidence: float
    is_reversible: bool
    touches_external: bool
    # Trust evaluation results (written by trust_gate_node)
    trust_verdict: str
    trust_should_proceed: bool
    trust_tier: str
    trust_route: str
    trust_record_id: str
    trust_reasons: list[str]
    # Action results
    result: str


# -- Agent nodes --
def analyze_task(state: AgentState) -> dict[str, Any]:
    """Simulate LLM analyzing a task and producing a confidence level."""
    task = state.get("task", "unknown task")
    # In production, this would call an LLM and extract confidence
    # using cognilateral_trust.extract_confidence_from_text()
    print(f"  Analyzing: {task}")
    return {
        "confidence": state.get("confidence", 0.7),
        "is_reversible": state.get("is_reversible", True),
        "touches_external": state.get("touches_external", False),
    }


def execute_action(state: AgentState) -> dict[str, str]:
    """Execute the action — trust gate approved."""
    print(f"  ACTING: Trust verdict={state['trust_verdict']}, "
          f"tier={state['trust_tier']}, record={state['trust_record_id'][:8]}...")
    return {"result": f"Action completed for: {state.get('task', 'unknown')}"}


def escalate_to_human(state: AgentState) -> dict[str, str]:
    """Escalate to human — trust gate blocked."""
    reasons = state.get("trust_reasons", [])
    print(f"  ESCALATING: {state['trust_verdict']}, reasons={reasons}")
    return {"result": f"Escalated to human: {', '.join(reasons) or 'insufficient confidence'}"}


# -- Build the graph --
def build_trust_agent():
    """Build a LangGraph agent with trust gate.

    Flow: analyze_task → trust_gate → (execute | escalate)
    """
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError:
        print("LangGraph not installed. Install with: pip install langgraph")
        print("Falling back to manual execution...\n")
        return None

    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("analyze", analyze_task)
    workflow.add_node("trust_gate", trust_gate_node)
    workflow.add_node("execute", execute_action)
    workflow.add_node("escalate", escalate_to_human)

    # Wire edges
    workflow.add_edge(START, "analyze")
    workflow.add_edge("analyze", "trust_gate")
    workflow.add_conditional_edges(
        "trust_gate",
        should_proceed,
        {"execute": "execute", "escalate": "escalate"},
    )
    workflow.add_edge("execute", END)
    workflow.add_edge("escalate", END)

    return workflow.compile()


def run_manual(task: str, confidence: float, is_reversible: bool, touches_external: bool):
    """Run without LangGraph — demonstrates the trust primitives directly."""
    state: dict[str, Any] = {
        "task": task,
        "confidence": confidence,
        "is_reversible": is_reversible,
        "touches_external": touches_external,
    }

    # Analyze
    state.update(analyze_task(state))

    # Trust gate
    state.update(trust_gate_node(state))

    # Route
    route = should_proceed(state)
    if route == "execute":
        state.update(execute_action(state))
    else:
        state.update(escalate_to_human(state))

    return state


def main():
    print("=" * 60)
    print("Cognilateral Trust Agent — LangGraph Integration Demo")
    print("=" * 60)

    agent = build_trust_agent()

    scenarios = [
        ("Send daily report email", 0.92, True, False),
        ("Delete production database", 0.85, False, True),
        ("Update user preferences", 0.78, True, False),
        ("Transfer $50,000 to vendor", 0.6, False, True),
    ]

    for task, confidence, reversible, external in scenarios:
        print(f"\nTask: {task}")
        print(f"  Confidence: {confidence}, Reversible: {reversible}, External: {external}")

        if agent:
            result = agent.invoke({
                "task": task,
                "confidence": confidence,
                "is_reversible": reversible,
                "touches_external": external,
            })
        else:
            result = run_manual(task, confidence, reversible, external)

        print(f"  Result: {result.get('result', 'N/A')}")
        print(f"  Accountability: {result.get('trust_record_id', 'N/A')}")

    print("\n" + "=" * 60)
    print("Every decision has an accountability record.")
    print("pip install cognilateral-trust")
    print("=" * 60)


if __name__ == "__main__":
    main()
