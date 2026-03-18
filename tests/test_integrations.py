"""Tests for LangGraph TrustNode and CrewAI TrustTool integrations.

These test the integration wrappers without requiring LangGraph or CrewAI
to be installed — the wrappers only import the frameworks at runtime,
and the core logic is pure cognilateral_trust.
"""

from __future__ import annotations


class TestLangGraphTrustNode:
    """TrustNode must evaluate trust and route correctly."""

    def test_trust_gate_node_returns_act_for_high_confidence(self) -> None:
        from cognilateral_trust.integrations.langgraph import trust_gate_node

        state = {"confidence": 0.85, "is_reversible": True, "touches_external": False}
        result = trust_gate_node(state)

        assert result["trust_verdict"] == "ACT"
        assert result["trust_should_proceed"] is True
        assert result["trust_tier"] != ""
        assert result["trust_record_id"] != ""

    def test_trust_gate_node_returns_escalate_for_external_action(self) -> None:
        from cognilateral_trust.integrations.langgraph import trust_gate_node

        state = {"confidence": 0.85, "is_reversible": False, "touches_external": True}
        result = trust_gate_node(state)

        assert result["trust_verdict"] == "ESCALATE"
        assert result["trust_should_proceed"] is False
        assert len(result["trust_reasons"]) > 0

    def test_trust_gate_node_defaults_confidence(self) -> None:
        from cognilateral_trust.integrations.langgraph import trust_gate_node

        result = trust_gate_node({})
        assert result["trust_verdict"] in ("ACT", "ESCALATE")
        assert "trust_tier" in result

    def test_should_proceed_routes_execute(self) -> None:
        from cognilateral_trust.integrations.langgraph import should_proceed

        assert should_proceed({"trust_should_proceed": True}) == "execute"

    def test_should_proceed_routes_escalate(self) -> None:
        from cognilateral_trust.integrations.langgraph import should_proceed

        assert should_proceed({"trust_should_proceed": False}) == "escalate"

    def test_should_proceed_defaults_to_escalate(self) -> None:
        from cognilateral_trust.integrations.langgraph import should_proceed

        assert should_proceed({}) == "escalate"

    def test_full_node_to_router_pipeline(self) -> None:
        """Simulate the full LangGraph flow: node → conditional edge."""
        from cognilateral_trust.integrations.langgraph import should_proceed, trust_gate_node

        state = {"confidence": 0.3, "is_reversible": True}
        state.update(trust_gate_node(state))
        route = should_proceed(state)

        assert route in ("execute", "escalate")
        assert state["trust_record_id"] != ""


class TestCrewAITrustTool:
    """TrustTool must evaluate trust and return structured results."""

    def test_tool_has_name_and_description(self) -> None:
        from cognilateral_trust.integrations.crewai import TrustEvaluationTool

        tool = TrustEvaluationTool()
        assert tool.name == "trust_evaluation"
        assert "confidence" in tool.description.lower()

    def test_tool_run_returns_act(self) -> None:
        from cognilateral_trust.integrations.crewai import TrustEvaluationTool

        tool = TrustEvaluationTool()
        result = tool.run(confidence=0.85, is_reversible=True, touches_external=False)

        assert result["verdict"] == "ACT"
        assert result["should_proceed"] is True
        assert result["record_id"] != ""

    def test_tool_run_returns_escalate(self) -> None:
        from cognilateral_trust.integrations.crewai import TrustEvaluationTool

        tool = TrustEvaluationTool()
        result = tool.run(confidence=0.85, is_reversible=False, touches_external=True)

        assert result["verdict"] == "ESCALATE"
        assert result["should_proceed"] is False

    def test_tool_run_with_defaults(self) -> None:
        from cognilateral_trust.integrations.crewai import TrustEvaluationTool

        tool = TrustEvaluationTool()
        result = tool.run()

        assert "verdict" in result
        assert "tier" in result
        assert result["confidence"] == 0.5

    def test_tool_private_run(self) -> None:
        from cognilateral_trust.integrations.crewai import TrustEvaluationTool

        tool = TrustEvaluationTool()
        result = tool._run(confidence=0.7)

        assert result["verdict"] in ("ACT", "ESCALATE")
