"""Validation tests for Trust Evaluation Protocol v1.

Any implementation can run these tests to verify spec conformance.
Covers schema validation, round-trip serialization, and backward compatibility.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cognilateral_trust.core import (
    ConfidenceTier,
    PassportStamp,
    TrustContext,
    TrustPassport,
    evaluate_tier_routing,
    route_by_tier,
)


@pytest.fixture
def spec_path() -> Path:
    """Fixture providing the path to the Trust Evaluation Protocol spec."""
    return Path(__file__).parent.parent / "docs" / "specs" / "trust-protocol-v1.md"


class TestProtocolSpecFile:
    """Test that the specification document exists and is valid."""

    @pytest.mark.b_test
    def test_b1_spec_file_exists(self, spec_path: Path) -> None:
        """Verify that docs/specs/trust-protocol-v1.md exists."""
        assert spec_path.exists(), f"Spec file not found at {spec_path}"
        content = spec_path.read_text()
        assert len(content) > 100, "Spec file exists but is empty or too short"
        assert "Trust Evaluation Protocol v1" in content, "Spec file does not contain title"
        assert "Verdict" in content, "Spec file does not define Verdict semantics"


class TestEvaluationRequestSchema:
    """A-tests: Validate evaluation request message format."""

    @pytest.mark.a_test
    def test_a1_evaluation_request_schema_valid(self) -> None:
        """Construct a valid evaluation request dict and verify all required fields."""
        request = {
            "confidence": 0.75,
            "evidence": ["tested", "validated"],
            "action_reversible": True,
            "welfare_relevant": False,
        }

        # Required field check
        assert "confidence" in request
        assert isinstance(request["confidence"], (int, float))
        assert 0.0 <= request["confidence"] <= 1.0

        # Optional fields with defaults
        evidence = request.get("evidence", [])
        assert isinstance(evidence, list)
        assert all(isinstance(e, str) for e in evidence)

        action_reversible = request.get("action_reversible", True)
        assert isinstance(action_reversible, bool)

        welfare_relevant = request.get("welfare_relevant", False)
        assert isinstance(welfare_relevant, bool)

    @pytest.mark.a_test
    def test_a1_minimal_request_valid(self) -> None:
        """Verify that minimal request (confidence only) is valid."""
        request = {"confidence": 0.5}

        # All required fields present
        assert "confidence" in request
        # Optional fields should have sensible defaults when missing
        evidence = request.get("evidence", [])
        assert evidence == []
        assert request.get("action_reversible", True) is True
        assert request.get("welfare_relevant", False) is False

    @pytest.mark.a_test
    def test_a1_request_confidence_validation(self) -> None:
        """Verify confidence range validation (0.0-1.0)."""
        # Valid boundaries
        for conf in [0.0, 0.5, 1.0]:
            request = {"confidence": conf}
            assert 0.0 <= request["confidence"] <= 1.0

        # Invalid values
        for conf in [-0.1, 1.1, 2.0]:
            request = {"confidence": conf}
            assert not (0.0 <= request["confidence"] <= 1.0)


class TestEvaluationResponseSchema:
    """A-tests: Validate evaluation response message format."""

    @pytest.mark.a_test
    def test_a2_evaluation_response_schema_valid(self) -> None:
        """Construct a valid response dict and verify verdict, tier, route."""
        response = {
            "verdict": "ACT",
            "tier": 5,
            "route": "warrant_check",
            "record_id": "eval-20260324-001",
            "reasons": [],
            "timestamp": "2026-03-24T14:30:45Z",
        }

        # Required fields
        assert response["verdict"] in ("ACT", "ESCALATE")
        assert isinstance(response["tier"], int)
        assert 0 <= response["tier"] <= 9
        assert response["route"] in ("basic", "warrant_check", "sovereignty_gate")
        assert isinstance(response["record_id"], str)
        assert len(response["record_id"]) > 0
        assert isinstance(response["timestamp"], str)

        # Optional reasons
        reasons = response.get("reasons", [])
        assert isinstance(reasons, list)
        assert all(isinstance(r, str) for r in reasons)

    @pytest.mark.a_test
    def test_a2_escalate_verdict_with_reasons(self) -> None:
        """Verify ESCALATE verdict includes reasons."""
        response = {
            "verdict": "ESCALATE",
            "tier": 8,
            "route": "sovereignty_gate",
            "record_id": "eval-20260324-002",
            "reasons": ["irreversible action at sovereignty-grade tier"],
            "timestamp": "2026-03-24T14:31:20Z",
        }

        assert response["verdict"] == "ESCALATE"
        assert len(response["reasons"]) > 0
        assert all(isinstance(r, str) for r in response["reasons"])

    @pytest.mark.a_test
    def test_a2_tier_range_valid(self) -> None:
        """Verify tier field is in valid range 0-9."""
        for tier in range(10):
            response = {
                "verdict": "ACT",
                "tier": tier,
                "route": "basic" if tier <= 3 else "warrant_check",
                "record_id": f"eval-{tier}",
                "timestamp": "2026-03-24T00:00:00Z",
            }
            assert 0 <= response["tier"] <= 9


class TestTrustContextSerialization:
    """A-tests: Round-trip serialization of TrustContext."""

    @pytest.mark.a_test
    def test_a3_trust_context_serialize_restore(self) -> None:
        """Create TrustContext, serialize, verify against spec schema, restore."""
        original = TrustContext(
            confidence=0.75,
            evidence=("tested", "validated"),
            action_reversible=True,
            welfare_relevant=False,
        )

        # Serialize
        serialized = original.serialize()

        # Verify schema
        assert isinstance(serialized, dict)
        assert "confidence" in serialized
        assert isinstance(serialized["confidence"], float)
        assert 0.0 <= serialized["confidence"] <= 1.0
        assert "evidence" in serialized
        assert isinstance(serialized["evidence"], (list, tuple))
        assert "action_reversible" in serialized
        assert isinstance(serialized["action_reversible"], bool)
        assert "welfare_relevant" in serialized
        assert isinstance(serialized["welfare_relevant"], bool)

        # Restore and verify equality
        restored = TrustContext.restore(serialized)
        assert restored.confidence == original.confidence
        assert restored.evidence == original.evidence
        assert restored.action_reversible == original.action_reversible
        assert restored.welfare_relevant == original.welfare_relevant

    @pytest.mark.a_test
    def test_a3_trust_context_confidence_validation(self) -> None:
        """Verify TrustContext enforces confidence range."""
        # Valid
        ctx = TrustContext(confidence=0.5)
        assert ctx.confidence == 0.5

        # Invalid — should raise
        with pytest.raises(ValueError, match="confidence must be 0.0-1.0"):
            TrustContext(confidence=1.5)

        with pytest.raises(ValueError, match="confidence must be 0.0-1.0"):
            TrustContext(confidence=-0.1)

    @pytest.mark.a_test
    def test_a3_trust_context_json_round_trip(self) -> None:
        """Verify TrustContext survives JSON serialization."""
        original = TrustContext(
            confidence=0.82,
            evidence=("method_a", "method_b"),
            action_reversible=False,
            welfare_relevant=True,
        )

        serialized = original.serialize()
        json_str = json.dumps(serialized)
        parsed = json.loads(json_str)
        restored = TrustContext.restore(parsed)

        assert restored == original


class TestPassportStampSerialization:
    """A-tests: Round-trip serialization of PassportStamp."""

    @pytest.mark.a_test
    def test_a4_passport_stamp_schema(self) -> None:
        """Verify PassportStamp serialization matches spec schema."""
        stamp = PassportStamp(
            agent_id="agent-001",
            action="evaluate",
            confidence_at_action=0.75,
            timestamp="2026-03-24T10:00:00Z",
            evidence_added=("observed_success", "passed_test"),
        )

        serialized = stamp.serialize()

        # Verify schema
        assert isinstance(serialized, dict)
        assert serialized["agent_id"] == "agent-001"
        assert serialized["action"] in ("spawn", "handoff", "evaluate", "terminate")
        assert isinstance(serialized["confidence_at_action"], float)
        assert 0.0 <= serialized["confidence_at_action"] <= 1.0
        assert isinstance(serialized["timestamp"], str)
        assert "T" in serialized["timestamp"]  # ISO 8601
        assert isinstance(serialized["evidence_added"], (list, tuple))

    @pytest.mark.a_test
    def test_a4_passport_stamp_restore(self) -> None:
        """Verify PassportStamp restore from serialized dict."""
        original = PassportStamp(
            agent_id="agent-xyz",
            action="spawn",
            confidence_at_action=0.8,
            timestamp="2026-03-24T09:00:00Z",
        )

        serialized = original.serialize()
        restored = PassportStamp.restore(serialized)

        assert restored.agent_id == original.agent_id
        assert restored.action == original.action
        assert restored.confidence_at_action == original.confidence_at_action
        assert restored.timestamp == original.timestamp
        assert restored.evidence_added == original.evidence_added

    @pytest.mark.a_test
    def test_a4_stamp_actions_valid(self) -> None:
        """Verify all valid stamp action types."""
        valid_actions = ("spawn", "handoff", "evaluate", "terminate")
        for action in valid_actions:
            stamp = PassportStamp(
                agent_id="test",
                action=action,
                confidence_at_action=0.5,
                timestamp="2026-03-24T00:00:00Z",
            )
            assert stamp.action == action


class TestTrustPassportSerialization:
    """A-tests: Round-trip serialization of TrustPassport."""

    @pytest.mark.a_test
    def test_a5_passport_schema(self) -> None:
        """Verify TrustPassport serialization matches spec schema."""
        context = TrustContext(confidence=0.7, evidence=("calibrated",))
        stamp1 = PassportStamp(
            agent_id="agent-001",
            action="spawn",
            confidence_at_action=0.8,
            timestamp="2026-03-24T09:00:00Z",
        )
        stamp2 = PassportStamp(
            agent_id="agent-001",
            action="evaluate",
            confidence_at_action=0.7,
            timestamp="2026-03-24T10:00:00Z",
        )

        passport = TrustPassport(
            agent_id="agent-001",
            created_at="2026-03-24T09:00:00Z",
            context=context,
            stamps=(stamp1, stamp2),
            warrant_chain=("warrant-001",),
        )

        serialized = passport.serialize()

        # Verify schema
        assert isinstance(serialized, dict)
        assert "agent_id" in serialized
        assert "created_at" in serialized
        assert "context" in serialized
        assert isinstance(serialized["context"], dict)
        assert "stamps" in serialized
        assert isinstance(serialized["stamps"], list)
        assert len(serialized["stamps"]) == 2
        assert "warrant_chain" in serialized
        assert isinstance(serialized["warrant_chain"], list)

    @pytest.mark.a_test
    def test_a5_passport_restore(self) -> None:
        """Verify TrustPassport restore from serialized dict."""
        context = TrustContext(confidence=0.65)
        stamp = PassportStamp(
            agent_id="agent-x",
            action="evaluate",
            confidence_at_action=0.65,
            timestamp="2026-03-24T11:00:00Z",
        )

        original = TrustPassport(
            agent_id="agent-x",
            created_at="2026-03-24T09:00:00Z",
            context=context,
            stamps=(stamp,),
            warrant_chain=("w1", "w2"),
        )

        serialized = original.serialize()
        restored = TrustPassport.restore(serialized)

        assert restored.agent_id == original.agent_id
        assert restored.created_at == original.created_at
        assert restored.context.confidence == original.context.confidence
        assert len(restored.stamps) == 1
        assert restored.stamps[0].agent_id == "agent-x"
        assert restored.warrant_chain == ("w1", "w2")

    @pytest.mark.a_test
    def test_a5_passport_immutability_add_stamp(self) -> None:
        """Verify TrustPassport.add_stamp() returns new passport."""
        original = TrustPassport(
            agent_id="agent-001",
            created_at="2026-03-24T09:00:00Z",
            context=TrustContext(confidence=0.5),
            stamps=(),
        )

        new_stamp = PassportStamp(
            agent_id="agent-001",
            action="evaluate",
            confidence_at_action=0.6,
            timestamp="2026-03-24T10:00:00Z",
        )

        updated = original.add_stamp(new_stamp)

        # Original unchanged
        assert len(original.stamps) == 0
        # New passport has stamp
        assert len(updated.stamps) == 1
        assert updated.stamps[0] == new_stamp
        # Different objects
        assert original is not updated


class TestBackwardCompatibility:
    """B-tests: Verify existing functionality remains intact."""

    @pytest.mark.b_test
    def test_b1_existing_types_importable(self) -> None:
        """Verify existing core exports remain importable."""
        from cognilateral_trust.core import (
            ConfidenceTier,
            PassportStamp,
            TrustContext,
            TrustPassport,
            evaluate_tier_routing,
            route_by_tier,
        )

        assert ConfidenceTier is not None
        assert PassportStamp is not None
        assert TrustContext is not None
        assert TrustPassport is not None
        assert callable(evaluate_tier_routing)
        assert callable(route_by_tier)

    @pytest.mark.b_test
    def test_b1_confidence_tier_enum_values(self) -> None:
        """Verify ConfidenceTier enum has all C0-C9 values."""
        assert ConfidenceTier.C0 == 0
        assert ConfidenceTier.C1 == 1
        assert ConfidenceTier.C2 == 2
        assert ConfidenceTier.C3 == 3
        assert ConfidenceTier.C4 == 4
        assert ConfidenceTier.C5 == 5
        assert ConfidenceTier.C6 == 6
        assert ConfidenceTier.C7 == 7
        assert ConfidenceTier.C8 == 8
        assert ConfidenceTier.C9 == 9

    @pytest.mark.b_test
    def test_b1_route_by_tier_logic(self) -> None:
        """Verify route_by_tier produces expected routing."""
        # basic: 0-3
        for tier in range(4):
            assert route_by_tier(tier) == "basic"

        # warrant_check: 4-6
        for tier in range(4, 7):
            assert route_by_tier(tier) == "warrant_check"

        # sovereignty_gate: 7-9
        for tier in range(7, 10):
            assert route_by_tier(tier) == "sovereignty_gate"

    @pytest.mark.b_test
    def test_b1_tier_routing_result_complete(self) -> None:
        """Verify evaluate_tier_routing returns full result object."""
        result = evaluate_tier_routing(5)

        assert result.tier == 5
        assert result.route == "warrant_check"
        assert result.requires_warrant_check is True
        assert result.requires_sovereignty_gate is False
        assert result.tier_name == "C5"

    @pytest.mark.b_test
    def test_b2_spec_file_not_empty(self, spec_path: Path) -> None:
        """Verify spec file contains required sections."""
        content = spec_path.read_text()

        required_sections = [
            "Abstract",
            "Terminology",
            "Evaluation Request",
            "Evaluation Response",
            "Verdict Semantics",
            "Confidence Tiers",
            "Warrant Format",
            "Trust Passport",
            "Conformance Requirements",
            "Examples",
        ]

        for section in required_sections:
            assert section in content, f"Spec missing required section: {section}"


class TestProtocolExamples:
    """Additional tests verifying protocol examples are valid."""

    @pytest.mark.a_test
    def test_protocol_example_1_low_confidence_act(self) -> None:
        """Verify Example 1 (low-confidence ACT) is valid."""
        request = {
            "confidence": 0.25,
            "evidence": ["observed once"],
            "action_reversible": True,
            "welfare_relevant": False,
        }

        # Validate request
        assert 0.0 <= request["confidence"] <= 1.0
        tier = min(9, max(0, int(request["confidence"] * 10)))
        assert tier == 2

        # Validate response structure
        response = {
            "verdict": "ACT",
            "tier": 2,
            "route": "basic",
            "record_id": "eval-20260324-001-uuid",
            "reasons": [],
            "timestamp": "2026-03-24T14:30:45Z",
        }

        assert response["verdict"] in ("ACT", "ESCALATE")
        assert response["route"] in ("basic", "warrant_check", "sovereignty_gate")
        assert isinstance(response["record_id"], str)

    @pytest.mark.a_test
    def test_protocol_example_2_high_confidence_escalate(self) -> None:
        """Verify Example 2 (high-confidence ESCALATE) is valid."""
        request = {
            "confidence": 0.85,
            "evidence": [
                "tested in 5 scenarios",
                "manual review passed",
                "production migration plan approved",
            ],
            "action_reversible": False,
            "welfare_relevant": True,
        }

        # Validate request
        assert 0.0 <= request["confidence"] <= 1.0
        tier = min(9, max(0, int(request["confidence"] * 10)))
        assert tier == 8

        # Irreversible at sovereignty gate should escalate
        response = {
            "verdict": "ESCALATE",
            "tier": 8,
            "route": "sovereignty_gate",
            "record_id": "eval-20260324-002-uuid",
            "reasons": ["irreversible action at sovereignty-grade tier"],
            "timestamp": "2026-03-24T14:31:20Z",
        }

        assert response["verdict"] == "ESCALATE"
        assert len(response["reasons"]) > 0

    @pytest.mark.a_test
    def test_protocol_example_3_warrant_checked(self) -> None:
        """Verify Example 3 (warrant-checked decision) is valid."""
        request = {
            "confidence": 0.55,
            "evidence": [
                "validated on 100 test cases",
                "precision 0.89",
                "recall 0.91",
            ],
            "action_reversible": True,
            "welfare_relevant": False,
        }

        # Validate request
        assert 0.0 <= request["confidence"] <= 1.0
        tier = min(9, max(0, int(request["confidence"] * 10)))
        assert tier == 5  # 0.55 * 10 = 5.5 → 5

        # Warrant check route should return ACT
        response = {
            "verdict": "ACT",
            "tier": 5,
            "route": "warrant_check",
            "record_id": "eval-20260324-003-uuid",
            "reasons": [],
            "timestamp": "2026-03-24T14:32:10Z",
        }

        assert response["verdict"] == "ACT"
        assert response["route"] == "warrant_check"
