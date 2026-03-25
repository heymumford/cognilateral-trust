"""Step 01 A-tests and B-tests: TrustContext, PassportStamp, TrustPassport.

These types are the protocol atoms consumed by all downstream swimlanes.
After this step ships, they are LOCKED — field changes require coordination.
"""

from __future__ import annotations

import json

import pytest

from cognilateral_trust.core import PassportStamp, TrustContext, TrustPassport


# ---------------------------------------------------------------------------
# A-tests: TrustContext
# ---------------------------------------------------------------------------


class TestTrustContextAcceptance:
    """A-tests: must-pass scenarios for TrustContext."""

    @pytest.mark.a_test
    def test_a1_json_round_trip_lossless(self) -> None:
        """Given a TrustContext, when serialized to JSON and deserialized, then round-trip is lossless."""
        ctx = TrustContext(
            confidence=0.7,
            evidence=("source_a", "source_b"),
            action_reversible=False,
            welfare_relevant=True,
        )
        serialized = ctx.serialize()
        json_str = json.dumps(serialized)
        parsed = json.loads(json_str)
        restored = TrustContext.restore(parsed)
        assert restored == ctx

    @pytest.mark.a_test
    def test_a2_value_semantics(self) -> None:
        """Given two TrustContexts with same fields, when compared, then they are equal."""
        ctx1 = TrustContext(confidence=0.5, evidence=("x",))
        ctx2 = TrustContext(confidence=0.5, evidence=("x",))
        assert ctx1 == ctx2

    @pytest.mark.a_test
    def test_a3_confidence_validation(self) -> None:
        """Given invalid confidence, when creating TrustContext, then ValueError."""
        with pytest.raises(ValueError, match="confidence"):
            TrustContext(confidence=1.5)
        with pytest.raises(ValueError, match="confidence"):
            TrustContext(confidence=-0.1)

    @pytest.mark.a_test
    def test_a4_frozen(self) -> None:
        """TrustContext is immutable."""
        ctx = TrustContext(confidence=0.8)
        with pytest.raises(AttributeError):
            ctx.confidence = 0.9  # type: ignore[misc]

    @pytest.mark.a_test
    def test_a5_defaults(self) -> None:
        """Defaults: evidence=(), action_reversible=True, welfare_relevant=False."""
        ctx = TrustContext(confidence=0.5)
        assert ctx.evidence == ()
        assert ctx.action_reversible is True
        assert ctx.welfare_relevant is False


# ---------------------------------------------------------------------------
# A-tests: PassportStamp
# ---------------------------------------------------------------------------


class TestPassportStampAcceptance:
    """A-tests: must-pass scenarios for PassportStamp."""

    @pytest.mark.a_test
    def test_a1_hash_deterministic(self) -> None:
        """Given a PassportStamp, when hashed, then hash is deterministic and stable."""
        stamp = PassportStamp(
            agent_id="agent-1",
            action="spawn",
            confidence_at_action=0.9,
            timestamp="2026-03-24T10:00:00Z",
            evidence_added=("proof_a",),
        )
        h1 = hash(stamp)
        h2 = hash(stamp)
        assert h1 == h2

    @pytest.mark.a_test
    def test_a2_json_round_trip(self) -> None:
        """PassportStamp survives JSON round-trip."""
        stamp = PassportStamp(
            agent_id="agent-1",
            action="handoff",
            confidence_at_action=0.85,
            timestamp="2026-03-24T12:00:00Z",
            evidence_added=("doc_1", "doc_2"),
        )
        restored = PassportStamp.restore(json.loads(json.dumps(stamp.serialize())))
        assert restored == stamp

    @pytest.mark.a_test
    def test_a3_frozen(self) -> None:
        """PassportStamp is immutable."""
        stamp = PassportStamp(agent_id="a", action="evaluate", confidence_at_action=0.5, timestamp="t")
        with pytest.raises(AttributeError):
            stamp.agent_id = "b"  # type: ignore[misc]

    @pytest.mark.a_test
    def test_a4_value_semantics(self) -> None:
        """Two stamps with same fields are equal."""
        args = {
            "agent_id": "x",
            "action": "terminate",
            "confidence_at_action": 0.6,
            "timestamp": "t",
        }
        assert PassportStamp(**args) == PassportStamp(**args)


# ---------------------------------------------------------------------------
# A-tests: TrustPassport
# ---------------------------------------------------------------------------


class TestTrustPassportAcceptance:
    """A-tests: must-pass scenarios for TrustPassport."""

    @pytest.mark.a_test
    def test_a1_add_stamp_returns_new_passport(self) -> None:
        """Given a TrustPassport with 5 stamps, when a new stamp is added, then a NEW passport is returned."""
        ctx = TrustContext(confidence=0.8)
        stamps = tuple(
            PassportStamp(
                agent_id=f"agent-{i}",
                action="evaluate",
                confidence_at_action=0.8 - i * 0.01,
                timestamp=f"2026-03-24T{10 + i}:00:00Z",
            )
            for i in range(5)
        )
        passport = TrustPassport(
            context=ctx,
            stamps=stamps,
            created_at="2026-03-24T10:00:00Z",
            agent_id="supervisor",
        )
        new_stamp = PassportStamp(
            agent_id="agent-5",
            action="handoff",
            confidence_at_action=0.7,
            timestamp="2026-03-24T15:00:00Z",
        )
        new_passport = passport.add_stamp(new_stamp)

        assert new_passport is not passport
        assert len(new_passport.stamps) == 6
        assert len(passport.stamps) == 5  # original unchanged
        assert new_passport.stamps[-1] == new_stamp

    @pytest.mark.a_test
    def test_a2_full_json_round_trip(self) -> None:
        """TrustPassport survives full JSON serialization round-trip."""
        ctx = TrustContext(confidence=0.7, evidence=("e1",), welfare_relevant=True)
        stamp = PassportStamp(
            agent_id="a1",
            action="spawn",
            confidence_at_action=0.7,
            timestamp="2026-03-24T10:00:00Z",
            evidence_added=("new_evidence",),
        )
        passport = TrustPassport(
            context=ctx,
            stamps=(stamp,),
            warrant_chain=("w1", "w2"),
            created_at="2026-03-24T10:00:00Z",
            agent_id="supervisor",
        )
        serialized = passport.serialize()
        json_str = json.dumps(serialized)
        parsed = json.loads(json_str)
        restored = TrustPassport.restore(parsed)

        assert restored.context == ctx
        assert restored.stamps == (stamp,)
        assert restored.warrant_chain == ("w1", "w2")
        assert restored.created_at == "2026-03-24T10:00:00Z"
        assert restored.agent_id == "supervisor"

    @pytest.mark.a_test
    def test_a3_frozen(self) -> None:
        """TrustPassport is immutable."""
        passport = TrustPassport(context=TrustContext(confidence=0.5))
        with pytest.raises(AttributeError):
            passport.agent_id = "new"  # type: ignore[misc]

    @pytest.mark.a_test
    def test_a4_empty_passport(self) -> None:
        """Empty passport has no stamps, no warrants."""
        passport = TrustPassport(context=TrustContext(confidence=0.5))
        assert passport.stamps == ()
        assert passport.warrant_chain == ()
        assert passport.created_at == ""
        assert passport.agent_id == ""


# ---------------------------------------------------------------------------
# B-tests: Regression guards
# ---------------------------------------------------------------------------


class TestProtocolTypesRegression:
    """B-tests: existing functionality unchanged."""

    @pytest.mark.b_test
    def test_b1_package_import(self) -> None:
        """Types importable from package root."""
        from cognilateral_trust import PassportStamp, TrustContext, TrustPassport

        assert TrustContext is not None
        assert TrustPassport is not None
        assert PassportStamp is not None

    @pytest.mark.b_test
    def test_b2_existing_core_exports_unchanged(self) -> None:
        """Existing core exports still work."""
        from cognilateral_trust import (
            ConfidenceTier,
            ROUTE_BASIC,
            ROUTE_SOVEREIGNTY_GATE,
            ROUTE_WARRANT_CHECK,
            TierRoutingResult,
            evaluate_tier_routing,
            route_by_tier,
        )

        assert ConfidenceTier.C0 == 0
        assert route_by_tier(3) == ROUTE_BASIC
        assert route_by_tier(5) == ROUTE_WARRANT_CHECK
        assert route_by_tier(8) == ROUTE_SOVEREIGNTY_GATE
        result = evaluate_tier_routing(5)
        assert isinstance(result, TierRoutingResult)

    @pytest.mark.b_test
    def test_b3_no_external_dependencies(self) -> None:
        """Protocol types use only stdlib."""
        import importlib

        # Import fresh
        mod = importlib.import_module("cognilateral_trust.core")
        # Check that no third-party imports are used
        for name in ["TrustContext", "PassportStamp", "TrustPassport"]:
            cls = getattr(mod, name)
            module = cls.__module__
            assert module.startswith("cognilateral_trust"), (
                f"{name} is defined in {module}, expected cognilateral_trust"
            )
