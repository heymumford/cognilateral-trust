"""A-tests and B-tests for lifecycle decorators (Step 05: @spawn_gate)."""

from __future__ import annotations

import asyncio
import time
import pytest
from cognilateral_trust.core import TrustContext
from cognilateral_trust.middleware import TrustEscalation
from cognilateral_trust.lifecycle import spawn_gate, TerminationBlocked


class TestSpawnGateAcceptance:
    """A-tests: must-pass scenarios for @spawn_gate."""

    def test_a1_low_confidence_raises_escalation(self):
        """Given @spawn_gate with low confidence, when called, then TrustEscalation raised."""

        @spawn_gate(min_tier="verified")
        def create_worker(trust_context: TrustContext | None = None):
            return "worker_created"

        ctx = TrustContext(confidence=0.3)
        with pytest.raises(TrustEscalation):
            create_worker(trust_context=ctx)

    def test_a2_sufficient_confidence_proceeds(self):
        """Given @spawn_gate with sufficient confidence, when called, then function executes."""

        @spawn_gate(min_tier="basic")
        def create_worker(trust_context: TrustContext | None = None):
            return "worker_created"

        ctx = TrustContext(confidence=0.7)
        result = create_worker(trust_context=ctx)
        assert result == "worker_created"

    def test_a3_async_function_support(self):
        """Given @spawn_gate on an async function, when called, then async behavior preserved."""

        @spawn_gate(min_tier="basic")
        async def create_worker_async(trust_context: TrustContext | None = None):
            return "async_worker_created"

        async def run_test():
            ctx = TrustContext(confidence=0.8)
            return await create_worker_async(trust_context=ctx)

        result = asyncio.run(run_test())
        assert result == "async_worker_created"

    def test_a4_no_context_proceeds_with_default(self):
        """Given @spawn_gate with no trust_context kwarg, when called, then proceeds (backward compat)."""

        @spawn_gate(min_tier="basic")
        def create_worker():
            return "worker_created"

        result = create_worker()
        assert result == "worker_created"

    def test_a5_welfare_relevant_escalates(self):
        """Given welfare_relevant=True and low confidence, when spawn_gate checks, then ESCALATE."""

        @spawn_gate(min_tier="basic")
        def create_welfare_worker(trust_context: TrustContext | None = None):
            return "welfare_worker"

        ctx = TrustContext(confidence=0.5, welfare_relevant=True)
        with pytest.raises(TrustEscalation):
            create_welfare_worker(trust_context=ctx)


class TestSpawnGatePerformance:
    """FF-SG-01: spawn_gate resolves in < 5ms."""

    def test_ff_sg_01_performance(self):
        @spawn_gate(min_tier="basic")
        def fast_spawn(trust_context: TrustContext | None = None):
            return True

        ctx = TrustContext(confidence=0.9)
        start = time.perf_counter_ns()
        for _ in range(1000):
            fast_spawn(trust_context=ctx)
        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000 / 1000
        assert elapsed_ms < 5, f"spawn_gate took {elapsed_ms:.2f}ms avg, must be < 5ms"


class TestSpawnGateRegression:
    """B-tests: regression guards."""

    def test_b1_trust_escalation_has_confidence_and_threshold(self):
        """TrustEscalation exception carries confidence and threshold attrs."""

        @spawn_gate(min_tier="verified")
        def create_worker(trust_context: TrustContext | None = None):
            return "worker_created"

        ctx = TrustContext(confidence=0.2)
        with pytest.raises(TrustEscalation) as exc_info:
            create_worker(trust_context=ctx)
        assert hasattr(exc_info.value, "confidence")
        assert hasattr(exc_info.value, "threshold")

    def test_b2_existing_middleware_unchanged(self):
        """Existing trust_gate decorator still works."""
        from cognilateral_trust.middleware import trust_gate

        @trust_gate(min_confidence=0.5)
        def guarded(confidence: float) -> str:
            return "ok"

        assert guarded(confidence=0.8) == "ok"

    def test_b3_tier_thresholds_basic(self):
        """Tier 'basic' threshold is 0.0 (always proceeds unless other gates)."""

        @spawn_gate(min_tier="basic")
        def spawn(trust_context: TrustContext | None = None):
            return "ok"

        # Should proceed at any confidence >= 0.0
        assert spawn(trust_context=TrustContext(confidence=0.0)) == "ok"
        assert spawn(trust_context=TrustContext(confidence=0.1)) == "ok"

    def test_b3a_invalid_min_tier_raises_value_error(self):
        """Invalid min_tier configuration raises ValueError with a clear message."""
        with pytest.raises(ValueError, match="min_tier"):

            @spawn_gate(min_tier="invalid")
            def spawn(trust_context: TrustContext | None = None):
                return "ok"

    def test_b4_tier_thresholds_observed(self):
        """Tier 'observed' threshold is 0.3."""

        @spawn_gate(min_tier="observed")
        def spawn(trust_context: TrustContext | None = None):
            return "ok"

        # Should fail below 0.3
        with pytest.raises(TrustEscalation):
            spawn(trust_context=TrustContext(confidence=0.2))
        # Should proceed at 0.3+
        assert spawn(trust_context=TrustContext(confidence=0.3)) == "ok"

    def test_b5_tier_thresholds_verified(self):
        """Tier 'verified' threshold is 0.5."""

        @spawn_gate(min_tier="verified")
        def spawn(trust_context: TrustContext | None = None):
            return "ok"

        # Should fail below 0.5
        with pytest.raises(TrustEscalation):
            spawn(trust_context=TrustContext(confidence=0.4))
        # Should proceed at 0.5+
        assert spawn(trust_context=TrustContext(confidence=0.5)) == "ok"

    def test_b6_tier_thresholds_governance(self):
        """Tier 'governance' threshold is 0.7."""

        @spawn_gate(min_tier="governance")
        def spawn(trust_context: TrustContext | None = None):
            return "ok"

        # Should fail below 0.7
        with pytest.raises(TrustEscalation):
            spawn(trust_context=TrustContext(confidence=0.6))
        # Should proceed at 0.7+
        assert spawn(trust_context=TrustContext(confidence=0.7)) == "ok"

    def test_b7_termination_blocked_is_exception(self):
        """TerminationBlocked exception exists."""
        assert issubclass(TerminationBlocked, Exception)

    def test_b8_function_arguments_preserved(self):
        """Function args/kwargs work normally with spawn_gate."""

        @spawn_gate(min_tier="basic")
        def spawn_with_args(x: int, y: str, trust_context: TrustContext | None = None):
            return f"{x}:{y}"

        result = spawn_with_args(42, "hello", trust_context=TrustContext(confidence=0.8))
        assert result == "42:hello"

    def test_b9_async_with_args(self):
        """Async functions with args work with spawn_gate."""

        @spawn_gate(min_tier="basic")
        async def async_spawn(x: int, trust_context: TrustContext | None = None):
            return x * 2

        async def run_test():
            ctx = TrustContext(confidence=0.8)
            return await async_spawn(21, trust_context=ctx)

        result = asyncio.run(run_test())
        assert result == 42

    def test_b10_welfare_with_sufficient_confidence(self):
        """Welfare-relevant action with confidence >= 0.7 still proceeds."""

        @spawn_gate(min_tier="basic")
        def welfare_spawn(trust_context: TrustContext | None = None):
            return "welfare_ok"

        ctx = TrustContext(confidence=0.8, welfare_relevant=True)
        assert welfare_spawn(trust_context=ctx) == "welfare_ok"


# ============================================================================
# Tests for @handoff_trust Decorator
# ============================================================================


class TestHandoffTrustAcceptance:
    """A-tests: must-pass scenarios for @handoff_trust."""

    def test_ht_a1_without_passport_returns_result(self):
        """Given @handoff_trust without passport kwarg, when called, then result returned unchanged."""
        from cognilateral_trust.lifecycle import handoff_trust

        @handoff_trust
        def process_task(task: str):
            return task.upper()

        result = process_task("hello")
        assert result == "HELLO"

    def test_ht_a2_with_passport_returns_tuple(self):
        """Given @handoff_trust with passport kwarg, when called, then (result, new_passport) returned."""
        from cognilateral_trust.lifecycle import handoff_trust
        from cognilateral_trust.core import TrustContext, TrustPassport

        @handoff_trust
        def process_task(task: str, passport=None):
            return task.upper()

        ctx = TrustContext(confidence=0.8)
        passport = TrustPassport(context=ctx, agent_id="a1", created_at="2026-01-01T00:00:00Z")
        result, new_passport = process_task("hello", passport=passport)

        assert result == "HELLO"
        assert new_passport is not None
        assert len(new_passport.stamps) == len(passport.stamps) + 1
        assert new_passport.stamps[-1].action == "handoff"

    def test_ht_a3_async_with_passport(self):
        """Given @handoff_trust on async function with passport, when called, then (result, passport) returned."""
        from cognilateral_trust.lifecycle import handoff_trust
        from cognilateral_trust.core import TrustContext, TrustPassport

        @handoff_trust
        async def process_task_async(task: str, passport=None):
            return task.upper()

        async def run_test():
            ctx = TrustContext(confidence=0.8)
            passport = TrustPassport(context=ctx, agent_id="a2", created_at="2026-01-01T00:00:00Z")
            result, new_passport = await process_task_async("hello", passport=passport)
            return result, new_passport

        result, new_passport = asyncio.run(run_test())
        assert result == "HELLO"
        assert new_passport is not None
        assert new_passport.stamps[-1].action == "handoff"


# ============================================================================
# Tests for @kill_warrant Decorator
# ============================================================================


class TestKillWarrantAcceptance:
    """A-tests: must-pass scenarios for @kill_warrant."""

    def test_kw_a1_welfare_relevant_blocks_termination(self):
        """Given welfare_relevant=True, when @kill_warrant checks, then TerminationBlocked raised."""
        from cognilateral_trust.lifecycle import kill_warrant, TerminationBlocked
        from cognilateral_trust.core import TrustContext, TrustPassport

        @kill_warrant
        def terminate_agent(passport=None):
            return "terminated"

        ctx = TrustContext(confidence=0.9, welfare_relevant=True)
        passport = TrustPassport(context=ctx, agent_id="a3", created_at="2026-01-01T00:00:00Z")

        with pytest.raises(TerminationBlocked) as exc_info:
            terminate_agent(passport=passport)
        assert "D-05" in exc_info.value.reason
        assert exc_info.value.agent_id == "a3"

    def test_kw_a2_no_unresolved_allows_termination(self):
        """Given no unresolved obligations, when @kill_warrant checks, then proceeds."""
        from cognilateral_trust.lifecycle import kill_warrant
        from cognilateral_trust.core import TrustContext, TrustPassport

        @kill_warrant
        def terminate_agent(passport=None):
            return "terminated"

        ctx = TrustContext(confidence=0.9, welfare_relevant=False)
        passport = TrustPassport(context=ctx, agent_id="a4", created_at="2026-01-01T00:00:00Z")

        result = terminate_agent(passport=passport)
        assert result == "terminated"

    def test_kw_a3_unresolved_obligations_block(self):
        """Given unresolved evaluation stamps, when @kill_warrant checks, then TerminationBlocked raised."""
        from cognilateral_trust.lifecycle import kill_warrant, TerminationBlocked
        from cognilateral_trust.core import TrustContext, TrustPassport, PassportStamp

        @kill_warrant
        def terminate_agent(passport=None):
            return "terminated"

        ctx = TrustContext(confidence=0.9, welfare_relevant=False)
        passport = TrustPassport(context=ctx, agent_id="a5", created_at="2026-01-01T00:00:00Z")
        # Add an evaluation stamp without a subsequent resolution
        eval_stamp = PassportStamp(
            agent_id="a5",
            action="evaluated",
            confidence_at_action=0.8,
            timestamp="2026-01-01T00:01:00Z",
        )
        passport = passport.add_stamp(eval_stamp)

        with pytest.raises(TerminationBlocked) as exc_info:
            terminate_agent(passport=passport)
        assert "unresolved" in exc_info.value.reason.lower()
        assert exc_info.value.agent_id == "a5"
        assert len(exc_info.value.unresolved) > 0


# ============================================================================
# Tests for suspend_trust / wake_trust
# ============================================================================


class TestSuspendWakeTrust:
    """Tests for suspend_trust and wake_trust serialization."""

    def test_sw_a1_suspend_and_wake_roundtrip(self):
        """Given a TrustPassport, when suspend then wake, then passport restored."""
        from cognilateral_trust.lifecycle import suspend_trust, wake_trust
        from cognilateral_trust.core import TrustContext, TrustPassport

        ctx = TrustContext(confidence=0.8, welfare_relevant=False)
        original_passport = TrustPassport(context=ctx, agent_id="a6", created_at="2026-01-01T00:00:00Z")

        # Suspend
        suspended = suspend_trust(original_passport)
        assert isinstance(suspended, str)

        # Wake
        restored_passport = wake_trust(suspended)
        assert restored_passport.agent_id == original_passport.agent_id
        assert restored_passport.context.confidence == original_passport.context.confidence


class TestTerminationBlockedAPI:
    """Tests for TerminationBlocked exception API."""

    def test_tb_a1_backward_compat_agent_id_optional(self):
        """Given TerminationBlocked is raised, when agent_id is None, then still valid."""
        from cognilateral_trust.lifecycle import TerminationBlocked

        exc = TerminationBlocked(reason="test", agent_id=None)
        assert exc.reason == "test"
        assert exc.agent_id is None

    def test_tb_a2_agent_id_present_in_exception(self):
        """Given TerminationBlocked with agent_id, when raised, then agent_id accessible."""
        from cognilateral_trust.lifecycle import TerminationBlocked

        exc = TerminationBlocked(reason="test", agent_id="a7")
        assert exc.agent_id == "a7"

    def test_tb_a3_unresolved_obligations_included(self):
        """Given TerminationBlocked with unresolved, when raised, then unresolved accessible."""
        from cognilateral_trust.lifecycle import TerminationBlocked

        exc = TerminationBlocked(reason="test", unresolved=("eval at time1", "check at time2"))
        assert len(exc.unresolved) == 2
        assert "eval" in exc.unresolved[0]
