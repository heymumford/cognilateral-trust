"""Sprint S3: Tests for cognilateral_trust.middleware module.

Covers:
- @trust_gate decorator (allow above threshold, raise below)
- TrustEscalation exception with attached accountability record
- async_evaluate_trust() matching sync behavior
- TrustMiddleware callable pipeline pattern
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from cognilateral_trust.middleware import (
    TrustEscalation,
    TrustMiddleware,
    async_evaluate_trust,
    trust_gate,
)


# ---------------------------------------------------------------------------
# TrustEscalation exception
# ---------------------------------------------------------------------------


class TestTrustEscalation:
    def test_is_exception(self) -> None:
        exc = TrustEscalation(confidence=0.3, threshold=0.7)
        assert isinstance(exc, Exception)

    def test_has_confidence_and_threshold(self) -> None:
        exc = TrustEscalation(confidence=0.3, threshold=0.7)
        assert exc.confidence == pytest.approx(0.3)
        assert exc.threshold == pytest.approx(0.7)

    def test_has_accountability_record(self) -> None:
        from cognilateral_trust.accountability import AccountabilityRecord

        exc = TrustEscalation(confidence=0.3, threshold=0.7)
        assert exc.accountability_record is not None
        assert isinstance(exc.accountability_record, AccountabilityRecord)

    def test_accountability_record_verdict_is_escalate(self) -> None:
        exc = TrustEscalation(confidence=0.3, threshold=0.7)
        assert exc.accountability_record.verdict == "ESCALATE"

    def test_message_includes_confidence_and_threshold(self) -> None:
        exc = TrustEscalation(confidence=0.3, threshold=0.7)
        msg = str(exc)
        assert "0.3" in msg or "0.30" in msg
        assert "0.7" in msg or "0.70" in msg

    def test_optional_context(self) -> None:
        exc = TrustEscalation(confidence=0.3, threshold=0.7, context={"task": "deploy"})
        assert exc.accountability_record is not None


# ---------------------------------------------------------------------------
# @trust_gate decorator
# ---------------------------------------------------------------------------


class TestTrustGateDecorator:
    def test_allows_execution_above_threshold(self) -> None:
        @trust_gate(min_confidence=0.5)
        def action(confidence: float) -> str:
            return "executed"

        result = action(confidence=0.8)
        assert result == "executed"

    def test_raises_trust_escalation_below_threshold(self) -> None:
        @trust_gate(min_confidence=0.7)
        def action(confidence: float) -> str:
            return "executed"

        with pytest.raises(TrustEscalation):
            action(confidence=0.4)

    def test_escalation_has_correct_confidence(self) -> None:
        @trust_gate(min_confidence=0.7)
        def action(confidence: float) -> str:
            return "executed"

        with pytest.raises(TrustEscalation) as exc_info:
            action(confidence=0.35)

        assert exc_info.value.confidence == pytest.approx(0.35)

    def test_escalation_has_correct_threshold(self) -> None:
        @trust_gate(min_confidence=0.8)
        def action(confidence: float) -> str:
            return "executed"

        with pytest.raises(TrustEscalation) as exc_info:
            action(confidence=0.5)

        assert exc_info.value.threshold == pytest.approx(0.8)

    def test_exactly_at_threshold_proceeds(self) -> None:
        @trust_gate(min_confidence=0.7)
        def action(confidence: float) -> str:
            return "at-threshold"

        result = action(confidence=0.7)
        assert result == "at-threshold"

    def test_decorated_function_passes_args_through(self) -> None:
        @trust_gate(min_confidence=0.5)
        def greet(name: str, *, confidence: float) -> str:
            return f"Hello, {name}"

        result = greet("World", confidence=0.9)
        assert result == "Hello, World"

    def test_default_threshold_is_reasonable(self) -> None:
        """Default threshold should be 0.5 — the neutral midpoint."""

        @trust_gate()
        def action(confidence: float) -> str:
            return "ok"

        # 0.6 above default threshold
        assert action(confidence=0.6) == "ok"
        # 0.3 below default threshold
        with pytest.raises(TrustEscalation):
            action(confidence=0.3)

    def test_accountability_record_attached_to_exception(self) -> None:
        @trust_gate(min_confidence=0.7)
        def action(confidence: float) -> None:
            pass

        with pytest.raises(TrustEscalation) as exc_info:
            action(confidence=0.2)

        assert exc_info.value.accountability_record is not None
        assert exc_info.value.accountability_record.record_id

    def test_decorator_preserves_function_name(self) -> None:
        @trust_gate(min_confidence=0.5)
        def my_special_action(confidence: float) -> None:
            pass

        assert my_special_action.__name__ == "my_special_action"

    def test_non_reversible_kwarg_forwarded(self) -> None:
        @trust_gate(min_confidence=0.5, is_reversible=False)
        def irreversible_action(confidence: float) -> str:
            return "done"

        # High confidence but irreversible at sovereignty tier → escalate
        with pytest.raises(TrustEscalation):
            irreversible_action(confidence=0.95)

    def test_zero_confidence_escalates(self) -> None:
        @trust_gate(min_confidence=0.5)
        def action(confidence: float) -> None:
            pass

        with pytest.raises(TrustEscalation):
            action(confidence=0.0)

    def test_full_confidence_proceeds(self) -> None:
        @trust_gate(min_confidence=0.9)
        def action(confidence: float) -> str:
            return "full-confidence"

        result = action(confidence=1.0)
        assert result == "full-confidence"


# ---------------------------------------------------------------------------
# async_evaluate_trust
# ---------------------------------------------------------------------------


class TestAsyncEvaluateTrust:
    def test_returns_same_result_as_sync(self) -> None:
        from cognilateral_trust.evaluate import evaluate_trust

        sync_result = evaluate_trust(0.75)
        async_result = asyncio.run(async_evaluate_trust(0.75))

        assert async_result.confidence == sync_result.confidence
        assert async_result.tier == sync_result.tier
        assert async_result.route == sync_result.route
        assert async_result.should_proceed == sync_result.should_proceed

    def test_async_is_awaitable(self) -> None:
        import inspect

        coro = async_evaluate_trust(0.6)
        assert inspect.iscoroutine(coro)
        # clean up
        asyncio.run(coro)

    def test_async_with_kwargs(self) -> None:
        result = asyncio.run(async_evaluate_trust(0.9, is_reversible=False, touches_external=True))
        assert result is not None
        # Sovereignty tier + irreversible → escalate
        assert result.should_proceed is False

    def test_async_low_confidence(self) -> None:
        result = asyncio.run(async_evaluate_trust(0.1))
        assert result.should_proceed is True  # C1, basic route, still proceeds

    def test_async_result_has_accountability_record(self) -> None:
        result = asyncio.run(async_evaluate_trust(0.5))
        assert result.accountability_record is not None


# ---------------------------------------------------------------------------
# TrustMiddleware callable pipeline
# ---------------------------------------------------------------------------


class TestTrustMiddleware:
    def _make_call(self, middleware: TrustMiddleware, confidence: float, **kwargs: Any) -> Any:
        """Simulate a middleware pipeline call."""
        return middleware(confidence=confidence, **kwargs)

    def test_allows_call_above_threshold(self) -> None:
        middleware = TrustMiddleware(min_confidence=0.6)
        result = self._make_call(middleware, 0.8)
        assert result is not None
        assert result.should_proceed is True

    def test_raises_below_threshold(self) -> None:
        middleware = TrustMiddleware(min_confidence=0.7)
        with pytest.raises(TrustEscalation):
            self._make_call(middleware, 0.3)

    def test_middleware_is_callable(self) -> None:
        middleware = TrustMiddleware(min_confidence=0.5)
        assert callable(middleware)

    def test_returns_trust_evaluation(self) -> None:
        from cognilateral_trust.evaluate import TrustEvaluation

        middleware = TrustMiddleware(min_confidence=0.5)
        result = middleware(confidence=0.8)
        assert isinstance(result, TrustEvaluation)

    def test_pipeline_chaining(self) -> None:
        """Multiple middleware instances applied in sequence."""
        gate1 = TrustMiddleware(min_confidence=0.4)
        gate2 = TrustMiddleware(min_confidence=0.6)

        confidence = 0.7
        result1 = gate1(confidence=confidence)
        assert result1.should_proceed is True
        result2 = gate2(confidence=confidence)
        assert result2.should_proceed is True

    def test_pipeline_short_circuits_on_low_confidence(self) -> None:
        gate1 = TrustMiddleware(min_confidence=0.3)
        gate2 = TrustMiddleware(min_confidence=0.7)

        confidence = 0.5
        result1 = gate1(confidence=confidence)
        assert result1.should_proceed is True

        # Gate2 should block
        with pytest.raises(TrustEscalation):
            gate2(confidence=confidence)

    def test_custom_threshold_per_instance(self) -> None:
        strict = TrustMiddleware(min_confidence=0.95)
        lenient = TrustMiddleware(min_confidence=0.3)

        confidence = 0.6
        lenient_result = lenient(confidence=confidence)
        assert lenient_result.should_proceed is True

        with pytest.raises(TrustEscalation):
            strict(confidence=confidence)
