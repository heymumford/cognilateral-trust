"""Tests for @sovereign_worker decorator — leaf-level self-governance.

TDD — tests written before implementation. Covers:
- Decorator on sync and async functions
- Raises SovereignRefusal when confidence < min_confidence
- D-05 welfare constraint: welfare_relevant=True forces min_confidence=0.9 (immutable)
- No parameter can override the D-05 hard constraint
- Function is called if confidence sufficient
- Works with various argument signatures
"""

from __future__ import annotations

import asyncio

import pytest

from cognilateral_trust.network.sovereign import SovereignRefusal, sovereign_worker


class TestSovereignWorkerSync:
    """@sovereign_worker on synchronous functions."""

    def test_insufficient_confidence_raises(self) -> None:
        """Function raises SovereignRefusal when confidence < min_confidence."""

        @sovereign_worker(min_confidence=0.7)
        def do_work(confidence: float) -> str:
            return "done"

        with pytest.raises(SovereignRefusal):
            do_work(confidence=0.5)

    def test_sufficient_confidence_executes(self) -> None:
        """Function executes when confidence >= min_confidence."""

        @sovereign_worker(min_confidence=0.7)
        def do_work(confidence: float) -> str:
            return "done"

        result = do_work(confidence=0.8)
        assert result == "done"

    def test_default_min_confidence(self) -> None:
        """Default min_confidence is 0.5."""

        @sovereign_worker()
        def do_work(confidence: float) -> str:
            return "done"

        # Should pass at 0.5
        result = do_work(confidence=0.5)
        assert result == "done"

        # Should fail below 0.5
        with pytest.raises(SovereignRefusal):
            do_work(confidence=0.49)

    def test_welfare_constraint_hardcoded(self) -> None:
        """D-05 welfare constraint: welfare_relevant=True forces min_confidence=0.9."""

        @sovereign_worker(welfare_relevant=True, min_confidence=0.5)
        def critical_action(confidence: float) -> str:
            return "done"

        # Even though min_confidence=0.5, welfare_relevant forces 0.9
        with pytest.raises(SovereignRefusal):
            critical_action(confidence=0.8)

        # Only 0.9+ passes
        result = critical_action(confidence=0.9)
        assert result == "done"

    def test_welfare_constraint_no_override(self) -> None:
        """D-05 welfare constraint cannot be overridden by any parameter."""

        @sovereign_worker(welfare_relevant=True, welfare_escalation=0.5)
        def critical_action(confidence: float) -> str:
            return "done"

        # welfare_escalation parameter is ignored; D-05 is always 0.9
        with pytest.raises(SovereignRefusal):
            critical_action(confidence=0.8)

    def test_error_message_includes_context(self) -> None:
        """SovereignRefusal error includes confidence and threshold."""

        @sovereign_worker(min_confidence=0.7)
        def do_work(confidence: float) -> str:
            return "done"

        try:
            do_work(confidence=0.5)
            pytest.fail("Should have raised SovereignRefusal")
        except SovereignRefusal as e:
            assert "0.5" in str(e) or "confidence" in str(e).lower()

    def test_with_args_and_kwargs(self) -> None:
        """Decorator works with various function signatures."""

        @sovereign_worker(min_confidence=0.7)
        def work_with_args(x: int, y: int, confidence: float) -> int:
            return x + y

        result = work_with_args(1, 2, confidence=0.8)
        assert result == 3

    def test_with_positional_confidence(self) -> None:
        """Decorator extracts confidence from positional or keyword args."""

        @sovereign_worker(min_confidence=0.7)
        def work(confidence: float, data: str) -> str:
            return f"{data}:{confidence}"

        result = work(0.8, "test")
        assert result == "test:0.8"

        with pytest.raises(SovereignRefusal):
            work(0.5, "test")


class TestSovereignWorkerAsync:
    """@sovereign_worker on async functions."""

    def test_async_insufficient_confidence_raises(self) -> None:
        """Async function raises SovereignRefusal when confidence < min_confidence."""

        @sovereign_worker(min_confidence=0.7)
        async def async_work(confidence: float) -> str:
            await asyncio.sleep(0.001)
            return "done"

        with pytest.raises(SovereignRefusal):
            asyncio.run(async_work(confidence=0.5))

    def test_async_sufficient_confidence_executes(self) -> None:
        """Async function executes when confidence >= min_confidence."""

        @sovereign_worker(min_confidence=0.7)
        async def async_work(confidence: float) -> str:
            await asyncio.sleep(0.001)
            return "done"

        result = asyncio.run(async_work(confidence=0.8))
        assert result == "done"

    def test_async_welfare_constraint(self) -> None:
        """D-05 welfare constraint applies to async functions."""

        @sovereign_worker(welfare_relevant=True)
        async def critical_async(confidence: float) -> str:
            await asyncio.sleep(0.001)
            return "done"

        with pytest.raises(SovereignRefusal):
            asyncio.run(critical_async(confidence=0.8))

        result = asyncio.run(critical_async(confidence=0.9))
        assert result == "done"


class TestSovereignRefusalException:
    """SovereignRefusal exception."""

    def test_create(self) -> None:
        """SovereignRefusal can be instantiated."""
        exc = SovereignRefusal("Low confidence: 0.5 < 0.7")
        assert "0.5" in str(exc)

    def test_is_exception(self) -> None:
        """SovereignRefusal is an Exception."""
        assert issubclass(SovereignRefusal, Exception)
