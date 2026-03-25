"""Tests for trust decay — confidence attenuation with agent-to-agent distance.

TDD — tests written before implementation. Covers:
- exponential_decay: confidence decays exponentially with hops
- linear_decay: confidence decays linearly per hop
- decay dispatch: route to appropriate function by curve type
- Identity property: decay(c, 0) == c
- Monotonicity: decay(c, n) >= decay(c, n+1) for all n (FF-TD-01)
- Bounds: result always in [0.0, 1.0]
- Edge cases: zero confidence, max confidence, half-life parameters
"""

from __future__ import annotations

import pytest

from cognilateral_trust.network.decay import decay, exponential_decay, linear_decay


class TestExponentialDecay:
    """Exponential decay curve: confidence * (0.5 ^ (hops / half_life))."""

    def test_identity_at_zero_hops(self) -> None:
        """decay(c, 0) == c regardless of curve."""
        assert exponential_decay(0.9, 0) == 0.9
        assert exponential_decay(0.5, 0) == 0.5
        assert exponential_decay(1.0, 0) == 1.0

    def test_decreases_with_hops(self) -> None:
        """Confidence strictly decreases as hops increase."""
        c = exponential_decay(0.9, 5, half_life=3.0)
        assert c < 0.9
        assert c > 0.0

    def test_monotonicity(self) -> None:
        """decay(c, n) >= decay(c, n+1) for all n (FF-TD-01)."""
        confidence = 0.9
        prev = confidence
        for hops in range(1, 10):
            curr = exponential_decay(confidence, hops)
            assert prev >= curr, f"Monotonicity violated at hops={hops}"
            prev = curr

    def test_bounds(self) -> None:
        """Result always in [0.0, 1.0]."""
        for c in [0.0, 0.1, 0.5, 0.9, 1.0]:
            for hops in range(20):
                result = exponential_decay(c, hops)
                assert 0.0 <= result <= 1.0, f"Out of bounds: {result}"

    def test_custom_half_life(self) -> None:
        """Shorter half_life → faster decay."""
        c1 = exponential_decay(0.9, 5, half_life=2.0)
        c2 = exponential_decay(0.9, 5, half_life=5.0)
        assert c1 < c2


class TestLinearDecay:
    """Linear decay curve: max(0, confidence - (rate * hops))."""

    def test_identity_at_zero_hops(self) -> None:
        """decay(c, 0) == c regardless of curve."""
        assert linear_decay(0.9, 0) == 0.9
        assert linear_decay(0.5, 0) == 0.5
        assert linear_decay(1.0, 0) == 1.0

    def test_decreases_with_hops(self) -> None:
        """Confidence decreases linearly by rate per hop."""
        c = linear_decay(0.9, 5, rate=0.1)
        assert c == pytest.approx(0.4)  # 0.9 - (0.1 * 5) = 0.4

    def test_monotonicity(self) -> None:
        """decay(c, n) >= decay(c, n+1) for all n (FF-TD-01)."""
        confidence = 0.9
        prev = confidence
        for hops in range(1, 10):
            curr = linear_decay(confidence, hops, rate=0.1)
            assert prev >= curr, f"Monotonicity violated at hops={hops}"
            prev = curr

    def test_bounds(self) -> None:
        """Result always in [0.0, 1.0]."""
        for c in [0.0, 0.1, 0.5, 0.9, 1.0]:
            for hops in range(20):
                result = linear_decay(c, hops, rate=0.1)
                assert 0.0 <= result <= 1.0, f"Out of bounds: {result}"

    def test_custom_rate(self) -> None:
        """Higher rate → faster decay."""
        c1 = linear_decay(0.9, 5, rate=0.05)
        c2 = linear_decay(0.9, 5, rate=0.1)
        assert c1 > c2


class TestDecayDispatch:
    """Test the dispatch function routing to exponential or linear."""

    def test_dispatch_exponential(self) -> None:
        """decay(..., curve='exponential') delegates to exponential_decay."""
        c1 = decay(0.9, 5, curve="exponential", half_life=3.0)
        c2 = exponential_decay(0.9, 5, half_life=3.0)
        assert c1 == pytest.approx(c2)

    def test_dispatch_linear(self) -> None:
        """decay(..., curve='linear') delegates to linear_decay."""
        c1 = decay(0.9, 5, curve="linear", rate=0.1)
        c2 = linear_decay(0.9, 5, rate=0.1)
        assert c1 == pytest.approx(c2)

    def test_default_is_exponential(self) -> None:
        """Default curve is exponential."""
        c1 = decay(0.9, 5)
        c2 = exponential_decay(0.9, 5, half_life=3.0)
        assert c1 == pytest.approx(c2)

    def test_invalid_curve_raises(self) -> None:
        """Invalid curve type raises ValueError."""
        with pytest.raises(ValueError):
            decay(0.9, 5, curve="sigmoid")

    def test_identity_dispatch(self) -> None:
        """Both curves return identity at zero hops via dispatch."""
        assert decay(0.9, 0, curve="exponential") == 0.9
        assert decay(0.9, 0, curve="linear") == 0.9
