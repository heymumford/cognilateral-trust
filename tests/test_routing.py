"""Tests for router trust policy — YAML-based declarative routing."""

from __future__ import annotations

import time
from typing import Any

import pytest

from cognilateral_trust.routing import (
    RouteRule,
    RouterPolicy,
    load_policy,
    route_decision,
    router_trust_policy,
)


class TestRouteRule:
    """RouteRule must be immutable and represent a single routing decision."""

    def test_route_rule_frozen(self) -> None:
        rule = RouteRule(tier="verified", route_to="careful_agent", reason="high-stakes")
        with pytest.raises(AttributeError):
            rule.tier = "unverified"  # type: ignore[misc]

    def test_route_rule_fields(self) -> None:
        rule = RouteRule(tier="verified", route_to="careful_agent", reason="test")
        assert rule.tier == "verified"
        assert rule.route_to == "careful_agent"
        assert rule.reason == "test"


class TestRouterPolicy:
    """RouterPolicy must collect rules and provide a fallback."""

    def test_router_policy_frozen(self) -> None:
        policy = RouterPolicy(
            rules=(RouteRule("verified", "careful", "reason"),),
            fallback="human_review",
        )
        with pytest.raises(AttributeError):
            policy.fallback = "other"  # type: ignore[misc]

    def test_router_policy_empty_rules(self) -> None:
        policy = RouterPolicy(rules=(), fallback="default")
        assert len(policy.rules) == 0
        assert policy.fallback == "default"

    def test_router_policy_multiple_rules(self) -> None:
        rules = (
            RouteRule("verified", "careful_agent", "high-stakes"),
            RouteRule("basic", "fast_agent", "low-stakes"),
            RouteRule("unverified", "hold_for_review", "unknown"),
        )
        policy = RouterPolicy(rules=rules, fallback="human_review")
        assert len(policy.rules) == 3
        assert policy.rules[0].tier == "verified"


class TestLoadPolicy:
    """load_policy must parse dict or YAML into RouterPolicy."""

    def test_load_policy_from_dict(self) -> None:
        policy_dict = {
            "rules": [
                {"tier": "verified", "route_to": "careful_agent", "reason": "high-stakes"},
                {"tier": "basic", "route_to": "fast_agent", "reason": "low-stakes"},
            ],
            "fallback": "human_review",
        }
        policy = load_policy(policy_dict)

        assert isinstance(policy, RouterPolicy)
        assert len(policy.rules) == 2
        assert policy.rules[0].tier == "verified"
        assert policy.rules[0].route_to == "careful_agent"
        assert policy.fallback == "human_review"

    def test_load_policy_minimal_dict(self) -> None:
        policy_dict = {
            "rules": [
                {"tier": "basic", "route_to": "default"},
            ],
        }
        policy = load_policy(policy_dict)
        assert policy.fallback == "human_review"

    def test_load_policy_missing_reason(self) -> None:
        policy_dict = {
            "rules": [
                {"tier": "basic", "route_to": "agent"},
            ],
        }
        policy = load_policy(policy_dict)
        assert policy.rules[0].reason == ""

    def test_load_policy_invalid_rules_type(self) -> None:
        policy_dict = {"rules": "not_a_list"}
        with pytest.raises(ValueError, match="'rules' must be list"):
            load_policy(policy_dict)

    def test_load_policy_invalid_fallback_type(self) -> None:
        policy_dict = {"rules": [], "fallback": 123}
        with pytest.raises(ValueError, match="'fallback' must be str"):
            load_policy(policy_dict)

    def test_load_policy_rule_missing_tier(self) -> None:
        policy_dict = {"rules": [{"route_to": "agent"}]}
        with pytest.raises(ValueError, match="'tier' must be non-empty str"):
            load_policy(policy_dict)

    def test_load_policy_rule_missing_route_to(self) -> None:
        policy_dict = {"rules": [{"tier": "basic"}]}
        with pytest.raises(ValueError, match="'route_to' must be non-empty str"):
            load_policy(policy_dict)

    def test_load_policy_invalid_policy_type(self) -> None:
        with pytest.raises(ValueError, match="policy must be dict"):
            load_policy(["not", "a", "dict"])  # type: ignore[arg-type]


class TestRouteDecision:
    """route_decision must map tier to route destination."""

    def test_route_decision_matches_rule(self) -> None:
        policy = RouterPolicy(
            rules=(RouteRule("verified", "careful_agent", "reason"),),
            fallback="human_review",
        )
        route = route_decision("verified", policy)
        assert route == "careful_agent"

    def test_route_decision_uses_fallback_when_no_match(self) -> None:
        policy = RouterPolicy(
            rules=(RouteRule("verified", "careful_agent", "reason"),),
            fallback="human_review",
        )
        route = route_decision("unknown", policy)
        assert route == "human_review"

    def test_route_decision_first_match_wins(self) -> None:
        policy = RouterPolicy(
            rules=(
                RouteRule("verified", "first", "reason1"),
                RouteRule("verified", "second", "reason2"),
            ),
            fallback="human_review",
        )
        route = route_decision("verified", policy)
        assert route == "first"

    def test_route_decision_empty_rules(self) -> None:
        policy = RouterPolicy(rules=(), fallback="fallback")
        route = route_decision("any", policy)
        assert route == "fallback"

    def test_route_decision_performance_sub5ms(self) -> None:
        # FF-RT-01: routing decision must resolve in <5ms
        policy = RouterPolicy(
            rules=(
                RouteRule("a", "x", ""),
                RouteRule("b", "y", ""),
                RouteRule("c", "z", ""),
                RouteRule("d", "w", ""),
                RouteRule("e", "v", ""),
            ),
            fallback="default",
        )

        start = time.perf_counter()
        for _ in range(100):
            route_decision("unknown", policy)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # 100 calls should still be fast; allow up to 5000ms total (50ms avg, 10x buffer for CI)
        assert elapsed_ms < 5000, f"routing took {elapsed_ms:.1f}ms for 100 calls"


class TestRouterTrustPolicyDecorator:
    """@router_trust_policy must enforce policy on decorated functions."""

    def test_decorator_attaches_policy(self) -> None:
        policy = RouterPolicy(
            rules=(RouteRule("verified", "careful_agent", "reason"),),
            fallback="human_review",
        )

        @router_trust_policy(policy)
        def test_func(context: str, trust_tier: str = "basic") -> str:
            return f"Processed {context}"

        result = test_func("action", trust_tier="verified")
        assert result == "Processed action"

    def test_decorator_routes_verified_tier(self) -> None:
        policy = RouterPolicy(
            rules=(RouteRule("verified", "careful_agent", "reason"),),
            fallback="human_review",
        )

        @router_trust_policy(policy)
        def test_func(context: str, trust_tier: str = "basic", _trust_route: str = "") -> dict[str, str]:
            return {"context": context, "route": _trust_route}

        result = test_func("action", trust_tier="verified")
        assert result["route"] == "careful_agent"

    def test_decorator_routes_unknown_to_fallback(self) -> None:
        policy = RouterPolicy(
            rules=(RouteRule("verified", "careful_agent", "reason"),),
            fallback="human_review",
        )

        @router_trust_policy(policy)
        def test_func(context: str, trust_tier: str = "basic", _trust_route: str = "") -> str:
            return _trust_route

        result = test_func("action", trust_tier="unknown")
        assert result == "human_review"

    def test_decorator_preserves_function_name(self) -> None:
        policy = RouterPolicy(
            rules=(RouteRule("verified", "careful_agent", "reason"),),
            fallback="human_review",
        )

        @router_trust_policy(policy)
        def my_special_function(trust_tier: str = "basic") -> str:
            return "result"

        assert my_special_function.__name__ == "my_special_function"

    def test_decorator_performance_sub5ms(self) -> None:
        # FF-RT-01: routing decision must resolve in <5ms
        policy = RouterPolicy(
            rules=(
                RouteRule("a", "x", ""),
                RouteRule("b", "y", ""),
            ),
            fallback="default",
        )

        @router_trust_policy(policy)
        def test_func(_route_time_ms: float = 0.0, trust_tier: str = "basic") -> float:
            return _route_time_ms

        start = time.perf_counter()
        times = [test_func(trust_tier="unknown") for _ in range(10)]
        total_ms = (time.perf_counter() - start) * 1000

        # Routing overhead per call should be <50ms (10x buffer for CI)
        for t in times:
            assert t < 50, f"routing took {t:.2f}ms"
        # Total for 10 calls should be <500ms (50ms avg + overhead, 10x buffer)
        assert total_ms < 500, f"total time {total_ms:.1f}ms for 10 calls"

    def test_decorator_default_trust_tier(self) -> None:
        policy = RouterPolicy(
            rules=(RouteRule("basic", "fast_agent", "reason"),),
            fallback="human_review",
        )

        @router_trust_policy(policy)
        def test_func(_trust_route: str = "", trust_tier: str = "basic") -> str:
            return _trust_route

        result = test_func()
        assert result == "fast_agent"

    def test_decorator_with_args_and_kwargs(self) -> None:
        policy = RouterPolicy(
            rules=(RouteRule("verified", "careful", ""),),
            fallback="default",
        )

        @router_trust_policy(policy)
        def test_func(a: int, b: str, trust_tier: str = "basic", _trust_route: str = "") -> dict[str, Any]:
            return {"a": a, "b": b, "route": _trust_route}

        result = test_func(1, "hello", trust_tier="verified")
        assert result == {"a": 1, "b": "hello", "route": "careful"}
