"""Router trust policy — YAML-based declarative routing by trust tier."""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, TypeVar

__all__ = [
    "RouteRule",
    "RouterPolicy",
    "load_policy",
    "router_trust_policy",
    "route_decision",
]


@dataclass(frozen=True)
class RouteRule:
    """Immutable routing rule mapping tier to destination."""

    tier: str
    route_to: str
    reason: str


@dataclass(frozen=True)
class RouterPolicy:
    """Immutable collection of routing rules with fallback."""

    rules: tuple[RouteRule, ...]
    fallback: str


def load_policy(policy_source: dict[str, Any] | str) -> RouterPolicy:
    """Parse routing policy from dict or YAML path.

    Args:
        policy_source: Either a dict with 'rules' and 'fallback' keys,
                      or a file path to a YAML policy file.

    Returns:
        RouterPolicy with parsed rules and fallback.

    Raises:
        ValueError: If policy structure is invalid.
    """
    if isinstance(policy_source, str):
        try:
            import yaml
        except ImportError:
            msg = "yaml not installed; pass policy as dict instead"
            raise ImportError(msg) from None
        with open(policy_source) as f:
            policy_dict = yaml.safe_load(f)
    else:
        policy_dict = policy_source

    if not isinstance(policy_dict, dict):
        msg = f"policy must be dict, got {type(policy_dict).__name__}"
        raise ValueError(msg)

    rules_data = policy_dict.get("rules", [])
    if not isinstance(rules_data, list):
        msg = f"'rules' must be list, got {type(rules_data).__name__}"
        raise ValueError(msg)

    fallback = policy_dict.get("fallback", "human_review")
    if not isinstance(fallback, str):
        msg = f"'fallback' must be str, got {type(fallback).__name__}"
        raise ValueError(msg)

    rules: list[RouteRule] = []
    for i, rule_data in enumerate(rules_data):
        if not isinstance(rule_data, dict):
            msg = f"rule {i} must be dict, got {type(rule_data).__name__}"
            raise ValueError(msg)

        tier = rule_data.get("tier")
        route_to = rule_data.get("route_to")
        reason = rule_data.get("reason", "")

        if not isinstance(tier, str) or not tier:
            msg = f"rule {i}: 'tier' must be non-empty str, got {tier!r}"
            raise ValueError(msg)
        if not isinstance(route_to, str) or not route_to:
            msg = f"rule {i}: 'route_to' must be non-empty str, got {route_to!r}"
            raise ValueError(msg)

        rules.append(RouteRule(tier=tier, route_to=route_to, reason=reason))

    return RouterPolicy(rules=tuple(rules), fallback=fallback)


def route_decision(tier: str, policy: RouterPolicy) -> str:
    """Evaluate routing decision for a trust tier.

    Searches policy.rules for matching tier and returns route_to destination.
    Falls back to policy.fallback if no rule matches.

    Args:
        tier: Confidence tier string (e.g., "verified", "basic", "unverified").
        policy: RouterPolicy with rules and fallback.

    Returns:
        Route destination string (e.g., "careful_agent", "human_review").
    """
    for rule in policy.rules:
        if rule.tier == tier:
            return rule.route_to
    return policy.fallback


F = TypeVar("F", bound=Callable[..., Any])


def router_trust_policy(policy: RouterPolicy) -> Callable[[F], F]:
    """Decorator that enforces trust routing policy on a function.

    The decorated function can optionally receive 'trust_tier' kwarg.
    Before execution, the decorator evaluates the routing decision.
    Optional routing metadata (_trust_route, _route_time_ms) is only
    injected if the function's signature accepts it.

    Args:
        policy: RouterPolicy to enforce.

    Returns:
        Decorator function.

    Example:
        policy = RouterPolicy(
            rules=(
                RouteRule("verified", "careful_agent", "High-stakes"),
                RouteRule("basic", "fast_agent", "Low-stakes"),
            ),
            fallback="human_review",
        )

        @router_trust_policy(policy)
        def process_decision(context: str, trust_tier: str = "basic") -> str:
            return f"Processing {context} via tier {trust_tier}"

        result = process_decision("action", trust_tier="verified")
        # Result routes via "careful_agent" per policy
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Extract trust_tier from kwargs or use default
            trust_tier = kwargs.pop("trust_tier", "basic")

            start = time.perf_counter()
            route = route_decision(trust_tier, policy)
            elapsed_ms = (time.perf_counter() - start) * 1000

            # Prepare call arguments — only include trust_tier/route metadata if func accepts it
            import inspect

            sig = inspect.signature(func)
            param_names = set(sig.parameters.keys())

            # Always pass trust_tier if the function accepts it
            if "trust_tier" in param_names:
                kwargs["trust_tier"] = trust_tier

            # Only inject route metadata if function signature explicitly accepts it
            if "_trust_route" in param_names:
                kwargs["_trust_route"] = route
            if "_route_time_ms" in param_names:
                kwargs["_route_time_ms"] = elapsed_ms

            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
