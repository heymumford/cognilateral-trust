"""Lifecycle decorators for trust-gated agent spawn and termination.

Provides:
- @spawn_gate: decorator that evaluates TrustContext before allowing agent creation
- @handoff_trust: decorator that transfers TrustPassport across agent handoffs
- @kill_warrant: decorator that checks trust obligations before termination
- suspend_trust/wake_trust: serialization for passport suspension/restoration
- TerminationBlocked: exception raised when an agent cannot be terminated

Zero external dependencies.
"""

from __future__ import annotations

import functools
import inspect
import json
from datetime import datetime, timezone
from typing import Any, Callable, Literal, TypeVar

from cognilateral_trust.core import PassportStamp, TrustPassport
from cognilateral_trust.middleware import TrustEscalation

__all__ = ["spawn_gate", "handoff_trust", "kill_warrant", "suspend_trust", "wake_trust", "TerminationBlocked"]

F = TypeVar("F", bound=Callable[..., Any])

# Tier name to confidence threshold mapping
_TIER_THRESHOLDS = {
    "basic": 0.0,
    "observed": 0.3,
    "verified": 0.5,
    "governance": 0.7,
}

# Welfare constraint threshold (D-05)
WELFARE_THRESHOLD = 0.7


class TerminationBlocked(Exception):
    """Raised when an agent cannot be terminated.

    Attributes:
        reason: Brief explanation of why termination is blocked.
        agent_id: Optional identifier of the agent being terminated.
        unresolved: Tuple of unresolved obligation descriptions (if applicable).
    """

    def __init__(self, reason: str, agent_id: str | None = None, unresolved: tuple[str, ...] = ()) -> None:
        self.reason = reason
        self.agent_id = agent_id
        self.unresolved = unresolved
        super().__init__(reason)


def spawn_gate(
    min_tier: Literal["basic", "observed", "verified", "governance"] = "verified",
) -> Callable[[F], F]:
    """Decorator factory that gates function execution on trust context evaluation.

    The decorated function may accept a `trust_context` keyword argument (TrustContext | None).
    If trust_context is provided, its confidence is evaluated against the tier threshold
    and welfare constraint (D-05).

    If no trust_context is provided, the function proceeds (backward compatibility).

    Welfare constraint (D-05): If welfare_relevant=True, confidence must be >= 0.7
    to proceed, regardless of min_tier.

    Tier thresholds:
    - "basic": 0.0 (always proceeds, unless welfare gate blocks)
    - "observed": 0.3
    - "verified": 0.5
    - "governance": 0.7

    Usage:
        @spawn_gate(min_tier="verified")
        def create_worker(trust_context: TrustContext | None = None) -> Worker:
            ...

        ctx = TrustContext(confidence=0.65)
        worker = create_worker(trust_context=ctx)  # proceeds

        ctx = TrustContext(confidence=0.4)
        worker = create_worker(trust_context=ctx)  # raises TrustEscalation
    """
    threshold = _TIER_THRESHOLDS.get(min_tier)
    if threshold is None:
        msg = f"Invalid min_tier: {min_tier}. Must be one of {list(_TIER_THRESHOLDS.keys())}"
        raise ValueError(msg)

    def decorator(func: F) -> F:
        def _enforce_trust_gate(trust_context: Any) -> None:
            """Shared trust gate logic for both sync and async wrappers.

            If no trust_context is provided, returns early (backward compatible).
            Applies welfare gate (D-05): hard floor at WELFARE_THRESHOLD if welfare_relevant.
            Raises TrustEscalation when the effective confidence is below the gate.
            """
            # If no trust_context provided, proceed (backward compat)
            if trust_context is None:
                return

            # Allow trust_context to omit attributes gracefully; fall back to sane defaults
            confidence = getattr(trust_context, "confidence", None)
            welfare_relevant = getattr(trust_context, "welfare_relevant", False)

            # Apply welfare gate (D-05): hard floor at WELFARE_THRESHOLD if welfare_relevant
            effective_threshold = threshold
            if welfare_relevant and effective_threshold is not None:
                effective_threshold = max(effective_threshold, WELFARE_THRESHOLD)

            # If threshold is somehow still None, we treat it as "no gating"
            if effective_threshold is None:
                return

            # If we can't read confidence, or it's below threshold, escalate
            if confidence is None or confidence < effective_threshold:
                gate_name = "welfare" if welfare_relevant else "spawn"
                gate_context: dict[str, Any] = {
                    "gate": gate_name,
                    "welfare_relevant": welfare_relevant,
                }
                if gate_name == "spawn":
                    gate_context["min_tier"] = min_tier

                raise TrustEscalation(
                    confidence=confidence or 0.0,
                    threshold=effective_threshold,
                    context=gate_context,
                )

        # Detect if function is async
        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                trust_context = kwargs.get("trust_context")
                _enforce_trust_gate(trust_context)
                return await func(*args, **kwargs)

            return async_wrapper  # type: ignore

        else:

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                trust_context = kwargs.get("trust_context")
                _enforce_trust_gate(trust_context)
                return func(*args, **kwargs)

            return sync_wrapper  # type: ignore

    return decorator


# ============================================================================
# Step 07: @handoff_trust Decorator
# ============================================================================


def handoff_trust(func: F) -> F:
    """Decorator: transfers TrustPassport across agent handoffs.

    Return-shape behavior:
    - If no `passport` kwarg is provided, the function executes normally and
      returns exactly whatever `func` returns (fully backward compatible).
    - If a `passport` kwarg is provided, the decorator always returns a
      2-tuple: `(result, new_passport)`, where `result` is the original
      return value from `func` and `new_passport` is the updated passport.

    This avoids overloading tuple semantics and prevents silent loss of the
    original result.

    Works on both sync and async functions.
    Stamps accumulate across chained handoffs.

    Usage:
        @handoff_trust
        def transfer_to_worker(task: str, passport: TrustPassport | None = None):
            return task

        ctx = TrustContext(confidence=0.8)
        passport = TrustPassport(context=ctx, agent_id="a1", created_at="...")
        task_result, new_passport = transfer_to_worker("task_data", passport=passport)
        # new_passport now has a handoff stamp appended
    """

    def _create_handoff_stamp(passport: TrustPassport) -> PassportStamp:
        """Create a handoff stamp with current timestamp."""
        return PassportStamp(
            agent_id=passport.agent_id,
            action="handoff",
            confidence_at_action=passport.context.confidence,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def _process_handoff(passport: TrustPassport | None, result: Any) -> Any:
        """Process passport stamping and return shape.

        If passport is None, return result unchanged.
        If passport is provided, always return (result, new_passport) tuple.
        """
        if passport is None:
            return result
        stamp = _create_handoff_stamp(passport)
        new_passport = passport.add_stamp(stamp)
        return (result, new_passport)

    # Detect if function is async
    if inspect.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            passport = kwargs.get("passport")
            result = await func(*args, **kwargs)
            return _process_handoff(passport, result)

        return async_wrapper  # type: ignore

    else:

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            passport = kwargs.get("passport")
            result = func(*args, **kwargs)
            return _process_handoff(passport, result)

        return sync_wrapper  # type: ignore


# ============================================================================
# Step 08: @kill_warrant Decorator
# ============================================================================


def kill_warrant(func: F) -> F:
    """Decorator: evaluates trust obligations before agent termination.

    D-05 welfare constraint: if passport.context.welfare_relevant is True,
    termination is ALWAYS blocked. This is HARDCODED. No config. No override.
    No parameter. Immutable.

    For non-welfare agents: checks if passport has unresolved evaluations
    (stamps where action is not a resolution action and no subsequent resolution).

    Resolution actions include: "terminate", "resolved", "completed"

    Works on both sync and async functions.

    Usage:
        @kill_warrant
        def terminate_agent(passport: TrustPassport | None = None):
            return "terminated"

        # Will raise TerminationBlocked if welfare_relevant=True
        # or if unresolved obligations exist
    """

    # Resolution actions that clear obligations
    _RESOLUTION_ACTIONS = {"terminate", "resolved", "completed"}

    def _check_unresolved_obligations(passport: TrustPassport) -> tuple[bool, tuple[str, ...]]:
        """Check if passport has unresolved obligations.

        Returns (is_blocked, unresolved_list).
        """
        if not passport.stamps:
            return False, ()

        unresolved = []
        for i, stamp in enumerate(passport.stamps):
            if stamp.action in _RESOLUTION_ACTIONS:
                continue  # Resolution action clears all prior obligations
            # Check if there's a subsequent resolution action
            has_resolution = any(s.action in _RESOLUTION_ACTIONS for s in passport.stamps[i + 1 :])
            if not has_resolution:
                unresolved.append(f"{stamp.action} at {stamp.timestamp}")

        return bool(unresolved), tuple(unresolved)

    # Detect if function is async
    if inspect.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            passport = kwargs.get("passport")

            # If no passport provided, proceed (backward compat)
            if passport is None:
                return await func(*args, **kwargs)

            # D-05: welfare_relevant is ALWAYS a hard block
            welfare_relevant = getattr(passport.context, "welfare_relevant", False)
            if welfare_relevant:
                reason = "D-05 welfare constraint: agent cannot be terminated because welfare_relevant=True"
                raise TerminationBlocked(reason=reason, agent_id=passport.agent_id, unresolved=())

            # Check for unresolved obligations
            is_blocked, unresolved = _check_unresolved_obligations(passport)
            if is_blocked:
                reason = f"Agent has {len(unresolved)} unresolved obligation(s)"
                raise TerminationBlocked(reason=reason, agent_id=passport.agent_id, unresolved=unresolved)

            # All checks passed
            return await func(*args, **kwargs)

        return async_wrapper  # type: ignore

    else:

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            passport = kwargs.get("passport")

            # If no passport provided, proceed (backward compat)
            if passport is None:
                return func(*args, **kwargs)

            # D-05: welfare_relevant is ALWAYS a hard block
            welfare_relevant = getattr(passport.context, "welfare_relevant", False)
            if welfare_relevant:
                reason = "D-05 welfare constraint: agent cannot be terminated because welfare_relevant=True"
                raise TerminationBlocked(reason=reason, agent_id=passport.agent_id, unresolved=())

            # Check for unresolved obligations
            is_blocked, unresolved = _check_unresolved_obligations(passport)
            if is_blocked:
                reason = f"Agent has {len(unresolved)} unresolved obligation(s)"
                raise TerminationBlocked(reason=reason, agent_id=passport.agent_id, unresolved=unresolved)

            # All checks passed
            return func(*args, **kwargs)

        return sync_wrapper  # type: ignore


# ============================================================================
# Step 09: Suspend/Wake + Decorator Composition
# ============================================================================


def suspend_trust(passport: TrustPassport) -> str:
    """Serialize passport for agent suspension. Returns JSON string."""
    data = passport.serialize()
    return json.dumps(data)


def wake_trust(data: str) -> TrustPassport:
    """Restore passport from suspension. Returns TrustPassport."""
    parsed = json.loads(data)
    return TrustPassport.restore(parsed)
