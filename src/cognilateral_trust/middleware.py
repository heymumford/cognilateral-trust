"""Framework middleware for trust-gated execution — zero dependencies.

Provides:
- TrustEscalation: exception raised when confidence is below threshold
- @trust_gate: decorator that enforces a minimum confidence before execution
- async_evaluate_trust: async wrapper around evaluate_trust (asyncio, no ext deps)
- TrustMiddleware: callable class for pipeline integration patterns

Public API:
    TrustEscalation
    TrustMiddleware
    async_evaluate_trust
    trust_gate
"""

from __future__ import annotations

import asyncio
import functools
from typing import Any, Callable, TypeVar

from cognilateral_trust.accountability import create_accountability_record
from cognilateral_trust.evaluate import TrustEvaluation, evaluate_trust

__all__ = [
    "TrustEscalation",
    "TrustMiddleware",
    "async_evaluate_trust",
    "trust_gate",
]

F = TypeVar("F", bound=Callable[..., Any])

_DEFAULT_MIN_CONFIDENCE = 0.5


# ---------------------------------------------------------------------------
# TrustEscalation exception
# ---------------------------------------------------------------------------


class TrustEscalation(Exception):
    """Raised when a trust gate blocks execution due to insufficient confidence.

    Attributes:
        confidence:          The effective confidence that failed the gate.
        threshold:           The minimum confidence required.
        accountability_record: The accountability record created at escalation.
    """

    def __init__(
        self,
        confidence: float,
        threshold: float,
        *,
        context: dict[str, Any] | None = None,
    ) -> None:
        self.confidence = confidence
        self.threshold = threshold
        ctx = dict(context or {})
        ctx.setdefault("min_confidence_threshold", threshold)
        self.accountability_record = create_accountability_record(
            verdict="ESCALATE",
            reasons=(f"confidence {confidence:.2f} below threshold {threshold:.2f}",),
            context=ctx,
            confidence=confidence,
            confidence_tier=min(9, max(0, int(confidence * 10))),
        )
        super().__init__(f"Trust gate blocked: confidence {confidence:.2f} < threshold {threshold:.2f}")


# ---------------------------------------------------------------------------
# trust_gate decorator
# ---------------------------------------------------------------------------


def trust_gate(
    min_confidence: float = _DEFAULT_MIN_CONFIDENCE,
    *,
    is_reversible: bool = True,
    touches_external: bool = False,
) -> Callable[[F], F]:
    """Decorator factory that gates function execution on a confidence threshold.

    The decorated function must accept a `confidence` keyword argument (float).
    If the confidence is below `min_confidence`, `TrustEscalation` is raised
    with an attached accountability record.

    evaluate_trust() is applied to the confidence value before the threshold
    check — this means sovereignty-grade actions (C7-C9) with is_reversible=False
    or touches_external=True will escalate even if confidence >= min_confidence.

    Usage:
        @trust_gate(min_confidence=0.7)
        def deploy(environment: str, *, confidence: float) -> None:
            ...

        deploy("prod", confidence=0.85)       # proceeds
        deploy("prod", confidence=0.4)        # raises TrustEscalation
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            confidence = kwargs.get("confidence", _DEFAULT_MIN_CONFIDENCE)
            evaluation = evaluate_trust(
                float(confidence),
                is_reversible=is_reversible,
                touches_external=touches_external,
            )
            if not evaluation.should_proceed or float(confidence) < min_confidence:
                raise TrustEscalation(
                    float(confidence),
                    min_confidence,
                    context={
                        "function": func.__name__,
                        "requires_sovereignty_gate": evaluation.requires_sovereignty_gate,
                        "route": evaluation.route,
                    },
                )
            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


# ---------------------------------------------------------------------------
# async_evaluate_trust
# ---------------------------------------------------------------------------


async def async_evaluate_trust(
    confidence: float,
    *,
    is_reversible: bool = True,
    touches_external: bool = False,
    context: dict[str, Any] | None = None,
) -> TrustEvaluation:
    """Async wrapper around evaluate_trust.

    Runs evaluate_trust() in the event loop executor so async callers
    can await it without blocking. Behavior is identical to the sync version.

    Usage:
        result = await async_evaluate_trust(0.75)
        if result.should_proceed:
            await perform_action()
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: evaluate_trust(
            confidence,
            is_reversible=is_reversible,
            touches_external=touches_external,
            context=context,
        ),
    )


# ---------------------------------------------------------------------------
# TrustMiddleware callable class
# ---------------------------------------------------------------------------


class TrustMiddleware:
    """Callable middleware for framework pipeline integration.

    Acts as a gate in any pipeline that passes `confidence` through.
    Can be chained — each gate independently evaluates confidence.

    Raises TrustEscalation when confidence is below threshold.
    Returns TrustEvaluation on success.

    Usage:
        gate = TrustMiddleware(min_confidence=0.7)

        # In a pipeline / request handler:
        evaluation = gate(confidence=0.85)
        # → TrustEvaluation(should_proceed=True, ...)

        evaluation = gate(confidence=0.4)
        # → raises TrustEscalation
    """

    def __init__(
        self,
        min_confidence: float = _DEFAULT_MIN_CONFIDENCE,
        *,
        is_reversible: bool = True,
        touches_external: bool = False,
    ) -> None:
        self.min_confidence = min_confidence
        self.is_reversible = is_reversible
        self.touches_external = touches_external

    def __call__(
        self,
        *,
        confidence: float,
        context: dict[str, Any] | None = None,
    ) -> TrustEvaluation:
        """Evaluate trust. Raises TrustEscalation if blocked."""
        evaluation = evaluate_trust(
            confidence,
            is_reversible=self.is_reversible,
            touches_external=self.touches_external,
            context=context,
        )
        if not evaluation.should_proceed or confidence < self.min_confidence:
            raise TrustEscalation(
                confidence,
                self.min_confidence,
                context=dict(context or {}),
            )
        return evaluation
