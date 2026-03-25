"""@sovereign_worker decorator — leaf-level self-governance.

Workers that refuse when they don't know. D-05 welfare constraint:
if welfare_relevant=True, min_confidence is ALWAYS 0.9 (hardcoded, immutable).
"""

from __future__ import annotations

import asyncio
import functools
import inspect
from typing import Any, Callable, TypeVar

__all__ = [
    "SovereignRefusal",
    "sovereign_worker",
]

F = TypeVar("F", bound=Callable[..., Any])


class SovereignRefusal(Exception):
    """Raised when a worker refuses execution due to insufficient confidence."""

    pass


def sovereign_worker(
    min_confidence: float = 0.5,
    welfare_relevant: bool = False,
    **_kwargs,
) -> Callable[[F], F]:
    """Decorator: worker self-refuses when confidence is insufficient.

    D-05 welfare constraint: if welfare_relevant=True, min_confidence is
    ALWAYS 0.9 (hardcoded, immutable, no override possible).

    Args:
        min_confidence: Minimum required confidence [0.0, 1.0]
        welfare_relevant: If True, min_confidence forced to 0.9 (D-05 constraint)
        **_kwargs: Ignored (welfare_escalation parameter is not used)

    Returns:
        Decorated function that checks confidence before execution

    Raises:
        SovereignRefusal: If confidence < effective_min_confidence
    """
    # D-05 welfare constraint: HARDCODED, IMMUTABLE
    if welfare_relevant:
        effective_min_confidence = 0.9
    else:
        effective_min_confidence = min_confidence

    def decorator(func: F) -> F:
        # Check if function is async
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                # Extract confidence from args or kwargs using signature binding
                sig = inspect.signature(func)
                try:
                    bound = sig.bind(*args, **kwargs)
                    bound.apply_defaults()
                    confidence = bound.arguments.get("confidence")
                except TypeError:
                    confidence = None

                if confidence is None:
                    msg = f"Function {func.__name__} requires 'confidence' argument"
                    raise ValueError(msg)

                if confidence < effective_min_confidence:
                    msg = f"Insufficient confidence: {confidence:.2f} < {effective_min_confidence:.2f}"
                    raise SovereignRefusal(msg)

                return await func(*args, **kwargs)

            return async_wrapper  # type: ignore

        else:

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                # Extract confidence from args or kwargs using signature binding
                sig = inspect.signature(func)
                try:
                    bound = sig.bind(*args, **kwargs)
                    bound.apply_defaults()
                    confidence = bound.arguments.get("confidence")
                except TypeError:
                    confidence = None

                if confidence is None:
                    msg = f"Function {func.__name__} requires 'confidence' argument"
                    raise ValueError(msg)

                if confidence < effective_min_confidence:
                    msg = f"Insufficient confidence: {confidence:.2f} < {effective_min_confidence:.2f}"
                    raise SovereignRefusal(msg)

                return func(*args, **kwargs)

            return sync_wrapper  # type: ignore

    return decorator
