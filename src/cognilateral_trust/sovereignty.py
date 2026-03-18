"""S9: Sovereignty Gate — D-07 with D-05 welfare hard constraint.

Zero external dependencies.

The sovereignty gate is the executable governance contract that decides
whether an agent may ACT autonomously or must ESCALATE to human review.

D-05 HARD CONSTRAINT: welfare_affected=True + policy.welfare_critical=True
ALWAYS produces ESCALATE, regardless of confidence or any other gate.
This constraint is immutable and must never be bypassed.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any, Callable, Literal


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

Verdict = Literal["ACT", "ESCALATE"]


@dataclass(frozen=True)
class SovereigntyPolicy:
    """Configuration for the sovereignty gate.

    Attributes:
        min_confidence: Minimum confidence to proceed autonomously.
        require_reversible_for_auto: If True, irreversible actions must escalate.
        require_tests_pass: If True, failing tests must escalate.
        welfare_critical: If True, any welfare impact triggers D-05 hard gate.
        max_external_impact: 0 = no external touches allowed; >0 = allowed.
    """

    min_confidence: float
    require_reversible_for_auto: bool
    require_tests_pass: bool
    welfare_critical: bool
    max_external_impact: int


#: Reasonable defaults for autonomous operation.
#: Not too restrictive: clean, reversible, passing-test actions at >=0.5 confidence ACT.
#: External touches are blocked by default (max_external_impact=0).
#: Welfare gate is active by default.
DEFAULT_POLICY = SovereigntyPolicy(
    min_confidence=0.5,
    require_reversible_for_auto=True,
    require_tests_pass=True,
    welfare_critical=True,
    max_external_impact=0,
)


@dataclass(frozen=True)
class SovereigntyDecision:
    """Result of a sovereignty gate evaluation.

    Attributes:
        verdict: "ACT" if autonomous execution is permitted, "ESCALATE" otherwise.
        reasons: Tuple of human-readable reason strings explaining the decision.
        confidence: The confidence value that was evaluated.
        checks_passed: Dict mapping check name to bool result.
    """

    verdict: Verdict
    reasons: tuple[str, ...]
    confidence: float
    checks_passed: dict[str, bool]


# ---------------------------------------------------------------------------
# Sovereignty error (raised by decorator on ESCALATE)
# ---------------------------------------------------------------------------


class SovereigntyError(RuntimeError):
    """Raised by sovereignty_gate decorator when verdict is ESCALATE."""

    def __init__(self, decision: SovereigntyDecision) -> None:
        super().__init__(f"Sovereignty gate blocked execution: verdict={decision.verdict}, reasons={decision.reasons}")
        self.decision = decision


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------


def evaluate_sovereignty(
    confidence: float,
    *,
    policy: SovereigntyPolicy | None = None,
    is_reversible: bool = True,
    tests_passing: bool = True,
    touches_external: bool = False,
    welfare_affected: bool = False,
) -> SovereigntyDecision:
    """Evaluate whether the agent may act autonomously.

    Applies each gate in order. The D-05 welfare gate is checked first
    and is an unconditional hard block when both welfare_affected=True
    and policy.welfare_critical=True.

    Args:
        confidence: Agent's self-reported confidence (0.0–1.0).
        policy: SovereigntyPolicy to use. Defaults to DEFAULT_POLICY.
        is_reversible: True if the action can be undone without human help.
        tests_passing: True if the current test suite is green.
        touches_external: True if the action affects external systems.
        welfare_affected: True if the action may affect human welfare.

    Returns:
        SovereigntyDecision with verdict ACT or ESCALATE and per-check results.
    """
    if policy is None:
        policy = DEFAULT_POLICY

    failed_reasons: list[str] = []
    checks: dict[str, bool] = {}

    # --- D-05 HARD GATE: welfare constraint (IMMUTABLE) ---
    welfare_ok = not (welfare_affected and policy.welfare_critical)
    checks["welfare"] = welfare_ok
    if not welfare_ok:
        failed_reasons.append(
            "D-05 welfare hard gate: welfare_affected=True with welfare_critical policy. "
            "Welfare-critical agents cannot be terminated or modified autonomously. "
            "Human authorization required."
        )
        # Return immediately — no other check can override this
        return SovereigntyDecision(
            verdict="ESCALATE",
            reasons=tuple(failed_reasons),
            confidence=confidence,
            checks_passed=checks,
        )

    # --- Confidence gate ---
    confidence_ok = confidence >= policy.min_confidence
    checks["confidence"] = confidence_ok
    if not confidence_ok:
        failed_reasons.append(f"Confidence {confidence:.3f} below minimum threshold {policy.min_confidence:.3f}")

    # --- Reversibility gate ---
    reversible_ok = (not policy.require_reversible_for_auto) or is_reversible
    checks["reversible"] = reversible_ok
    if not reversible_ok:
        failed_reasons.append("Action is irreversible and policy requires reversibility for autonomous execution")

    # --- Tests gate ---
    tests_ok = (not policy.require_tests_pass) or tests_passing
    checks["tests_passing"] = tests_ok
    if not tests_ok:
        failed_reasons.append("Tests are not passing and policy requires green tests for autonomous execution")

    # --- External impact gate ---
    external_ok = (not touches_external) or (policy.max_external_impact > 0)
    checks["external_impact"] = external_ok
    if not external_ok:
        failed_reasons.append(
            "Action touches external systems and policy disallows external impact "
            f"(max_external_impact={policy.max_external_impact})"
        )

    if failed_reasons:
        return SovereigntyDecision(
            verdict="ESCALATE",
            reasons=tuple(failed_reasons),
            confidence=confidence,
            checks_passed=checks,
        )

    return SovereigntyDecision(
        verdict="ACT",
        reasons=("All sovereignty gates passed",),
        confidence=confidence,
        checks_passed=checks,
    )


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------


def sovereignty_gate(
    confidence: float,
    *,
    policy: SovereigntyPolicy | None = None,
    is_reversible: bool = True,
    tests_passing: bool = True,
    touches_external: bool = False,
    welfare_affected: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that gates function execution on sovereignty evaluation.

    Raises SovereigntyError if verdict is ESCALATE; otherwise executes
    the wrapped function normally.

    Args:
        confidence: Agent confidence for this action.
        policy: SovereigntyPolicy to apply. Defaults to DEFAULT_POLICY.
        is_reversible: True if the action can be undone.
        tests_passing: True if the test suite is green.
        touches_external: True if external systems are affected.
        welfare_affected: True if human welfare may be affected.

    Returns:
        Decorator that raises SovereigntyError on ESCALATE.
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            decision = evaluate_sovereignty(
                confidence,
                policy=policy,
                is_reversible=is_reversible,
                tests_passing=tests_passing,
                touches_external=touches_external,
                welfare_affected=welfare_affected,
            )
            if decision.verdict == "ESCALATE":
                raise SovereigntyError(decision)
            return fn(*args, **kwargs)

        return wrapper

    return decorator
