"""Open consent — end user configures their own trust sensitivity (D5).

The person downstream of an AI decision can set their own threshold
for when actions should escalate. This inverts the usual pattern:
instead of the developer deciding what's "safe enough," the affected
person does.

Usage:
    from cognilateral_trust import ConsentProfile, evaluate_with_consent

    # The end user says: "I want stricter evaluation for medical decisions"
    profile = ConsentProfile(
        min_confidence=0.8,
        always_escalate_irreversible=True,
        require_calibration=True,
    )

    result = evaluate_with_consent(0.7, profile)
    # ESCALATE — user requires 0.8, system has 0.7

The welfare constraint (D-05) still applies unconditionally. Open consent
can only make evaluation stricter, never weaker. The floor is non-negotiable.
"""

from __future__ import annotations

from dataclasses import dataclass

from cognilateral_trust.evaluate import evaluate_trust


@dataclass(frozen=True)
class ConsentProfile:
    """End user's trust sensitivity configuration.

    All fields make evaluation stricter, never weaker.
    The D-05 welfare constraint is always enforced regardless of profile.
    """

    min_confidence: float = 0.0
    always_escalate_irreversible: bool = False
    always_escalate_external: bool = False
    require_calibration: bool = False
    max_tier_for_auto_act: int = 9


@dataclass(frozen=True)
class ConsentResult:
    """Result of a consent-aware trust evaluation."""

    should_proceed: bool
    verdict: str
    consent_override: bool
    override_reason: str | None
    original_verdict: str
    profile: ConsentProfile


DEFAULT_PROFILE = ConsentProfile()


def evaluate_with_consent(
    confidence: float,
    profile: ConsentProfile | None = None,
    *,
    is_reversible: bool = True,
    touches_external: bool = False,
    calibration_accuracy: float | None = None,
) -> ConsentResult:
    """Evaluate trust with end-user consent profile applied.

    The profile can only make evaluation stricter. If the base evaluation
    says ESCALATE, consent cannot override it to ACT.

    Args:
        confidence: Agent confidence (0.0-1.0)
        profile: End user's sensitivity configuration (None = default)
        is_reversible: Whether the action can be undone
        touches_external: Whether the action affects external systems
        calibration_accuracy: Historical calibration accuracy, if known

    Returns:
        ConsentResult with verdict and any consent overrides
    """
    if profile is None:
        profile = DEFAULT_PROFILE

    result = evaluate_trust(
        confidence,
        is_reversible=is_reversible,
        touches_external=touches_external,
    )

    original_verdict = "ACT" if result.should_proceed else "ESCALATE"

    # If base evaluation says ESCALATE, consent cannot weaken it
    if not result.should_proceed:
        return ConsentResult(
            should_proceed=False,
            verdict="ESCALATE",
            consent_override=False,
            override_reason=None,
            original_verdict=original_verdict,
            profile=profile,
        )

    # Apply consent checks — each can only escalate, never permit
    override_reason = None

    if confidence < profile.min_confidence:
        override_reason = f"Below user minimum confidence ({profile.min_confidence})"

    elif profile.always_escalate_irreversible and not is_reversible:
        override_reason = "User requires escalation for irreversible actions"

    elif profile.always_escalate_external and touches_external:
        override_reason = "User requires escalation for external actions"

    elif profile.require_calibration and calibration_accuracy is None:
        override_reason = "User requires calibration data before acting"

    elif hasattr(result.tier, "value") and result.tier.value > profile.max_tier_for_auto_act:
        override_reason = f"Tier {result.tier.value} exceeds user max ({profile.max_tier_for_auto_act})"

    if override_reason:
        return ConsentResult(
            should_proceed=False,
            verdict="ESCALATE",
            consent_override=True,
            override_reason=override_reason,
            original_verdict=original_verdict,
            profile=profile,
        )

    return ConsentResult(
        should_proceed=True,
        verdict="ACT",
        consent_override=False,
        override_reason=None,
        original_verdict=original_verdict,
        profile=profile,
    )


# ---------------------------------------------------------------------------
# D75-W7: Named consent presets (convenience wrappers over ConsentProfile)
# ---------------------------------------------------------------------------

CONSERVATIVE = ConsentProfile(
    min_confidence=0.8,
    always_escalate_irreversible=True,
    always_escalate_external=True,
    require_calibration=True,
    max_tier_for_auto_act=7,
)

BALANCED = ConsentProfile(
    min_confidence=0.5,
    always_escalate_irreversible=True,
    always_escalate_external=False,
    require_calibration=False,
    max_tier_for_auto_act=9,
)

AGGRESSIVE = ConsentProfile(
    min_confidence=0.15,
    always_escalate_irreversible=False,
    always_escalate_external=False,
    require_calibration=False,
    max_tier_for_auto_act=9,
)
